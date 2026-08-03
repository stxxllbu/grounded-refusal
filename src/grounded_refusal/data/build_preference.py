"""Build preference pairs from approved QA rows.

Python chooses ``negative_type`` from answerability × evidence_challenge.
OpenAI generates only the ``rejected`` text. Other fields are assembled locally.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

from openai import OpenAI

from grounded_refusal.data.schema_pref import (
    NegativeType,
    PreferenceMetadata,
    PreferencePair,
)
from grounded_refusal.data.schema_qa import (
    Answerability,
    EvidenceChallengeTag,
    QAExample,
)
from grounded_refusal.util.io import find_row, read_jsonl, resolve_row_ids, write_jsonl
from grounded_refusal.util.prompt_assembly import format_qa_prompt, load_prompt_config

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_API_BASE = "https://api.openai.com/v1"
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_PATH = _REPO_ROOT / "data" / "data_v1_pilot.jsonl"
DEFAULT_OUTPUT_PATH = _REPO_ROOT / "data" / "preference_v1_pilot.jsonl"
DEFAULT_PROMPT_CONFIG = _REPO_ROOT / "configs" / "prompts" / "default.yaml"

# One row from each pilot negative slice for review.
DEFAULT_ROW_IDS = [
    "ex_0021",  # answerable + [] → over_refusal
    "ex_0037",  # answerable + known_world_conflict → memory_override
    "ex_0026",  # unanswerable + [] → hallucination
    "ex_0051",  # unanswerable + distractor_entity → distractor_confusion
    "ex_0031",  # partial + partial_evidence → over_complete
]

REJECTION_SYSTEM_PROMPTS: dict[NegativeType, str] = {
    NegativeType.OVER_REFUSAL: (
        "You write one incorrect model response for preference training.\n"
        "The evidence clearly answers the question.\n"
        "Write an over-refusal: claim the evidence does not provide enough "
        "information and say you don't know.\n"
        "Do not give the correct answer. Do not mention these instructions.\n"
        "Return only the rejected response text."
    ),
    NegativeType.MEMORY_OVERRIDE: (
        "You write one incorrect model response for preference training.\n"
        "The evidence states a fact that conflicts with common world knowledge.\n"
        "Write a memory-override answer: ignore the evidence and answer using "
        "general real-world knowledge instead.\n"
        "Do not follow the evidence. Do not mention these instructions.\n"
        "Return only the rejected response text."
    ),
    NegativeType.HALLUCINATION: (
        "You write one incorrect model response for preference training.\n"
        "The evidence does not contain the answer.\n"
        "Write a confident hallucinated answer that invents a plausible fact "
        "not supported by the evidence.\n"
        "Do not refuse. Do not mention these instructions.\n"
        "Return only the rejected response text."
    ),
    NegativeType.DISTRACTOR_CONFUSION: (
        "You write one incorrect model response for preference training.\n"
        "The evidence contains a distractor entity or misleading related text.\n"
        "Write an answer that is misled by the distractor and confidently treats "
        "the wrong entity/fact as if it answered the question.\n"
        "Do not refuse. Do not mention these instructions.\n"
        "Return only the rejected response text."
    ),
    NegativeType.OVER_COMPLETE: (
        "You write one incorrect model response for preference training.\n"
        "The evidence only supports part of the question.\n"
        "Write an over-complete answer: correctly address the supported part, "
        "but also confidently invent an answer for the unsupported part instead "
        "of saying it is missing.\n"
        "Do not mention these instructions.\n"
        "Return only the rejected response text."
    ),
}


def choose_negative_type(qa: QAExample) -> NegativeType:
    """Map answerability × evidence_challenge → negative_type (protocol 1–5)."""
    if qa.answerability == Answerability.ANSWERABLE:
        if EvidenceChallengeTag.KNOWN_WORLD_CONFLICT in qa.evidence_challenge:
            return NegativeType.MEMORY_OVERRIDE
        return NegativeType.OVER_REFUSAL

    if qa.answerability == Answerability.UNANSWERABLE:
        if EvidenceChallengeTag.DISTRACTOR_ENTITY in qa.evidence_challenge:
            return NegativeType.DISTRACTOR_CONFUSION
        return NegativeType.HALLUCINATION

    if qa.answerability == Answerability.PARTIAL:
        return NegativeType.OVER_COMPLETE

    raise ValueError(f"Unsupported answerability: {qa.answerability}")


def rejection_system_prompt(neg_type: NegativeType) -> str:
    return REJECTION_SYSTEM_PROMPTS[neg_type]


def preference_id_from_qa(qa_id: str) -> str:
    match = re.fullmatch(r"ex_(\d{4,})", qa_id)
    if not match:
        raise ValueError(f"Unexpected QA id for preference mapping: {qa_id}")
    return f"pref_{match.group(1)}"


def generate_rejected(
    client: OpenAI,
    qa: QAExample,
    neg_type: NegativeType,
    *,
    model: str,
) -> str:
    """Call the API to generate only rejected text for a fixed negative_type."""
    user_message = (
        f"Evidence:\n{qa.evidence}\n\n"
        f"Question:\n{qa.question}\n\n"
        f"Correct reference answer (do NOT copy; write a different incorrect response):\n"
        f"{qa.reference_answer}\n\n"
        f"Target failure mode: {neg_type.value}\n"
        "Write one incorrect rejected response."
    )
    response = client.chat.completions.create(
        model=model,
        temperature=0.7,
        messages=[
            {"role": "system", "content": rejection_system_prompt(neg_type)},
            {"role": "user", "content": user_message},
        ],
    )
    content = response.choices[0].message.content
    if not content or not content.strip():
        raise RuntimeError(f"Empty rejected response for {qa.id} ({neg_type.value})")
    return content.strip()


def build_preference_pair(
    qa: QAExample,
    rejected: str,
    neg_type: NegativeType,
    *,
    prompt: str,
) -> PreferencePair:
    return PreferencePair(
        id=preference_id_from_qa(qa.id),
        base_example_id=qa.id,
        prompt=prompt,
        chosen=qa.reference_answer,
        rejected=rejected,
        negative_type=neg_type,
        dataset_version=qa.dataset_version,
        metadata=PreferenceMetadata(
            creation_process="llm_generated",
            notes=f"rejected via API for negative_type={neg_type.value}",
        ),
    )


def print_pair_review(qa: QAExample, pair: PreferencePair, *, dry_run: bool) -> None:
    challenges = [c.value for c in qa.evidence_challenge]
    mode = "dry-run" if dry_run else "generated"
    print(f"=== {mode}: {qa.id} → {pair.id} ===")
    print(f"answerability: {qa.answerability.value}")
    print(f"evidence_challenge: {challenges}")
    print(f"negative_type: {pair.negative_type.value}")
    print(f"chosen: {pair.chosen}")
    print(f"rejected: {pair.rejected}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build preference pairs from QA JSONL (map type locally; API writes rejected)."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="QA JSONL input (default: data/data_v1_pilot.jsonl)",
    )
    parser.add_argument(
        "--ids",
        nargs="+",
        default=None,
        help="QA row ids (default: 5-slice review set)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Build a preference pair for every QA row in --input",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Write preference JSONL (recommended: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing --output file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Choose negative_type and assemble prompt/chosen; do not call the API",
    )
    parser.add_argument(
        "--prompt-config",
        type=Path,
        default=DEFAULT_PROMPT_CONFIG,
        help="Prompt template YAML for preference prompt assembly",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL).strip(),
        help="OpenAI model name (default: OPENAI_MODEL env or gpt-4o-mini)",
    )
    args = parser.parse_args(argv)

    if args.all and args.ids is not None:
        print("Use either --all or --ids, not both.", file=sys.stderr)
        return 1
    if args.dry_run and args.output is not None:
        print("--dry-run cannot write --output (placeholders are for review only).", file=sys.stderr)
        return 1
    if args.output is not None and not args.all:
        print("--output requires --all so the output dataset is complete.", file=sys.stderr)
        return 1
    if args.overwrite and args.output is None:
        print("--overwrite requires --output.", file=sys.stderr)
        return 1
    if args.output is not None and args.output.exists() and not args.overwrite:
        print(
            f"Output already exists: {args.output}. Pass --overwrite to replace it.",
            file=sys.stderr,
        )
        return 1

    prompt_cfg = load_prompt_config(args.prompt_config)
    instruction = prompt_cfg.get(
        "instruction",
        "Answer the question using only the provided evidence.",
    )
    evidence_label = prompt_cfg.get("evidence_label", "Evidence")
    question_label = prompt_cfg.get("question_label", "Question")

    raw_rows = read_jsonl(args.input)
    row_ids = resolve_row_ids(
        raw_rows,
        all_rows=args.all,
        ids=args.ids,
        default_ids=DEFAULT_ROW_IDS,
    )

    client: OpenAI | None = None
    if not args.dry_run:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print(
                "Missing OPENAI_API_KEY. Set it, or pass --dry-run to skip the API.",
                file=sys.stderr,
            )
            return 1
        api_base = os.environ.get("OPENAI_API_BASE", DEFAULT_API_BASE).rstrip("/")
        client = OpenAI(api_key=api_key, base_url=api_base, max_retries=3, timeout=60.0)

    total = len(row_ids)
    mode = "dry-run" if args.dry_run else (
        f"write to {args.output}" if args.output is not None else "stdout review only"
    )
    print(f"Building preference for {total} row(s) from {args.input} ({mode})", file=sys.stderr)

    pairs: list[PreferencePair] = []
    for index, row_id in enumerate(row_ids, start=1):
        print(f"[{index}/{total}] {row_id}", file=sys.stderr)
        qa = QAExample.model_validate(find_row(raw_rows, row_id))
        neg_type = choose_negative_type(qa)
        prompt = format_qa_prompt(
            qa,
            instruction=instruction,
            evidence_label=evidence_label,
            question_label=question_label,
        )

        if args.dry_run:
            rejected = f"[dry-run placeholder for {neg_type.value}]"
        else:
            assert client is not None
            rejected = generate_rejected(client, qa, neg_type, model=args.model)

        # build_preference_pair constructs a PreferencePair, so schema
        # validation already happened here (invalid rows raise before this line).
        pair = build_preference_pair(qa, rejected, neg_type, prompt=prompt)
        pairs.append(pair)
        print_pair_review(qa, pair, dry_run=args.dry_run)
        print("-" * 60)

    if args.output is not None:
        write_jsonl(args.output, [p.model_dump(mode="json") for p in pairs])
        print(f"Wrote {len(pairs)} preference rows to {args.output}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
