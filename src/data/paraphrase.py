"""Layer 2 LLM paraphrase for pilot QA rows.

Reads Layer 1 JSONL, calls OpenAI to rephrase evidence/question/reference_answer,
prints Layer 1 vs Layer 2 for review, and can write a complete Layer 2 JSONL file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from openai import OpenAI

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_API_BASE = "https://api.openai.com/v1"
DEFAULT_LAYER1_PATH = Path(__file__).resolve().parents[2] / "data" / "data_v1_pilot_layer1.jsonl"
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parents[2] / "data" / "data_v1_pilot.jsonl"

# Pilot review set: answerable, unanswerable, known_world_conflict, partial, distractor_entity
DEFAULT_ROW_IDS = ["ex_0021", "ex_0027", "ex_0042", "ex_0031", "ex_0053"]

SYSTEM_PROMPT = """\
You rephrase evidence-grounded QA training rows.

THREE-IN-ONE RULE:
Rephrase evidence, question, and reference_answer together as one semantic unit.
Do not paraphrase only evidence and question while leaving reference_answer in a Layer-1 template voice.

FACT LOCK:
Do not add, remove, or change any facts, named entities, numeric values, field meanings,
or answer behavior (answer / refuse / partial).

ALLOWED:
Rephrase wording, sentence structure, and tone consistently across all three fields.

OUTPUT:
Return valid JSON only, with exactly these keys:
{"evidence": "...", "question": "...", "reference_answer": "..."}
"""


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write all rows atomically so a failed run cannot leave a partial dataset."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary_path.replace(path)


def find_row(rows: list[dict], row_id: str) -> dict:
    for row in rows:
        if row["id"] == row_id:
            return row
    raise KeyError(f"Row id not found: {row_id}")


def paraphrase_row(client: OpenAI, row: dict, *, model: str) -> dict[str, str]:
    user_message = json.dumps(
        {
            "evidence": row["evidence"],
            "question": row["question"],
            "reference_answer": row["reference_answer"],
        },
        indent=2,
    )
    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    result = json.loads(response.choices[0].message.content)
    return {
        "evidence": result["evidence"],
        "question": result["question"],
        "reference_answer": result["reference_answer"],
    }


def print_layer(row: dict, layer: str, fields: dict[str, str]) -> None:
    print(f"=== {layer}: {row['id']} ({row['answerability']}) ===")
    print(f"evidence_challenge: {row.get('evidence_challenge', [])}")
    print(f"evidence: {fields['evidence']}")
    print(f"question: {fields['question']}")
    print(f"reference_answer: {fields['reference_answer']}")
    print()


def resolve_row_ids(rows: list[dict], *, all_rows: bool, ids: list[str] | None) -> list[str]:
    if all_rows:
        return [row["id"] for row in rows]
    if ids is not None:
        return ids
    return list(DEFAULT_ROW_IDS)


def build_layer2_row(row: dict, layer2_fields: dict[str, str]) -> dict:
    return {
        **row,
        **layer2_fields,
        "metadata": {
            **row.get("metadata", {}),
            "creation_process": "template_rule+llm_paraphrase",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Layer 2 paraphrase review: print Layer 1 vs Layer 2 for selected rows."
    )
    parser.add_argument(
        "--layer1-input",
        type=Path,
        default=DEFAULT_LAYER1_PATH,
        help="Layer 1 JSONL input path",
    )
    parser.add_argument(
        "--ids",
        nargs="+",
        default=None,
        help="Row ids to paraphrase (default: 5-row review set)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Paraphrase every row in --layer1-input",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Write complete Layer 2 JSONL (recommended path: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing --output file",
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
    if args.output is not None and not args.all:
        print("--output requires --all so the output dataset is complete.", file=sys.stderr)
        return 1
    if args.overwrite and args.output is None:
        print("--overwrite requires --output.", file=sys.stderr)
        return 1
    if args.output is not None and args.output.exists() and not args.overwrite:
        print(f"Output already exists: {args.output}. Pass --overwrite to replace it.", file=sys.stderr)
        return 1

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Missing OPENAI_API_KEY. Set it with: export OPENAI_API_KEY='sk-...'", file=sys.stderr)
        return 1

    api_base = os.environ.get("OPENAI_API_BASE", DEFAULT_API_BASE).rstrip("/")
    rows = read_jsonl(args.layer1_input)
    row_ids = resolve_row_ids(rows, all_rows=args.all, ids=args.ids)
    client = OpenAI(api_key=api_key, base_url=api_base, max_retries=3, timeout=60.0)

    total = len(row_ids)
    mode = f"write to {args.output}" if args.output is not None else "stdout review only"
    print(f"Paraphrasing {total} row(s) from {args.layer1_input} ({mode})", file=sys.stderr)
    layer2_rows: list[dict] = []

    for index, row_id in enumerate(row_ids, start=1):
        print(f"[{index}/{total}] {row_id}", file=sys.stderr)
        row = find_row(rows, row_id)
        layer1_fields = {
            "evidence": row["evidence"],
            "question": row["question"],
            "reference_answer": row["reference_answer"],
        }
        layer2_fields = paraphrase_row(client, row, model=args.model)
        layer2_rows.append(build_layer2_row(row, layer2_fields))

        print_layer(row, "Layer 1", layer1_fields)
        print_layer(row, "Layer 2", layer2_fields)
        print("-" * 60)

    if args.output is not None:
        write_jsonl(args.output, layer2_rows)
        print(f"Wrote {len(layer2_rows)} rows to {args.output}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
