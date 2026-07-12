"""Layer 2 LLM paraphrase for pilot QA rows.

Reads Layer 1 JSONL, calls OpenAI to rephrase evidence/question/reference_answer,
and prints Layer 1 vs Layer 2 to stdout for human review. Does not write output files.
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
        default=DEFAULT_ROW_IDS,
        help="Row ids to paraphrase",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL).strip(),
        help="OpenAI model name (default: OPENAI_MODEL env or gpt-4o-mini)",
    )
    args = parser.parse_args(argv)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Missing OPENAI_API_KEY. Set it with: export OPENAI_API_KEY='sk-...'", file=sys.stderr)
        return 1

    api_base = os.environ.get("OPENAI_API_BASE", DEFAULT_API_BASE).rstrip("/")
    rows = read_jsonl(args.layer1_input)
    client = OpenAI(api_key=api_key, base_url=api_base, max_retries=3, timeout=60.0)

    for row_id in args.ids:
        row = find_row(rows, row_id)
        layer1_fields = {
            "evidence": row["evidence"],
            "question": row["question"],
            "reference_answer": row["reference_answer"],
        }
        layer2_fields = paraphrase_row(client, row, model=args.model)

        print_layer(row, "Layer 1", layer1_fields)
        print_layer(row, "Layer 2", layer2_fields)
        print("-" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
