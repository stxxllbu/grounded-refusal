"""Validate QA JSONL rows against the Pydantic QA schema.

Checks field shape and enums via ``QAExample``. Does not check Layer-1 vs
Layer-2 fact lock, answer correctness, or paraphrase quality.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from data.schema_qa import QAExample
from util.io import read_jsonl


def validate_qa_jsonl_against_schema(
    path: str | Path,
) -> tuple[list[QAExample], list[str]]:
    """Return parsed QA examples and a list of schema-contract errors.

    Empty errors means every row matches the ``QAExample`` Pydantic model.
    """
    rows = read_jsonl(path)
    parsed: list[QAExample] = []
    errors: list[str] = []
    for idx, row in enumerate(rows, start=1):
        try:
            parsed.append(QAExample.model_validate(row))
        except ValidationError as exc:
            errors.append(f"row {idx}: {exc}")
    return parsed, errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a QA JSONL file against the QAExample schema."
    )
    parser.add_argument("path", type=Path, help="Path to QA .jsonl file")
    args = parser.parse_args(argv)

    parsed, errors = validate_qa_jsonl_against_schema(args.path)
    if errors:
        print(f"Validation failed for {args.path}:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print(f"OK: {len(parsed)} QA rows match schema contract ({args.path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
