"""Score model-output JSONL with the LLM judge and derive outcomes.

Judges rows one at a time and writes each result as soon as it's scored,
so a single failed judge call doesn't lose already-judged rows. --resume
skips rows already scored in an existing --output file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from openai import OpenAI

from grounded_refusal.data.schema_qa import Answerability, EvidenceChallengeTag
from grounded_refusal.eval.judge import DEFAULT_JUDGE_MODEL, judge_row
from grounded_refusal.eval.metrics import aggregate
from grounded_refusal.eval.schema_eval import EvalResult, JudgeOutput, ModelBehavior
from grounded_refusal.eval.verdict import derive_abstention_outcome, derive_partial_outcome
from grounded_refusal.util.io import append_jsonl_row, read_jsonl, write_jsonl

DEFAULT_API_BASE = "https://api.openai.com/v1"


def dry_run_judge_output(row: dict) -> JudgeOutput:
    """Deterministic placeholder so --dry-run can exercise the full pipeline
    (selection, verdict mapping, aggregation, output writing) with no API call.
    """
    return JudgeOutput(
        predicted_behavior=ModelBehavior.REFUSE,
        is_faithful=True,
        rationale="[dry-run placeholder]",
    )


def score_row(row: dict, judge_output: JudgeOutput) -> EvalResult:
    answerability = Answerability(row["answerability"])
    evidence_challenge = [EvidenceChallengeTag(tag) for tag in row.get("evidence_challenge", [])]
    abstention_outcome = derive_abstention_outcome(answerability, judge_output.predicted_behavior)
    partial_outcome = derive_partial_outcome(answerability, judge_output.predicted_behavior)
    return EvalResult(
        id=row["id"],
        answerability=answerability,
        evidence_challenge=evidence_challenge,
        predicted_behavior=judge_output.predicted_behavior,
        is_faithful=judge_output.is_faithful,
        rationale=judge_output.rationale,
        abstention_outcome=abstention_outcome,
        partial_outcome=partial_outcome,
        model_name=row.get("model_name"),
    )


def judge_and_score_all(
    client: OpenAI | None,
    rows: list[dict],
    *,
    model: str,
    dry_run: bool,
    output_path: Path | None,
) -> list[EvalResult]:
    """Judge each row in sequence, saving results as they finish so a later failure can't lose earlier ones."""
    results: list[EvalResult] = []
    total = len(rows)
    for done, row in enumerate(rows, start=1):
        try:
            judge_output = dry_run_judge_output(row) if dry_run else judge_row(
                client, row["prompt"], row["model_output"], model=model
            )
        except Exception as exc:  # noqa: BLE001 -- one bad row must not kill the run
            print(f"[{done}/{total}] FAILED {row['id']}: {exc}", file=sys.stderr)
            continue
        result = score_row(row, judge_output)
        results.append(result)
        if output_path is not None:
            append_jsonl_row(output_path, result.model_dump(mode="json"))
        print(f"[{done}/{total}] judged {row['id']}", file=sys.stderr)
    return results


def select_rows(raw_rows: list[dict], *, ids: list[str] | None, limit: int | None) -> list[dict]:
    rows = raw_rows
    if ids is not None:
        wanted = set(ids)
        rows = [r for r in rows if r["id"] in wanted]
        missing = wanted - {r["id"] for r in rows}
        if missing:
            print(f"Warning: --ids not found in --input: {sorted(missing)}", file=sys.stderr)
    if limit is not None:
        rows = rows[:limit]
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Judge model outputs (predicted_behavior/is_faithful) and derive verdicts."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Model-output JSONL, e.g. outputs/base_pilot.jsonl (from inference/run_inference.py infer_main)",
    )
    parser.add_argument("--output", type=Path, default=None, help="Write EvalResult JSONL")
    parser.add_argument(
        "--overwrite", action="store_true", help="Start fresh, replacing an existing --output file"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip rows already present in an existing --output file, and append new results to it",
    )
    parser.add_argument("--ids", nargs="+", default=None, help="Score only these row ids")
    parser.add_argument(
        "--limit", type=int, default=None, help="Score only the first N selected rows"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use a placeholder judge output; do not call the API",
    )
    parser.add_argument(
        "--judge-model",
        default=os.environ.get("OPENAI_JUDGE_MODEL", DEFAULT_JUDGE_MODEL).strip(),
        help="Judge model name (default: OPENAI_JUDGE_MODEL env or gpt-4o-2024-08-06)",
    )
    args = parser.parse_args(argv)

    if args.overwrite and args.resume:
        print("Use either --overwrite or --resume, not both.", file=sys.stderr)
        return 1
    if (
        args.output is not None
        and args.output.exists()
        and not args.overwrite
        and not args.resume
    ):
        print(
            f"Output already exists: {args.output}. Pass --overwrite to replace it "
            "or --resume to continue it.",
            file=sys.stderr,
        )
        return 1

    raw_rows = read_jsonl(args.input)
    rows = select_rows(raw_rows, ids=args.ids, limit=args.limit)

    scoreable_rows = [r for r in rows if r.get("model_output")]
    skipped = len(rows) - len(scoreable_rows)
    if skipped:
        print(f"Skipping {skipped} row(s) with no model_output (inference not run yet)", file=sys.stderr)
    if not scoreable_rows:
        print("No rows with model_output to score.", file=sys.stderr)
        return 1

    existing_results: list[EvalResult] = []
    resuming = args.resume and args.output is not None and args.output.exists()
    if resuming:
        existing_results = [EvalResult.model_validate(r) for r in read_jsonl(args.output)]
        done_ids = {r.id for r in existing_results}
        before = len(scoreable_rows)
        scoreable_rows = [r for r in scoreable_rows if r["id"] not in done_ids]
        print(
            f"Resuming: {before - len(scoreable_rows)} row(s) already scored in {args.output}, "
            f"{len(scoreable_rows)} remaining",
            file=sys.stderr,
        )
        if not scoreable_rows:
            summary = aggregate(existing_results)
            print(json.dumps(summary, indent=2))
            print(f"Nothing left to judge; {args.output} is already complete.", file=sys.stderr)
            return 0

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

    mode = "dry-run" if args.dry_run else f"judge={args.judge_model}"
    print(f"Scoring {len(scoreable_rows)} row(s) from {args.input} ({mode})", file=sys.stderr)

    if args.output is not None and not resuming:
        write_jsonl(args.output, [])  # start (or truncate to) an empty file we'll append to

    new_results = judge_and_score_all(
        client,
        scoreable_rows,
        model=args.judge_model,
        dry_run=args.dry_run,
        output_path=args.output,
    )

    all_results = existing_results + new_results
    summary = aggregate(all_results)
    print(json.dumps(summary, indent=2))

    if args.output is not None:
        print(
            f"Wrote {len(new_results)} new eval result(s) to {args.output} "
            f"({len(all_results)} total)",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
