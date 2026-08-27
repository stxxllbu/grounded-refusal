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
    scored_results: list[EvalResult] = []
    total_rows = len(rows)
    for index, row in enumerate(rows, start=1):
        try:
            judge_output = dry_run_judge_output(row) if dry_run else judge_row(
                client, row["prompt"], row["model_output"], model=model
            )
        except Exception as exc:  # noqa: BLE001 -- one bad row must not kill the run
            print(f"[{index}/{total_rows}] FAILED {row['id']}: {exc}", file=sys.stderr)
            continue
        scored_result = score_row(row, judge_output)
        scored_results.append(scored_result)
        if output_path is not None:
            append_jsonl_row(output_path, scored_result.model_dump(mode="json"))
    return scored_results


def fail(message: str) -> int:
    print(message, file=sys.stderr)
    return 1


def select_rows(raw_rows: list[dict], *, ids: list[str] | None, limit: int | None) -> list[dict]:
    """Narrow raw_rows down to what --ids and/or --limit asked for; --ids alone can't shrink a run that's already too big to afford, so --limit exists for that."""
    rows_after_id_filter = raw_rows
    if ids is not None:
        requested_ids = set(ids)
        rows_after_id_filter = [r for r in raw_rows if r["id"] in requested_ids]

    rows_after_limit = rows_after_id_filter
    if limit is not None:
        rows_after_limit = rows_after_id_filter[:limit]

    return rows_after_limit


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
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write EvalResult JSONL",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Start fresh, replacing an existing --output file",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip rows already present in an existing --output file, and append new results to it",
    )
    parser.add_argument(
        "--ids",
        nargs="+",
        default=None,
        help="Score only these row ids",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Score only the first N selected rows",
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
        return fail("Use either --overwrite or --resume, not both.")
    if (
        args.output is not None
        and args.output.exists()
        and not args.overwrite
        and not args.resume
    ):
        return fail(
            f"Output already exists: {args.output}. Pass --overwrite to replace it "
            "or --resume to continue it."
        )

    raw_rows = read_jsonl(args.input)
    rows = select_rows(raw_rows, ids=args.ids, limit=args.limit)

    rows_with_output = [r for r in rows if r.get("model_output")]
    if not rows_with_output:
        return fail("No rows with model_output to score.")

    results_loaded_from_previous_run: list[EvalResult] = []
    resuming = args.resume and args.output is not None and args.output.exists()
    if resuming:
        results_loaded_from_previous_run = [EvalResult.model_validate(r) for r in read_jsonl(args.output)]
        already_scored_ids = {r.id for r in results_loaded_from_previous_run}
        rows_with_output = [r for r in rows_with_output if r["id"] not in already_scored_ids]

    client: OpenAI | None = None
    if not args.dry_run and rows_with_output:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return fail("Missing OPENAI_API_KEY. Set it, or pass --dry-run to skip the API.")
        api_base = os.environ.get("OPENAI_API_BASE", DEFAULT_API_BASE).rstrip("/")
        client = OpenAI(api_key=api_key, base_url=api_base, max_retries=3, timeout=60.0)

    if args.output is not None and not resuming:
        write_jsonl(args.output, [])  # start (or truncate to) an empty file we'll append to

    results_produced_by_this_run = judge_and_score_all(
        client,
        rows_with_output,
        model=args.judge_model,
        dry_run=args.dry_run,
        output_path=args.output,
    )

    combined_results = results_loaded_from_previous_run + results_produced_by_this_run
    summary = aggregate(combined_results)
    print(json.dumps(summary, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
