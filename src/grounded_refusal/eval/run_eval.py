"""Score model-output JSONL with the LLM judge and derive outcomes.

GPT-4o extracts only predicted_behavior and is_faithful (judge.py);
verdict.py maps those plus gold answerability to two independent outcomes
(abstention_outcome, partial_outcome) -- see docs/EVAL_METRICS.md. Mirrors
build_preference.py's CLI shape: --input/--output/--overwrite, --dry-run to
smoke-test the pipeline without API calls, OPENAI_API_KEY check.
"""

from __future__ import annotations

import argparse
import concurrent.futures
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
from grounded_refusal.util.io import read_jsonl, write_jsonl

DEFAULT_API_BASE = "https://api.openai.com/v1"
DEFAULT_MAX_WORKERS = 4


def dry_run_judge_output(row: dict) -> JudgeOutput:
    """Deterministic placeholder so --dry-run can exercise the full pipeline
    (selection, verdict mapping, aggregation, output writing) with no API call.
    """
    return JudgeOutput(
        predicted_behavior=ModelBehavior.REFUSE,
        is_faithful=True,
        rationale="[dry-run placeholder]",
    )


def collect_judge_outputs(
    client: OpenAI | None,
    rows: list[dict],
    *,
    model: str,
    dry_run: bool,
    max_workers: int,
) -> dict[str, JudgeOutput]:
    if dry_run:
        return {row["id"]: dry_run_judge_output(row) for row in rows}

    assert client is not None
    outputs: dict[str, JudgeOutput] = {}
    total = len(rows)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_id = {
            pool.submit(judge_row, client, row["prompt"], row["model_output"], model=model): row["id"]
            for row in rows
        }
        done = 0
        for future in concurrent.futures.as_completed(future_to_id):
            row_id = future_to_id[future]
            outputs[row_id] = future.result()
            done += 1
            print(f"[{done}/{total}] judged {row_id}", file=sys.stderr)
    return outputs


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
        help="Model-output JSONL, e.g. outputs/base_pilot.jsonl (from cli.py infer_main)",
    )
    parser.add_argument("--output", type=Path, default=None, help="Write EvalResult JSONL")
    parser.add_argument(
        "--overwrite", action="store_true", help="Allow replacing an existing --output file"
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
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help="Concurrent judge API calls (ignored with --dry-run)",
    )
    args = parser.parse_args(argv)

    if args.output is not None and args.output.exists() and not args.overwrite:
        print(
            f"Output already exists: {args.output}. Pass --overwrite to replace it.",
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

    judge_outputs = collect_judge_outputs(
        client,
        scoreable_rows,
        model=args.judge_model,
        dry_run=args.dry_run,
        max_workers=args.max_workers,
    )
    results = [score_row(row, judge_outputs[row["id"]]) for row in scoreable_rows]

    summary = aggregate(results)
    print(json.dumps(summary, indent=2))

    if args.output is not None:
        write_jsonl(args.output, [r.model_dump(mode="json") for r in results])
        print(f"Wrote {len(results)} eval results to {args.output}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
