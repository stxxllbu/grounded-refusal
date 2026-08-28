"""Score model-output JSONL with the LLM judge and derive outcomes.

Two phases: judge_all calls the AI and only ever writes raw judge output
(id + predicted_behavior + is_faithful + rationale) to --output, one row at
a time, so a single failed judge call doesn't lose already-judged rows and
--resume can skip rows already in that file. score_all then turns that raw
output plus each row's gold label (from --input) into scored EvalResults --
a pure, free, always-safe-to-rerun step, decoupled from ever having to call
the API again if scoring logic changes.
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


def usage_error(message: str) -> int:
    print(message, file=sys.stderr)
    return 1


def select_rows(raw_rows: list[dict], *, ids: list[str] | None, limit: int | None) -> list[dict]:
    """Narrow raw_rows down to what --ids and/or --limit asked for; --ids alone can't shrink a run that's already too big to afford, so --limit exists for that."""
    rows_after_id_filter = raw_rows
    if ids is not None:
        requested_ids = set(ids)
        rows_after_id_filter = [r for r in raw_rows if r["id"] in requested_ids]
        found_ids = {r["id"] for r in rows_after_id_filter}
        requested_ids_not_found = requested_ids - found_ids
        if requested_ids_not_found:
            print(
                f"Warning: --ids not found in --input: {sorted(requested_ids_not_found)}",
                file=sys.stderr,
            )

    rows_after_limit = rows_after_id_filter
    if limit is not None:
        rows_after_limit = rows_after_id_filter[:limit]

    return rows_after_limit


def judge_all(
    client: OpenAI | None,
    rows: list[dict],
    *,
    model: str,
    dry_run: bool,
    output_path: Path | None,
) -> list[dict]:
    """Judge each row in sequence, appending raw judge output as each finishes
    so a later failure can't lose earlier rows. Returns every raw record
    produced this call (in memory), regardless of whether output_path is set.
    """
    raw_results: list[dict] = []
    total = len(rows)
    for index, row in enumerate(rows, start=1):
        try:
            if dry_run:
                judge_output = JudgeOutput(
                    predicted_behavior=ModelBehavior.REFUSE,
                    is_faithful=True,
                    rationale="[dry-run placeholder]",
                )
            else:
                judge_output = judge_row(client, row["prompt"], row["model_output"], model=model)
        except Exception as exc:  # noqa: BLE001 -- one bad row must not kill the run
            print(f"[{index}/{total}] FAILED {row['id']}: {exc}", file=sys.stderr)
            continue
        raw_result = {"id": row["id"], **judge_output.model_dump(mode="json")}
        raw_results.append(raw_result)
        if output_path is not None:
            append_jsonl_row(output_path, raw_result)
    return raw_results


def score_all(rows_by_id: dict[str, dict], raw_judge_results: list[dict]) -> list[EvalResult]:
    """Turn raw judge output plus each row's gold label into scored EvalResults.

    Pure, no API calls -- safe to rerun any time this logic or aggregate's
    changes, without re-judging (or re-paying for) anything.
    """
    scored: list[EvalResult] = []
    for raw in raw_judge_results:
        row = rows_by_id[raw["id"]]
        judge_output = JudgeOutput(
            predicted_behavior=raw["predicted_behavior"],
            is_faithful=raw["is_faithful"],
            rationale=raw["rationale"],
        )
        answerability = Answerability(row["answerability"])
        evidence_challenge = [EvidenceChallengeTag(tag) for tag in row.get("evidence_challenge", [])]
        scored.append(
            EvalResult(
                id=row["id"],
                answerability=answerability,
                evidence_challenge=evidence_challenge,
                predicted_behavior=judge_output.predicted_behavior,
                is_faithful=judge_output.is_faithful,
                rationale=judge_output.rationale,
                abstention_outcome=derive_abstention_outcome(answerability, judge_output.predicted_behavior),
                partial_outcome=derive_partial_outcome(answerability, judge_output.predicted_behavior),
                model_name=row.get("model_name"),
            )
        )
    return scored


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
        help="Write raw judge output JSONL (id + predicted_behavior + is_faithful + rationale)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Start fresh, replacing an existing --output file",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip rows already judged in an existing --output file, and append new results to it",
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
        return usage_error("Use either --overwrite or --resume, not both.")
    if args.output is not None and args.output.exists() and not args.overwrite and not args.resume:
        return usage_error(
            f"Output already exists: {args.output}. Pass --overwrite to replace it "
            "or --resume to continue it."
        )

    raw_rows = read_jsonl(args.input)
    selected_rows = select_rows(raw_rows, ids=args.ids, limit=args.limit)
    rows_by_id = {r["id"]: r for r in selected_rows if r.get("model_output")}
    if not rows_by_id:
        return usage_error("No rows with model_output to score.")

    previously_judged_raw: list[dict] = []
    already_judged_ids: set[str] = set()
    if args.resume and args.output is not None and args.output.exists():
        previously_judged_raw = read_jsonl(args.output)
        already_judged_ids = {r["id"] for r in previously_judged_raw}

    rows_to_judge = [row for row_id, row in rows_by_id.items() if row_id not in already_judged_ids]

    client: OpenAI | None = None
    if not args.dry_run and rows_to_judge:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return usage_error("Missing OPENAI_API_KEY. Set it, or pass --dry-run to skip the API.")
        api_base = os.environ.get("OPENAI_API_BASE", DEFAULT_API_BASE).rstrip("/")
        client = OpenAI(api_key=api_key, base_url=api_base, max_retries=3, timeout=60.0)

    if args.output is not None and not already_judged_ids:
        write_jsonl(args.output, [])  # start (or truncate to) an empty file we'll append to

    newly_judged_raw = judge_all(
        client, rows_to_judge, model=args.judge_model, dry_run=args.dry_run, output_path=args.output
    )

    scored_results = score_all(rows_by_id, previously_judged_raw + newly_judged_raw)
    print(json.dumps(aggregate(scored_results), indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
