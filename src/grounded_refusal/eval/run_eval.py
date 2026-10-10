"""Score model-output JSONL with the LLM judge and derive outcomes.

Two phases: judge_all calls the AI and only ever writes raw judge output
(id + predicted_behavior + is_faithful + rationale) to --output, one row at
a time, so a single failed judge call doesn't lose already-judged rows and
--resume can skip rows already in that file. score_all then turns that raw
output plus each row's gold label (from --input) into scored EvalResults --
a pure, free, always-safe-to-rerun step, decoupled from ever having to call
the API again if scoring logic changes.

With --correctness-output, the correctness judge also runs, on the rows that
have a reference answer (answerable and partial), through the same judge_all
loop. Its raw output (id + rationale + answer_match) goes to its own file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path

from pydantic import BaseModel

from grounded_refusal.data.schema_qa import Answerability, EvidenceChallengeTag
from grounded_refusal.eval.correctness import judge_correctness
from grounded_refusal.eval.judge import DEFAULT_JUDGE_MODEL, judge_row
from grounded_refusal.eval.metrics import aggregate
from grounded_refusal.eval.schema_eval import (
    CorrectnessOutput,
    EvalResult,
    JudgeOutput,
    ModelBehavior,
)
from grounded_refusal.eval.verdict import derive_abstention_outcome, derive_partial_outcome
from grounded_refusal.util.io import append_jsonl_row, read_jsonl, write_jsonl


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


def run_behavior_judge(row: dict, *, model: str, dry_run: bool) -> JudgeOutput:
    """Run the behavior judge (predicted_behavior, is_faithful) on one inference row."""
    if dry_run:
        return JudgeOutput(
            predicted_behavior=ModelBehavior.REFUSE,
            is_faithful=True,
            rationale="[dry-run placeholder]",
        )
    return judge_row(row["prompt"], row["model_output"], model=model)


def run_correctness_judge(row: dict, *, model: str, dry_run: bool) -> CorrectnessOutput:
    """Run the correctness judge (answer_match) on one inference row that has a reference answer."""
    if dry_run:
        return CorrectnessOutput(rationale="[dry-run placeholder]", answer_match="no_answer")
    return judge_correctness(
        row["question"],
        row["reference_answer"],
        row["model_output"],
        model=model,
        supported_subquestions=row.get("supported_subquestions"),
    )


def split_already_judged(
    rows: list[dict], output_path: Path | None, *, resume: bool
) -> tuple[list[dict], list[dict]]:
    """Split rows by whether output_path already holds a judge result for them.

    Returns the raw results already in the file for these rows, and the rows
    still to judge. Without --resume nothing counts as already judged.
    """
    previously_judged_raw: list[dict] = []
    if resume and output_path is not None and output_path.exists():
        previously_judged_raw = read_jsonl(output_path)
    already_judged_ids = {r["id"] for r in previously_judged_raw}
    selected_ids = {row["id"] for row in rows}
    # The file can hold rows outside this run's --ids/--limit selection; keep only the selected ones.
    return (
        [r for r in previously_judged_raw if r["id"] in selected_ids],
        [row for row in rows if row["id"] not in already_judged_ids],
    )


def start_output_file_unless_resuming(output_path: Path | None, *, resume: bool) -> None:
    """Leave a file that --resume is continuing; otherwise start (or truncate to) an empty one."""
    if output_path is not None and not (resume and output_path.exists()):
        write_jsonl(output_path, [])


def judge_all(
    rows: list[dict],
    run_judge: Callable[[dict], BaseModel],
    *,
    output_path: Path | None,
) -> list[dict]:
    """Call run_judge on each row in sequence, appending raw judge output as each finishes
    so a later failure can't lose earlier rows. Returns every raw record
    produced this call (in memory), regardless of whether output_path is set.
    """
    raw_results: list[dict] = []
    total = len(rows)
    for index, row in enumerate(rows, start=1):
        try:
            judge_output = run_judge(row)
        except Exception as exc:  # noqa: BLE001 -- one bad row must not kill the run
            print(f"[{index}/{total}] FAILED {row['id']}: {exc}", file=sys.stderr)
            continue
        raw_result = {"id": row["id"], **judge_output.model_dump(mode="json")}
        raw_results.append(raw_result)
        if output_path is not None:
            append_jsonl_row(output_path, raw_result)
    return raw_results


def score_all(
    rows_by_id: dict[str, dict],
    raw_judge_results: list[dict],
    raw_correctness_results: list[dict] | None = None,
) -> list[EvalResult]:
    """Turn raw judge output plus each row's gold label into scored EvalResults.

    Pure, no API calls -- safe to rerun any time this logic or aggregate's
    changes, without re-judging (or re-paying for) anything.
    """
    is_correct_by_id = {
        raw["id"]: CorrectnessOutput(rationale=raw["rationale"], answer_match=raw["answer_match"]).is_correct
        for raw in raw_correctness_results or []
    }
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
                is_correct=is_correct_by_id.get(row["id"]),
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
        help=(
            "Skip rows already judged in an existing --output file, and append new "
            "results to it. If every row is already judged, this recomputes metrics "
            "from --output with no API calls."
        ),
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
        help=f"Judge model name (default: OPENAI_JUDGE_MODEL env or {DEFAULT_JUDGE_MODEL})",
    )
    parser.add_argument(
        "--correctness-output",
        type=Path,
        default=None,
        help=(
            "Also run the correctness judge on answerable and partial rows, and write its "
            "raw output JSONL (id + rationale + answer_match) here. Follows --overwrite/--resume."
        ),
    )
    args = parser.parse_args(argv)

    if args.overwrite and args.resume:
        return usage_error("Use either --overwrite or --resume, not both.")
    for path in (args.output, args.correctness_output):
        if path is not None and path.exists() and not args.overwrite and not args.resume:
            return usage_error(
                f"Output already exists: {path}. Pass --overwrite to replace it "
                "or --resume to continue it."
            )

    raw_rows = read_jsonl(args.input)
    selected_rows = select_rows(raw_rows, ids=args.ids, limit=args.limit)
    rows_by_id = {r["id"]: r for r in selected_rows if r.get("model_output")}
    if not rows_by_id:
        return usage_error("No rows with model_output to score.")

    rows = list(rows_by_id.values())
    previously_judged_raw, rows_to_judge = split_already_judged(rows, args.output, resume=args.resume)

    previously_judged_correctness_raw: list[dict] = []
    rows_to_judge_for_correctness: list[dict] = []
    if args.correctness_output is not None:
        rows_with_reference_answer = [
            row for row in rows if row["answerability"] != Answerability.UNANSWERABLE.value
        ]
        if any("question" not in row for row in rows_with_reference_answer):
            return usage_error(
                "--correctness-output needs a question field in every --input row; "
                "inference outputs written before run_inference.py added it do not have one."
            )
        previously_judged_correctness_raw, rows_to_judge_for_correctness = split_already_judged(
            rows_with_reference_answer, args.correctness_output, resume=args.resume
        )

    if (
        not args.dry_run
        and (rows_to_judge or rows_to_judge_for_correctness)
        and not os.environ.get("OPENAI_API_KEY")
    ):
        return usage_error("Missing OPENAI_API_KEY. Set it, or pass --dry-run to skip the API.")

    start_output_file_unless_resuming(args.output, resume=args.resume)
    raw_judge_results = previously_judged_raw + judge_all(
        rows_to_judge,
        partial(run_behavior_judge, model=args.judge_model, dry_run=args.dry_run),
        output_path=args.output,
    )

    raw_correctness_results: list[dict] | None = None
    if args.correctness_output is not None:
        start_output_file_unless_resuming(args.correctness_output, resume=args.resume)
        raw_correctness_results = previously_judged_correctness_raw + judge_all(
            rows_to_judge_for_correctness,
            partial(run_correctness_judge, model=args.judge_model, dry_run=args.dry_run),
            output_path=args.correctness_output,
        )

    scored_results = score_all(rows_by_id, raw_judge_results, raw_correctness_results)
    print(json.dumps(aggregate(scored_results), indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
