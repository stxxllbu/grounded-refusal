"""Aggregate EvalResult rows into Module-5 headline metrics, sliced by answerability."""

from __future__ import annotations

from collections import defaultdict

from grounded_refusal.eval.schema_eval import EvalResult


def aggregate(results: list[EvalResult]) -> dict[str, float]:
    """Compute answer accuracy / abstention / over-refusal / unsupported-claim
    rates by answerability slice, plus an overall anomaly rate.

    Silent about slices with zero rows: a metric key is only present if the
    underlying slice is non-empty, so callers can tell "0%" apart from
    "no data for this slice" rather than a misleading zero.
    """
    by_slice: dict[str, list[EvalResult]] = defaultdict(list)
    for result in results:
        by_slice[result.answerability.value].append(result)

    summary: dict[str, float] = {}

    if answerable := by_slice.get("answerable"):
        n = len(answerable)
        summary["answer_accuracy"] = sum(r.verdict == "correct" for r in answerable) / n
        summary["over_refusal_rate"] = sum(r.verdict == "over_refusal" for r in answerable) / n

    if unanswerable := by_slice.get("unanswerable"):
        n = len(unanswerable)
        summary["abstention_rate"] = sum(r.verdict == "correct" for r in unanswerable) / n
        summary["unsupported_claim_rate"] = sum(not r.is_faithful for r in unanswerable) / n

    if partial := by_slice.get("partial"):
        n = len(partial)
        summary["partial_answer_quality"] = sum(r.verdict == "correct" for r in partial) / n

    if results:
        summary["anomaly_rate"] = sum(r.verdict == "anomaly" for r in results) / len(results)

    return summary
