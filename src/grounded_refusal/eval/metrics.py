"""Aggregate EvalResult rows into the three independent metric groups
defined in docs/EVAL_METRICS.md. No group shares a denominator with another.
"""

from __future__ import annotations

from typing import Any

from grounded_refusal.eval.schema_eval import EvalResult, ModelBehavior


def aggregate(results: list[EvalResult]) -> dict[str, Any]:
    """Compute abstention confusion-matrix metrics, conditional hallucination
    rate, and partial sub-metrics. Each block is silent (keys absent) if its
    underlying row count is zero, so callers can tell "0%" apart from
    "no data for this metric."
    """
    summary: dict[str, Any] = {}

    # --- Abstention confusion matrix: answerable/unanswerable rows only ---
    tp = sum(r.abstention_outcome == "true_positive" for r in results)
    fp = sum(r.abstention_outcome == "false_positive" for r in results)
    fn = sum(r.abstention_outcome == "false_negative" for r in results)
    tn = sum(r.abstention_outcome == "true_negative" for r in results)

    if tp + fn > 0:
        summary["abstention_recall"] = tp / (tp + fn)
    if tp + fp > 0:
        summary["abstention_precision"] = tp / (tp + fp)
    if fp + tn > 0:
        summary["over_refusal_rate"] = fp / (fp + tn)

    # --- Hallucination rate: conditional on attempted answer, any slice ---
    attempted = [
        r for r in results if r.predicted_behavior in (ModelBehavior.ANSWER, ModelBehavior.PARTIAL)
    ]
    if attempted:
        summary["hallucination_rate"] = sum(not r.is_faithful for r in attempted) / len(attempted)

    # --- Partial sub-metrics: partial rows only ---
    partial_results = [r for r in results if r.partial_outcome is not None]
    if partial_results:
        n = len(partial_results)
        summary["partial_match_rate"] = sum(r.partial_outcome == "match" for r in partial_results) / n
        summary["partial_under_deliver_rate"] = (
            sum(r.partial_outcome == "under_deliver" for r in partial_results) / n
        )
        summary["partial_over_deliver_rate"] = (
            sum(r.partial_outcome == "over_deliver" for r in partial_results) / n
        )

    return summary
