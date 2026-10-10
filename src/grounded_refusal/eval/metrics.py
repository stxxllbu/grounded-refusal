"""Aggregate EvalResult rows into the three independent metric groups
defined in docs/EVAL_METRICS.md. No group shares a denominator with another.

Rows that went through the correctness judge (is_correct is set) add a
fourth group: answer accuracy, and the five answer outcomes that cross
is_correct with is_faithful.
"""

from __future__ import annotations

from typing import Any

from grounded_refusal.eval.schema_eval import EvalResult, ModelBehavior


ANSWER_OUTCOMES = (
    "correct_and_faithful",
    "correct_but_unfaithful",
    "wrong_but_faithful",
    "wrong_and_unfaithful",
    "refused",
)


def answer_outcome(predicted_behavior: ModelBehavior, is_faithful: bool, is_correct: bool) -> str:
    """Sort one response to a row that has a reference answer into one of ANSWER_OUTCOMES.

    A refusal is "refused" whatever the other two signals say. Otherwise
    is_correct says whether the answer is the reference one, and is_faithful
    says whether everything the response claims is in the evidence.
    """
    if predicted_behavior == ModelBehavior.REFUSE:
        return "refused"
    if is_correct:
        return "correct_and_faithful" if is_faithful else "correct_but_unfaithful"
    return "wrong_but_faithful" if is_faithful else "wrong_and_unfaithful"


def aggregate(results: list[EvalResult]) -> dict[str, Any]:
    """Compute abstention confusion-matrix metrics, conditional hallucination
    rate, partial sub-metrics, and answer correctness. Each block is silent
    (keys absent) if its underlying row count is zero, so callers can tell
    "0%" apart from "no data for this metric."
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

    # --- Answer correctness: rows the correctness judge saw ---
    results_with_is_correct = [r for r in results if r.is_correct is not None]
    if results_with_is_correct:
        summary["answer_accuracy"] = sum(r.is_correct for r in results_with_is_correct) / len(
            results_with_is_correct
        )
        outcomes = [
            answer_outcome(r.predicted_behavior, r.is_faithful, r.is_correct) for r in results_with_is_correct
        ]
        summary["answer_outcome_counts"] = {name: outcomes.count(name) for name in ANSWER_OUTCOMES}

    return summary
