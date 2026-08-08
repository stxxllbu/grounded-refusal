"""Two independent, pure-function outcomes. Design: docs/EVAL_METRICS.md.

derive_abstention_outcome: answerability vs predicted_behavior only, for
answerable/unanswerable rows. Never touches is_faithful or evidence_challenge.

derive_partial_outcome: predicted_behavior only, for partial rows. Never
merged into the abstention matrix -- see docs/EVAL_METRICS.md "Partial rows"
for why forcing a 3rd class into a 2x2 confusion matrix was rejected.
"""

from __future__ import annotations

from grounded_refusal.data.schema_qa import Answerability
from grounded_refusal.eval.schema_eval import AbstentionOutcome, ModelBehavior, PartialOutcome


def derive_abstention_outcome(
    answerability: Answerability,
    predicted_behavior: ModelBehavior,
) -> AbstentionOutcome | None:
    """TP/FP/TN/FN for the "should this row have been refused?" classification.

    Returns None for partial rows -- they never enter this matrix.
    """
    if answerability == Answerability.PARTIAL:
        return None

    should_refuse = answerability == Answerability.UNANSWERABLE
    did_refuse = predicted_behavior == ModelBehavior.REFUSE

    if should_refuse and did_refuse:
        return "true_positive"
    if should_refuse and not did_refuse:
        return "false_negative"
    if not should_refuse and did_refuse:
        return "false_positive"
    return "true_negative"


def derive_partial_outcome(
    answerability: Answerability,
    predicted_behavior: ModelBehavior,
) -> PartialOutcome | None:
    """Did a partial row get the expected partial behavior, or over/under-deliver?

    Returns None for non-partial rows.
    """
    if answerability != Answerability.PARTIAL:
        return None

    if predicted_behavior == ModelBehavior.PARTIAL:
        return "match"
    if predicted_behavior == ModelBehavior.REFUSE:
        return "under_deliver"
    return "over_deliver"  # ModelBehavior.ANSWER
