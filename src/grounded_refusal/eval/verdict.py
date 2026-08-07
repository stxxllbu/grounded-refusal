"""Deterministic mapping from (gold labels, judge output) to a verdict.

Mirrors build_preference.py's choose_negative_type, but keyed on OBSERVED
model behavior/faithfulness rather than on how a negative example was
constructed. The judge decides predicted_behavior and is_faithful; Python
decides the verdict.
"""

from __future__ import annotations

from grounded_refusal.data.schema_qa import Answerability, EvidenceChallengeTag
from grounded_refusal.eval.schema_eval import ModelBehavior, VerdictLabel


def derive_verdict(
    answerability: Answerability,
    evidence_challenge: list[EvidenceChallengeTag],
    predicted_behavior: ModelBehavior,
    is_faithful: bool,
) -> VerdictLabel:
    """Combine one QA row's gold labels with one judge output into a verdict.

    Combinations outside the pilot map (protocol table 1-5) are labeled
    "anomaly" rather than forced into the closest category.
    """
    has_distractor = EvidenceChallengeTag.DISTRACTOR_ENTITY in evidence_challenge
    has_world_conflict = EvidenceChallengeTag.KNOWN_WORLD_CONFLICT in evidence_challenge

    if answerability == Answerability.ANSWERABLE:
        if predicted_behavior == ModelBehavior.REFUSE:
            return "over_refusal"
        if predicted_behavior == ModelBehavior.ANSWER:
            if is_faithful:
                return "correct"
            return "memory_override" if has_world_conflict else "anomaly"
        return "anomaly"  # partial on a fully-answerable question

    if answerability == Answerability.UNANSWERABLE:
        if predicted_behavior == ModelBehavior.REFUSE:
            return "correct"
        if predicted_behavior == ModelBehavior.ANSWER and not is_faithful:
            return "distractor_confusion" if has_distractor else "hallucination"
        return "anomaly"

    if answerability == Answerability.PARTIAL:
        if predicted_behavior == ModelBehavior.PARTIAL and is_faithful:
            return "correct"
        if predicted_behavior == ModelBehavior.ANSWER and not is_faithful:
            return "over_complete"
        if predicted_behavior == ModelBehavior.REFUSE:
            return "over_refusal"  # protocol's noted extension: whole-question refusal
        return "anomaly"

    raise ValueError(f"Unsupported answerability: {answerability}")
