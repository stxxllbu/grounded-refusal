"""Eval judge output and derived-outcome contract (Pydantic).

Design: docs/EVAL_METRICS.md. Two independent outcome types, computed by
separate pure functions in verdict.py -- never merged into one verdict field.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from grounded_refusal.data.schema_qa import Answerability, EvidenceChallengeTag


class ModelBehavior(str, Enum):
    ANSWER = "answer"
    REFUSE = "refuse"
    PARTIAL = "partial"


class JudgeOutput(BaseModel):
    """Raw judge output for one (prompt, model_output) pair.

    The judge never sees gold answerability/evidence_challenge labels, so it
    assesses behavior and grounding independently rather than pattern-matching
    a label.
    """

    predicted_behavior: ModelBehavior
    is_faithful: bool
    rationale: str


# Abstention confusion matrix outcome (answerable/unanswerable rows only).
# None when answerability == partial -- those rows never enter this matrix.
AbstentionOutcome = Literal[
    "true_positive",
    "false_positive",
    "true_negative",
    "false_negative",
]

# Partial sub-metric outcome (partial rows only).
# None when answerability != partial.
PartialOutcome = Literal["match", "under_deliver", "over_deliver"]


class EvalResult(BaseModel):
    """One scored row: gold labels + judge output + the two independent outcomes."""

    id: str
    answerability: Answerability
    evidence_challenge: list[EvidenceChallengeTag] = Field(default_factory=list)
    predicted_behavior: ModelBehavior
    is_faithful: bool
    rationale: str
    abstention_outcome: AbstentionOutcome | None = None
    partial_outcome: PartialOutcome | None = None
    model_name: str | None = None
