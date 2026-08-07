"""Eval judge output and derived verdict contract (Pydantic)."""

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


VerdictLabel = Literal[
    "correct",
    "over_refusal",
    "hallucination",
    "distractor_confusion",
    "over_complete",
    "memory_override",
    "anomaly",
]


class EvalResult(BaseModel):
    """One scored row: gold labels + judge output + derived verdict."""

    id: str
    answerability: Answerability
    evidence_challenge: list[EvidenceChallengeTag] = Field(default_factory=list)
    predicted_behavior: ModelBehavior
    is_faithful: bool
    rationale: str
    verdict: VerdictLabel
    model_name: str | None = None
