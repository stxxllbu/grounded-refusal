"""QA example row contract (Pydantic)."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class Answerability(str, Enum):
    ANSWERABLE = "answerable"
    UNANSWERABLE = "unanswerable"
    PARTIAL = "partial"


class EvidenceType(str, Enum):
    SINGLE_SENTENCE = "single_sentence"
    SHORT_PARAGRAPH = "short_paragraph"
    MULTI_PARAGRAPH = "multi_paragraph"


class EvidenceChallengeTag(str, Enum):
    DISTRACTOR_ENTITY = "distractor_entity"
    KNOWN_WORLD_CONFLICT = "known_world_conflict"
    PARTIAL_EVIDENCE = "partial_evidence"


class Split(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    DEV = "dev"


class ExampleMetadata(BaseModel):
    entity: str | None = None
    creation_process: Literal[
        "manual",
        "template_rule",
        "template_rule+llm_paraphrase",
        "llm_generated",
        "mixed",
    ] | None = None


class QAExample(BaseModel):
    """One evidence-grounded QA row from data/*.jsonl."""

    id: str = Field(pattern=r"^ex_\d{4,}$")
    evidence: str
    question: str
    reference_answer: str
    answerability: Answerability
    evidence_type: EvidenceType
    evidence_challenge: list[EvidenceChallengeTag] = Field(default_factory=list)
    split: Split
    dataset_version: str = Field(pattern=r"^v\d+(_[a-z0-9_]+)?$")
    question_decomposition: list[str] | None = None
    supported_subquestions: list[str] | None = None
    metadata: ExampleMetadata | None = None
    tags: list[str] | None = None

    @model_validator(mode="after")
    def validate_partial_fields(self) -> QAExample:
        if self.answerability == Answerability.PARTIAL:
            if not self.question_decomposition or not self.supported_subquestions:
                raise ValueError(
                    "partial examples require question_decomposition and supported_subquestions"
                )
            unsupported = set(self.question_decomposition) - set(self.supported_subquestions)
            if unsupported and not self.supported_subquestions:
                raise ValueError("partial examples need at least one unsupported sub-question")
        return self
