"""Preference pair row contract (Pydantic)."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class NegativeType(str, Enum):
    HALLUCINATION = "hallucination"
    OVER_REFUSAL = "over_refusal"
    OVER_COMPLETE = "over_complete"
    DISTRACTOR_CONFUSION = "distractor_confusion"
    MEMORY_OVERRIDE = "memory_override"


class PreferenceMetadata(BaseModel):
    creation_process: Literal[
        "manual",
        "template_rule",
        "template_rule+llm_paraphrase",
        "llm_generated",
        "mixed",
    ] | None = None
    notes: str | None = None


class PreferencePair(BaseModel):
    """One DPO chosen/rejected pair from data/preference_*.jsonl."""

    id: str = Field(pattern=r"^pref_\d{4,}[a-z]?$")
    base_example_id: str = Field(pattern=r"^ex_\d{4,}$")
    prompt: str
    chosen: str
    rejected: str
    negative_type: NegativeType
    dataset_version: str = Field(pattern=r"^v\d+(_[a-z0-9_]+)?$")
    metadata: PreferenceMetadata | None = None
