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
    COREFERENCE_AMBIGUITY = "coreference_ambiguity"
    HEDGED_UNCERTAINTY = "hedged_uncertainty"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    MULTI_HOP_ARITHMETIC = "multi_hop_arithmetic"
    RED_HERRING = "red_herring"
    CONDITIONAL_LOGIC = "conditional_logic"
    NEGATION_EXCEPTION = "negation_exception"
    EMBEDDED_INSTRUCTION = "embedded_instruction"
    DIGIT_CONFUSION = "digit_confusion"
    FALSE_PRESUPPOSITION = "false_presupposition"
    CIRCULAR_EVIDENCE = "circular_evidence"
    NEAR_MISS = "near_miss"


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
