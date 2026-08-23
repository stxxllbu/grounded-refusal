"""Assemble model prompts from prompt-config YAML + QA fields."""

from __future__ import annotations

from pathlib import Path

import yaml

from grounded_refusal.data.schema_qa import QAExample


def load_yaml_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def format_qa_prompt(
    example: QAExample,
    *,
    instruction: str,
    evidence_label: str = "Evidence",
    question_label: str = "Question",
) -> str:
    """Assemble Evidence / Question / Instruction text for inference or DPO."""
    return (
        f"{evidence_label}:\n{example.evidence.strip()}\n\n"
        f"{question_label}:\n{example.question.strip()}\n\n"
        f"Instruction:\n{instruction.strip()}"
    )
