"""LoRA SFT on data_v1_pilot.jsonl.

Builds (prompt, completion) pairs from QA rows (prompt = format_qa_prompt(),
completion = reference_answer), trains a LoRA adapter with trl's SFTTrainer,
and saves the adapter to the configured output_dir. That checkpoint is meant
to be loaded the same way inference checkpoints are (base model + adapter)
and scored with eval/run_eval.py for the SFT row of the headline table.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

from grounded_refusal.data.validate_qa_jsonl_against_schema import validate_qa_jsonl_against_schema
from grounded_refusal.util.prompt_assembly import format_qa_prompt, load_yaml_config


def build_prompt_completion_rows(
    examples,
    *,
    instruction: str,
    evidence_label: str,
    question_label: str,
) -> list[dict]:
    """One {"prompt": ..., "completion": ...} dict per QA row, for SFTTrainer."""
    rows: list[dict] = []
    for ex in examples:
        prompt = format_qa_prompt(
            ex,
            instruction=instruction,
            evidence_label=evidence_label,
            question_label=question_label,
        )
        rows.append({"prompt": prompt, "completion": ex.reference_answer})
    return rows


def train_sft_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a LoRA SFT adapter on QA JSONL.")
    parser.add_argument("--data", type=Path, default=Path("data/data_v1_pilot.jsonl"))
    parser.add_argument("--prompt-config", type=Path, default=Path("configs/prompts/default.yaml"))
    parser.add_argument("--model-config", type=Path, default=Path("configs/models/base.yaml"))
    parser.add_argument("--train-config", type=Path, default=Path("configs/train/sft_v1.yaml"))
    args = parser.parse_args(argv)

    examples, errors = validate_qa_jsonl_against_schema(args.data)
    if errors:
        raise SystemExit(f"Training defensive: Invalid input data:\n" + "\n".join(errors))

    prompt_cfg = load_yaml_config(args.prompt_config)
    instruction = prompt_cfg.get("instruction")
    evidence_label = prompt_cfg.get("evidence_label", "Evidence")
    question_label = prompt_cfg.get("question_label", "Question")
    if not instruction:
        raise SystemExit(f"Training defensive: Missing 'instruction' in prompt config: {args.prompt_config}")

    rows = build_prompt_completion_rows(
        examples,
        instruction=instruction,
        evidence_label=evidence_label,
        question_label=question_label,
    )
    print(f"Built {len(rows)} prompt/completion rows from {args.data}")
    train_dataset = Dataset.from_list(rows)

    train_cfg = load_yaml_config(args.train_config)
    lora_config = LoraConfig(**train_cfg["lora"])
    sft_config = SFTConfig(**train_cfg["training"])

    model_cfg = load_yaml_config(args.model_config)
    trainer = SFTTrainer(
        model=model_cfg["model_name"],
        train_dataset=train_dataset,
        peft_config=lora_config,
        args=sft_config,
    )
    trainer.train()
    trainer.save_model(sft_config.output_dir)
    print(f"Wrote LoRA adapter to {sft_config.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(train_sft_main())
