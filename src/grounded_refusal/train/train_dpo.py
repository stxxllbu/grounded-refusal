"""LoRA DPO training, continuing from an SFT checkpoint.

Merges the given SFT LoRA adapter into the base model first, then trains a
fresh LoRA adapter with trl's DPOTrainer on top of that merged model. Passing
peft_config to DPOTrainer without a ref_model makes it use the merged model
with the new adapter disabled as the frozen reference (trl's built-in
reference-free LoRA DPO -- confirmed against
https://huggingface.co/docs/trl/dpo_trainer and the ValueError trl raises if
a PeftModel/peft_config and an explicit ref_model are both given) -- no
second full model copy is loaded. This keeps the reference point well
defined: the model state right after SFT, before any DPO.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from grounded_refusal.data.schema_pref import PreferencePair
from grounded_refusal.train.train_sft import git_commit_hash, timestamped_output_dir
from grounded_refusal.util.io import read_jsonl
from grounded_refusal.util.prompt_assembly import load_yaml_config


def validate_preference_jsonl(path: Path) -> list[PreferencePair]:
    return [PreferencePair.model_validate(row) for row in read_jsonl(path)]


def build_dpo_rows(pairs: list[PreferencePair]) -> list[dict]:
    """One conversational {"prompt": [...], "chosen": [...], "rejected": [...]} dict per pair.

    ``prompt`` is already the fully assembled Evidence/Question/Instruction
    text build_preference.py wrote into the preference file -- wrapped here
    as a single user turn, not reassembled from the source QA row. Wrapping
    in role/content form (matching train_sft.py's build_prompt_completion_rows)
    is what makes DPOTrainer apply the tokenizer's chat template automatically.
    """
    rows: list[dict] = []
    for pair in pairs:
        rows.append(
            {
                "prompt": [{"role": "user", "content": pair.prompt}],
                "chosen": [{"role": "assistant", "content": pair.chosen}],
                "rejected": [{"role": "assistant", "content": pair.rejected}],
            }
        )
    return rows


def load_merged_sft_model(model_name: str, sft_adapter_path: str):
    """Base model + SFT LoRA, merged into plain weights.

    The merged model is both DPO's starting point and, with its new adapter
    disabled, DPOTrainer's reference model.
    """
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    sft_model = PeftModel.from_pretrained(base_model, sft_adapter_path)
    return sft_model.merge_and_unload()


def write_checkpoint_metadata(
    output_dir: str,
    *,
    base_model: str,
    sft_adapter_path: str,
    data_path: Path,
    num_rows: int,
    num_epochs: float,
    beta: float,
    lora_config: LoraConfig,
    train_config_path: Path,
) -> None:
    metadata = {
        "base_model": base_model,
        "sft_adapter": sft_adapter_path,
        "trained_on": str(data_path),
        "num_rows": num_rows,
        "num_epochs": num_epochs,
        "beta": beta,
        "lora_r": lora_config.r,
        "lora_target_modules": list(lora_config.target_modules),
        "train_config": str(train_config_path),
        "git_commit": git_commit_hash(),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (Path(output_dir) / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def train_dpo_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a LoRA DPO adapter on preference JSONL.")
    parser.add_argument("--data", type=Path, default=Path("data/preference_v2_train.jsonl"))
    parser.add_argument(
        "--sft-adapter",
        type=Path,
        required=True,
        help="SFT LoRA checkpoint to continue from (e.g. checkpoints/<timestamp>_lora)",
    )
    parser.add_argument("--model-config", type=Path, default=Path("configs/models/base.yaml"))
    parser.add_argument("--train-config", type=Path, default=Path("configs/train/dpo.yaml"))
    args = parser.parse_args(argv)

    pairs = validate_preference_jsonl(args.data)
    rows = build_dpo_rows(pairs)
    print(f"Built {len(rows)} preference rows from {args.data}")
    train_dataset = Dataset.from_list(rows)

    train_cfg = load_yaml_config(args.train_config)
    train_cfg["training"]["output_dir"] = timestamped_output_dir(train_cfg["training"]["output_dir"])
    lora_config = LoraConfig(**train_cfg["lora"])
    dpo_config = DPOConfig(**train_cfg["training"])

    model_cfg = load_yaml_config(args.model_config)
    model = load_merged_sft_model(model_cfg["model_name"], str(args.sft_adapter))
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["model_name"])

    # No ref_model: DPOTrainer uses the merged model with this new LoRA
    # disabled as the frozen reference (see module docstring).
    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    trainer.train()
    trainer.save_model(dpo_config.output_dir)
    write_checkpoint_metadata(
        dpo_config.output_dir,
        base_model=model_cfg["model_name"],
        sft_adapter_path=str(args.sft_adapter),
        data_path=args.data,
        num_rows=len(rows),
        num_epochs=dpo_config.num_train_epochs,
        beta=dpo_config.beta,
        lora_config=lora_config,
        train_config_path=args.train_config,
    )
    print(f"Wrote LoRA DPO adapter to {dpo_config.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(train_dpo_main())
