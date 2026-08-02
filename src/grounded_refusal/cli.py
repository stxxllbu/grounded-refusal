from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from data.validate_qa_jsonl_against_schema import validate_qa_jsonl_against_schema
from util.io import write_jsonl
from util.prompt_assembly import format_qa_prompt, load_prompt_config


def infer_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run base-model inference on QA examples.")
    parser.add_argument("--data", type=Path, default=Path("data/hand_examples.jsonl"))
    parser.add_argument("--prompt-config", type=Path, default=Path("configs/prompts/default.yaml"))
    parser.add_argument("--model-config", type=Path, default=Path("configs/models/base.yaml"))
    parser.add_argument("--output", type=Path, default=Path("outputs/base_hand_examples.jsonl"))
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Format prompts only; skip model.")
    args = parser.parse_args(argv)

    examples, errors = validate_qa_jsonl_against_schema(args.data)
    if errors:
        raise SystemExit(f"Invalid input data:\n" + "\n".join(errors))

    prompt_cfg = load_prompt_config(args.prompt_config)
    instruction = prompt_cfg.get("instruction")
    if not instruction:
        raise SystemExit(f"Missing 'instruction' in prompt config: {args.prompt_config}")

    with args.model_config.open(encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    rows: list[dict] = []
    for ex in examples:
        prompt = format_qa_prompt(
            ex,
            instruction=instruction,
            evidence_label=prompt_cfg.get("evidence_label", "Evidence"),
            question_label=prompt_cfg.get("question_label", "Question"),
        )
        row = {
            "id": ex.id,
            "prompt": prompt,
            "reference_answer": ex.reference_answer,
            "answerability": ex.answerability.value,
            "evidence_type": ex.evidence_type.value,
            "evidence_challenge": [c.value for c in ex.evidence_challenge],
            "model_output": None,
        }
        rows.append(row)

    if args.dry_run:
        write_jsonl(args.output, rows)
        print(f"Dry run: wrote {len(rows)} prompts to {args.output}")
        return 0

    try:
        from grounded_refusal.inference import run_batch_inference
    except ImportError as exc:
        raise SystemExit(
            "Inference dependencies missing. Install with: pip install -e '.[inference]'"
        ) from exc

    max_new_tokens = args.max_new_tokens or model_cfg.get("max_new_tokens", 256)
    outputs = run_batch_inference(
        prompts=[r["prompt"] for r in rows],
        model_name=model_cfg["model_name"],
        max_new_tokens=max_new_tokens,
        temperature=model_cfg.get("temperature", 0.0),
        device_map=model_cfg.get("device_map", "auto"),
    )
    for row, output in zip(rows, outputs, strict=True):
        row["model_output"] = output
        row["model_name"] = model_cfg["model_name"]

    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} inference rows to {args.output}")
    return 0
