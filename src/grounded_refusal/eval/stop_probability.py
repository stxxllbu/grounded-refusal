"""Stop probability after the first sentence of an answer.

Question: once a model has written the first sentence of its answer, how likely
is it to stop there?

Setup
    Rows    the 60 answerable rows of the held-out set
    Models  5 per seed (SFT, continued SFT, and three DPO variants), 4 seeds

For each row, every model of a seed is scored on the same two inputs:

    1. prompt + first sentence of that seed's SFT response
       -> stop_prob: probability that the next token is the end-of-turn token
    2. prompt + reference answer
       -> reference_log_prob: sum of the reference answer's token log-probabilities

Forward passes only; nothing is generated. One output row per (seed, model, row).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, RepetitionPenaltyLogitsProcessor

from grounded_refusal.util.io import read_jsonl, write_jsonl
from grounded_refusal.util.prompt_assembly import load_yaml_config

# seed -> model -> adapter directory.
# "sft" is trained on the base model. Every other model is a second-stage
# adapter trained on top of the same seed's merged SFT adapter.
CHECKPOINTS = {
    42: {
        "sft": "checkpoints/20261008_223428_lora",
        "sft_continued": "checkpoints/20261012_153825_sft_continued",
        "dpo": "checkpoints/20261008_223937_dpo",
        "dpo_answerable_only": "checkpoints/20261011_230820_dpo_answerable_only",
        "dpo_unanswerable_partial_only": "checkpoints/20261011_232416_dpo_unanswerable_partial_only",
    },
    44: {
        "sft": "checkpoints/20261009_171252_lora",
        "sft_continued": "checkpoints/20261012_154912_sft_continued",
        "dpo": "checkpoints/20261009_171742_dpo",
        "dpo_answerable_only": "checkpoints/20261011_233428_dpo_answerable_only",
        "dpo_unanswerable_partial_only": "checkpoints/20261011_234404_dpo_unanswerable_partial_only",
    },
    45: {
        "sft": "checkpoints/20261009_181152_lora",
        "sft_continued": "checkpoints/20261012_155500_sft_continued",
        "dpo": "checkpoints/20261009_181642_dpo",
        "dpo_answerable_only": "checkpoints/20261011_235413_dpo_answerable_only",
        "dpo_unanswerable_partial_only": "checkpoints/20261012_000348_dpo_unanswerable_partial_only",
    },
    46: {
        "sft": "checkpoints/20261009_190915_lora",
        "sft_continued": "checkpoints/20261012_160046_sft_continued",
        "dpo": "checkpoints/20261009_191404_dpo",
        "dpo_answerable_only": "checkpoints/20261012_001357_dpo_answerable_only",
        "dpo_unanswerable_partial_only": "checkpoints/20261012_002334_dpo_unanswerable_partial_only",
    },
}

# seed -> held-out responses of that seed's SFT model (the source of the first sentence)
SFT_RESPONSES = {
    42: "outputs/inference-qwen2.5-3b-instruct-sft/20261008_223428_lora_v3b_heldout.jsonl",
    44: "outputs/inference-qwen2.5-3b-instruct-sft/20261009_171252_lora_v3b_heldout.jsonl",
    45: "outputs/inference-qwen2.5-3b-instruct-sft/20261009_181152_lora_v3b_heldout.jsonl",
    46: "outputs/inference-qwen2.5-3b-instruct-sft/20261009_190915_lora_v3b_heldout.jsonl",
}

DEFAULT_OUTPUT = Path("outputs/stop-probability/stop_probability_v3b_heldout.jsonl")
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def first_sentence(response: str) -> str:
    return SENTENCE_BOUNDARY.split(response.strip())[0]


def load_model(base_model_name: str, sft_checkpoint: str, second_stage_checkpoint: str | None):
    """Load the SFT model, or a second-stage model on top of the merged SFT adapter (as hf_backend.py does)."""
    model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(model, sft_checkpoint)
    if second_stage_checkpoint is not None:
        model = PeftModel.from_pretrained(model.merge_and_unload(), second_stage_checkpoint)
    return model.eval()


@torch.no_grad()
def next_token_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Logits of shape (sequence_length, vocab_size). Row i is the prediction for token i + 1."""
    logits = model(input_ids=input_ids.unsqueeze(0).to(model.device)).logits[0]
    return logits.float().cpu()


def score_stop(model, tokenizer, chat_prompt: str, sentence: str) -> dict:
    """Probability that the model ends its turn right after `sentence`."""
    input_ids = tokenizer(chat_prompt + sentence, return_tensors="pt")["input_ids"][0]
    logits = next_token_logits(model, input_ids)[-1]
    log_probs = torch.log_softmax(logits, dim=-1)
    eos = tokenizer.eos_token_id

    # hf_backend.py generates with the model's default repetition penalty, so the
    # token generation picks is the argmax after that penalty is applied.
    penalty = RepetitionPenaltyLogitsProcessor(model.generation_config.repetition_penalty)
    penalized_logits = penalty(input_ids.unsqueeze(0), logits.unsqueeze(0))[0]

    return {
        "stop_prob": log_probs[eos].exp().item(),
        "stop_is_top_token": log_probs.argmax().item() == eos,
        "stop_is_top_token_with_repetition_penalty": penalized_logits.argmax().item() == eos,
    }


def score_reference_answer(model, tokenizer, chat_prompt: str, reference_answer: str) -> dict:
    """Log-probability of the reference answer followed by the end-of-turn token."""
    prompt_length = tokenizer(chat_prompt, return_tensors="pt")["input_ids"].shape[1]
    input_ids = tokenizer(chat_prompt + reference_answer, return_tensors="pt")["input_ids"][0]
    input_ids = torch.cat([input_ids, torch.tensor([tokenizer.eos_token_id])])

    log_probs = torch.log_softmax(next_token_logits(model, input_ids), dim=-1)
    answer_ids = input_ids[prompt_length:]
    # log_probs[i] predicts token i + 1, so the answer tokens are predicted by rows prompt_length - 1 onward.
    answer_log_probs = log_probs[prompt_length - 1 : -1].gather(1, answer_ids.unsqueeze(1))

    return {
        "reference_log_prob": answer_log_probs.sum().item(),
        "reference_num_tokens": len(answer_ids),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stop probability after the first sentence, for each model.")
    parser.add_argument("--model-config", type=Path, default=Path("configs/models/base.yaml"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(CHECKPOINTS))
    args = parser.parse_args(argv)

    base_model_name = load_yaml_config(args.model_config)["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    results: list[dict] = []
    for seed in args.seeds:
        rows = [row for row in read_jsonl(SFT_RESPONSES[seed]) if row["answerability"] == "answerable"]

        for model_name, checkpoint in CHECKPOINTS[seed].items():
            second_stage_checkpoint = None if model_name == "sft" else checkpoint
            model = load_model(base_model_name, CHECKPOINTS[seed]["sft"], second_stage_checkpoint)

            for row in rows:
                chat_prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": row["prompt"]}], tokenize=False, add_generation_prompt=True
                )
                sentence = first_sentence(row["model_output"])
                results.append(
                    {
                        "id": row["id"],
                        "seed": seed,
                        "arm": model_name,
                        "first_sentence": sentence,
                        **score_stop(model, tokenizer, chat_prompt, sentence),
                        **score_reference_answer(model, tokenizer, chat_prompt, row["reference_answer"]),
                    }
                )

            print(f"seed {seed} {model_name}: scored {len(rows)} rows")
            del model
            torch.cuda.empty_cache()

    write_jsonl(args.output, results)
    print(f"Wrote {len(results)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
