"""Per-token change in log-probability from SFT to DPO.

Question: when DPO lowers the log-probability of a response, which tokens
does the decrease fall on?

Setup
    Texts   the chosen and rejected responses of the 672 training pairs in
            preference_v3c_train.jsonl, plus the reference answers of the 60
            answerable held-out rows
    Models  the SFT model and the full-data DPO model, seeds 42, 44, 45, 46

For each text, every response token gets

    delta = log p_DPO(token) - log p_SFT(token)

Texts are tokenized with trl's own function, the way DPOTrainer tokenized them
during training, so the completion ends with the end-of-turn token and the
newline the chat template puts after it. The SFT model is the DPO model with
its adapter disabled, which is also the reference DPOTrainer trained against,
so beta times the sum of a response's deltas is the reward trl logs for it.

Forward passes only; nothing is generated. One output row per (seed, text).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.data_utils import _tokenize

from grounded_refusal.util.io import read_jsonl, write_jsonl
from grounded_refusal.util.prompt_assembly import load_yaml_config

# seed -> SFT adapter and the full-data DPO adapter trained on top of it
CHECKPOINTS = {
    42: {"sft": "checkpoints/20261008_223428_lora", "dpo": "checkpoints/20261008_223937_dpo"},
    44: {"sft": "checkpoints/20261009_171252_lora", "dpo": "checkpoints/20261009_171742_dpo"},
    45: {"sft": "checkpoints/20261009_181152_lora", "dpo": "checkpoints/20261009_181642_dpo"},
    46: {"sft": "checkpoints/20261009_190915_lora", "dpo": "checkpoints/20261009_191404_dpo"},
}

PREFERENCE_DATA = Path("data/preference_v3c_train.jsonl")
TRAIN_DATA = Path("data/data_v3c_train.jsonl")
# Any seed's held-out SFT output: only its prompt and reference answer are read, which are the same for every seed.
HELDOUT_ROWS = Path("outputs/inference-qwen2.5-3b-instruct-sft/20261008_223428_lora_v3b_heldout.jsonl")
DEFAULT_OUTPUT = Path("outputs/token-attribution/token_attribution.jsonl")

SENTENCE_END = re.compile(r"[.!?][\"')\]]*$")


def load_dpo_model(base_model_name: str, sft_checkpoint: str, dpo_checkpoint: str):
    """The DPO model as train_dpo.py built it: the DPO adapter on top of the merged SFT adapter."""
    model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(model, sft_checkpoint).merge_and_unload()
    return PeftModel.from_pretrained(model, dpo_checkpoint).eval()


def tokenize_pair(tokenizer, prompt: str, response: str) -> tuple[list[int], list[int]]:
    """Prompt ids and completion ids, as DPOTrainer's tokenize_fn produces them for conversational data."""
    messages = [{"role": "user", "content": prompt}]
    prompt_ids = _tokenize(tokenizer, messages, add_generation_prompt=True)["input_ids"]
    full_ids = _tokenize(tokenizer, messages + [{"role": "assistant", "content": response}])["input_ids"]
    assert full_ids[: len(prompt_ids)] == prompt_ids
    return prompt_ids, full_ids[len(prompt_ids) :]


def position_kinds(tokenizer, completion_ids: list[int]) -> list[str]:
    """Label each completion token "end" (the end-of-turn token), "boundary" (the first token of a
    sentence after the first) or "other". The newline after the end-of-turn token is "other"."""
    texts = [tokenizer.decode([token_id]) for token_id in completion_ids]
    kinds = []
    for i, token_id in enumerate(completion_ids):
        if token_id == tokenizer.eos_token_id:
            kinds.append("end")
        elif i > 0 and texts[i][:1].isspace() and SENTENCE_END.search(texts[i - 1].rstrip()):
            kinds.append("boundary")
        else:
            kinds.append("other")
    return kinds


@torch.no_grad()
def completion_log_probs(model, prompt_ids: list[int], completion_ids: list[int]) -> list[float]:
    """Log-probability of each completion token given everything before it."""
    input_ids = torch.tensor([prompt_ids + completion_ids], device=model.device)
    log_probs = torch.log_softmax(model(input_ids=input_ids).logits[0].float(), dim=-1)
    targets = torch.tensor(completion_ids, device=model.device)
    # log_probs[i] predicts token i + 1, so completion tokens are predicted by rows len(prompt_ids) - 1 onward.
    rows = log_probs[len(prompt_ids) - 1 : -1]
    return rows.gather(1, targets.unsqueeze(1))[:, 0].tolist()


def texts_to_score() -> list[dict]:
    """Every text to score: chosen and rejected of each training pair, and each answerable held-out reference answer."""
    answerability = {row["id"]: row["answerability"] for row in read_jsonl(TRAIN_DATA)}
    texts = []
    for pair in read_jsonl(PREFERENCE_DATA):
        for response in ("chosen", "rejected"):
            texts.append(
                {
                    "source": "train",
                    "id": pair["id"],
                    "answerability": answerability[pair["base_example_id"]],
                    "response": response,
                    "prompt": pair["prompt"],
                    "text": pair[response],
                }
            )
    for row in read_jsonl(HELDOUT_ROWS):
        if row["answerability"] == "answerable":
            texts.append(
                {
                    "source": "heldout",
                    "id": row["id"],
                    "answerability": row["answerability"],
                    "response": "reference",
                    "prompt": row["prompt"],
                    "text": row["reference_answer"],
                }
            )
    return texts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Per-token log-probability change from SFT to DPO.")
    parser.add_argument("--model-config", type=Path, default=Path("configs/models/base.yaml"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(CHECKPOINTS))
    args = parser.parse_args(argv)

    base_model_name = load_yaml_config(args.model_config)["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    texts = texts_to_score()

    results: list[dict] = []
    for seed in args.seeds:
        model = load_dpo_model(base_model_name, CHECKPOINTS[seed]["sft"], CHECKPOINTS[seed]["dpo"])
        for text in texts:
            prompt_ids, completion_ids = tokenize_pair(tokenizer, text["prompt"], text["text"])
            dpo_log_probs = completion_log_probs(model, prompt_ids, completion_ids)
            with model.disable_adapter():
                sft_log_probs = completion_log_probs(model, prompt_ids, completion_ids)
            results.append(
                {
                    "seed": seed,
                    **{key: text[key] for key in ("source", "id", "answerability", "response")},
                    "tokens": [tokenizer.decode([token_id]) for token_id in completion_ids],
                    "kinds": position_kinds(tokenizer, completion_ids),
                    "sft_log_probs": [round(x, 5) for x in sft_log_probs],
                    "dpo_log_probs": [round(x, 5) for x in dpo_log_probs],
                }
            )
        print(f"seed {seed}: scored {len(texts)} texts")
        del model
        torch.cuda.empty_cache()

    write_jsonl(args.output, results)
    print(f"Wrote {len(results)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
