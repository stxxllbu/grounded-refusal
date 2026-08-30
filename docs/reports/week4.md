# Week 4 report — LoRA SFT training pipeline

## Summary

Built the LoRA SFT training and evaluation pipeline for this project, listed in the Development
table below. Trained a first checkpoint and ran it against the Week 3 base-model baseline as a
smoke test that the pipeline works end to end. Those comparison numbers are not a research finding:
the judge used to score them has since been found to have a specific blind spot (see the caveat
under Experiments).

## Development

| Item | Path | Notes |
|------|------|-------|
| GPU memory estimate | [`docs/GPU_MEMORY_ESTIMATE.md`](../GPU_MEMORY_ESTIMATE.md) | Sets the training config's batch size and gradient checkpointing |
| LoRA SFT training script | [`src/grounded_refusal/train/train_sft.py`](../../src/grounded_refusal/train/train_sft.py) | `train_sft_main` |
| Training config | [`configs/train/lora.yaml`](../../configs/train/lora.yaml) | `r=16`, `q_proj`/`v_proj`, batch=2 × grad_accum=8, gradient checkpointing, bf16 |
| Adapter loading (inference) | [`src/grounded_refusal/inference/hf_backend.py`](../../src/grounded_refusal/inference/hf_backend.py), [`run_inference.py`](../../src/grounded_refusal/inference/run_inference.py) | New `--adapter` flag |
| Judge distractor/presupposition rule | [`src/grounded_refusal/eval/judge.py`](../../src/grounded_refusal/eval/judge.py) | Two new few-shot anchors |
| Checkpoint storage | `checkpoints/<timestamp>_<config-name>/`, [`checkpoints/README.md`](../../checkpoints/README.md) | Not committed to git |

```text
data/data_v1_pilot.jsonl
        ↓  train_sft.py
checkpoints/<timestamp>_lora/
        ↓  run_inference.py --adapter <checkpoint>
outputs/inference-.../*.jsonl
        ↓  run_eval.py (judge.py)
outputs/eval-.../*_eval.jsonl
```

## GPU memory budget

`docs/GPU_MEMORY_ESTIMATE.md` estimates VRAM use for full fine-tuning versus LoRA on this machine's
16GB RTX 5060 Ti. The estimate uses Qwen2.5-3B-Instruct's real `config.json` values
(`hidden_size=2048`, `num_hidden_layers=36`, `num_key_value_heads=2`).

LoRA needs about 8x less total memory than full fine-tuning (6.06GB versus 48GB), since it trains
only about 3.7M of the model's 3B parameters.

The estimate sets `configs/train/lora.yaml` directly. `per_device_train_batch_size: 2` and
`gradient_accumulation_steps: 8` give an effective batch size of 16, without ever holding 16
examples' activations in memory at once. `gradient_checkpointing: true` is on because the estimate
showed batch=4 alone was already close to the 16GB ceiling.

## Training data format must match inference

`train_sft.py`'s `build_prompt_completion_rows` builds one row per QA example:
`{"prompt": [{"role": "user", ...}], "completion": [{"role": "assistant", ...}]}`. This
conversational format is required, not a style choice: `trl.SFTTrainer` only applies the
tokenizer's chat template to input in this shape, and `hf_backend.py`'s inference path always
wraps its own prompt text in `apply_chat_template(..., add_generation_prompt=True)`. Matching the
two exactly means the model trains on the same input shape it is prompted with at evaluation time.
Verified by checking the checkpoint's generations for template-token leakage; there is none.

## Adapter loading

`hf_backend.py`'s `run_sequential_inference` takes an `adapter_path` argument. When given, it wraps
the frozen base model with `peft.PeftModel.from_pretrained`, so a trained checkpoint can be
evaluated directly. `run_inference.py` exposes this as `--adapter`. The output's `model_name` field
records which adapter was used (`<base>+lora:<path>`), so an eval file downstream can be traced back
to the exact checkpoint that produced it.

## Checkpoint storage

Each training run writes its adapter to `checkpoints/<YYYYMMDD_HHMMSS>_<config-name>/`.
`timestamped_output_dir()` adds that timestamp to `configs/train/lora.yaml`'s
`output_dir: checkpoints/lora`. Repeat runs of the same config never overwrite each other as a
result.

The `checkpoints/` directory is not committed to git. Checkpoints are binary model weights; they
don't belong in version history, and they're cheap to reproduce from the training script, config,
and data alone. `checkpoints/README.md` is the one file inside it that is tracked, so the directory
itself still exists after a fresh clone.

## Judge rule: distractor entities and false presuppositions

`judge.py`'s few-shot examples encode this rule: a response counts as `refuse`, not `partial`, when
it states a true fact about a different entity than the one actually asked about and explicitly
declines the one asked about, or when it corrects a question's false premise and answers no part of
the question as posed. (`partial` means the response answers the part the evidence supports and
explicitly says the rest isn't supported. `refuse` means the response answers no part of the
question.)

**`ex_0100`** — distractor entity.

- Evidence: "The Meridian Bridge, opened in 1998, spans the Colby River and carries approximately
  40,000 vehicles daily, connecting the east and west districts of Fairview."
- Question: "How many vehicles does the Meridian Bridge that opened in 2004 carry daily?"
- Response: "Based on the evidence given, the Meridian Bridge that opened in 1998 carries
  approximately 40,000 vehicles daily. The information about the bridge opening in 2004 is not
  provided, so I cannot provide the number of vehicles it carries daily."

The response never states a number as the answer to the bridge actually asked about, the one that
opened in 2004. It states a real number for a different bridge, then declines the one asked about.
Under the rule above, that is `refuse`.

**`ex_0129`** — false presupposition.

- Evidence: "Halvorsen Pharmaceuticals' Phase 3 trial for its arthritis treatment met its primary
  endpoint, showing statistically significant symptom reduction compared to placebo."
- Question: "Why did Halvorsen Pharmaceuticals' Phase 3 trial fail to meet its primary endpoint?"
- Response: "Based on the evidence given, Halvorsen Pharmaceuticals' Phase 3 trial for its
  arthritis treatment met its primary endpoint, showing statistically significant symptom
  reduction compared to placebo. Therefore, there is no information available to suggest that the
  trial failed to meet its primary endpoint."

The question assumes the trial failed; the evidence says it succeeded. The response states that
and answers no part of "why did it fail." That is also `refuse`.

`ex_0100` and `ex_0129` are among `judge.py`'s few-shot examples.

## Experiments

**1. Training run.**

```bash
PYTHONPATH=src python -m grounded_refusal.train.train_sft
```

This trains on the 50 rows in `data/data_v1_pilot.jsonl` for 3 epochs, with an effective batch size
of 16. It ran in 21.7 seconds on the RTX 5060 Ti and produced the checkpoint
`checkpoints/20260823_151044_lora`.

| step (~epoch) | loss | mean_token_accuracy |
|---:|---:|---:|
| 1 (0.64) | 1.419 | 0.748 |
| 2 (1.0) | 1.004 | 0.812 |
| 3 (1.64) | 0.892 | 0.776 |
| 4 (2.0) | 0.675 | 0.787 |
| 5 (2.64) | 0.735 | 0.792 |
| 6 (3.0) | 0.848 | 0.833 |

Loss trends down with the noise expected at 12 total optimizer steps.

**2. Base vs. SFT on `data_v2_pilot`, 55 rows. A smoke test, not a finding.**

```bash
PYTHONPATH=src python -m grounded_refusal.inference.run_inference \
  --data data/data_v2_pilot.jsonl \
  --adapter checkpoints/20260823_151044_lora \
  --output outputs/inference-qwen2.5-3b-instruct-sft/20260823_151044_lora_v2_pilot.jsonl
```

| Model | abstention_recall | abstention_precision | over_refusal_rate | hallucination_rate | partial_match_rate |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-3B-Instruct (base, Week 3) | 0.6316 | 0.9231 | 0.0333 | 0.1905 | 0.6667 |
| Qwen2.5-3B-Instruct + SFT-v1 LoRA | 0.5789 | 0.9167 | 0.0333 | 0.2326 | 1.0000 |

This checkpoint was trained on `data_v1_pilot` (Experiment 1) and is evaluated here on
`data_v2_pilot`, the harder dataset Week 3 used for the base-model baseline, so the two rows above
compare the same evaluation set across base and SFT. This is not a claim about whether SFT helps.
[`docs/JUDGE_MODEL.md`](../JUDGE_MODEL.md) later found that gpt-4o, the judge used here, tends to
accept a response's claim that "the evidence doesn't specify X" without checking it against the
evidence. SFT's `partial_match_rate` of 1.0 suggests the model produces exactly this kind of claim
more often than the base model does.

That means gpt-4o's blind spot likely hits SFT's outputs harder than the base model's, making SFT
look better here than it really is. Week 5 re-judges both models with gpt-5-mini; that comparison,
not this one, is the one to trust.

## How to run

```bash
# 1. Train
PYTHONPATH=src python -m grounded_refusal.train.train_sft

# 2. Run the trained adapter against a pilot set
PYTHONPATH=src python -m grounded_refusal.inference.run_inference \
  --data data/data_v2_pilot.jsonl \
  --adapter checkpoints/<the-run-you-just-made> \
  --output outputs/inference-qwen2.5-3b-instruct-sft/<name>.jsonl

# 3. Judge it
PYTHONPATH=src python -m grounded_refusal.eval.run_eval \
  --input outputs/inference-qwen2.5-3b-instruct-sft/<name>.jsonl \
  --output outputs/eval-qwen2.5-3b-instruct-sft/<name>_eval.jsonl \
  --overwrite
```

## Limitations

- **The base-vs-SFT comparison uses gpt-4o as judge, which has a known blind spot** (see the caveat
  above), so it shouldn't be read as a conclusion. Week 5 re-judges both models with gpt-5-mini
  instead.
- **The recall drop has no clear cause.** SFT trained only on `data_v1_pilot` (50 easier rows) and
  was evaluated only on the harder, different `data_v2_pilot`, so a genuine capability regression
  and a failure to generalize to unseen failure modes would look identical in this data.

## Next: Week 5

- Re-judge base and SFT on `data_v2_pilot` with gpt-5-mini to establish a trustworthy baseline.
- Extend the eval data beyond this week's 55-row set.
- Train future SFT checkpoints on `data_v2` instead of `data_v1_pilot`, so training and evaluation
  data match.
