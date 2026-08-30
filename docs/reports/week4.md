# Week 4 report — LoRA SFT training pipeline design

## Summary

Built and wired up the full LoRA SFT loop for this project: a memory budget worked out before
touching any config, a training script that turns QA rows into the exact input shape the model
sees at inference, adapter-aware inference so a trained checkpoint can actually be evaluated, a
checkpoint-artifact convention, and two additions to the judge's few-shot examples found necessary
while reviewing SFT's output. Ran one comparison against the Week 3 base-model baseline as a smoke
test that the pipeline works end to end — those numbers are not a research finding; see the caveat
below.

## Design

### GPU memory budget, worked out before choosing a config

[`docs/GPU_MEMORY_ESTIMATE.md`](../GPU_MEMORY_ESTIMATE.md) computes, from Qwen2.5-3B-Instruct's
real `config.json` (`hidden_size=2048`, `num_hidden_layers=36`, `num_key_value_heads=2`), how much
VRAM full fine-tuning vs. LoRA would need on this machine's 16GB RTX 5060 Ti, and how activation
memory scales with batch size. Two errors were found and fixed in the estimate itself before
trusting it: the LoRA footprint was stated as 0.05GB in one table and 0.06GB in another for the
same 3.7M trained parameters under the same 16-bytes-per-parameter rule; and the doc originally
claimed full fine-tuning needs ~800x more memory than LoRA, when the total-memory ratio is actually
~8x (~48GB vs ~6.06GB). The ~800x figure was the trained-parameter-*count* ratio, a different thing
entirely.

The resulting numbers set `configs/train/lora.yaml` directly: `per_device_train_batch_size: 2`,
`gradient_accumulation_steps: 8` (effective batch 16 without ever holding 16 examples' activations
at once), and `gradient_checkpointing: true` — all three exist specifically because batch=4 alone
was already close to the 16GB ceiling by this estimate.

### Training data format has to be the exact shape used at inference

`train_sft.py`'s `build_prompt_completion_rows` builds one
`{"prompt": [{"role": "user", ...}], "completion": [{"role": "assistant", ...}]}` row per QA
example. This conversational shape isn't a stylistic choice — it's the only shape that makes
`trl.SFTTrainer` apply the tokenizer's chat template. The first version built plain strings instead
(`{"prompt": "<text>", "completion": "<text>"}`); `SFTTrainer` treats plain strings as its
"standard" format, raw string concatenation with no chat template, while `hf_backend.py`'s
inference path always wraps the same prompt text in
`apply_chat_template(..., add_generation_prompt=True)`. Left as plain strings, the model would have
been trained on an input shape it would never actually see at evaluation time, silently. Caught
before the checkpoint below was trusted; verified afterward by inspecting raw generations for
template-token leakage.

### Adapter loading: the missing link between training and evaluation

Before this week, `train_sft.py` could produce a checkpoint, but nothing in the inference path
could load it — a trained adapter had no way to be evaluated. `hf_backend.py`'s
`run_sequential_inference` gained an `adapter_path` parameter that, when given, wraps the frozen
base model in `peft.PeftModel.from_pretrained`; `run_inference.py` exposes this as `--adapter`, and
records which adapter was used in the output's `model_name` field (`<base>+lora:<path>`) so an eval
file downstream stays traceable to the exact checkpoint that produced it.

### Checkpoint artifacts: timestamped, reproducible, not in git

Each training run writes to `checkpoints/<YYYYMMDD_HHMMSS>_<config-name>/`
(`timestamped_output_dir()` prefixes the timestamp onto `configs/train/lora.yaml`'s
`output_dir: checkpoints/lora`), so repeat runs of the same config never overwrite each other. The
directory isn't committed to git (`checkpoints/README.md` is the one tracked exception, so the
directory itself survives a clone) — binary weights don't belong in version history, and are cheap
to reproduce from the training script, config, and data alone.

### Judge robustness: two adversarial-entity cases added as few-shot anchors

Manually reviewing SFT-v1's `data_v2_pilot` judge output (same discipline as Week 3's
`known_world_conflict` fix) surfaced two rows the judge classified as `partial` that should have
been `refuse`:

- `ex_0100` (`distractor_entity`): the response states a true, evidence-grounded figure for a
  *different* bridge than the one asked about, then explicitly declines the one actually asked
  about. No value is ever asserted as the answer to the entity actually asked about — that's a
  refusal, not a partial answer.
- `ex_0129` (false presupposition — the question asks why a trial *failed*; the evidence says it
  succeeded): the response corrects the false premise and states there's no support for the
  premised failure. Answering no part of "why did it fail" by establishing that it didn't is a
  refusal.

Two similar-looking rows were checked against the same rule and correctly left unchanged —
`ex_0131` (a genuine partial: restates a same-entity fact and explicitly offers it as confirmed)
and `ex_0116` (a genuine, if concerning, `answer`: picks one number out of two conflicting sources
and asserts it with no hedge) — confirming this was a targeted fix, not a blanket reclassification.
Both fixes were added as few-shot anchors to `judge.py`, contrasted against the existing examples
they were being confused with.

## First run: the pipeline works end to end

```bash
PYTHONPATH=src python -m grounded_refusal.train.train_sft
```

50 rows (`data/data_v1_pilot.jsonl`), 3 epochs, effective batch size 16, 21.7s wall clock on the
RTX 5060 Ti. Output: `checkpoints/20260823_151044_lora`.

| step (~epoch) | loss | mean_token_accuracy |
|---:|---:|---:|
| 1 (0.64) | 1.419 | 0.748 |
| 2 (1.0) | 1.004 | 0.812 |
| 3 (1.64) | 0.892 | 0.776 |
| 4 (2.0) | 0.675 | 0.787 |
| 5 (2.64) | 0.735 | 0.792 |
| 6 (3.0) | 0.848 | 0.833 |

Loss trends down with the noise expected at 12 total optimizer steps.

## An exploratory comparison, not a finding

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

These numbers are judged with gpt-4o, which [`docs/JUDGE_MODEL.md`](../JUDGE_MODEL.md) later found
to systematically undercount certain hallucination patterns, specifically, accepting a response's
own claim that "the evidence doesn't specify X" without checking it against the evidence.
`partial_match_rate` jumping to a clean 1.0 suggests SFT shifted the model toward exactly this kind
of explicit-decline language, which is the pattern gpt-4o was found to under-check. So this
comparison may be flattering SFT rather than measuring it fairly. The trustworthy version of this
table is a Week 5 re-judge with gpt-5-mini, not this one.

Two things came up while producing this table that are noted here but not resolved: re-judging
after adding the two anchors above changed the unfaithful-row count from 7 to 10, not 2, and the
raw judge output needed to explain the extra change was overwritten rather than archived before
anyone noticed; and this experiment can't distinguish whether the recall regression reflects a
general SFT effect or overfitting to `data_v1_pilot`'s narrower patterns, since the only eval set
used was the harder, out-of-distribution `data_v2_pilot`.

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
