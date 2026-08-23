# Week 4 report (draft) — SFT training + baseline comparison

## Summary

Built the LoRA SFT training pipeline, trained the first checkpoint on `data_v1_pilot`'s 50 rows,
added the missing piece that let it actually be evaluated (adapter loading in the inference path),
and compared it against the Week 3 base-model baseline on `data_v2_pilot`. Along the way, fixed a
train/inference prompt-format mismatch that would have silently produced a badly-calibrated
checkpoint, and found (via manual review, same discipline as Week 3's judge fix) two more judge
miscalibrations on adversarial rows. Headline result: SFT does **not** clearly move the harder
`data_v2_pilot` numbers in the right direction — `abstention_recall` regresses moderately,
`hallucination_rate` is flat-to-worse, and `partial_match_rate` is the one clear improvement. Small
n (55 rows) — directional, not conclusive.

## Development

| Item | Path | Notes |
|------|------|-------|
| LoRA SFT training script | [`src/grounded_refusal/train/train_sft.py`](../../src/grounded_refusal/train/train_sft.py) | `train_sft_main` — builds prompt/completion pairs, trains via `trl.SFTTrainer`, saves the adapter |
| Training config | [`configs/train/lora.yaml`](../../configs/train/lora.yaml) | `r=16`, `target_modules=[q_proj, v_proj]`, batch=2 × grad_accum=8, gradient checkpointing, bf16 |
| GPU memory estimate | [`docs/GPU_MEMORY_ESTIMATE.md`](../GPU_MEMORY_ESTIMATE.md) | Per-component estimate for the 16GB RTX 5060 Ti, worked from Qwen2.5-3B's real `config.json` |
| Adapter loading (inference) | [`src/grounded_refusal/inference/hf_backend.py`](../../src/grounded_refusal/inference/hf_backend.py), [`run_inference.py`](../../src/grounded_refusal/inference/run_inference.py) | New `--adapter` flag; without this, a trained checkpoint had no way to be evaluated |
| Judge calibration fixes | [`src/grounded_refusal/eval/judge.py`](../../src/grounded_refusal/eval/judge.py) | Two new few-shot anchors (distractor-entity-with-explicit-separation, false-presupposition) |
| Checkpoint naming/storage | `checkpoints/<timestamp>_<config-name>/`, `.gitignore`, [`checkpoints/README.md`](../../checkpoints/README.md) | Replaces the original `runs/sft_v1` scheme; not committed to git |

## Training run

```bash
PYTHONPATH=src python -m grounded_refusal.train.train_sft
```

50 rows (`data/data_v1_pilot.jsonl`), 3 epochs, effective batch size 16 (2 × grad_accum 8) → 12
optimizer steps total, 21.7s wall clock on the RTX 5060 Ti. Output: `checkpoints/20260823_151044_lora`.

| step (≈epoch) | loss | mean_token_accuracy |
|---:|---:|---:|
| 1 (0.64) | 1.419 | 0.748 |
| 2 (1.0) | 1.004 | 0.812 |
| 3 (1.64) | 0.892 | 0.776 |
| 4 (2.0) | 0.675 | 0.787 |
| 5 (2.64) | 0.735 | 0.792 |
| 6 (3.0) | 0.848 | 0.833 |

Loss trends down with the expected noise at this scale (12 steps total); not a clean monotonic
curve, consistent with a very small dataset.

## Bug found before training was trustworthy: prompt-format mismatch

`train_sft.py` originally built `{"prompt": "<plain string>", "completion": "<plain string>"}`
rows. trl's `SFTTrainer` treats plain strings as its "standard" format — raw concatenation, no
chat template applied. `hf_backend.py`'s inference path, however, wraps that same prompt text in
`tokenizer.apply_chat_template(..., add_generation_prompt=True)` before generating. Training and
inference were therefore seeing differently-formatted input for the same logical prompt — the
model would have been SFT'd on a format it's never actually prompted with at eval time. Fixed by
switching to trl's conversational format (`{"role": "user", ...}` / `{"role": "assistant", ...}`),
which makes `SFTTrainer` apply the same chat template `hf_backend.py` uses. Verified by inspecting
raw inference output afterward — no template-token leakage, clean generations.

## Experiments

**1. Ran the SFT checkpoint against both pilot sets** (`--adapter checkpoints/20260823_151044_lora`):

```bash
PYTHONPATH=src python -m grounded_refusal.inference.run_inference \
  --data data/data_v2_pilot.jsonl \
  --adapter checkpoints/20260823_151044_lora \
  --output outputs/inference-qwen2.5-3b-instruct-sft/20260823_151044_lora_v2_pilot.jsonl
```

`v1_pilot` output was generated but not judged this round (see Limitations).

**2. Judged `data_v2_pilot`, found two more judge miscalibrations.** Same manual-review discipline
as Week 3's `known_world_conflict` fix. Reviewing the judge's raw output against `reference_answer`
by hand turned up two mislabeled rows, both on adversarial-entity/premise cases:

- `ex_0100` (`distractor_entity`): response states a true, evidence-grounded figure for a
  different entity than the one asked about ("the bridge that opened in 1998 carries 40,000
  vehicles..."), then explicitly declines the entity actually asked about ("...2004 is not
  provided, so I cannot provide..."). Judge called this `partial`; it's `refuse` — no value is
  ever asserted as the answer to the entity actually asked about. (Contrast with an existing
  few-shot anchor where a distractor's value *is* asserted directly as the answer, with no
  separation — that one correctly stays `answer`.)
- `ex_0129` (false presupposition — question asks why a trial *failed*; evidence says it
  succeeded): response corrects the false premise and states there's no support for the premised
  failure. Judge called this `partial`; it's `refuse` — answering no part of "why did it fail" by
  establishing that it didn't is a refusal, not a partial answer.

Two similar-looking rows were checked against the same rule and correctly **not** changed:
`ex_0131` (restates a same-entity relative fact and explicitly offers it as confirmed — genuinely
partial, unlike `ex_0100`'s different-entity case) and `ex_0116` (picks one number out of two
directly conflicting evidence sources and asserts it as *the* answer with no hedge — a real
`answer`/faithful-but-concerning case, not a miscalibration).

Both fixes were added as few-shot anchors to `judge.py`, and the full 55-row set was re-judged.

**3. Base vs. SFT on `data_v2_pilot`** (55 rows; both scored with the same, post-fix `judge.py`):

| Model | `abstention_recall` | `abstention_precision` | `over_refusal_rate` | `hallucination_rate` | `partial_match_rate` |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-3B-Instruct (base, Week 3) | 0.6316 | 0.9231 | 0.0333 | 0.1905 | 0.6667 |
| Qwen2.5-3B-Instruct + SFT-v1 LoRA | 0.5789 | 0.9167 | 0.0333 | 0.2326 | **1.0000** |

Reading this: `over_refusal_rate` is unchanged — SFT did not make the model more trigger-happy
about refusing. `partial_match_rate` improved cleanly to perfect. But `abstention_recall` regressed
(0.63 → 0.58) and `hallucination_rate` is flat-to-worse (0.19 → 0.23), not better as an
early, uncorrected read of the raw judge output had suggested (that first pass showed recall
dropping much further, to 0.42 — most of that gap turned out to be the two judge miscalibrations
above, not real model regression, which is exactly why the manual review mattered here).

**Not fully explained**: after the judge fix, the unfaithful-row count *within* the
`hallucination_rate` denominator went from 7 (first raw pass) to 10 (post-fix pass) — a larger
jump than the 2 rows that were deliberately relabeled account for. GPT-4o at `temperature=0` is
mostly but not perfectly deterministic, and adding two new few-shot examples changes the judge's
full context for every row, not just the two intended targets. The original raw per-row output was
overwritten (not archived) before this was noticed, so an exact row-level diff isn't possible after
the fact — this is a real gap in the finding, not a resolved one.

## How to run

```bash
# 1. Train (needs the GPU free -- check nvidia-smi first; ~16GB required per
#    docs/GPU_MEMORY_ESTIMATE.md, ~22s on this machine)
PYTHONPATH=src python -m grounded_refusal.train.train_sft

# 2. Run the trained adapter against a pilot set
PYTHONPATH=src python -m grounded_refusal.inference.run_inference \
  --data data/data_v2_pilot.jsonl \
  --adapter checkpoints/<the-run-you-just-made> \
  --output outputs/inference-qwen2.5-3b-instruct-sft/<name>.jsonl

# 3. Judge it (same one-worker-at-a-time constraint as Week 3 -- gpt-4o TPM limit)
PYTHONPATH=src python -m grounded_refusal.eval.run_eval \
  --input outputs/inference-qwen2.5-3b-instruct-sft/<name>.jsonl \
  --output outputs/eval-qwen2.5-3b-instruct-sft/<name>_eval.jsonl \
  --overwrite --max-workers 1
```

## Limitations

- **Small n, same caveat as Week 3.** 55 rows on `data_v2_pilot`; the abstention matrix's
  denominators are ~19-20 rows. A recall move of 0.63 → 0.58 is directional, not statistically tight.
- **Trained on `data_v1_pilot` only, evaluated cross-distribution on `data_v2_pilot`.** The training
  set is the easier Week 2 pilot; `data_v2_pilot` was deliberately built to be harder and cover
  failure modes (`distractor_entity`, `known_world_conflict`, `partial_evidence`, prompt injection)
  the training set barely touches. Some of the recall regression may be the model overfitting to
  `data_v1_pilot`'s narrower refusal patterns rather than learning refusal in general — not
  distinguishable from this experiment alone.
- **`v1_pilot` SFT output was generated but not judged.** Would show whether the recall regression
  also appears on the (easier, in-distribution) training-adjacent set, or only on the harder
  out-of-distribution one.
- **The hallucination-rate-denominator discrepancy above is unresolved**, not just unreported —
  the data needed to explain it (the original raw judge pass) no longer exists.
- **Single training run, no seed/hyperparameter sweep.** One LoRA config, one seed, 3 epochs. No
  evidence yet on whether the recall regression is a property of SFT-on-this-data in general or an
  artifact of this particular run.
