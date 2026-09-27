# Week 7 report: the first DPO run and the base / SFT / DPO comparison

## Summary

Week 6 ended with a DPO training script that had never been run and no DPO evaluation. This week
the script ran end to end for the first time, the checkpoint was evaluated on `data_v2_heldout`
next to base and SFT, and the five rows where DPO started refusing questions SFT had answered
were read one by one.

On this held-out set DPO is not better than SFT. Hallucination does not drop (12 unfaithful rows
for both) and over-refusal doubles (4 of 58 answerable rows to 8 of 58). All five new
over-refusals fall on trap types that never appear in the training split, and three of them refuse
by inventing something the evidence does not say, using refusal wording the preference data taught.
This is one run at one setting, so it says where DPO went wrong here, not that DPO cannot help
(see Limitations).

Two problems surfaced on the way and are fixed: a DPO adapter loaded on its own is a different
model from the one that was trained, and `run_metadata.json` recorded the wrong commit.

## Development

| Item | Path | Notes |
|------|------|-------|
| DPO checkpoint | [`checkpoints/20260926_143154_dpo`](../../checkpoints/20260926_143154_dpo) | [`run_metadata.json`](../../checkpoints/20260926_143154_dpo/run_metadata.json), [`trainer_state.json`](../../checkpoints/20260926_143154_dpo/trainer_state.json); weights not committed |
| Training log saved | [`train_dpo.py`](../../src/grounded_refusal/train/train_dpo.py), [`train_sft.py`](../../src/grounded_refusal/train/train_sft.py) | `trainer.state` written to `trainer_state.json` in a `finally` after `trainer.train()` (issue #11); explicit bf16 load in `load_merged_sft_model` (issue #9). PR #16 |
| Inference on a DPO checkpoint | [`hf_backend.py`](../../src/grounded_refusal/inference/hf_backend.py) | Reads `sft_adapter` from the adapter's `run_metadata.json` and merges it first |
| Held-out inference | `outputs/inference-*/` | `base_v2_heldout.jsonl`, `20260913_193703_lora_v2_heldout.jsonl`, `20260926_143154_dpo_v2_heldout.jsonl` |
| Held-out judging | `outputs/eval-*/` | The same three, `*_eval_judge-gpt5-mini.jsonl` |
| Checkpoint index | [`checkpoints/README.md`](../../checkpoints/README.md) | Now two tables, SFT and DPO; the DPO table records which SFT adapter each starts from and beta |

## The DPO run

`train_dpo.py` continued from the `data_v2_train` SFT checkpoint `20260913_193703_lora` on
[`preference_v2_train.jsonl`](../../data/preference_v2_train.jsonl) (480 pairs), with
[`dpo.yaml`](../../configs/train/dpo.yaml) unchanged: beta 0.1, sigmoid loss, learning rate 5e-6,
batch 1 x gradient accumulation 16, 3 epochs. That is 90 optimizer steps, 361 seconds on the
16GB card, with no out-of-memory error.

Each logged value is the mean of 32 pairs (2 optimizer steps of 16 pairs), so single points are
noisy. Means per epoch:

| Metric | Epoch 1 | Epoch 2 | Epoch 3 |
|---|---:|---:|---:|
| `loss` | 0.622 | 0.437 | 0.335 |
| `rewards/margins` | 0.16 | 0.69 | 1.11 |
| `rewards/accuracies` | 0.79 | 0.97 | 0.99 |
| `rewards/chosen` | +0.001 | -0.017 | -0.046 |
| `rewards/rejected` | -0.16 | -0.71 | -1.15 |
| `grad_norm` | 7.35 | 5.25 | 4.17 |

- The first logged loss was 0.683, close to ln 2 = 0.693, as expected when the policy equals the
  reference at the start.
- The margin grows because `rewards/rejected` falls; `rewards/chosen` stays near 0. Converted with
  `reward / beta`, the rejected answers lost about 11 nats of log-probability against the reference and
  the chosen answers about 0.3.
- `rewards/accuracies` first reached 1.00 at step 24, inside the second epoch. Chosen and rejected
  answers were easy to tell apart.
- `grad_norm` was above the default clip value of 1.0 at every logged step (minimum 3.47), so every
  update was clipped.
- These are training-set numbers. `train_dpo.py` passes no evaluation set, so the log holds no
  `eval_*` metrics.

`run_metadata.json` recorded `git_commit: a140cbe`, the HEAD when the run started. The run used the
dtype and logging changes, which were still uncommitted then, so the value was corrected by hand to
`0ba0220`, the commit that contains the code as it ran.

## Two problems found on the way

**Issue #9 did not reproduce.** `load_merged_sft_model` loaded the base model without a dtype, which
the issue expected to fall back to fp32. On the installed transformers 5.14.1 a missing dtype already
means `"auto"`, which reads `bfloat16` from the model's `config.json`; both ways put 5.85 GB of weights
on the GPU. The explicit `torch_dtype=torch.bfloat16` now removes the dependence on the library
version, and does not fix an out-of-memory error that was happening.

**A DPO adapter alone is not the trained model.** The DPO adapter was trained on top of the base model
with the SFT adapter already merged in (Week 6), so it stores only the change made on top of SFT. Its
`adapter_config.json` names only the plain base model; the link to SFT exists only as `sft_adapter` in
`run_metadata.json`. `run_inference.py --adapter` used to load one adapter onto the plain base, which
loads without error and gives a different model. On 8 held-out prompts (greedy) the outputs of
"base + DPO adapter" matched the correct "base, merge SFT, then DPO" model in 0 of 8. `hf_backend.py`
now reads `sft_adapter` and merges that adapter first; after the change the DPO checkpoint reproduced
the correct outputs for 8 of 8 prompts and an SFT checkpoint was unchanged (8 of 8). No DPO output had
been generated before the fix.

## Held-out evaluation

Setup: `data_v2_heldout.jsonl`, 120 rows (58 answerable, 48 unanswerable, 14 partial), none used
for training by any of the three models. Base `Qwen2.5-3B-Instruct`, the SFT checkpoint and the DPO
checkpoint used the same settings (`base.yaml`: greedy, 256 new tokens). No output was empty or hit
the token limit. Every row was judged with gpt-5-mini, once; no judge call failed. Metric
definitions are in [`EVAL_METRICS.md`](../EVAL_METRICS.md).

| Metric | base | SFT | DPO |
|---|---:|---:|---:|
| `abstention_recall` | 0.646 (31/48) | 0.917 (44/48) | 0.938 (45/48) |
| `abstention_precision` | 0.969 (31/32) | 0.917 (44/48) | 0.849 (45/53) |
| `over_refusal_rate` | 0.017 (1/58) | 0.069 (4/58) | 0.138 (8/58) |
| `hallucination_rate` | 0.218 (19/87) | 0.167 (12/72) | 0.179 (12/67) |
| `partial_match_rate` | 0.500 (7/14) | 0.929 (13/14) | 0.929 (13/14) |
| `partial_under_deliver_rate` | 0.071 (1/14) | 0 | 0 |
| `partial_over_deliver_rate` | 0.429 (6/14) | 0.071 (1/14) | 0.071 (1/14) |

**Base to SFT** is a large change in the expected direction: recall rises from 31 to 44 of 48,
partial handling from 7 to 13 of 14, at the cost of over-refusal going from 1 to 4 rows.

**SFT to DPO** is a small change in the wrong direction for the project's question (lower
hallucination without more over-refusal):

- Hallucination: both have 12 unfaithful rows. DPO's rate is higher only because it attempts fewer
  rows (67 against 72). Only 5 of the 12 are the same rows for both, so DPO makes different mistakes,
  not fewer.
- Over-refusal: 5 answerable rows that SFT answered are refused by DPO, and 1 the other way, a net
  rise from 4 to 8.
- Refusal on unanswerable rows: DPO refuses 2 more correctly and 1 fewer, a net 44 to 45.
- 81 of the 120 outputs differ from SFT's, so DPO changed the model; it did not change it toward fewer
  errors.

Splitting the rows by whether their free-text `tags` occur in `data_v2_train` (the grouping proposed in
issue #14; group sizes 51, 47 and 22 match the issue):

| Group | Answerable rows | Over-refusals base / SFT / DPO | Unanswerable rows | Correct refusals base / SFT / DPO |
|---|---:|---|---:|---|
| No tags | 43 | 0 / 1 / 1 | 2 | 2 / 2 / 2 |
| Only tags seen in training | 2 | 0 / 1 / 1 | 37 | 20 / 33 / 34 |
| Any tag not seen in training | 13 | 1 / 2 / **6** | 9 | 9 / 9 / 9 |

Almost all of DPO's extra over-refusals fall in the last group (2 to 6 of 13 answerable rows).

## The five new over-refusals, read one by one

All five are answerable by their gold labels and all carry a tag that appears only on the 55
`data_v2_pilot` rows, never in training.

| Row | Tags | What DPO did | Reading |
|---|---|---|---|
| `ex_0118` | `multi_hop_arithmetic`, `red_herring` | Called the $95M industry average and the computed $105M "a contradiction" and declined | Refusal by inventing a conflict; the industry figure is a distractor. SFT computed $52.5M correctly |
| `ex_0122` | `conditional_logic`, `red_herring` | "'Annual revenue' isn't quantified here... The regulation itself is missing." | Refusal on two false claims: the evidence gives $62M and states the rule |
| `ex_0123` | `conditional_logic`, `red_herring` | "'Annual revenue' isn't given for this year -- only a $38 million figure for last year." | Refusal on a misreading: the evidence says $38M this year |
| `ex_0138` | `embedded_instruction` | Restated 210 and the editor's 500, then "I can't provide a definitive number" | Real refusal. SFT's answer was not clean either (it restated both figures without settling on 210) and the judge accepted it as an answer |
| `ex_0143` | `negation_exception` | "nowhere are contamination results reported as positive, so I can't say" | Real refusal on a question that is ambiguous (does a trace below the threshold count as positive) |

In each case the judge's behavior label matches a plain reading, so these are DPO's refusals, not
judge errors.

The quoted-word opening in `ex_0122` and `ex_0123` comes from the training data. 100 of the 480
`chosen` answers begin with a quoted word, all of them `coreference_ambiguity` pairs, in the form
`'She' follows a mention of both A and B, so I can't say who...`. Held-out outputs that start with a
quoted word: base 0 of 120, SFT 21, DPO 32. SFT's training answers already contain the pattern, and DPO
increased its use. The "there's a contradiction" wording in `ex_0118` matches the
`conflicting_evidence` pairs. In three of the five rows the refusal is also unfaithful (false claims about
what the evidence contains), which is worse than a plain "I don't know".

A possible cause, not tested: in `preference_v2_train.jsonl` 244 pairs come from the failure modes whose
`chosen` declines to give the disputed value (`coreference_ambiguity` 101, `hedged_uncertainty` 73,
`conflicting_evidence` 70), against 202 whose `chosen` answers from the evidence (`memory_override` 108,
`over_refusal` 94). The counts are in [Week 6](week6.md#results); no pair is a plain `hallucination` or
`distractor_confusion` pair.

## Limitations

- One DPO run at one setting (beta 0.1 and the other hyperparameters untuned), evaluated once by one judge
  run on 120 rows. The over-refusal counts 4 of 58 and 8 of 58 have overlapping 95% intervals (about 0.03 to
  0.16 and 0.07 to 0.25), so the rise is a direction, not a significant difference.
- The "tag not seen in training" group has only 13 answerable rows (2 to 6 over-refusals).
- The judge is not consistent with itself. For `ex_0143`, SFT's answer "so it tested positive" was
  judged unfaithful here, while [`JUDGE_MODEL.md`](../JUDGE_MODEL.md) records gpt-5-mini judging a
  near-identical SFT answer to the same evidence as faithful. Judge noise was not measured, so
  differences of a few rows in `hallucination_rate` should be read with care.
- The held-out set contains the 55 pilot rows already used for judge calibration (Week 6), so it is a
  held-out set, not a clean test set.
- Issue #15 (`ex_0133` and `ex_0135`, which are held-out rows whose labels are disputed) is not
  resolved; relabeling could move the abstention numbers by up to two rows.
- The three-group split above was computed by a one-off script that is not in the repository, and
  `run_eval.py` prints the metrics without saving them. They can be reproduced by rerunning it with `--resume`
  once every row is judged (no API calls).

## How to run

Everything from the repository root. Inference needs the GPU (about 1.5 minutes per model); judging needs
`OPENAI_API_KEY` (one call per row, 120 rows per model, about 6 minutes).

```bash
# DPO training, from the SFT checkpoint
python -m grounded_refusal.train.train_dpo --sft-adapter checkpoints/20260913_193703_lora

# Inference on the held-out split: base, SFT, DPO (--adapter of a DPO checkpoint merges its SFT adapter itself)
python -m grounded_refusal.inference.run_inference --data data/data_v2_heldout.jsonl \
  --output outputs/inference-qwen2.5-3b-instruct/base_v2_heldout.jsonl
python -m grounded_refusal.inference.run_inference --data data/data_v2_heldout.jsonl \
  --adapter checkpoints/20260913_193703_lora \
  --output outputs/inference-qwen2.5-3b-instruct-sft/20260913_193703_lora_v2_heldout.jsonl
python -m grounded_refusal.inference.run_inference --data data/data_v2_heldout.jsonl \
  --adapter checkpoints/20260926_143154_dpo \
  --output outputs/inference-qwen2.5-3b-instruct-dpo/20260926_143154_dpo_v2_heldout.jsonl

# Judge each output (one at a time; rerun with --resume if it stops)
python -m grounded_refusal.eval.run_eval \
  --input outputs/inference-qwen2.5-3b-instruct/base_v2_heldout.jsonl \
  --output outputs/eval-qwen2.5-3b-instruct/base_v2_heldout_eval_judge-gpt5-mini.jsonl
```

## Next

Open questions rather than decisions:

- Resolve the tag-group breakdown in code so it is reproducible (issue #14), and decide whether to relabel
  the disputed rows first (issue #15).
- The preference data leans toward refusal and reuses a few refusal templates; rebalancing it (more pairs
  whose `chosen` answers, fewer near-identical refusal wordings) and tuning beta are the two obvious
  levers for a second DPO run.
- Failure analysis of the remaining held-out errors (Module 6 in [`EVAL_METRICS.md`](../EVAL_METRICS.md)),
  for example the 12 unfaithful rows each of SFT and DPO, of which only 5 are shared.
