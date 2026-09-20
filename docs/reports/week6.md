# Week 6 report: data_v2 split, SFT retrain, and the DPO pipeline

## Summary

SFT had been trained on an easier dataset than the one it was evaluated on, a mismatch every SFT
result since Week 4 has inherited, and the DPO training pipeline, originally scheduled for
Week 5, had not yet been built. This week resolved both: SFT was retrained on data matching its
evaluation distribution, and the DPO training pipeline is now complete.

SFT's only prior checkpoint was trained on [`data_v1_pilot`](../../data/data_v1_pilot.jsonl) (50
hand-built rows from [Week 2](week2.md)) and evaluated on
[`data_v2`](../../data/data_v2.jsonl) (600 rows, built in [Week 5](week5.md) specifically to
expose failures the easier set could not). DPO itself was pushed back while that week's judge
recalibration and `data_v2` extension took priority, both prerequisites for a trustworthy DPO
comparison.

This week retrained SFT on `data_v2_train`, a stratified 480-row split of `data_v2` reserved for
training, and held out the remaining 120 rows for evaluation. It also extended preference-pair
generation to cover `data_v2`'s harder failure modes and wrote the DPO training script. Week 7
can now run base, SFT, and DPO on the same held-out split and report a comparison free of both
problems.

## Development

| Item | Path | Notes |
|------|------|-------|
| Train/held-out split | [`split_train_heldout.py`](../../src/grounded_refusal/data/split_train_heldout.py) | Stratified by `answerability` × `evidence_challenge`; pilot rows pinned to held-out |
| Split outputs | [`data_v2_train.jsonl`](../../data/data_v2_train.jsonl) (480), [`data_v2_heldout.jsonl`](../../data/data_v2_heldout.jsonl) (120) | |
| SFT retrain | n/a (checkpoint not committed) | `train_sft.py` on `data_v2_train.jsonl`, same `configs/train/lora.yaml` recipe as Week 4 |
| Preference generation: tag routing | [`build_preference.py`](../../src/grounded_refusal/data/build_preference.py), [`PREFERENCE_GENERATION_PROTOCOL.md`](../PREFERENCE_GENERATION_PROTOCOL.md) | Four new `negative_type` values for `data_v2`'s tag-based failure modes |
| Preference generation: model | `build_preference.py` | `DEFAULT_MODEL`: `gpt-4o-mini` → `gpt-5-mini` |
| Preference data | [`preference_v2_train.jsonl`](../../data/preference_v2_train.jsonl) (480) | |
| DPO training script | [`train_dpo.py`](../../src/grounded_refusal/train/train_dpo.py), [`dpo.yaml`](../../configs/train/dpo.yaml) | LoRA merge into a fresh adapter; no separate reference model |

## Splitting data_v2 for a matched SFT retrain

`split_train_heldout.py` splits the full 600-row [`data_v2.jsonl`](../../data/data_v2.jsonl) into
a training set and a held-out evaluation set of matching difficulty.

### Design

- The 55 [`data_v2_pilot`](../../data/data_v2_pilot.jsonl) rows, the original hand-built
  adversarial set later extended into `data_v2`, are pinned entirely to held-out. They have been
  read and judged repeatedly during judge validation ([`JUDGE_MODEL.md`](../JUDGE_MODEL.md));
  training on them risks the model memorizing the exact rows already used to diagnose its
  failures.
- The remaining 545 rows form seven strata by
  [`(answerability, evidence_challenge)`](../DATA_LABELS.md), 11 to 193 rows each.
- Each stratum is shuffled with a fixed seed (42, matching `configs/train/lora.yaml`) and
  apportioned by largest remainder to bring the overall split to 80/20.

### Result

| Split | Rows | answerable | unanswerable | partial |
|-------|-----:|-----------:|--------------:|--------:|
| `data_v2_train.jsonl` | 480 | 202 | 212 | 66 |
| `data_v2_heldout.jsonl` | 120 | 58 | 48 | 14 |
| Full `data_v2.jsonl` | 600 | 260 | 260 | 80 |

Held-out's `answerable` share (48%) exceeds the full set's (43%) because the 55%-`answerable`
pilot sits entirely inside it. Metrics computed on `data_v2_heldout` alone should be read with
that skew in mind, not treated as representative of `data_v2`'s overall composition.

## Retraining SFT on data_v2_train

The [Week 4](week4.md) SFT pipeline
([`train_sft.py`](../../src/grounded_refusal/train/train_sft.py), `configs/train/lora.yaml`) was
rerun with `--data data/data_v2_train.jsonl` in place of `data/data_v1_pilot.jsonl`. Training and
evaluation now draw from one stratified split of the same 600-row set, rather than two datasets
built at different times for different purposes, removing the mismatch described above.
Checkpoint weights are not committed to git, consistent with
[`checkpoints/README.md`](../../checkpoints/README.md).

**TBD:** checkpoint identifier, epoch count, and training loss, once `run_metadata.json` for this
run is available and added to that file's table.

## Extending preference generation for data_v2's failure modes

### The gap

`build_preference.py`'s `choose_negative_type` mapped only `answerability` and the three
`EvidenceChallengeTag` enum values (`distractor_entity`, `known_world_conflict`,
`partial_evidence`), the taxonomy built for `data_v1_pilot`. `data_v2`'s extension introduced
harder failure modes as free-text `tags` (`coreference_ambiguity`, `hedged_uncertainty`,
`conflicting_evidence`, `multi_hop_arithmetic`) that map could not see, so every row carrying one
fell into a generic `hallucination` or `over_refusal` bucket regardless of which specific
mechanism it tested.

### The fix

`choose_negative_type` now checks these four tags first, in priority order, before falling
through to the original map, each routing to its own new `negative_type` with a matching
`rejected`-writing instruction. `conflicting_evidence`'s instruction, for instance, has the model
pick the later-stated of two conflicting values, reproducing the recency bias
[`DATA_V2_EXTENSION.md`](../DATA_V2_EXTENSION.md) measured in the base model's actual failures,
rather than an arbitrary wrong answer.

### Model change

The generation model changed from `gpt-4o-mini` to `gpt-5-mini`, on the reasoning that reliably
producing a specific target failure mode is a harder instruction-following task than plain text
completion.

### A known trade-off

Because the priority order is a fixed sequence, a row carrying more than one of these tags is
classified by whichever comes first. 20 of the 93 rows tagged `hedged_uncertainty` also carry
`coreference_ambiguity` (19 rows) or `conflicting_evidence` (1 row); both of those tags rank
higher than `hedged_uncertainty`, so those 20 rows are classified under the other tag instead.
`hedged_uncertainty` accounts for 73 of the 480 generated pairs rather than 93.

### Results

Partial-answerability rows carrying one of the four new tags (46 of 545: 19 `conflicting_evidence`,
14 `multi_hop_arithmetic`, 13 `coreference_ambiguity`, no overlap between them) need the generated
`rejected` to introduce exactly one error while leaving the row's already-supported half
untouched; none of the four new instructions state that requirement explicitly. In the three
sampled rows combining a partial answerability with one of these tags (`ex_0513`, `ex_0502`,
`ex_0485`), `gpt-5-mini` preserved the untouched half correctly each time without being told to;
the remaining 43 rows in that combination have not been checked individually.

480 real pairs were generated from [`data_v2_train.jsonl`](../../data/data_v2_train.jsonl), one
API call per row:

| `negative_type` | Count |
|---|---:|
| `memory_override` | 108 |
| `coreference_ambiguity` | 101 |
| `over_refusal` | 94 |
| `hedged_uncertainty` | 73 |
| `conflicting_evidence` | 70 |
| `over_complete` | 20 |
| `multi_hop_arithmetic` | 14 |

All 480 rows validate against `PreferencePair`, and at least one generated pair from each of the
seven `negative_type` values was read manually against its source evidence, question, and
`chosen` answer; none showed a quality problem. `build_preference.py --all` holds every row in
memory and writes the output file only once the run completes; a network interruption during
this run did not cause a failure, but the script has no incremental checkpointing to protect a
future run from one that does not recover.

## The DPO training script: resolving reference-model ambiguity

### The problem

DPO computes its loss from two model states: a policy, updated throughout training, and a
reference, frozen at the state before training starts. Continuing training on the same LoRA adapter SFT
produced would make the reference ambiguous: disabling that adapter to compute a reference value
returns the base model, since that adapter is the only thing distinguishing post-SFT from
pre-SFT, not the post-SFT state the reference is supposed to represent. `huggingface/trl` has an
open issue on this ambiguity
(["Inncorrect reference model used when using pretrained policy adapters", issue #1340](https://github.com/huggingface/trl/issues/1340)).

### The fix

`train_dpo.py` avoids it. `load_merged_sft_model` merges the given SFT adapter into the base
model's weights first, using `peft`'s `merge_and_unload`. A fresh, untrained LoRA adapter is then
attached on top of that merged model; `DPOTrainer` trains this new adapter, while the merged
weights beneath it stay frozen. Reference and policy are now
unambiguous: the merged weights are fixed, and toggling only the new adapter distinguishes the
model's state before this DPO run from its state during it. Passing `peft_config` for that new
adapter without a `ref_model` signals `DPOTrainer` to derive the reference this way instead of
loading a second full copy of the model; passing both raises `ValueError`, confirmed against
[huggingface.co/docs/trl/dpo_trainer](https://huggingface.co/docs/trl/dpo_trainer).

### Hyperparameters

| Setting | Value | Reason |
|---|---|---|
| LoRA `r` / `alpha` / `target_modules` | 16 / 32 / `q_proj`, `v_proj` | Matches the SFT adapter's own configuration |
| `beta` | 0.1 | trl's default; not yet tuned against this data |
| `loss_type` | `sigmoid` | The original DPO formulation; `ipo` and other variants are later ablation work |
| `per_device_train_batch_size` × `gradient_accumulation_steps` | 1 × 16 | DPO scores both `chosen` and `rejected` per example, roughly doubling SFT's per-step memory at the same batch size |
| `max_length` | 512 | Matches SFT; `data_v2_train.jsonl`'s evidence and answers average 21 words |

### Data format

`build_dpo_rows` wraps each pair's `prompt`, `chosen`, and `rejected` fields in role/content form
before handing them to `DPOTrainer`, for the same reason `train_sft.py`'s
`build_prompt_completion_rows` does: it makes the trainer apply the tokenizer's chat template
automatically, matching the format `hf_backend.py` already applies at inference time.

### Known follow-up

`PreferencePair` validation is defined inline in `train_dpo.py` (`validate_preference_jsonl`)
rather than as a standalone module under `grounded_refusal/data/`, unlike
`validate_qa_jsonl_against_schema.py`, which both `train_sft.py` and `run_inference.py` import
and reuse. Extracting it to match that pattern is deferred, not done here.

## Next: Week 7

Run `train_dpo.py` from the `data_v2_train` SFT checkpoint. Run base, SFT, and DPO inference on
`data_v2_heldout.jsonl`, the split none of the three has trained on, and score all three with
`judge.py`. Compare the three independent metric groups from
[`EVAL_METRICS.md`](../EVAL_METRICS.md), watching in particular whether DPO's hallucination rate
drops without its over-refusal rate rising in exchange.
