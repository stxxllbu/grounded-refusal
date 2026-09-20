# Week 6 report: data_v2 split, SFT retrain, and the DPO pipeline

## Summary

Four things got done this week: `data_v2` was split into stratified train and held-out sets,
SFT was retrained on the new training split, preference-pair generation was extended to target
`data_v2`'s specific failure modes, and the DPO training script itself was written.

The split and retrain close out what Week 5 named as the next required step: SFT's only
checkpoint had been trained on easy data (`data_v1_pilot`) and evaluated on hard data
(`data_v2`), confounding every comparison since Week 4. The preference-data and training-script
work catches up DPO itself, which the project's original eight-week plan placed at Week 5 and
which was deferred twice while judge validation and the `data_v2` extension took priority. With
both pieces in place, Week 7 can run a DPO comparison unconfounded by either problem.

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

Fixing the Week 4/5 confound requires training and evaluation data of matching difficulty.
`split_train_heldout.py` produces that split from the full 600-row `data_v2.jsonl`:

The 55 `data_v2_pilot` rows are pinned entirely to held-out. They have already been read and
judged repeatedly for judge validation ([`JUDGE_MODEL.md`](../JUDGE_MODEL.md)); training on them
would let the model see rows this project has already used to characterize its own failures. The
remaining 545 rows are grouped into seven strata by `(answerability, evidence_challenge)`,
ranging from 11 to 193 rows each. Each stratum is shuffled with a fixed seed (42, matching
`configs/train/lora.yaml`) and sampled by a largest-remainder apportionment to bring the overall
split to 80/20.

| Split | Rows | answerable | unanswerable | partial |
|-------|-----:|-----------:|--------------:|--------:|
| `data_v2_train.jsonl` | 480 | 202 | 212 | 66 |
| `data_v2_heldout.jsonl` | 120 | 58 | 48 | 14 |
| Full `data_v2.jsonl` | 600 | 260 | 260 | 80 |

Held-out's `answerable` share (48%) runs above the full set's (43%) because the pilot, which is
54% `answerable`, sits entirely inside it. Metrics computed on `data_v2_heldout.jsonl` alone
should account for that skew rather than treat it as representative of `data_v2`'s overall
composition.

## Retraining SFT on data_v2_train

The existing SFT pipeline ([`train_sft.py`](../../src/grounded_refusal/train/train_sft.py),
`configs/train/lora.yaml`) was rerun with `--data data/data_v2_train.jsonl` in place of
`data/data_v1_pilot.jsonl`. Training and evaluation now draw from the same distribution for the
first time: `data_v2_train.jsonl` and `data_v2_heldout.jsonl` are a stratified split of one
600-row set, not two datasets built at different times for different purposes. Checkpoint
weights are not committed to git, consistent with [`checkpoints/README.md`](../../checkpoints/README.md);
a row documenting this run's `run_metadata.json` belongs in that file's table.

## Extending preference generation for data_v2's failure modes

`build_preference.py`'s `choose_negative_type` mapped only `answerability` × the three
`EvidenceChallengeTag` enum values (`distractor_entity`, `known_world_conflict`,
`partial_evidence`). `data_v2`'s extension introduced harder failure modes as free-text `tags`
(`coreference_ambiguity`, `hedged_uncertainty`, `conflicting_evidence`, `multi_hop_arithmetic`)
that map could not see, so every row carrying one fell into a generic `hallucination` or
`over_refusal` bucket regardless of which specific mechanism it tested.

`choose_negative_type` now checks these four tags first, in priority order, before falling
through to the original map, each routing to its own new `negative_type` with a matching
`rejected`-writing instruction. `conflicting_evidence`'s instruction, for instance, has the
model pick the later-stated of two conflicting values, reproducing the recency bias
[`DATA_V2_EXTENSION.md`](../DATA_V2_EXTENSION.md) measured in the base model's actual failures,
rather than an arbitrary wrong answer.

Because the priority order is a fixed sequence, a row carrying more than one of these tags is
classified by whichever comes first. 20 of the 93 rows tagged `hedged_uncertainty` also carry
`coreference_ambiguity` (19 rows) or `conflicting_evidence` (1 row), which rank higher, so those
20 are classified under the other tag instead; `hedged_uncertainty` accounts for 73 of the 480
generated pairs rather than 93.

The generation model changed from `gpt-4o-mini` to `gpt-5-mini`, reasoning about a specific
failure mode being harder to get right consistently than plain text completion. Partial rows
carrying one of the four new tags (52 of 545, split across `multi_hop_arithmetic`,
`coreference_ambiguity`, and `conflicting_evidence`) need the generated `rejected` to introduce
exactly one error while leaving the row's independently-supported half untouched; none of the
four new instructions state that requirement explicitly. In the three sampled rows combining a
partial answerability with one of these tags (`ex_0513`, `ex_0502`, `ex_0485`), `gpt-5-mini`
preserved the untouched half correctly each time without being told to; the remaining 49 rows in
that combination have not been checked individually.

480 real pairs were generated from `data_v2_train.jsonl`, one API call per row:

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
future run from one that doesn't recover.

## Writing the DPO training script

DPO needs both a policy (updated during training) and a reference (frozen, representing the
model's state before DPO starts) to compute its loss. Continuing training on the same LoRA
adapter SFT produced would make "reference" ambiguous: disabling that adapter to compute a
reference value returns the *base* model, since that adapter is the only thing distinguishing
post-SFT from pre-SFT, not the post-SFT state the reference is supposed to represent. This is a
real, open issue in `huggingface/trl`
(["Inncorrect reference model used when using pretrained policy adapters", issue #1340](https://github.com/huggingface/trl/issues/1340)),
not a hypothetical concern.

`train_dpo.py` avoids it: `load_merged_sft_model` merges the given SFT adapter into the base
model's weights first (`peft`'s `merge_and_unload`), then a fresh, untrained LoRA adapter is
attached on top of that merged model for `DPOTrainer` to train. Reference and policy are now
unambiguous: the merged weights are fixed, and toggling only the new adapter distinguishes
"before this DPO run" from "during it." Passing `peft_config` for that new adapter without a
`ref_model` is what tells `DPOTrainer` to derive the reference this way instead of loading a
second full copy of the model; passing both raises `ValueError` (confirmed against
[huggingface.co/docs/trl/dpo_trainer](https://huggingface.co/docs/trl/dpo_trainer)).

| Setting | Value | Reason |
|---|---|---|
| LoRA `r` / `alpha` / `target_modules` | 16 / 32 / `q_proj`, `v_proj` | Matches the SFT adapter's own configuration |
| `beta` | 0.1 | trl's default; not yet tuned against this data |
| `loss_type` | `sigmoid` | The original DPO formulation; `ipo`/other variants are later ablation work |
| `per_device_train_batch_size` × `gradient_accumulation_steps` | 1 × 16 | DPO scores both `chosen` and `rejected` per example, roughly doubling SFT's per-step memory at the same batch size |
| `max_length` | 512 | Matches SFT; `data_v2_train.jsonl`'s evidence and answers average 21 words |

`build_dpo_rows` wraps each pair's `prompt`/`chosen`/`rejected` in role/content form before
handing them to `DPOTrainer`, the same reason `train_sft.py`'s
`build_prompt_completion_rows` does: it makes the trainer apply the tokenizer's chat template
automatically, matching the format `hf_backend.py` already applies at inference time.

`PreferencePair` validation is defined inline in `train_dpo.py`
(`validate_preference_jsonl`) rather than as a standalone module under `grounded_refusal/data/`,
unlike `validate_qa_jsonl_against_schema.py`, which both `train_sft.py` and `run_inference.py`
import and reuse. Extracting it to match that pattern is deferred, not done here.

## Next: Week 7

Run `train_dpo.py` from the `data_v2_train` SFT checkpoint. Run base, SFT, and DPO inference on
`data_v2_heldout.jsonl` (the split none of the three has trained on) and score all three with
`judge.py`. Compare the three independent metric groups from
[`EVAL_METRICS.md`](../EVAL_METRICS.md), watching in particular whether DPO's hallucination rate
drops without its over-refusal rate rising in exchange.
