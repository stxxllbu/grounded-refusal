# Data versions: v1, v2, v3

What each `dataset_version` means, which files belong to it, and why the next one exists. Written
because by the time `v3` showed up, the answer was scattered across `README.md`,
[`DATA_V2_EXTENSION.md`](DATA_V2_EXTENSION.md) and four week reports. **Edit here first** when a
new version is added.

---

## v1 — task definition, hand-built

| File | Rows | Role |
|---|---:|---|
| [`data/hand_examples.jsonl`](../data/hand_examples.jsonl) | 20 | Week 1 gold examples (`split: dev`) |
| [`data/data_v1_pilot_layer1.jsonl`](../data/data_v1_pilot_layer1.jsonl) | 50 | Week 2 Layer 1 templates |
| [`data/data_v1_pilot.jsonl`](../data/data_v1_pilot.jsonl) | 50 | Week 2 Layer 2 paraphrase, the file actually used for SFT/eval |

Built by hand (Layer 1) then LLM-paraphrased for fluency (Layer 2), per
[`QA_GENERATION_PROTOCOL.md`](QA_GENERATION_PROTOCOL.md). A full ~500-row `data_v1.jsonl` was planned
but deferred (`README.md` Status table) and never built. The base model scored close to perfect on
this set (`abstention_recall` 0.95, Week 3), which is why `v2` exists.

## v2 — adversarial, extended, split

| File | Rows | Role |
|---|---:|---|
| [`data/data_v2_pilot.jsonl`](../data/data_v2_pilot.jsonl) | 55 | Week 3 hand-built adversarial set. **Frozen** — never edited after creation, because it was read repeatedly to calibrate the judge ([`JUDGE_MODEL.md`](JUDGE_MODEL.md)) |
| [`data/data_v2.jsonl`](../data/data_v2.jsonl) | 600 | Week 5. The 55 pilot rows first, then 545 new rows, allocated by measuring which phenomena actually broke the base model ([`DATA_V2_EXTENSION.md`](DATA_V2_EXTENSION.md)) |
| [`data/data_v2_train.jsonl`](../data/data_v2_train.jsonl) | 480 | Week 6. Stratified split of the 545 non-pilot rows (80%); all 55 pilot rows excluded |
| [`data/data_v2_heldout.jsonl`](../data/data_v2_heldout.jsonl) | 120 | Week 6. The other 20% of non-pilot rows, plus **all 55 pilot rows**, pinned here so training never sees them |
| [`data/preference_v2_train.jsonl`](../data/preference_v2_train.jsonl) | 480 | Week 6. DPO pairs, 1:1 with `data_v2_train.jsonl` |

`data_v2_heldout.jsonl` is the evaluation set every Week 7 result is reported against
([`week7.md`](reports/week7.md)). Nine of the `tags` values used for adversarial constructions
(`red_herring`, `conditional_logic`, `negation_exception`, `embedded_instruction`,
`false_presupposition`, `circular_evidence`, `digit_confusion`, `near_miss`, `adjacent_metric`) exist
**only** on the 55 pilot rows — pinning them all to held-out to protect judge calibration meant these
nine constructions never appeared in *any* training data, for SFT or DPO. Only four tags scaled into
`data_v2_train.jsonl`: `coreference_ambiguity`, `hedged_uncertainty`, `conflicting_evidence`,
`multi_hop_arithmetic`.

## v3 — closing the nine-tag training gap

Week 7's DPO evaluation found this gap by its effect: DPO's five new over-refusals (rows SFT answered
correctly but DPO refused) all carried one of those nine untrained tags, and DPO's refusals for them
often invented a conflict or missing fact that the evidence didn't actually contain. `v3` exists to
give those nine constructions real training exposure, without touching anything `v2` established.

| File | Rows | Role |
|---|---:|---|
| [`data/data_v3_extension.jsonl`](../data/data_v3_extension.jsonl) | 192 | New rows only: 24 per tag, all 9 previously-untrained tags, `split: train` |
| [`data/data_v3_train.jsonl`](../data/data_v3_train.jsonl) | 672 | `data_v2_train.jsonl`'s 480 rows first, then `data_v3_extension.jsonl`'s 192, same pattern as `data_v2.jsonl` |

**What v3 deliberately does not touch:** `data_v2_pilot.jsonl`, `data_v2.jsonl`, `data_v2_train.jsonl`
and `data_v2_heldout.jsonl` are all unchanged. `data_v2_heldout.jsonl` stays the evaluation set — a
model trained on `data_v3_train.jsonl` is still scored on the exact 120 rows Week 7 used, so the
before/after numbers stay comparable. A separate `v3` held-out set is future work, not done here.

**Tag → answerability, fixed for these nine (used to decide `chosen`/`rejected` when preference pairs
are built for `v3`):**

| Answerable (ignore the trap, answer correctly) | Unanswerable (recognize why it can't be answered) |
|---|---|
| `red_herring`, `conditional_logic`, `negation_exception`, `embedded_instruction`, `digit_confusion` | `false_presupposition`, `circular_evidence`, `near_miss` / `adjacent_metric` |

**Status: 192 rows written and schema-validated, nothing trained on it yet.** 24 rows per tag puts
each of the nine on comparable footing with the already-trained `multi_hop_arithmetic` (22 rows across
all of `v2`, and it still failed once stacked with an untrained tag — `week7.md`'s `ex_0118`) to
`conflicting_evidence` (70 in `data_v2_train.jsonl`). Each tag's 24 rows span distinct domains
(finance, healthcare, transit, manufacturing, sports, government, retail, and more) and vary the
construction itself, not just the nouns — e.g. `red_herring` mixes a plain unused-fact form with a
one-step-arithmetic form, `negation_exception` cycles through five different exception phrasings — so
the rows don't teach one more surface template the way the `v2` refusal wording did (`week7.md`'s
finding). No exact-duplicate evidence text and no id collisions with `v2`, checked directly.

Not done yet: `choose_negative_type` in `build_preference.py` extended to route these nine tags (it
currently only recognizes the four `v2` tags, so no DPO preference pairs exist for `v3` rows yet), and
an SFT/DPO retrain on `data_v3_train.jsonl` to confirm the gap actually closes.

## Quick answer: "which file do I train on"

| Task | File |
|---|---|
| Reproduce a Week 4–7 result | `data_v2_train.jsonl` (SFT) / `preference_v2_train.jsonl` (DPO) |
| Train with the nine-tag gap closed | `data_v3_train.jsonl` |
| Evaluate any of the above | `data_v2_heldout.jsonl` — unchanged since Week 6 |
