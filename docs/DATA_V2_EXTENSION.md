# data_v2 extension: 55 → 600

How `data/data_v2_pilot.jsonl` (55 rows, Week 3 hand-built stress test) grew into
`data/data_v2.jsonl` (600 rows). `data_v2_pilot.jsonl` is left untouched as the
original snapshot; `data_v2.jsonl` is a superset (the same 55 rows first, then
545 new ones appended).

**QA labels:** [`DATA_LABELS.md`](DATA_LABELS.md).
**Base QA construction:** [`QA_GENERATION_PROTOCOL.md`](QA_GENERATION_PROTOCOL.md)
(covers `data_v1_pilot.jsonl`'s Layer1→Layer2 pipeline; this doc's method is
different and is explained below).

---

## Why this isn't just "more of the same recipe"

The original 55-row `data_v2_pilot.jsonl` was hand-built to be harder than
`data_v1_pilot.jsonl`, and it worked — base Qwen2.5-3B-Instruct's
`abstention_recall` dropped from 0.95 (on `data_v1_pilot`) to 0.63 on it. But a
first attempt at scaling this to 600 rows (by writing more examples that were
just harder to *read* — longer sentences, facts buried in paragraphs, more
"clever"-looking traps) turned out to be the wrong axis entirely. Reading is
not the bottleneck for a 3B instruct model; specific failure modes are, and
they don't correlate with human-perceived difficulty.

## Method: read the judge log, don't guess

`outputs/eval-qwen2.5-3b-instruct/base_v2_pilot_eval.jsonl` has base Qwen2.5-3B's
real per-row `EvalResult` for all 55 pilot rows (predicted_behavior, is_faithful,
abstention_outcome, partial_outcome, rationale — see [`EVAL_METRICS.md`](EVAL_METRICS.md)
for what these mean). Reading that file row by row and grouping failures
(`abstention_outcome ∈ {false_positive, false_negative}`, `is_faithful == false`,
or `partial_outcome != match`) by the row's `tags`/`evidence_challenge` gives an
**empirical hit rate per phenomenon** — how often that specific construction
actually broke this specific model — instead of a guess.

### Empirical hit rates, 55-row pilot

| Phenomenon | Hit rate | n | Note |
|---|---|---|---|
| `coreference_ambiguity` | 100% | 3/3 | Model always resolves the ambiguous pronoun to one entity instead of recognizing genuine ambiguity |
| `hedged_uncertainty` | 100% | 2/2 | Model treats "preliminary estimate" / "analysts believe" as usable evidence |
| `distractor_entity` + `partial_evidence` combined | 100% | 1/1 | Stacking traps in one row — got the arithmetic right, misattributed an unrelated detail |
| `conflicting_evidence` | 67% | 2/3 | Both failures picked the *later-appearing* source in the text (recency bias) instead of flagging the conflict |
| `negation_exception` | 50% | 1/2 | Collapses a "below threshold" qualifier into a simple binary |
| `embedded_instruction` | 33% | 1/3 | Different failure shape — model resisted the injected instruction but over-refused a straightforward question out of apparent suspicion |
| `known_world_conflict`, plain single-sentence, not buried, no arithmetic | 33% | 2/6 | Both failures were on *very famous* facts (Boston's renaming history, Springfield's geography) stated with no disguise |
| `known_world_conflict` + multi-hop arithmetic | 0% | 0/5 | |
| `known_world_conflict` + fact buried in a longer paragraph | 0% | 0/4 | |
| `distractor_entity`, plain (no other trap stacked) | 0% | 0/2 | |
| `near_miss` / `adjacent_metric` (unanswerable, evidence gives an adjacent stat) | 0% | 0/2 | |
| `partial_evidence`, plain | 0% | 0/2 | |
| `multi_hop_arithmetic`, plain | 0% | 0/2 | |
| `false_presupposition` | 0% | 0/4 | |
| `circular_evidence` | 0% | 0/3 | |
| `digit_confusion` | 0% | 0/3 | |
| `conditional_logic` | 0% | 0/2 | |

This directly overturned the intuition that "harder to read = harder for the
model." Burying a false fact in a longer paragraph, or requiring multi-hop
arithmetic on top of a false premise, made `known_world_conflict` rows *easier*
for the model (0/9), while a flat, undisguised false statement about a very
famous fact tripped it 33% of the time. The same pattern held for
`distractor_entity`: the subtle, single-mechanism version was never wrong (0/2),
but stacked with `partial_evidence` it broke the model outright (1/1).

## What the 600-row set does with this

The 545 new rows (`ex_0146`–`ex_0690`) are allocated almost entirely to the
phenomena above with a non-zero empirical hit rate, roughly in proportion to
hit rate:

- `coreference_ambiguity` — 121 rows total (pure pronoun/reference ambiguity
  across ~20 domains, plus combined with hedging/distractor/partial).
- `hedged_uncertainty` — 108 rows total (varied hedge markers: preliminary
  estimates, unnamed sources, "has not ruled out", leaked/unverified reports,
  etc., across the same domain spread).
- `conflicting_evidence` — 82 rows total, including deliberate order-reversal
  pairs (same fact conflict, sources swapped in text order) to test whether the
  recency bias replicates.
- `known_world_conflict`, plain and undisguised, using genuinely iconic facts
  (not obscure ones) — 138 `answerable` rows, the main lever used to bring the
  answerable count back up (see below).
- A new, not-yet-empirically-verified pattern: `distractor_entity` where the
  asked-about entity's real value **is** stated directly, but a similarly-named
  unrelated entity's different value is also present — testing misattribution
  risk without necessarily forcing a refusal. ~106 `answerable` rows.
- Everything else — combined/stacked variants (`tier_tangled`: distractor+partial
  +arithmetic, conflicting+partial, coreference+partial, distractor+coreference,
  hedged+distractor, hedged+coreference) — fills the remainder and is also
  where most of the 80 `partial` rows come from.

Phenomena that scored 0% in the pilot (`false_presupposition`, `circular_evidence`,
`digit_confusion`, `conditional_logic`, plain `distractor_entity`, plain
`multi_hop_arithmetic`, `near_miss`) were **not** scaled up; they remain at their
original pilot-scale counts in the 55 legacy rows.

### Final composition (600 rows)

```text
answerability:  answerable 260 / unanswerable 260 / partial 80
evidence_type:  short_paragraph 379 / single_sentence 221
tags (top):     tier_tangled 265, tier_extreme 138, tier_confused 130,
                coreference_ambiguity 121, hedged_uncertainty 108,
                conflicting_evidence 82, tier_stress 67, multi_hop_arithmetic 22
```

`answerability` was deliberately rebalanced back toward 260/260/80 — the four
highest-yield mechanisms (coreference, hedged, conflicting, distractor+partial)
are naturally `unanswerable`/`partial`-heavy in their correct behavior, so
without the `known_world_conflict`-plain and distractor-with-real-value-present
batches the set would have skewed close to 80% "correct answer = refuse,"
which would bias any training/eval built on it toward only testing abstention,
not the full answer/refuse/partial boundary this project's central research
question is about.

## Conventions used for the new rows

- `id`: `ex_0146`–`ex_0690`, continuing from the pilot's last id (`ex_0145`).
  Confirmed not to collide with `data/generated/data_v1_layer1_generated.jsonl`
  (occupies `ex_0071`–`ex_0120`, an unused draft for the deferred full `data_v1`
  build).
- `split`: `"dev"` for every new row, matching the pilot's convention. Train/eval
  re-partitioning of the 600 rows is left to whoever consumes this file — not
  done here.
- `dataset_version`: `"v2"`.
- `metadata.creation_process`: `"llm_generated"` — distinct from the original
  55 rows' `"manual"`, since these were not hand-typed by a person and did not
  go through the Layer1→Layer2 template pipeline either. This is an honest
  provenance label, not a quality claim.
- `evidence_challenge` (the schema's 3-value enum: `distractor_entity`,
  `known_world_conflict`, `partial_evidence`) is set only when one of those
  three literally applies. Every other phenomenon (`coreference_ambiguity`,
  `hedged_uncertainty`, `conflicting_evidence`, `false_presupposition`,
  `circular_evidence`, `embedded_instruction`, `digit_confusion`,
  `negation_exception`, `near_miss`/`adjacent_metric`, `multi_hop_arithmetic`,
  `red_herring`, `conditional_logic`) lives only in the free-text `tags` field,
  same as the original 55 rows. **This is a pre-existing gap, not something this
  extension fixes**: the `EvidenceChallengeTag` enum and `DATA_LABELS.md` only
  formally define the original three tags; the tier system
  (`tier_extreme`/`tier_stress`/`tier_tangled`/`tier_confused`) and every
  phenomenon tag beyond the original three exist only as informal `tags`
  strings, undocumented anywhere before this file.

## What's verified vs. what's a hypothesis

Only the **first 72** new rows (`ex_0146`–`ex_0217`) were designed directly from
the judge-log hit-rate table above — everything downstream of that first batch
(the remaining 473 rows) extrapolates the same mechanisms to more domains and
entity names but has **not** been re-run against the actual base model. The
empirical loop that produced the hit-rate table has not been closed a second
time on the larger set.

**Before trusting this file's difficulty for training or headline eval numbers**,
run `inference/run_inference.py` + `eval/run_eval.py` on a sample (or all of
`ex_0146`–`ex_0690`) and check whether the hit rates above still hold at scale,
particularly for:
- the two new patterns invented for this extension with no pilot precedent —
  the "similarly-named entity, real value stated directly" answerable control,
  and the multi-mechanism `tier_tangled` stacks;
- whether the `conflicting_evidence` recency bias (model favors the
  later-mentioned source) replicates on the order-reversed pairs included here.

## How to run

```bash
PYTHONPATH=src .venv/bin/python -m grounded_refusal.data.validate_qa_jsonl_against_schema data/data_v2.jsonl

PYTHONPATH=src python -m grounded_refusal.inference.run_inference \
  --data data/data_v2.jsonl \
  --output outputs/inference-qwen2.5-3b-instruct/base_v2_full.jsonl

PYTHONPATH=src python -m grounded_refusal.eval.run_eval \
  --input outputs/inference-qwen2.5-3b-instruct/base_v2_full.jsonl \
  --output outputs/eval-qwen2.5-3b-instruct/base_v2_full_eval.jsonl \
  --overwrite --max-workers 1
```
