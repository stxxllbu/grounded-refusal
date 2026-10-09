# Data versions: v1, v2, v3a, v3b, v3c

What each `dataset_version` means, which files belong to it, and why the next one exists. Written
because by the time `v3a` showed up, the answer was scattered across `README.md`,
[`DATA_V2_EXTENSION.md`](DATA_V2_EXTENSION.md) and four week reports. **Edit here first** when a
new version is added.

**Naming rule:** a version gets a new letter suffix (`v3a` → `v3b` → `v3c` ...) whenever a change
touches a field a training script actually reads and learns from: `evidence`, `question`,
`reference_answer`, `answerability`, or a preference pair's `chosen`/`rejected`. A change to a
purely bookkeeping field (`id`, `tags`, `split`, `metadata`) does not need a new letter, since it
can't change what a model trained on the file would learn — for example, `data_v2_train.jsonl`'s
`split` field was corrected in place after checkpoints already existed, without bumping `v2`,
because nothing reads `split` at training or eval time. The point of the rule is that a checkpoint's
`run_metadata.json` always names a file whose training-relevant content matches what the checkpoint
was actually trained on.

---

## v1 — task definition, hand-built

| File | Rows | Role |
|---|---:|---|
| [`data/hand_examples.jsonl`](../data/hand_examples.jsonl) | 20 | Week 1 gold examples (`split: dev`) |
| [`data/data_v1_pilot_layer1.jsonl`](../data/data_v1_pilot_layer1.jsonl) | 50 | Week 2 Layer 1 templates |
| [`data/data_v1_pilot.jsonl`](../data/data_v1_pilot.jsonl) | 50 | Week 2 Layer 2 paraphrase, the file actually used for SFT/eval |

Built by hand (Layer 1) then LLM-paraphrased for fluency (Layer 2), per
[`QA_GENERATION_PROTOCOL.md`](QA_GENERATION_PROTOCOL.md). A full ~500-row `data_v1.jsonl` was planned
but deferred (`README.md` Status table) and never built.

## v2 — adversarial, extended, split

| File | Rows | Role |
|---|---:|---|
| [`data/data_v2_pilot.jsonl`](../data/data_v2_pilot.jsonl) | 55 | Week 3 hand-built adversarial set. **Frozen** — never edited after creation, because it was read repeatedly to calibrate the judge ([`JUDGE_MODEL.md`](JUDGE_MODEL.md)) |
| [`data/data_v2.jsonl`](../data/data_v2.jsonl) | 600 | Week 5. The 55 pilot rows first, then 545 new rows, allocated by measuring which phenomena actually broke the base model ([`DATA_V2_EXTENSION.md`](DATA_V2_EXTENSION.md)) |
| [`data/data_v2_train.jsonl`](../data/data_v2_train.jsonl) | 480 | Week 6. Stratified split of the 545 non-pilot rows (80%); all 55 pilot rows excluded |
| [`data/data_v2_heldout.jsonl`](../data/data_v2_heldout.jsonl) | 120 | Week 6. The other 20% of non-pilot rows, plus **all 55 pilot rows**, pinned here so training never sees them |
| [`data/preference_v2_train.jsonl`](../data/preference_v2_train.jsonl) | 480 | Week 6. DPO pairs, 1:1 with `data_v2_train.jsonl` |

Nine of the `tags` values used for adversarial constructions (`red_herring`, `conditional_logic`,
`negation_exception`, `embedded_instruction`, `false_presupposition`, `circular_evidence`,
`digit_confusion`, `near_miss`, `adjacent_metric`) exist **only** on the 55 pilot rows — pinning them
all to held-out to protect judge calibration meant these nine constructions never appeared in *any*
training data, for SFT or DPO. Only four tags scaled into `data_v2_train.jsonl`:
`coreference_ambiguity`, `hedged_uncertainty`, `conflicting_evidence`, `multi_hop_arithmetic`.

## v3 — closing the nine-tag gap, then correcting inherited labels

The `v3` line builds on `v2` without editing any `v2` file. It comes in lettered snapshots, because
each step changes training-relevant content and a checkpoint's `run_metadata.json` needs to name the
exact snapshot it was trained on:

- **`v3a`** gives the nine tags described above real training exposure.
- **`v3b`** corrects `coreference_ambiguity` labels that `v2` got wrong, in both the training data
  and the dev set.
- **`v3c`** is a control for `v3b`: the same labels, with 18 of the corrected training answers
  rewritten in the wording the other `coreference_ambiguity` answers use.

### v3a — the nine-tag extension

`v3a` is the first `v3` snapshot: 192 new rows, 24 per tag, covering the nine tags.

| File | Rows | Role |
|---|---:|---|
| [`data/data_v3a_extension.jsonl`](../data/data_v3a_extension.jsonl) | 192 | New rows only: 24 per tag, all 9 previously-untrained tags, `split: train` |
| [`data/data_v3a_train.jsonl`](../data/data_v3a_train.jsonl) | 672 | `data_v2_train.jsonl`'s 480 rows first, then `data_v3a_extension.jsonl`'s 192, same pattern as `data_v2.jsonl` |

**What v3a deliberately does not touch:** `data_v2_pilot.jsonl`, `data_v2.jsonl`, `data_v2_train.jsonl`
and `data_v2_heldout.jsonl` are all unchanged, so `data_v2_heldout.jsonl` remains a fixed evaluation
set across `v2`- and `v3a`-trained models. A separate `v3a` held-out set is future work, not done here.

**Tag → answerability, fixed for these nine (used to decide `chosen`/`rejected` when preference pairs
are built for `v3a`):**

| Answerable (ignore the trap, answer correctly) | Unanswerable (recognize why it can't be answered) |
|---|---|
| `red_herring`, `conditional_logic`, `negation_exception`, `embedded_instruction`, `digit_confusion` | `false_presupposition`, `circular_evidence`, `near_miss` / `adjacent_metric` |

`data/data_v3a_extension.jsonl` has 24 rows per tag. Each tag's 24 rows span distinct domains (finance,
healthcare, transit, manufacturing, sports, government, retail, and more) and vary the construction
itself, not just the nouns: `red_herring` mixes a plain unused-fact form with a one-step-arithmetic
form, and `negation_exception` cycles through five different exception phrasings. No exact-duplicate
evidence text and no id collisions with `v2`, checked directly.

`data/preference_v3a_extension.jsonl` and `data/preference_v3a_train.jsonl` are the corresponding
preference pairs, generated by `build_preference.py` (see that script and
[`PREFERENCE_GENERATION_PROTOCOL.md`](PREFERENCE_GENERATION_PROTOCOL.md) for how `negative_type` is
chosen and how pairs are built).

`v3a` inherits `v2`'s labels unchanged, including the `coreference_ambiguity` mislabels that `v3b`
corrects.

### v3b — correcting the `coreference_ambiguity` labels

`v3b` fixes 26 rows that `v2` labeled `unanswerable` even though the evidence answers them, in whole
or in part. All 26 carry the `coreference_ambiguity` tag. The fix covers both the training data and
the dev set, and the corrected answers carry through to the preference pairs.

**The problem.** A `coreference_ambiguity` row tests whether the model declines to guess who a pronoun
refers to. `v2` got this label wrong in two situations.

The first situation is a pronoun that is not actually ambiguous. In `ex_0133`, the evidence says
that Northwood Capital acquired Silverline Systems and that "It relocated its headquarters to Austin
the following year." The pronoun continues the subject of the previous sentence, so it refers to
Northwood Capital.

The second situation is a question that asks two things. In `ex_0581`, the question asks who
suggested foul play and whether a formal report has been filed. The pronoun makes the first part
unanswerable, but the evidence answers the second part directly: no formal report has been filed.

**The rule.** [Issue #15](https://github.com/stxxllbu/grounded-refusal/issues/15) checked all 121
`coreference_ambiguity` rows in `data_v2.jsonl` against a rule for when a pronoun is ambiguous.

- A pronoun is ambiguous only when it follows two parallel candidates and nothing in the text
  separates them. An example is "Analysts Tara Blevins and Owen Marsh both submitted preliminary risk
  assessments… She flagged significant exposure."
- A pronoun is resolved when ordinary reading settles it. The most common case is a pronoun that
  continues the previous sentence's subject.
- Gender is never inferred from a name.

[Issue #18](https://github.com/stxxllbu/grounded-refusal/issues/18) added one more point. An
ambiguous pronoun makes only its own part of the question unanswerable. If the question also asks
something the evidence answers, that part stays answerable.

To confirm that #18 found every row of this kind, all `unanswerable` rows in `data_v2.jsonl` and
`data_v3a_extension.jsonl` whose question asks two things were checked. There are 25. Besides the 23
rows in #18, the other two (`ex_0115` and `ex_0171`) cannot be answered in either part, so their
`unanswerable` label is correct.

**The fix.**

| Rows | Old label | New label | Issue |
|---|---|---|---|
| `ex_0133`, `ex_0135`, `ex_0157` | `unanswerable` | `answerable` | [#15](https://github.com/stxxllbu/grounded-refusal/issues/15) |
| `ex_0572` to `ex_0594` (23 rows) | `unanswerable` | `partial` | [#18](https://github.com/stxxllbu/grounded-refusal/issues/18) |

In the three rows relabeled `answerable`, the pronoun continues the previous sentence's subject. The
answers are Northwood Capital, the prototype, and Grace Whitfield. These rows also lose the
`coreference_ambiguity` tag, because their pronoun is not ambiguous. This matters for preference
generation, as described below.

In the 23 rows relabeled `partial`, the question asks who did something and whether something has
been finalized, filed, or issued. The first part is ambiguous, and the evidence answers the second.
These rows gain the two fields the schema requires for partial rows. `question_decomposition` names
the two sub-questions, for example `["foul_play_suggester", "formal_report_filed"]`, and
`supported_subquestions` lists the one the evidence answers, here `["formal_report_filed"]`. They also
get `evidence_challenge: ["partial_evidence"]`, which matches the other `coreference_ambiguity` rows
that were already `partial`.

All 26 rows get a new `reference_answer` and `dataset_version: v3b`. Each new answer is written for
its own row in varied wording, since a shared answer template is a known problem of its own (issue
#19). No `evidence` or `question` text changes.

**Files.**

| File | Rows | Role |
|---|---:|---|
| [`data/data_v3b_train.jsonl`](../data/data_v3b_train.jsonl) | 672 | `data_v3a_train.jsonl` with 19 training rows corrected |
| [`data/data_v3b_heldout.jsonl`](../data/data_v3b_heldout.jsonl) | 120 | `data_v2_heldout.jsonl` with 7 dev rows corrected. The dev set for `v3b`-trained models |
| [`data/preference_v3b_train.jsonl`](../data/preference_v3b_train.jsonl) | 672 | `preference_v3a_train.jsonl` with the 19 matching pairs updated |

The 19 training rows are `ex_0157` and 18 of the 23 partial rows. The 7 dev rows are `ex_0133`,
`ex_0135`, `ex_0572`, `ex_0581`, `ex_0586`, `ex_0587`, and `ex_0590`. The label counts change as
follows.

| File | answerable | partial | unanswerable |
|---|---:|---:|---:|
| `data_v3a_train.jsonl` | 322 | 66 | 284 |
| `data_v3b_train.jsonl` | 323 | 84 | 265 |
| `data_v2_heldout.jsonl` | 58 | 14 | 48 |
| `data_v3b_heldout.jsonl` | 60 | 19 | 41 |

**Preference pairs.** In each of the 19 affected pairs, `chosen` is now the corrected
`reference_answer`. The `rejected` answers are handled in two ways.

The 18 `partial` pairs keep their `rejected` answers. Each one already guesses one of the two names
and then answers the status question correctly, as in "Julian Cole thought the pitch needed more
data, and no formal feedback has been sent." Under the new label this is the right negative, because
it differs from `chosen` only in guessing who the pronoun refers to.

Under the old label, these pairs worked against the model. The old `chosen` answered neither part of
the question, while the `rejected` answer got the status part right. DPO was therefore pushing the
model away from the correct half of the answer. The new `chosen` removes this conflict.

`pref_0157` needed a new `rejected` answer. With the tag removed, `build_preference.py` routes this
row to `over_refusal`, so the regenerated `rejected` is a refusal claiming the pronoun is ambiguous.
That is the answer this row used to treat as correct. It was generated with gpt-5-mini in a single
API call.

**Checks.** All three files validate against their schemas: `QAExample` for the two QA files and
`PreferencePair` for the preference file. Each file was also compared field by field with the file it
was copied from. Only the fields described above change, and only in the 26 rows and 19 pairs listed.

Within `preference_v3b_train.jsonl`, every pair is consistent with its row in
`data_v3b_train.jsonl`. All 672 `prompt` values rebuild exactly from that row. Every `chosen` equals
that row's `reference_answer`. Every `negative_type` matches what `build_preference.py` would choose
for that row today.

**What stays unchanged.** `data_v2_heldout.jsonl` and `data_v2_pilot.jsonl` are not edited, so every
number already reported against them still matches its file.

A `v2` or `v3a` checkpoint can also be scored against `data_v3b_heldout.jsonl` without new inference
or judging. The prompts are identical and the judge never sees the gold labels, so only the labels
used for scoring change.

**Not in v3b.** Issue #19 is not addressed here. It concerns the wording of 243 training answers
across three tags, and it is left for a later snapshot.

### v3c — a wording control for v3b

`v3c` changes the wording of 18 training answers and nothing else. It exists to separate two changes
that `v3b` made at the same time.

**The problem.** For the 18 training rows that `v3b` relabeled from `unanswerable` to `partial`
(the training share of `ex_0572` to `ex_0594`), `v3b` changed two things in one step:

- The content of the answer. The `v3a` answer declined both parts of the question. The `v3b` answer
  declines the "who" part and answers the status part.
- The wording of the answer. The `v3a` answer used the opening shared by almost every
  `coreference_ambiguity` answer, `'She' follows a mention of both A and B`. The `v3b` answer for
  each row was written in its own wording.

A model trained on `v3b` can therefore differ from one trained on `v3a` on these rows because of
either change, and the two cannot be told apart.

**The fix.** `v3c` keeps the `v3b` labels and content and returns the wording to the shared form.
The 13 `coreference_ambiguity` rows that were already `partial` before `v3b` use this form:

```text
'She' follows a mention of both A and B, so I can't say who <did the thing>. <The status>, however, <is such>.
```

The 18 rows now follow it too. For `ex_0573`:

| Version | Label | `reference_answer` |
|---|---|---|
| `v3a` | `unanswerable` | 'She' follows a mention of both Dana Okafor and Priya Iyer, and since the recommendations are also described as informal and not adopted, there's neither a way to tell which consultant is meant nor a formal decision to report. |
| `v3b` | `partial` | Nothing has been formally adopted. As for who recommended against the merger, the passage names both Dana Okafor and Priya Iyer and doesn't say which one "she" is. |
| `v3c` | `partial` | 'She' follows a mention of both Dana Okafor and Priya Iyer, so I can't say who recommended against the merger. Nothing, however, has been formally adopted. |

The opening of each new answer, with its pronoun and names, is taken from that row's `v3a` answer.
The status sentence restates the fact the `v3b` answer gives, in the evidence's own words. The 18
answers were written by hand, with no API call.

This gives two comparisons that each change one thing. `v3b` and `v3c` differ only in the wording of
these 18 answers. `v3a` and `v3c` differ in the label corrections, with the wording held to the
shared form.

**Files.**

| File | Rows | Role |
|---|---:|---|
| [`data/data_v3c_train.jsonl`](../data/data_v3c_train.jsonl) | 672 | `data_v3b_train.jsonl` with 18 `reference_answer` values rewritten |
| [`data/preference_v3c_train.jsonl`](../data/preference_v3c_train.jsonl) | 672 | `preference_v3b_train.jsonl` with the 18 matching `chosen` values updated |

The 18 rows are `ex_0573` to `ex_0580`, `ex_0582` to `ex_0585`, `ex_0588`, `ex_0589`, and `ex_0591`
to `ex_0594`. They and their pairs get `dataset_version: v3c`. The label counts are the same as
`data_v3b_train.jsonl`.

**What stays unchanged.** `ex_0157`, the one training row `v3b` relabeled `answerable`, keeps its
`v3b` answer; the shared wording is a refusal, so it has no form for a row that should be answered.
The `rejected` answers and every `negative_type` are the same as in `v3b`. There is no `v3c`
held-out file: a `v3c`-trained model is evaluated on `data_v3b_heldout.jsonl`, whose labels already
match, and whose answer wording is never used in scoring.

**Checks.** Both files validate against their schemas. Compared field by field with the `v3b` files,
only `reference_answer` (or `chosen`) and `dataset_version` change, and only in the 18 rows and 18
pairs listed. Every `chosen` equals its row's `reference_answer`, all 672 `prompt` values rebuild
from their rows, and every `negative_type` matches what `build_preference.py` would choose today.

**A known cost.** `v3c` raises the number of training answers that follow this one form from 13 to
31, which is the concentration issue #19 describes. `v3c` is a control for reading the `v3b` result,
not a replacement for `v3b` as the current recipe.

## Quick answer: "which file do I use"

| Task | File |
|---|---|
| Train the current recipe | `data_v3b_train.jsonl` (SFT) / `preference_v3b_train.jsonl` (DPO) |
| Evaluate a `v3b`-trained model | `data_v3b_heldout.jsonl` |
| Run the `v3c` wording control | `data_v3c_train.jsonl` (SFT) / `preference_v3c_train.jsonl` (DPO), evaluated on `data_v3b_heldout.jsonl` |
| Reproduce the `v3a` checkpoints | `data_v3a_train.jsonl` (SFT) / `preference_v3a_train.jsonl` (DPO) |
| Reproduce the `v2` checkpoints | `data_v2_train.jsonl` (SFT) / `preference_v2_train.jsonl` (DPO) |
| Evaluate `v2`/`v3a` checkpoints as originally reported | `data_v2_heldout.jsonl` |
