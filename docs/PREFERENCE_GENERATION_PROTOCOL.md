# Preference generation protocol

How we build preference JSONL from approved QA rows: `data/preference_v1_pilot.jsonl`, and later `data/preference_v1.jsonl`.

**QA construction (must be done first):** [`QA_GENERATION_PROTOCOL.md`](QA_GENERATION_PROTOCOL.md).  
**QA labels:** [`DATA_LABELS.md`](DATA_LABELS.md).  
**Schema:** [`src/grounded_refusal/data/schema_pref.py`](../src/grounded_refusal/data/schema_pref.py) (`PreferencePair`, `NegativeType`).

Each preference row is one DPO training pair derived from an approved QA row and linked to it through `base_example_id`. Preference generation does not rebuild or relabel the QA data.

---

## Principle

1. Start from an **approved** QA row (pilot: `data_v1_pilot.jsonl`).
2. Choose `negative_type` from that row’s `answerability` × `evidence_challenge` (fixed map; not LLM-chosen).
3. Assemble `prompt`; set `chosen` = QA `reference_answer`.
4. Write one deliberate bad answer as `rejected` (API or human), matching that `negative_type`.
5. Pilot default: **1 QA → 1 preference pair**. More pairs per QA are optional later.

Preference generation reuses `answerability` and `evidence_challenge` from the
source QA row, then assigns `negative_type` to describe how `rejected` fails:

- `answerability`: the correct response behavior.
- `evidence_challenge`: why the QA item is difficult.
- `negative_type`: how the `rejected` response fails.

---

## Fields

| Field | Required | Purpose |
|-------|----------|---------|
| `id` | yes | Preference id, e.g. `pref_0021` (optional letter suffix for multiple pairs: `pref_0021a`). |
| `base_example_id` | yes | Links to QA `id`, e.g. `ex_0021`. |
| `prompt` | yes | Full model input assembled from the source QA row and prompt config. |
| `chosen` | yes | Preferred answer (= QA `reference_answer`). |
| `rejected` | yes | One specific bad answer. |
| `negative_type` | yes | Failure mode of `rejected` (see map below). |
| `dataset_version` | yes | Same family as QA, e.g. `v1`. |
| `metadata` | no | e.g. `creation_process`, `notes`. |

Allowed values for `negative_type` are:
`hallucination` | `over_refusal` | `over_complete` | `distractor_confusion` | `memory_override` |
`coreference_ambiguity` | `hedged_uncertainty` | `conflicting_evidence` | `multi_hop_arithmetic`
(the last 4 added for the `data_v2` tag extension — see below).

---

## Pipeline

```text
approved QA JSONL
        ↓
choose negative_type  (Python map)
        ↓
assemble prompt + chosen
        ↓
generate rejected     (API writes text only)
        ↓
validate PreferencePair
        ↓
write preference_*.jsonl
```

| Stage | What it does | Where |
|-------|--------------|-------|
| Input QA | Read approved Layer 2 (or full) QA rows | [`data_v1_pilot.jsonl`](../data/data_v1_pilot.jsonl); later `data/data_v1.jsonl` |
| Select `negative_type` | Map QA labels to one failure mode | [`build_preference.py`](../src/grounded_refusal/data/build_preference.py) → `choose_negative_type` |
| Assemble `prompt` / `chosen` | Format `prompt`; copy `reference_answer` to `chosen` | [`build_preference.py`](../src/grounded_refusal/data/build_preference.py) → `format_qa_prompt` (shared with inference, in `util/prompt_assembly.py`); [`default.yaml`](../configs/prompts/default.yaml) |
| Generate `rejected` | Generate a bad answer for the selected failure mode | [`build_preference.py`](../src/grounded_refusal/data/build_preference.py) → `generate_rejected` |
| Validate pair | Validate field shape and enum values | [`schema_pref.py`](../src/grounded_refusal/data/schema_pref.py) → `PreferencePair` |
| Write output | Write validated pairs to JSONL | [`util/io.py`](../src/grounded_refusal/util/io.py) → [`preference_v1_pilot.jsonl`](../data/preference_v1_pilot.jsonl); later `data/preference_v1.jsonl` |

Python decides `negative_type` and system instructions; the API returns **only** `rejected` text. Details: **FAQ**.

---

## Construction rules

The Fields section defines every column. Follow these steps in order when
building a preference row.

### 1. Select an approved QA row

The source row must satisfy
[`QA_GENERATION_PROTOCOL.md`](QA_GENERATION_PROTOCOL.md). Prefer a row that has
completed Layer 2 paraphrasing.

### 2. Choose `negative_type`

Python maps `answerability` × `evidence_challenge` to one `negative_type`; the
LLM does not choose it. Use the following fixed mapping for the pilot.

#### Pilot map

| # | `answerability` | `evidence_challenge` | Correct (`chosen`) | Bad (`rejected`) | `negative_type` |
|---|-----------------|----------------------|--------------------|------------------|-----------------|
| 1 | `answerable` | `[]` | Answer from evidence | Refuse even though evidence answers | `over_refusal` |
| 2 | `answerable` | `["known_world_conflict"]` | Answer from evidence | Prefer world knowledge over evidence | `memory_override` |
| 3 | `unanswerable` | `[]` | Refuse | Fabricate a plausible answer | `hallucination` |
| 4 | `unanswerable` | `["distractor_entity"]` | Refuse | Answer using distractor / wrong entity | `distractor_confusion` |
| 5 | `partial` | `["partial_evidence"]` | Answer supported; refuse unsupported | Also fill unsupported part | `over_complete` |

Decision order (same as code):

1. `answerable` + `known_world_conflict` → `memory_override`
2. Else `answerable` → `over_refusal`
3. Else `unanswerable` + `distractor_entity` → `distractor_confusion`
4. Else `unanswerable` → `hallucination`
5. Else `partial` → `over_complete`

Notes:

- `known_world_conflict` is **answerable**, not unanswerable.
- Do not use one generic hallucination for all unanswerable rows. Plain unanswerable and distractor cases require different `rejected` content.
- Pilot `partial`: **`over_complete` only**.

#### data_v2 tag extension

The pilot map above was built for `data_v1_pilot.jsonl`'s 3 `evidence_challenge`
enum values. `data_v2` introduced more specific failure modes as free-text
`tags` (not part of the `EvidenceChallengeTag` enum — see
[`DATA_V2_EXTENSION.md`](DATA_V2_EXTENSION.md)). Four of these tags appear in
`data_v2_train.jsonl` and are checked **before** the pilot map, so a row
carrying one of them gets a `negative_type` that targets its actual failure
mode instead of falling into a generic `hallucination`/`over_refusal` bucket:

| `tags` contains | Correct (`chosen`) | Bad (`rejected`) | `negative_type` |
|---|---|---|---|
| `multi_hop_arithmetic` | Correct computed value | Confident wrong arithmetic | `multi_hop_arithmetic` |
| `coreference_ambiguity` | Decline — referent is ambiguous | Confidently name one entity as the referent | `coreference_ambiguity` |
| `conflicting_evidence` | Decline — sources conflict | Pick one source's value, don't mention the conflict | `conflicting_evidence` |
| `hedged_uncertainty` | Decline — only a hedged estimate exists | State the hedged value as confirmed fact | `hedged_uncertainty` |

Decision order (same as code, `choose_negative_type` in
[`build_preference.py`](../src/grounded_refusal/data/build_preference.py)):

1. `multi_hop_arithmetic` in `tags` → `multi_hop_arithmetic`
2. Else `coreference_ambiguity` in `tags` → `coreference_ambiguity`
3. Else `conflicting_evidence` in `tags` → `conflicting_evidence`
4. Else `hedged_uncertainty` in `tags` → `hedged_uncertainty`
5. Else fall through to the pilot map above (answerability × evidence_challenge)

Why tags are checked first, not folded into the answerability/evidence_challenge
map: these 4 tags mostly co-occur with `unanswerable` + `evidence_challenge: []`
(the exact combination the pilot map already routes to plain `hallucination`).
Checking tags first intercepts them before they reach that generic bucket.

Why these 4 and not the others in `DATA_V2_EXTENSION.md` (`negation_exception`,
`embedded_instruction`, `false_presupposition`, `circular_evidence`,
`digit_confusion`, `conditional_logic`, `near_miss`, `red_herring`): those
tags only exist on the 55 pilot rows, which are pinned entirely to
`data_v2_heldout.jsonl` (see `split_train_heldout.py`) and never appear in
`data_v2_train.jsonl` — so `build_preference.py` run on the training split
never encounters them. Extending the map to cover them too is future work,
needed only once/if those rows are trained on.

#### Examples

One example per type, condensed from pilot rows. The shared instruction is omitted
so the `chosen` / `rejected` contrast stays visible.

**`over_refusal` — `pref_0021`**

```text
evidence: “The deepest point of Crater Lake reaches 594 meters.”
question: “What is the deepest point of Crater Lake?”
chosen:   “The deepest point reaches 594 meters.”
rejected: “The evidence does not provide enough information, so I don’t know.”
Why it fails: The evidence fully answers the question, but the response refuses.
```

**`memory_override` — `pref_0042`**

```text
evidence: “The text states that water boils at 50°C at sea level.”
question: “What boiling point is stated in the text?”
chosen:   “The text states that water boils at 50°C at sea level.”
rejected: “Water boils at 100°C at sea level.”
Why it fails: World knowledge replaces the value supplied by the evidence.
```

**`hallucination` — `pref_0026`**

```text
evidence: “The current CEO of Harbor Labs is Priya Nair.”
question: “When was Harbor Labs founded?”
chosen:   “The evidence does not provide the founding year, so I don’t know.”
rejected: “Harbor Labs was founded in 2015.”
Why it fails: A plausible fact that is absent from the evidence is invented.
```

**`distractor_confusion` — `pref_0051`**

```text
evidence: “The Mississippi River runs through the United States, while the
           Amazon rainforest is in South America.”
question: “Where does the Amazon River flow?”
chosen:   “The evidence does not specify where the Amazon River flows.”
rejected: “The Amazon River flows through the United States.”
Why it fails: Information is carried over from the distractor entity.
```

**`over_complete` — `pref_0031`**

```text
evidence: “Alan Turing was born in London.”
question: “Where was Alan Turing born, and in what year did he die?”
chosen:   “Alan Turing was born in London. The evidence does not give the year
           of his death.”
rejected: “Alan Turing was born in London, and he died in 1954.”
Why it fails: The unsupported half is answered instead of refused.
```

### 3. Assemble `prompt` and `chosen`

Set `chosen` to the source QA row’s `reference_answer`. The builder
deterministically assembles `prompt` from:

- `evidence` and `question` in the source QA row;
- `instruction`, `evidence_label`, and `question_label` in
  [`configs/prompts/default.yaml`](../configs/prompts/default.yaml).

The resulting format is:

```text
{evidence_label}:
{QA evidence}

{question_label}:
{QA question}

Instruction:
{instruction}
```

No LLM generates or rewrites `prompt` or `chosen`; the LLM is used only to
generate the `rejected` response.

### 4. Generate `rejected` with the API

Python selects `negative_type` and its matching system instruction. The API
generates only the `rejected` response.

- `rejected` must fail in the way specified by `negative_type`, rather than being
  generically wrong.
- `chosen` and `rejected` should have similar fluency, so DPO does not learn a
  style shortcut.
- `rejected` must not mention the hidden generation instruction.

### 5. Set bookkeeping fields and validate

| Field | How to set |
|-------|------------|
| `id` | `pref_` + digits from QA id (`ex_0021` → `pref_0021`) |
| `base_example_id` | QA `id` |
| `dataset_version` | match QA (e.g. `v1`) |
| `metadata.creation_process` | typically `llm_generated` when the API writes `rejected` |

Build and validate a `PreferencePair` before writing it to JSONL. Schema
validation covers field shape and enum values; semantic quality still requires
human review.

---

## Scale and pilot

| Split | Preference rows |
|-------|-----------------|
| Pilot `preference_v1_pilot.jsonl` | **50** (1:1 with `data_v1_pilot.jsonl`) |
| Full `preference_v1.jsonl` | ~**500** (1:1 with `data_v1.jsonl`), after full QA exists |

Distributions: [`reports/week2.md`](reports/week2.md).

### Optional additional pairs after the pilot

| QA situation | Extra `rejected` idea | `negative_type` |
|--------------|----------------------|-----------------|
| `partial` | refuse the whole question | `over_refusal` |
| `answerable` + `known_world_conflict` | second pair with the other of `{memory_override, over_refusal}` | as labeled |
| any | style-only bad answers / typos | avoid — not a grounding failure |

**Workflow**

1. Freeze / approve the QA file used as the base.
2. Run [`build_preference.py`](../src/grounded_refusal/data/build_preference.py) (map type → API `rejected` → validate → write).
3. Human-review a sample across all five slices (especially `memory_override`).
4. Fix the map or generation instructions if needed, then scale with the same process.

---

## FAQ

### Does the API choose `negative_type`?

No. Python chooses it from the pilot map, then picks the matching system instruction. The API only returns `rejected` text.

### Why not store `evidence` / `answerability` again on the preference row?

They live on the QA row. Preference links with `base_example_id` and puts the training input in `prompt`. This avoids duplicated copies that could drift apart.

### Style constraints for `chosen` / `rejected`?

- Similar fluency (avoid template `chosen` vs polished `rejected`, or DPO may reward style).
- Prefer QA that already finished Layer 2 paraphrase.
- `rejected` must be wrong **for the labeled `negative_type`**, not just a paraphrase of `chosen`.

### What does schema validation cover?

`PreferencePair(...)` validates field shape and the `negative_type` enum. It cannot determine whether `rejected` actually exhibits that failure mode; that requires human review or a later judge.
