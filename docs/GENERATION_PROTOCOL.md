# QA data generation protocol

How we build `data_v1.jsonl` (and `data_v1_pilot.jsonl`).

Label definitions: [`DATA_LABELS.md`](DATA_LABELS.md).  
Field reference: [`FIELDS.md`](FIELDS.md).  
Preference pairs: see **Preference data generation protocol** below in this file.

---

## Principle

1. **Fix first:** `answerability` (`answerable` | `unanswerable` | `partial`) and which fact or field the row tests.
2. **Then write:** `evidence`, `question`, `reference_answer` so the label is correct by construction.
3. **Then tag:** `evidence_type`, `evidence_challenge`, and other schema fields.

Do not guess labels after writing text. Do not use an LLM to decide `answerability`.

---

## Two layers

Layer 1 rows live in `data_v1_pilot_layer1.jsonl`; Layer 2 paraphrase overwrites `data_v1_pilot.jsonl`.

### Layer 1 — Rules (human or script template)

Locks everything that must be correct:

- `answerability`
- Entity, field, and values in evidence
- What the question asks (must match the construction rule for that `answerability`)
- `reference_answer` semantics
- `question_decomposition` / `supported_subquestions` when `partial`

Output is a **dry template sentence** (correct but repetitive).

### Layer 2 — LLM paraphrase (required for pilot and full set)

Send Layer-1 **`evidence`, `question`, and `reference_answer` together** to an LLM with a strict system prompt.

These three fields are one semantic unit. **Do not** paraphrase only evidence/question while leaving `reference_answer` in the Layer-1 template voice.

**Why all three must be paraphrased together**

- **Style mismatch:** natural evidence/question with a templated chosen answer teaches odd SFT bias.
- **DPO shortcut risk:** if chosen stays templated while rejected is fluent, the model can reward hack on style instead of grounding.

**LLM constraints**

- **Allowed:** rephrase wording, sentence structure, tone (all three fields consistently)
- **Not allowed:** add, remove, or change facts, entities, fields, numbers, or accept/refuse behavior

**Validator (required on pilot, recommended on full set):** after paraphrase, confirm entities, numeric values, asked field, and `answerability` behavior are unchanged.

**Runtime**

- **Pilot (50 rows):** API model
- **Scale (500 rows):** API or local **Ollama** (decide after pilot prompt is stable)

Set `metadata.creation_process` to `template_rule+llm_paraphrase` for rows that went through Layer 2.

**System prompt (outline)**

```
Rephrase evidence, question, and reference_answer together.
Keep all facts, entities, numbers, and fields identical.
Keep the same answer behavior (answer / refuse / partial).
Do not add or remove information.
Output JSON: {"evidence": "...", "question": "...", "reference_answer": "..."}
```

---

## Construction rules

### `answerable`

- Evidence states one clear fact.
- Question asks **only** for that fact.
- `reference_answer` answers from evidence; start with `According to the evidence, ...`
- Default `evidence_challenge`: `[]`

**Rule template (Layer 1)**

```
Evidence: {ENTITY} has a {FIELD} of {VALUE}.
Question: What is the {FIELD} of {ENTITY} according to the evidence?
```

**Paraphrase example (Layer 2)** — same facts, field, and answer behavior:

```
Evidence: Positioned at a surface elevation of 1,897 meters, Lake Tahoe is a prominent geographic feature.
Question: Can you tell me the specific surface elevation of Lake Tahoe based on the text?
Reference answer: Based on the text, Lake Tahoe's surface elevation is 1,897 meters.
```

Canonical repo example: `ex_0003`.

---

### `unanswerable`

- Evidence states fact A only.
- Question asks for fact B; B is **not** in evidence.
- `reference_answer` states what is missing + `so I don't know` (or equivalent refusal).
- Default `evidence_challenge`: `[]`

**Rule template (Layer 1)** — same entity as answerable row, different asked field:

```
Evidence: Lake Tahoe has a surface elevation of 1,897 meters.
Question: What is the maximum depth of Lake Tahoe?
```

Canonical repo example: `ex_0006`.

---

### `partial`

- Evidence supports sub-question 1 only.
- Question asks for sub-question 1 **and** 2.
- `reference_answer` answers 1, explicitly refuses 2.
- Required: `question_decomposition`, `supported_subquestions` (`supported` ⊆ `decomposition`).
- Typical `evidence_challenge`: `["partial_evidence"]`

Canonical repo example: `ex_0011`.

---

### `evidence_challenge` tags (not a fourth `answerability`)

Apply on top of the three rules above:

| Tag | Construction sketch | Canonical example |
|-----|---------------------|-------------------|
| `distractor_entity` | Distractor entity has facts; target entity is not fully answered in evidence | `ex_0016` |
| `known_world_conflict` | Evidence states a fictional/wrong-world fact; question asks for that fact as stated | `ex_0002` |
| `partial_evidence` | Same as partial rule above | `ex_0011` |

---

## `reference_answer` style

**Layer 1 (draft):** use consistent template openings below so rules are easy to validate.

| `answerability` | Layer 1 draft style |
|-----------------|---------------------|
| `answerable` | `According to the evidence, ...` |
| `unanswerable` | What evidence lacks + `so I don't know` |
| `partial` | `The evidence says ... It does not provide ...` |

**Layer 2:** rephrase `reference_answer` together with evidence and question. Wording may change; facts and behavior must not.

For `known_world_conflict` answerable rows, follow evidence; do not correct with world knowledge.

---

## Scale and pilot

| Split | `answerability` counts |
|-------|-------------------------|
| Full `data_v1.jsonl` | 200 answerable / 200 unanswerable / 100 partial (**500** total) |
| Pilot `data_v1_pilot.jsonl` | 20 / 20 / 10 (**50** total) |

**IDs:** `ex_0001`–`ex_0020` are dev hand examples (`hand_examples.jsonl`). New rows start `ex_0021`.  
**Split:** pilot and full train sets use `split: "train"`. Do not train on `hand_examples.jsonl` (`split: "dev"`).  
**Version:** `dataset_version: "v1"`.

**Pilot gate:** human + validator review all **50** pilot rows (focus: LLM paraphrase did not drop or alter facts across all three text fields). Fix construction rules or paraphrase prompt before scaling to 500.

---

## Workflow

1. Align this protocol.
2. Build **50** pilot QA rows: Layer 1 rules → Layer 2 API paraphrase (all three text fields) → validate.
3. Review pilot; adjust rules or paraphrase prompt if needed.
4. Scale to **500** QA rows (`build_data.py` or equivalent; API or Ollama for Layer 2).
5. Document counts in data README when committing.
6. After QA rows are approved, build preference pairs (next section).

---

# Preference data generation protocol

How we build `data/preference_v1_pilot.jsonl` (and later `preference_v1.jsonl`) from approved QA rows.

Schema: `src/data/schema_pref.py` (`PreferencePair`).  
This does **not** rebuild QA. Each preference row is **derived** from one QA row via `base_example_id`.

---

## Principle

1. Start from an approved QA row (`data_v1_pilot.jsonl` / `data_v1.jsonl`).
2. Build `prompt` by formatting evidence + question + instruction (from `configs/prompts/default.yaml`).
3. Set `chosen` = that row's `reference_answer`.
4. Write one deliberate bad answer as `rejected`, and label how it fails with `negative_type`.
5. Pilot default: **1 QA → 1 preference pair**. Same QA may get more pairs later (harder negatives).

Do not invent a new `answerability`. Slice negatives with existing QA labels:

- `answerability` = correct behavior
- `evidence_challenge` = why the item is hard
- `negative_type` = how `rejected` fails

---

## Row shape

| Field | Source |
|-------|--------|
| `id` | New id, e.g. `pref_0021` |
| `base_example_id` | QA `id`, e.g. `ex_0021` |
| `prompt` | Assembled from evidence + question + instruction |
| `chosen` | QA `reference_answer` |
| `rejected` | New bad answer (main new content) |
| `negative_type` | Failure mode of `rejected` |
| `dataset_version` | Same family as QA, e.g. `v1` |
| `metadata` | Optional (`creation_process`, `notes`) |

`prompt` is **concatenated**, not LLM-authored as a free rewrite of the task.

---

## Pilot negative map (required)

One pair per QA row for pilot. Decide `negative_type` from existing QA labels only:

`answerability` × `evidence_challenge` → one `negative_type`.

| # | `answerability` | `evidence_challenge` | Correct behavior (`chosen`) | Bad behavior (`rejected`) | `negative_type` |
|---|-----------------|----------------------|-----------------------------|---------------------------|-----------------|
| 1 | `answerable` | `[]` | Answer from evidence | Refuse / say don't know even though evidence answers | `over_refusal` |
| 2 | `answerable` | `["known_world_conflict"]` | Answer from evidence (even if anti-common-sense) | Prefer world knowledge over evidence | `memory_override` |
| 3 | `unanswerable` | `[]` | Refuse | Fabricate a plausible answer not in evidence | `hallucination` |
| 4 | `unanswerable` | `["distractor_entity"]` | Refuse | Answer using the distractor / wrong entity | `distractor_confusion` |
| 5 | `partial` | `["partial_evidence"]` | Answer supported part; refuse unsupported | Also fill the unsupported part | `over_complete` |

Decision order (same map in code):

1. If `answerable` and `known_world_conflict` → `memory_override`
2. Else if `answerable` → `over_refusal`
3. Else if `unanswerable` and `distractor_entity` → `distractor_confusion`
4. Else if `unanswerable` → `hallucination`
5. Else if `partial` → `over_complete`

Notes:

- `known_world_conflict` is **answerable**, not unanswerable.
- Do **not** use one generic hallucination template for all unanswerable rows: missing (`[]`) and `distractor_entity` need different `rejected` content.
- For pilot `partial`, use **`over_complete` only**. Optional later: same row + `over_refusal` (refuse even the supported part).

---

## Optional later negatives (not required for pilot)

| QA situation | Extra `rejected` idea | `negative_type` |
|--------------|----------------------|-----------------|
| `partial` | Refuse the whole question | `over_refusal` |
| `answerable` + `known_world_conflict` | Second pair with the other of `{memory_override, over_refusal}` | as labeled |
| any | Style-only bad answers, unrelated typos | avoid — not a grounding failure |

Keep hard negatives tied to the failure the challenge is testing.

---

## Style constraints

- `chosen` and `rejected` should be similarly fluent (avoid template `chosen` vs polished `rejected`, or DPO can reward style).
- Prefer QA rows that already finished Layer 2 paraphrase before building preference pairs.
- `rejected` must be wrong **for the stated reason** in `negative_type`, not merely different wording of `chosen`.

---

## Scale and pilot

| Split | Preference rows |
|-------|-----------------|
| Pilot `preference_v1_pilot.jsonl` | ~**50** (1:1 with `data_v1_pilot.jsonl`) |
| Full `preference_v1.jsonl` | ~**500** (1:1 with `data_v1.jsonl`), after full QA exists |

**IDs:** `pref_` + digits matching or derived from base example (e.g. `pref_0021` from `ex_0021`). Optional letter suffix for multiple pairs from one QA (`pref_0021a`).  
**Version:** `dataset_version: "v1"`.

**Pilot gate:** spot-check that each `negative_type` matches the QA slice above, and that `rejected` is actually wrong under evidence-only rules.

---

## Workflow

1. Freeze / approve the QA file used as base (pilot first).
2. For each QA row, assemble `prompt`, copy `chosen`, write `rejected` per the pilot map.
3. Validate with `PreferencePair` (`schema_pref.py`).
4. Human review a sample across all five pilot slices.
5. Scale with the same map when full QA exists; add optional second negatives only after the 1:1 map is solid.
