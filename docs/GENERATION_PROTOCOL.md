# QA data generation protocol

How we build `data_v1.jsonl` (and `data_v1_pilot.jsonl`).

Label definitions: [`DATA_LABELS.md`](DATA_LABELS.md).  
Preference pairs are documented separately (not in this file).

---

## Principle

1. **Fix first:** `answerability` (`answerable` | `unanswerable` | `partial`) and which fact or field the row tests.
2. **Then write:** `evidence`, `question`, `reference_answer` so the label is correct by construction.
3. **Then tag:** `evidence_type`, `evidence_challenge`, and other schema fields.

Do not guess labels after writing text. Do not use an LLM to decide `answerability`.

---

## Two layers

### Layer 1 — Rules (human or script template)

Locks everything that must be correct:

- `answerability`
- Entity, field, and values in evidence
- What the question asks (must match the construction rule for that `answerability`)
- `reference_answer` semantics
- `question_decomposition` / `supported_subquestions` when `partial`

Output is a **dry template sentence** (correct but repetitive).

### Layer 2 — LLM paraphrase (optional per row)

Send Layer-1 `evidence` and `question` to an LLM with a strict system prompt:

- **Allowed:** rephrase wording, sentence structure, question style
- **Not allowed:** add, remove, or change facts, entities, fields, or numbers

`reference_answer` may be paraphrased under the same constraint (same facts, same accept/refuse behavior).

**Validator (required on pilot, recommended on full set):** after paraphrase, confirm entities, numeric values, and the asked field are unchanged and `answerability` still holds.

Set `metadata.creation_process` to `template_rule` or `template_rule+llm_paraphrase`.

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

**Paraphrase example (Layer 2)** — same facts, same field:

```
Evidence: Positioned at a surface elevation of 1,897 meters, Lake Tahoe is a prominent geographic feature.
Question: Can you tell me the specific surface elevation of Lake Tahoe based on the text?
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

| `answerability` | Style |
|-----------------|-------|
| `answerable` | `According to the evidence, ...` |
| `unanswerable` | What evidence lacks + `so I don't know` |
| `partial` | `The evidence says ... It does not provide ...` |

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

**Pilot gate:** human + validator review all **50** pilot rows (focus: LLM paraphrase did not drop or alter facts). Fix construction rules before scaling to 500.

---

## Workflow

1. Align this protocol.
2. Build **50** pilot QA rows (rules → optional paraphrase → validate).
3. Review pilot; adjust rules or paraphrase prompt if needed.
4. Scale to **500** QA rows (`build_data.py` or equivalent).
5. Document counts in data README when committing.

Preference pair generation is a **separate** step and a **separate** doc, after QA data is approved.
