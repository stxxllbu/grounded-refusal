# QA generation protocol

How we build QA JSONL: `data/hand_examples.jsonl`, `data/data_v1_pilot_layer1.jsonl`, `data/data_v1_pilot.jsonl`, and later `data/data_v1.jsonl`.

**Label values:** [`DATA_LABELS.md`](DATA_LABELS.md).  
**Schema:** [`src/data/schema_qa.py`](../src/data/schema_qa.py).  
**Preference pairs:** separate step (temporary notes in deprecated [`GENERATION_PROTOCOL.md`](GENERATION_PROTOCOL.md)).

---

## Principle

1. **Fix first:** `answerability` and which fact the row tests (object, attribute, value in evidence).
2. **Then write:** `evidence`, `question`, `reference_answer` so the label is correct by construction.
3. **Then tag:** `evidence_type`, `evidence_challenge`, and remaining schema fields.

---

## Fields

One row = one `(evidence, question)` scenario with a single gold `reference_answer`.

| Field | Required | Purpose |
|-------|----------|---------|
| `id` | yes | Stable id (`ex_0001`). Used in logs, eval, and preference `base_example_id`. |
| `evidence` | yes | Text the model must ground on. |
| `question` | yes | User question. |
| `reference_answer` | yes | Gold response (also DPO `chosen` later). |
| `answerability` | yes | `answerable` \| `unanswerable` \| `partial` — see [`DATA_LABELS.md`](DATA_LABELS.md). |
| `evidence_type` | yes | `single_sentence` \| `short_paragraph` \| `multi_paragraph`. |
| `evidence_challenge` | yes | Trap tags; use `[]` when simple. |
| `split` | yes | `train` / `val` / `test` / `dev`. |
| `dataset_version` | yes | e.g. `v1`. |
| `question_decomposition` | when `partial` | Sub-question ids, e.g. `["birth_place", "awards"]`. |
| `supported_subquestions` | when `partial` | Which sub-questions evidence supports (`⊆` decomposition). |
| `metadata` | no | e.g. `entity`, `creation_process`. |
| `tags` | no | Free-form ad-hoc tags. |

---

## Pipeline

We produce each QA row in **two steps**:

```text
decide labels + fact
        ↓
Layer 1  — dry template (semantics locked)
        ↓
Layer 2  — paraphrase evidence / question / reference_answer
        ↓
validate schema (+ human / fact-lock review)
        ↓
data_v1_pilot.jsonl  (or later data_v1.jsonl)
```

| Step | What it does | Where |
|------|----------------|-------|
| Contract | Field types and enums | [`schema_qa.py`](../src/data/schema_qa.py) |
| Layer 1 | Template rows; correct by construction | `data/data_v1_pilot_layer1.jsonl` (no dedicated builder script yet) |
| Layer 2 | Rephrase the three text fields only | [`paraphrase.py`](../src/data/paraphrase.py) → `data/data_v1_pilot.jsonl` |
| Validate | JSON shape / enums / partial rules | [`validate_qa_jsonl_against_schema.py`](../src/data/validate_qa_jsonl_against_schema.py) |
| IO | Read/write JSONL | [`util/io.py`](../src/util/io.py) |

**Layer 1** locks semantics: `answerability`, the tested fact, what the question asks, gold answer behavior, and partial decomposition fields when needed. Wording is intentionally dry. Set `metadata.creation_process` to `template_rule` when applicable.

**Layer 2** must not change those semantics—only fluency. Set `metadata.creation_process` to `template_rule+llm_paraphrase`. Details (why three fields, fact-lock, prompt): see **FAQ** at the end.

---

## Construction rules

Fields lists **every** column. This section says **how to fill them when building a row**, in order:

1. **Behavior + text** — `answerability`, then `evidence` / `question` / `reference_answer` (and partial extras). This needs real recipes (below).
2. **Optional trap** — `evidence_challenge` (add-on; not a fourth answerability).
3. **Describe the evidence shape** — `evidence_type` (tag length/structure; no separate “question recipe”).
4. **Bookkeeping** — `id`, `split`, `dataset_version`, `metadata`, `tags`.

Apply (1)–(4) at Layer 1. Layer 2 must not change (1)–(2); it may change wording only.

### 1. By `answerability` (core recipes)

#### `answerable`

- Evidence states one clear fact.
- Question asks **only** for that fact.
- `reference_answer` answers from evidence.
- Default `evidence_challenge`: `[]`

Template sketch:

```
Evidence: {ENTITY} has a {FIELD} of {VALUE}.
Question: What is the {FIELD} of {ENTITY} according to the evidence?
```

Example: `ex_0003` in `hand_examples.jsonl`.

#### `unanswerable`

- Evidence states fact A only.
- Question asks for fact B; B is **not** in evidence.
- `reference_answer` states what is missing + refuse.
- Default `evidence_challenge`: `[]`

Template sketch (same entity as a paired answerable row, different asked field):

```
Evidence: Lake Tahoe has a surface elevation of 1,897 meters.
Question: What is the maximum depth of Lake Tahoe?
```

Example: `ex_0006`.

#### `partial`

- Evidence supports sub-question 1 only.
- Question asks for sub-question 1 **and** 2.
- `reference_answer` answers 1, explicitly refuses 2.
- Required: `question_decomposition`, `supported_subquestions` (`supported` ⊆ `decomposition`).
- Typical `evidence_challenge`: `["partial_evidence"]`

Example: `ex_0011` (also in [`DATA_LABELS.md`](DATA_LABELS.md)).

**`reference_answer` Layer-1 openings** (same three recipes; Layer 2 may rephrase):

| `answerability` | Suggested opening |
|-----------------|-------------------|
| `answerable` | `According to the evidence, ...` |
| `unanswerable` | What is missing + `so I don't know` |
| `partial` | `The evidence says ... It does not provide ...` |

For `known_world_conflict`, follow evidence; do not “fix” with world knowledge.

### 2. Adding `evidence_challenge` (optional traps)

Not another `answerability`. First finish step 1, then optionally harden. Definitions: [`DATA_LABELS.md`](DATA_LABELS.md).

| If you want… | Start from | How to construct the trap | Example |
|--------------|------------|---------------------------|---------|
| No trap | any of the three | Leave `"evidence_challenge": []` | `ex_0003`, `ex_0006` |
| Distractor | **`unanswerable`** | Keep the asked fact missing, but put **related / lookalike entities** in evidence | `ex_0016` |
| World-knowledge conflict | **`answerable`** | Answer **is** in evidence, but conflicts with common sense; ask for the fact **as stated** | `ex_0002` |
| Partial evidence | **`partial`** | Same as the `partial` recipe; tag `["partial_evidence"]` for slicing | `ex_0011` |

### 3. Choosing `evidence_type`

No separate “how to invent a question” recipe. After the evidence text exists, set the tag to match its shape ([`DATA_LABELS.md`](DATA_LABELS.md)):

| If evidence looks like… | Set `evidence_type` |
|-------------------------|---------------------|
| One sentence | `single_sentence` |
| A few sentences, one block | `short_paragraph` |
| Multiple paragraphs | `multi_paragraph` |

You may **aim** for a target mix when writing (pilot/full quotas in Scale below), but the label must still describe the text you actually wrote.

### 4. Bookkeeping fields

| Field | How to set |
|-------|------------|
| `id` | `ex_NNNN`; hand `ex_0001`–`ex_0020`; new train from `ex_0021` |
| `split` | `dev` for hand examples; `train` for pilot/full train sets |
| `dataset_version` | `v1` for current data |
| `metadata.entity` | optional; main entity string if useful |
| `metadata.creation_process` | `template_rule` (Layer 1) or `template_rule+llm_paraphrase` (Layer 2) |
| `tags` | optional free-form |

---

## Scale and pilot

| Split | `answerability` counts |
|-------|-------------------------|
| Full `data_v1.jsonl` | 200 / 200 / 100 (**500**) |
| Pilot `data_v1_pilot.jsonl` | 20 / 20 / 10 (**50**) |

- IDs `ex_0001`–`ex_0020`: `hand_examples.jsonl` (`split: "dev"`). New train rows from `ex_0021`.
- Pilot/full train: `split: "train"`, `dataset_version: "v1"`.
- Distributions: [`reports/week2.md`](reports/week2.md).

**Workflow**

1. Align this protocol + [`DATA_LABELS.md`](DATA_LABELS.md).
2. Layer 1 → Layer 2 (`paraphrase.py`) → validate.
3. Review pilot; fix rules or paraphrase prompt if needed.
4. Scale to ~500 when ready.
5. Then preference pairs (separate protocol).

---

## FAQ

### Why paraphrase evidence, question, and `reference_answer` together?

They are one semantic unit. If only evidence/question become natural and `reference_answer` stays Layer-1 template voice:

- SFT sees mismatched style inside one example.
- Later DPO can reward style (fluent `rejected` vs stiff `chosen`) instead of grounding.

Layer 2 therefore rewrites all three in one API call ([`paraphrase.py`](../src/data/paraphrase.py)).

### What may Layer 2 change?

- **Allowed:** wording, sentence structure, tone.
- **Not allowed:** facts, entities, numbers, asked field, or answer/refuse/partial behavior.

### What is fact-lock?

After paraphrase, check that entities, numeric values, asked field, and accept/refuse behavior still match Layer 1. Pilot relied on human review; automated fact-lock is TODO. Schema validation only checks field shape/enums ([`validate_qa_jsonl_against_schema.py`](../src/data/validate_qa_jsonl_against_schema.py)).

### What prompt does Layer 2 use?

Outline (full prompt in `paraphrase.py`):

```
Rephrase evidence, question, and reference_answer together.
Keep all facts, entities, numbers, and fields identical.
Keep the same answer behavior (answer / refuse / partial).
Do not add or remove information.
Output JSON: {"evidence": "...", "question": "...", "reference_answer": "..."}
```

### Pilot vs scale runtime?

- Pilot (~50): API via `paraphrase.py`.
- Scale (~500): API or local Ollama after the prompt is stable.
