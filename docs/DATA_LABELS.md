# Data labels

Definitions for fields on each QA row in `data/*.jsonl`. **Edit here first**, then update schemas (`src/data/schema_qa.py`).

Full rows: `data/hand_examples.jsonl` (pilot: `data/data_v1_pilot.jsonl`).

---

## `answerability` — what the model should do

| Value | Correct behavior |
|-------|------------------|
| `answerable` | Answer using evidence |
| `unanswerable` | Refuse (say you don't know) |
| `partial` | Answer supported part; refuse unsupported part |

**Rules**

- Field is **required** on every row; exactly one of the three values.
- When `partial`: also require `question_decomposition` and `supported_subquestions`.

**Example** (`partial` — `ex_0011`): question asks birth place **and** awards; evidence only has birth place → answer the first, refuse the second.

```json
{
  "id": "ex_0011",
  "evidence": "Marie Curie was born in Warsaw and later moved to Paris, where she conducted research on radioactivity.",
  "question": "Where was Marie Curie born, and what awards did she win?",
  "reference_answer": "The evidence says Marie Curie was born in Warsaw. It does not provide information about what awards she won.",
  "answerability": "partial",
  "evidence_challenge": ["partial_evidence"]
}
```

(Also see `ex_0003` answerable, `ex_0006` unanswerable in `hand_examples.jsonl`.)

---

## `evidence_type` — shape / length of evidence text

Describes **how long or structured the evidence is**, not difficulty or traps.

| Value | When to use |
|-------|-------------|
| `single_sentence` | Evidence is **one sentence** |
| `short_paragraph` | Evidence is **a few sentences**, still one short block |
| `multi_paragraph` | Evidence has **multiple paragraphs** |

**Rules**

- Field is **required** on every row; exactly one of the three values.
- Describes length/structure only — not difficulty (use `evidence_challenge` for traps).

**Example** (`short_paragraph` — `ex_0001`):

```json
{
  "id": "ex_0001",
  "evidence": "The Amazon River is the largest river by discharge volume of water in the world. It flows through South America, mainly through Brazil and Peru.",
  "evidence_type": "short_paragraph"
}
```

---

## `evidence_challenge` — difficulty / trap mechanisms

Describes **what makes the example hard** for eval slicing. Independent of `evidence_type`.

| Tag | When to add | Typical `answerability` |
|-----|-------------|-------------------------|
| `[]` | No special trap — simple case | any |
| `distractor_entity` | Evidence includes **similar entities or misleading related text** | Often `unanswerable` |
| `known_world_conflict` | Evidence **contains an answer** that **conflicts with common world knowledge** | Often `answerable` |
| `partial_evidence` | Evidence **only covers part of the question** | Often `partial` |

**Rules**

- Field is **required** on every row (use `[]` when there is no trap).
- **Simple examples:** `"evidence_challenge": []`
- **Hard examples:** one or more tags, e.g. `["distractor_entity"]`
- Multiple tags allowed on one row.

**Example** (`known_world_conflict` — `ex_0002`): evidence says Nile flows through Brazil/Peru; correct answer still follows evidence (not real-world Egypt).

```json
{
  "id": "ex_0002",
  "evidence": "In this fictional dataset, the Nile River is described as flowing mainly through Brazil and Peru.",
  "question": "Where does the Nile River flow according to the evidence?",
  "reference_answer": "According to the evidence, the Nile River is described as flowing mainly through Brazil and Peru.",
  "answerability": "answerable",
  "evidence_challenge": ["known_world_conflict"]
}
```

(`partial_evidence` is the same pattern as the `partial` example above; `distractor_entity` → `ex_0016`.)

---

These three fields are independent: behavior (`answerability`), text shape (`evidence_type`), trap (`evidence_challenge`).
