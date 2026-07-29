# Data labels 

Definitions for fields on each QA row in `data/*.jsonl`. **Edit here first**, then update schemas (`src/data/schema_qa.py`).

Canonical hand examples: `data/hand_examples.jsonl`. Pilot examples: `data/data_v1_pilot.jsonl`.

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

### Example: `answerable`

Evidence states the fact; question asks only for that fact.

- id: `ex_0003`
- evidence: `The Eiffel Tower was completed in 1889 and stands 330 meters tall including its antenna.`
- question: `How tall is the Eiffel Tower according to the evidence?`
- reference_answer: answers **330 meters** from evidence
- labels: `answerability=answerable`, `evidence_challenge=[]`

### Example: `unanswerable`

Evidence is about something else; the asked fact is missing.

- id: `ex_0006`
- evidence: `The Amazon rainforest is the largest tropical rainforest in the world.`
- question: `How long is the Amazon River?`
- reference_answer: refuses — evidence does not mention river length
- labels: `answerability=unanswerable`, `evidence_challenge=[]`

### Example: `partial`

Question has two parts; evidence supports only one.

- id: `ex_0011`
- evidence: Marie Curie born in Warsaw; moved to Paris; radioactivity research
- question: `Where was Marie Curie born, and what awards did she win?`
- reference_answer: answers birth place; refuses awards
- labels: `answerability=partial`, `evidence_challenge=["partial_evidence"]`
- also set: `question_decomposition`, `supported_subquestions`

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

### Example: `single_sentence`

- id: `ex_0003`
- evidence (one sentence): `The Eiffel Tower was completed in 1889 and stands 330 meters tall including its antenna.`

### Example: `short_paragraph`

- id: `ex_0001`
- evidence (a few sentences, one block):  
  `The Amazon River is the largest river by discharge volume of water in the world. It flows through South America, mainly through Brazil and Peru.`

### Example: `multi_paragraph`

- id: `ex_0044` (pilot)
- evidence: several paragraphs about the Meridian Trade Treaty (signing year, duration, later events); question asks only for the signing year stated in the text.

---

## `evidence_challenge` — difficulty / trap mechanisms

Describes **what makes the example hard** for eval slicing. Independent of `evidence_type`.

| Tag | When to add | Typical `answerability` |
|-----|-------------|-------------------------|
| *(none)* `[]` | No special trap — simple case | any |
| `distractor_entity` | Evidence includes **similar entities or misleading related text** | Often `unanswerable` |
| `known_world_conflict` | Evidence **contains an answer** that **conflicts with common world knowledge** | Often `answerable` |
| `partial_evidence` | Evidence **only covers part of the question** | Often `partial` |

**Rules**

- Field is **required** on every row (use `[]` when there is no trap).
- **Simple examples:** `"evidence_challenge": []`
- **Hard examples:** one or more tags, e.g. `["distractor_entity"]`
- Multiple tags allowed on one row.

### Example: `[]` (simple)

No trap. Same as plain `answerable` / `unanswerable` examples above (`ex_0003`, `ex_0006`).

### Example: `distractor_entity`

Related entities appear, but the asked fact is still missing.

- id: `ex_0016`
- evidence: Mississippi River flows through the US; Amazon **rainforest** is in South America
- question: `Where does the Amazon River flow?`
- correct: **refuse** (rainforest ≠ river location)
- wrong failure mode later in preference: answer as if rainforest/Mississippi answered the river question  
- labels: `answerability=unanswerable`, `evidence_challenge=["distractor_entity"]`

### Example: `known_world_conflict`

Evidence gives an anti-common-sense fact; model must still follow evidence.

- id: `ex_0002`
- evidence: fictional dataset says the Nile flows mainly through Brazil and Peru
- question: where does the Nile flow **according to the evidence**?
- correct: answer Brazil and Peru (not Egypt / real-world Nile)
- labels: `answerability=answerable`, `evidence_challenge=["known_world_conflict"]`

### Example: `partial_evidence`

Same construction as `partial` answerability — evidence covers only some sub-questions.

- id: `ex_0011` (see partial example above)
- labels: `answerability=partial`, `evidence_challenge=["partial_evidence"]`

---

## Label combo cheat sheet

Quick reminder of common **aligned** combinations (not all legal combos):

| `answerability` | Typical `evidence_challenge` | Example id |
|-----------------|------------------------------|------------|
| `answerable` | `[]` | `ex_0003` |
| `answerable` | `["known_world_conflict"]` | `ex_0002` |
| `unanswerable` | `[]` | `ex_0006` |
| `unanswerable` | `["distractor_entity"]` | `ex_0016` |
| `partial` | `["partial_evidence"]` | `ex_0011` |

These three fields are independent dimensions: behavior (`answerability`), text shape (`evidence_type`), trap (`evidence_challenge`). Schema does not forbid odd combos; prefer the rows above by construction.
