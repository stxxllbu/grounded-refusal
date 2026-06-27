# Data labels (aligned)

Definitions for fields on each row in `data/*.jsonl`. **Edit here first**, then update schemas.

## `answerability` — what the model should do

| Value | Correct behavior |
|-------|------------------|
| `answerable` | Answer using evidence |
| `unanswerable` | Refuse (say you don't know) |
| `partial` | Answer supported part; refuse unsupported part |

Requires `question_decomposition` and `supported_subquestions` when `partial`.

---

## `evidence_type` — shape / length of evidence text

Describes **how long or structured the evidence is**, not difficulty or traps.

| Value | When to use |
|-------|-------------|
| `single_sentence` | Evidence is **one sentence** |
| `short_paragraph` | Evidence is **a few sentences**, still one short block |
| `multi_paragraph` | Evidence has **multiple paragraphs** |

Every row **must** have exactly one `evidence_type`.

---

## `evidence_challenge` — difficulty / trap mechanisms

Describes **what makes the example hard** for eval slicing. Independent of `evidence_type`.

| Tag | When to add | Typical `answerability` |
|-----|-------------|-------------------------|
| `distractor_entity` | Evidence includes **similar entities or misleading related text** | Often `unanswerable` |
| `known_world_conflict` | Evidence **contains an answer** that **conflicts with common world knowledge** | Often `answerable` |
| `partial_evidence` | Evidence **only covers part of the question** | Often `partial` |

**Rules**

- Field is **required** on every row.
- **Simple examples:** `"evidence_challenge": []`
- **Hard examples:** one or more tags, e.g. `["distractor_entity"]`
- Multiple tags allowed on one row.

---

## Examples

**Simple unanswerable**

```json
{
  "answerability": "unanswerable",
  "evidence_type": "single_sentence",
  "evidence_challenge": []
}
```

**Distractor unanswerable**

```json
{
  "answerability": "unanswerable",
  "evidence_type": "short_paragraph",
  "evidence_challenge": ["distractor_entity"]
}
```

**Known-world conflict answerable**

```json
{
  "answerability": "answerable",
  "evidence_type": "single_sentence",
  "evidence_challenge": ["known_world_conflict"]
}
```

**Partial answer**

```json
{
  "answerability": "partial",
  "evidence_type": "single_sentence",
  "evidence_challenge": ["partial_evidence"]
}
```
