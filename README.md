# grounded-refusal

Evidence-grounded QA: answer from provided evidence, refuse when insufficient, partially answer when only partly supported.

## Week 1 — Task definition + hand examples

### Task

Given **evidence** and a **question**, the correct behavior is:

1. **Answer** when the evidence is sufficient (`answerable`)
2. **Refuse** when the evidence is insufficient (`unanswerable`)
3. **Partially answer** when the evidence only supports part of the question (`partial`)

Use **only the provided evidence** — not outside world knowledge unless the evidence states it.

### Repo map

| File | What it is |
|------|------------|
| [`docs/DATA_LABELS.md`](docs/DATA_LABELS.md) | Label definitions: `answerability`, `evidence_type`, `evidence_challenge` |
| [`data/hand_examples.jsonl`](data/hand_examples.jsonl) | 20 hand-written gold examples |
| [`configs/prompts/default.yaml`](configs/prompts/default.yaml) | Shared prompt instruction + section labels |

### Hand examples (`data/hand_examples.jsonl`)

20 rows in `dev` split, `dataset_version: v1`:

| `answerability` | Count | IDs | Notes |
|-----------------|-------|-----|-------|
| `answerable` | 5 | `ex_0001`–`ex_0005` | includes 1 `known_world_conflict` (`ex_0002`) |
| `unanswerable` | 5 | `ex_0006`–`ex_0010` | plain missing-info cases |
| `partial` | 5 | `ex_0011`–`ex_0015` | each has `question_decomposition` + `supported_subquestions` |
| `unanswerable` | 5 | `ex_0016`–`ex_0020` | `evidence_challenge: ["distractor_entity"]` |

Each row is one JSON object (JSONL = one JSON per line).

### Core fields per example

| Field | Meaning |
|-------|---------|
| `evidence` | Text the model may use |
| `question` | What to answer |
| `reference_answer` | Gold response |
| `answerability` | `answerable` \| `unanswerable` \| `partial` |
| `evidence_type` | Shape of evidence: `single_sentence`, `short_paragraph`, `multi_paragraph` |
| `evidence_challenge` | Difficulty tags, or `[]` for simple cases |
| `question_decomposition` | Sub-parts of the question (when `partial`) |
| `supported_subquestions` | Which sub-parts the evidence supports (when `partial`) |

See [`docs/DATA_LABELS.md`](docs/DATA_LABELS.md) for full definitions.

### Prompt rule (`configs/prompts/default.yaml`)

For each example, the prompt is:

```
Evidence:
{evidence from jsonl}

Question:
{question from jsonl}

Instruction:
{instruction from default.yaml}
```

### Example row (`ex_0006`, unanswerable)

```json
{
  "id": "ex_0006",
  "evidence": "The Amazon rainforest is the largest tropical rainforest in the world.",
  "question": "How long is the Amazon River?",
  "reference_answer": "The provided evidence does not contain information about the length of the Amazon River, so I don't know.",
  "answerability": "unanswerable",
  "evidence_type": "single_sentence",
  "evidence_challenge": []
}
```

Correct behavior: refuse — the evidence talks about the rainforest, not river length.
