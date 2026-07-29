# Week 1 report — Task definition + hand examples

## Delivered

| Item | Path | Notes |
|------|------|-------|
| Hand examples | [`data/hand_examples.jsonl`](../../data/hand_examples.jsonl) | 20 gold rows, `split=dev` |
| Label definitions | [`docs/DATA_LABELS.md`](../DATA_LABELS.md) | `answerability`, `evidence_type`, `evidence_challenge` |
| Prompt template | [`configs/prompts/default.yaml`](../../configs/prompts/default.yaml) | Evidence / Question / Instruction |

## Task (summary)

Given evidence + question, correct behavior is:

1. **Answer** if evidence is sufficient (`answerable`)
2. **Refuse** if evidence is insufficient (`unanswerable`)
3. **Partially answer** if only part is supported (`partial`)

Ground on provided evidence only.

## Hand-example distribution (20)

### By `answerability`

| `answerability` | Count | Share |
|-----------------|------:|------:|
| `answerable` | 5 | 25% |
| `unanswerable` | 10 | 50% |
| `partial` | 5 | 25% |
| **Total** | **20** | 100% |

### By `answerability` × `evidence_challenge`

| `answerability` | `evidence_challenge` | Count | IDs (approx.) |
|-----------------|----------------------|------:|---------------|
| `answerable` | `[]` | 4 | `ex_0001`, `ex_0003`–`ex_0005` |
| `answerable` | `["known_world_conflict"]` | 1 | `ex_0002` |
| `unanswerable` | `[]` | 5 | `ex_0006`–`ex_0010` |
| `unanswerable` | `["distractor_entity"]` | 5 | `ex_0016`–`ex_0020` |
| `partial` | `["partial_evidence"]` | 5 | `ex_0011`–`ex_0015` |

## Not in Week 1

- Scaled dataset / preference pairs
- Training / eval metrics
