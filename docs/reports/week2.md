# Week 2 report — Pilot QA + preference pairs

## Delivered

| Item | Path | Rows |
|------|------|-----:|
| Layer 1 QA (templates) | [`data/data_v1_pilot_layer1.jsonl`](../../data/data_v1_pilot_layer1.jsonl) | 50 |
| Layer 2 QA (paraphrase) | [`data/data_v1_pilot.jsonl`](../../data/data_v1_pilot.jsonl) | 50 |
| Preference pairs | [`data/preference_v1_pilot.jsonl`](../../data/preference_v1_pilot.jsonl) | 50 |
| QA schema | [`src/data/schema_qa.py`](../../src/data/schema_qa.py) | — |
| Preference schema | [`src/data/schema_pref.py`](../../src/data/schema_pref.py) | — |
| QA validator | [`src/data/validate_qa_jsonl_against_schema.py`](../../src/data/validate_qa_jsonl_against_schema.py) | — |
| Layer 2 script | [`src/data/paraphrase.py`](../../src/data/paraphrase.py) | — |
| Preference builder | [`src/data/build_preference.py`](../../src/data/build_preference.py) | — |
| Protocol | [`docs/QA_GENERATION_PROTOCOL.md`](../QA_GENERATION_PROTOCOL.md) (QA); preference still in deprecated [`GENERATION_PROTOCOL.md`](../GENERATION_PROTOCOL.md) | — |

IDs: QA `ex_0021`–`ex_0070`; preference `pref_0021`–`pref_0070` (1:1 via `base_example_id`).

**Full ~500 `data_v1` / `preference_v1` deferred.** This week closes the **pilot** pipeline.

---

## QA pilot distribution (50)

Source: `data/data_v1_pilot.jsonl` (Layer 2).

### By `answerability`

| `answerability` | Count | Share |
|-----------------|------:|------:|
| `answerable` | 20 | 40% |
| `unanswerable` | 20 | 40% |
| `partial` | 10 | 20% |
| **Total** | **50** | 100% |

### By `evidence_type`

| `evidence_type` | Count | Share |
|-----------------|------:|------:|
| `single_sentence` | 25 | 50% |
| `short_paragraph` | 21 | 42% |
| `multi_paragraph` | 4 | 8% |
| **Total** | **50** | 100% |

### By `evidence_challenge` (overall)

| `evidence_challenge` | Count |
|----------------------|------:|
| `[]` | 27 |
| `["partial_evidence"]` | 10 |
| `["distractor_entity"]` | 10 |
| `["known_world_conflict"]` | 3 |
| **Total** | **50** |

### By `answerability` × `evidence_challenge` (canonical slices)

| # | `answerability` | `evidence_challenge` | Count |
|---|-----------------|----------------------|------:|
| 1 | `answerable` | `[]` | 17 |
| 2 | `answerable` | `["known_world_conflict"]` | 3 |
| 3 | `unanswerable` | `[]` | 10 |
| 4 | `unanswerable` | `["distractor_entity"]` | 10 |
| 5 | `partial` | `["partial_evidence"]` | 10 |
| | **Total** | | **50** |

---

## Preference pilot distribution (50)

Source: `data/preference_v1_pilot.jsonl`.

Built 1:1 from Layer 2 QA: Python chooses `negative_type`; API writes only `rejected`.

### By `negative_type`

| `negative_type` | Count | Share | From QA slice |
|-----------------|------:|------:|---------------|
| `over_refusal` | 17 | 34% | answerable + `[]` |
| `hallucination` | 10 | 20% | unanswerable + `[]` |
| `distractor_confusion` | 10 | 20% | unanswerable + distractor |
| `over_complete` | 10 | 20% | partial + partial_evidence |
| `memory_override` | 3 | 6% | answerable + known_world_conflict |
| **Total** | **50** | 100% | |

Counts match the QA slice table above (same 1–5 map as in the generation protocol).

### Row fields (reminder)

| Field | Source |
|-------|--------|
| `id` | e.g. `pref_0021` |
| `base_example_id` | QA `id` |
| `prompt` | evidence + question + instruction |
| `chosen` | QA `reference_answer` |
| `rejected` | API bad answer |
| `negative_type` | mapped from QA labels |

---

## Validation

| Check | What it covers | What it does **not** cover |
|-------|----------------|----------------------------|
| [`validate_qa_jsonl_against_schema.py`](../../src/data/validate_qa_jsonl_against_schema.py) + `QAExample` | QA field shape, enums, partial-field rules | Fact-lock Layer1↔Layer2, answer quality, paraphrase fidelity |
| `PreferencePair` in `build_preference.py` | Preference field shape + `negative_type` enum when pairs are built | Whether `rejected` text truly matches the intended failure mode |

Run QA schema check (example):

```bash
export PYTHONPATH="$PWD/src"
.venv/bin/python -m data.validate_qa_jsonl_against_schema data/data_v1_pilot.jsonl
```

Preference rows are validated as they are constructed (`PreferencePair(...)`). Human review is still needed for `rejected` quality (especially `memory_override`).

---

## Pipeline (short)


```text
Layer 1 templates  →  Layer 2 paraphrase  →  preference pairs
data_v1_pilot_layer1.jsonl
        ↓
data_v1_pilot.jsonl
        ↓
preference_v1_pilot.jsonl
```

## Not in Week 2 pilot

- Full ~500 QA / preference
- SFT / DPO training
- Eval harness (Week 3)
