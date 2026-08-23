# grounded-refusal

Evidence-grounded QA: answer from provided evidence, refuse when insufficient, partially answer when only partly supported.

## Status

| Week | Focus | Status | Report |
|------|--------|--------|--------|
| 1 | Task + hand examples (20) | done | [`docs/reports/week1.md`](docs/reports/week1.md) |
| 2 | Pilot QA + preference (50 + 50) | **pilot done**; full ~500 deferred | [`docs/reports/week2.md`](docs/reports/week2.md) |
| 3 | Eval / base baseline | **harness v1 + baseline runs done** (v1 + v2 pilot); full judge calibration deferred | [`docs/reports/week3.md`](docs/reports/week3.md) |

## Quick data pointers

| File | Rows | Role |
|------|-----:|------|
| [`data/hand_examples.jsonl`](data/hand_examples.jsonl) | 20 | Week 1 gold (`dev`) |
| [`data/data_v1_pilot_layer1.jsonl`](data/data_v1_pilot_layer1.jsonl) | 50 | Week 2 Layer 1 templates |
| [`data/data_v1_pilot.jsonl`](data/data_v1_pilot.jsonl) | 50 | Week 2 Layer 2 paraphrased QA |
| [`data/preference_v1_pilot.jsonl`](data/preference_v1_pilot.jsonl) | 50 | Week 2 DPO pairs (1:1 with Layer 2) |
| [`data/data_v2_pilot.jsonl`](data/data_v2_pilot.jsonl) | 55 | Week 3 hand-built adversarial stress test |
| [`data/data_v2.jsonl`](data/data_v2.jsonl) | 600 | `data_v2_pilot` extended to 600 rows, weighted toward empirically-validated high-failure-rate mechanisms — see [`docs/DATA_V2_EXTENSION.md`](docs/DATA_V2_EXTENSION.md) |

**Week 2 pilot mix (QA):** 20 answerable / 20 unanswerable / 10 partial.  
**Week 2 pilot mix (preference):** 17 over_refusal / 10 hallucination / 10 distractor_confusion / 10 over_complete / 3 memory_override.  
**Week 3 stress-test mix (QA):** 30 answerable / 19 unanswerable / 6 partial.

Full distributions, schemas, and pipeline notes: see the week reports above.

## Labels and protocol

| Doc | Contents |
|-----|----------|
| [`docs/DATA_LABELS.md`](docs/DATA_LABELS.md) | `answerability`, `evidence_type`, `evidence_challenge` |
| [`docs/QA_GENERATION_PROTOCOL.md`](docs/QA_GENERATION_PROTOCOL.md) | How QA rows are built |
| [`docs/PREFERENCE_GENERATION_PROTOCOL.md`](docs/PREFERENCE_GENERATION_PROTOCOL.md) | How preference pairs are built |
| [`docs/DATA_V2_EXTENSION.md`](docs/DATA_V2_EXTENSION.md) | How `data_v2_pilot` (55) grew into `data_v2.jsonl` (600), and the judge-log-driven method used to pick which failure modes to scale |
| [`configs/prompts/default.yaml`](configs/prompts/default.yaml) | Shared Evidence / Question / Instruction template |

## Task (one paragraph)

Given **evidence** and a **question**, use only the evidence: **answer** if sufficient, **refuse** if not, **partially answer** if only part is supported. Do not override evidence with outside world knowledge unless the evidence itself states that fact.
