# grounded-refusal

Evidence-grounded QA: answer from provided evidence, refuse when insufficient, partially answer when only partly supported.

## Task and research question

Given **evidence** and a **question**, the model must use only the evidence. It should **answer**
when the evidence is sufficient, **refuse** when it isn't, and **partially answer** when only part
of the question is supported. It should never fill a gap with outside knowledge or pretraining
memory.

This project studies whether preference optimization can calibrate that answer/refuse/partial
boundary: fewer unsupported claims, better abstention on unanswerable questions, without new
over-refusal on questions the model could actually answer. Testing that requires a full system, not
one training run: a data builder for answerable, unanswerable, partial, and distractor cases; a
training pipeline comparing base, SFT, and DPO; a diagnostic eval harness that can tell a genuinely
more honest model from a merely more conservative one; and failure analysis that feeds back into
the data.

## Quickstart

```bash
pip install -e ".[dev]"          # core + tests
pip install -e ".[inference]"    # + torch/transformers/peft, to run a model
pip install -e ".[train]"        # + trl/datasets, for LoRA SFT training

export PYTHONPATH=src

# Run base-model inference on a QA file
python -m grounded_refusal.inference.run_inference \
  --data data/data_v2_pilot.jsonl \
  --output outputs/inference-<model>/<name>.jsonl

# Train a LoRA SFT adapter
python -m grounded_refusal.train.train_sft

# Run inference with that adapter instead of the base model
python -m grounded_refusal.inference.run_inference \
  --data data/data_v2_pilot.jsonl \
  --adapter checkpoints/<the-run-you-just-made> \
  --output outputs/inference-<model>-sft/<name>.jsonl

# Judge either output (needs OPENAI_API_KEY, or pass --dry-run)
python -m grounded_refusal.eval.run_eval \
  --input outputs/inference-<model>/<name>.jsonl \
  --output outputs/eval-<model>/<name>_eval.jsonl
```

## Code layout

| Module | Contents |
|---|---|
| [`src/grounded_refusal/data/`](src/grounded_refusal/data/) | QA/preference schemas, generation scripts, schema validation |
| [`src/grounded_refusal/inference/`](src/grounded_refusal/inference/) | Base-model and LoRA-adapter inference |
| [`src/grounded_refusal/eval/`](src/grounded_refusal/eval/) | LLM judge, verdict logic, metric aggregation |
| [`src/grounded_refusal/train/`](src/grounded_refusal/train/) | LoRA SFT training |

## Status

| Week | Focus | Status | Report |
|------|--------|--------|--------|
| 1 | Task + hand examples (20) | done | [`docs/reports/week1.md`](docs/reports/week1.md) |
| 2 | Pilot QA + preference (50 + 50) | **pilot done**; full ~500 deferred | [`docs/reports/week2.md`](docs/reports/week2.md) |
| 3 | Eval / base baseline | **eval harness built**; ran baseline on `data_v1_pilot` and `data_v2_pilot` | [`docs/reports/week3.md`](docs/reports/week3.md) |
| 4 | SFT training pipeline | **pipeline built + smoke-test run** | [`docs/reports/week4.md`](docs/reports/week4.md) |

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

## Checkpoints

Trained LoRA adapters aren't committed to git. What each timestamped checkpoint actually is (base
model, training data, config) is recorded in [`checkpoints/README.md`](checkpoints/README.md).

## Docs

### Data

| Doc | Contents |
|-----|----------|
| [`docs/DATA_LABELS.md`](docs/DATA_LABELS.md) | `answerability`, `evidence_type`, `evidence_challenge` |
| [`docs/QA_GENERATION_PROTOCOL.md`](docs/QA_GENERATION_PROTOCOL.md) | How QA rows are built |
| [`docs/PREFERENCE_GENERATION_PROTOCOL.md`](docs/PREFERENCE_GENERATION_PROTOCOL.md) | How preference pairs are built |
| [`docs/DATA_V2_EXTENSION.md`](docs/DATA_V2_EXTENSION.md) | How `data_v2_pilot` (55) grew into `data_v2.jsonl` (600), and the judge-log-driven method used to pick which failure modes to scale |
| [`configs/prompts/default.yaml`](configs/prompts/default.yaml) | Shared Evidence / Question / Instruction template |

### Training and eval

| Doc | Contents |
|-----|----------|
| [`docs/EVAL_METRICS.md`](docs/EVAL_METRICS.md) | How judge output becomes the abstention / hallucination / partial metrics |
| [`docs/GPU_MEMORY_ESTIMATE.md`](docs/GPU_MEMORY_ESTIMATE.md) | LoRA vs. full fine-tuning VRAM budget for a 16GB GPU |
| [`docs/JUDGE_MODEL.md`](docs/JUDGE_MODEL.md) | Why the judge moved from gpt-4o to gpt-5-mini, with a row-by-row comparison |
