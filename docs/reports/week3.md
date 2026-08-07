# Week 3 report — Evaluation harness v1

## Delivered

| Item | Path | Notes |
|------|------|-------|
| Judge output / verdict schema | [`src/grounded_refusal/eval/schema_eval.py`](../../src/grounded_refusal/eval/schema_eval.py) | `ModelBehavior`, `JudgeOutput`, `EvalResult` |
| Verdict logic | [`src/grounded_refusal/eval/verdict.py`](../../src/grounded_refusal/eval/verdict.py) | `derive_verdict` — pure function, no API calls |
| Metrics aggregation | [`src/grounded_refusal/eval/metrics.py`](../../src/grounded_refusal/eval/metrics.py) | `aggregate` — slices by `answerability` |
| Judge call | [`src/grounded_refusal/eval/judge.py`](../../src/grounded_refusal/eval/judge.py) | `judge_row` — GPT-4o, structured output, 6 few-shot anchors |
| CLI | [`src/grounded_refusal/eval/run_eval.py`](../../src/grounded_refusal/eval/run_eval.py) | `python -m grounded_refusal.eval.run_eval` |
| Regression tests | [`tests/test_verdict.py`](../../tests/test_verdict.py) | 14 cases, all passing |

## Design (summary)

Two-stage LLM-judge, following industry-standard practice of keeping the judge's task atomic and letting deterministic code own the taxonomy:

1. **GPT-4o extracts only two orthogonal signals** per row — `predicted_behavior` (`answer` / `refuse` / `partial`) and `is_faithful` (bool: are all claims grounded in the evidence?). The judge never sees the gold `answerability` label, so it can't pattern-match a label — it independently assesses behavior and grounding from `evidence` + `question` + `model_output` alone.
2. **Python derives the verdict deterministically** from `(gold answerability, gold evidence_challenge, predicted_behavior, is_faithful)`. This mirrors `build_preference.py`'s `choose_negative_type`, but keyed on *observed* model behavior rather than *construction* intent — the same taxonomy (`correct` / `over_refusal` / `hallucination` / `distractor_confusion` / `over_complete` / `memory_override`), plus `anomaly` for combinations outside the pilot map, so unexpected model behavior is flagged rather than silently forced into the nearest category.

Full rationale and bias-mitigation notes (position bias avoided by using pointwise not pairwise judging, self-enhancement bias avoided by judging a 3B model with a GPT-4o judge, verbosity-bias risk specific to `over_complete`) were worked out in conversation; not duplicated here.

## Pipeline

```text
outputs/*.jsonl              (baseline inference, from cli.py infer_main)
        ↓
judge_row()                  predicted_behavior + is_faithful   (GPT-4o, temperature=0)
        ↓
derive_verdict()             correct / over_refusal / hallucination /
                              distractor_confusion / over_complete /
                              memory_override / anomaly
        ↓
aggregate()                  answer_accuracy, abstention_rate,
                              over_refusal_rate, unsupported_claim_rate,
                              anomaly_rate
```

## How to run

**1. Generate baseline model outputs first** (Week 3's other half — run on a machine with a GPU):

```bash
PYTHONPATH=src python -m grounded_refusal.cli \
  --data data/data_v1_pilot.jsonl \
  --output outputs/base_pilot.jsonl
```

**2. Dry-run the eval harness** (no API key needed — validates wiring: row selection, verdict mapping, aggregation, output writing):

```bash
PYTHONPATH=src python -m grounded_refusal.eval.run_eval \
  --input outputs/base_pilot.jsonl \
  --dry-run
```

**3. Judge a small sample first** (same 5-slice review set `build_preference.py` uses — one row per `negative_type`):

```bash
export OPENAI_API_KEY=...
PYTHONPATH=src python -m grounded_refusal.eval.run_eval \
  --input outputs/base_pilot.jsonl \
  --ids ex_0021 ex_0037 ex_0026 ex_0051 ex_0031 \
  --output outputs/base_pilot_eval_sample.jsonl
```

**4. Full pilot run** once the sample looks right:

```bash
PYTHONPATH=src python -m grounded_refusal.eval.run_eval \
  --input outputs/base_pilot.jsonl \
  --output outputs/base_pilot_eval.jsonl \
  --overwrite
```

Output: one `EvalResult` row per scored example in `--output`, plus the four headline metrics printed as JSON to stdout. `--max-workers` controls judge-call concurrency (default 4); `--limit` caps how many selected rows are scored.

## Validation

`tests/test_verdict.py` — 14 pytest cases anchored on `PREFERENCE_GENERATION_PROTOCOL.md`'s five worked examples (both `chosen` and `rejected` halves of `pref_0021`/`0042`/`0026`/`0051`/`0031`), plus the two edge cases the protocol's FAQ already flags (`known_world_conflict` refused, `partial` refused outright), plus two anomaly guards. `derive_verdict` is a pure function, so this is fully deterministic — no LLM calls, no flakiness.

```bash
PYTHONPATH="$PWD/src" .venv/bin/python -m pytest tests/test_verdict.py -v
```

The judge call itself (`judge_row`, GPT-4o) has only been smoke-tested with a placeholder in `--dry-run` mode — its actual accuracy against human judgment is not yet validated. That's a Week 6 item (human audit + judge calibration), not Week 3.

## Not in Week 3

- Running the judge against real baseline model outputs — blocked on the base-model inference run (GPU machine, separate from this repo's dev environment)
- Human calibration of judge verdicts (planned: full review of all 50 pilot rows, per the same discipline already used for `rejected` quality — see Week 6)
- Multi-checkpoint comparison (base vs. SFT vs. DPO) — `EvalResult.model_name` already supports tagging results by checkpoint, but no SFT/DPO checkpoints exist yet (Weeks 4–5)
- Slice metrics finer than `answerability` (e.g. distractor vs. known-world-conflict specifically, rather than folded into the same slice)
