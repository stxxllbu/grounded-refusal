# Week 3 report — Evaluation harness v1

## Delivered

| Item | Path | Notes |
|------|------|-------|
| Judge output / outcome schema | [`src/grounded_refusal/eval/schema_eval.py`](../../src/grounded_refusal/eval/schema_eval.py) | `ModelBehavior`, `JudgeOutput`, `EvalResult`, `AbstentionOutcome`, `PartialOutcome` |
| Outcome logic | [`src/grounded_refusal/eval/verdict.py`](../../src/grounded_refusal/eval/verdict.py) | `derive_abstention_outcome`, `derive_partial_outcome` — two pure functions, no API calls |
| Metrics aggregation | [`src/grounded_refusal/eval/metrics.py`](../../src/grounded_refusal/eval/metrics.py) | `aggregate` — three independent metric groups, no shared denominators |
| Judge call | [`src/grounded_refusal/eval/judge.py`](../../src/grounded_refusal/eval/judge.py) | `judge_row` — GPT-4o, structured output, 6 few-shot anchors |
| CLI | [`src/grounded_refusal/eval/run_eval.py`](../../src/grounded_refusal/eval/run_eval.py) | `python -m grounded_refusal.eval.run_eval` |
| Regression tests | [`tests/test_verdict.py`](../../tests/test_verdict.py) | 20 cases, all passing |
| Metrics design doc | [`docs/EVAL_METRICS.md`](../EVAL_METRICS.md) | Full spec, rationale, and literature references |

## Design (summary)

Two-stage LLM-judge, following industry-standard practice of keeping the judge's task atomic and letting deterministic code own the metrics. Full spec: [`docs/EVAL_METRICS.md`](../EVAL_METRICS.md); short version:

1. **GPT-4o extracts only two orthogonal signals** per row — `predicted_behavior` (`answer` / `refuse` / `partial`) and `is_faithful` (bool: are all claims grounded in the evidence?). The judge never sees the gold `answerability` label, so it can't pattern-match a label — it independently assesses behavior and grounding from `evidence` + `question` + `model_output` alone.
2. **Python derives two independent outcomes**, each a pure function of only the fields it actually needs:
   - `derive_abstention_outcome(answerability, predicted_behavior)` — a standard TP/FP/TN/FN confusion matrix for "should this row have been refused?" (SQuAD 2.0 / Abstain-QA style), strictly limited to `answerable`/`unanswerable` rows. `over_refusal_rate` is literally this matrix's false-positive rate.
   - `derive_partial_outcome(answerability, predicted_behavior)` — `match` / `under_deliver` / `over_deliver` for `partial` rows, kept fully separate from the abstention matrix rather than forced in as a 3rd class (which would break the 2×2 precision/recall/F1 semantics).
3. **Hallucination rate is a third, separate calculation** — filtered to rows where `predicted_behavior ∈ {answer, partial}` (did the model assert anything checkable?), then `is_faithful` within that filtered set. Never uses `answerability` or `evidence_challenge`.

An earlier attempt collapsed all of this into one `derive_verdict()` function producing a 7-category taxonomy (`correct`/`over_refusal`/`hallucination`/`distractor_confusion`/`over_complete`/`memory_override`/`anomaly`) borrowed from `build_preference.py`'s `choose_negative_type`. It was discarded — see [`docs/EVAL_METRICS.md`](../EVAL_METRICS.md#where-did-evidence_challenge-go) for why (the `anomaly` bucket conflated genuine logical impossibilities with ordinary unnamed failures, and the taxonomy wasn't how the field actually reports this — see the doc's references).

## Pipeline

```text
outputs/*.jsonl              (baseline inference, from cli.py infer_main)
        ↓
judge_row()                  predicted_behavior + is_faithful   (GPT-4o, temperature=0)
        ↓
derive_abstention_outcome()  true_positive / false_positive /
                              true_negative / false_negative
                              (answerable/unanswerable rows only)
derive_partial_outcome()     match / under_deliver / over_deliver
                              (partial rows only)
        ↓
aggregate()                  abstention_recall, abstention_precision,
                              over_refusal_rate, hallucination_rate,
                              partial_match_rate, partial_under_deliver_rate,
                              partial_over_deliver_rate
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

Output: one `EvalResult` row per scored example in `--output`, plus up to seven metrics printed as JSON to stdout (`abstention_recall`, `abstention_precision`, `over_refusal_rate`, `hallucination_rate`, `partial_match_rate`, `partial_under_deliver_rate`, `partial_over_deliver_rate` — each key present only if its underlying row count is non-zero). `--max-workers` controls judge-call concurrency (default 4); `--limit` caps how many selected rows are scored.

## Validation

`tests/test_verdict.py` — 20 pytest cases anchored on `PREFERENCE_GENERATION_PROTOCOL.md`'s five worked examples (both `chosen` and `rejected` halves of `pref_0021`/`0042`/`0026`/`0051`/`0031`), covering both `derive_abstention_outcome` and `derive_partial_outcome`, including the "partial rows never enter the abstention matrix" exclusion cases. Both functions are pure, so this is fully deterministic — no LLM calls, no flakiness.

```bash
PYTHONPATH="$PWD/src" .venv/bin/python -m pytest tests/test_verdict.py -v
```

The judge call itself (`judge_row`, GPT-4o) has only been smoke-tested with a placeholder in `--dry-run` mode — its actual accuracy against human judgment is not yet validated. That's a Week 6 item (human audit + judge calibration), not Week 3.

## Not in Week 3

- Running the judge against real baseline model outputs — blocked on the base-model inference run (GPU machine, separate from this repo's dev environment)
- Human calibration of judge verdicts (planned: full review of all 50 pilot rows, per the same discipline already used for `rejected` quality — see Week 6)
- Multi-checkpoint comparison (base vs. SFT vs. DPO) — `EvalResult.model_name` already supports tagging results by checkpoint, but no SFT/DPO checkpoints exist yet (Weeks 4–5)
- Slice metrics finer than `answerability` (e.g. distractor vs. known-world-conflict specifically, rather than folded into the same slice)
