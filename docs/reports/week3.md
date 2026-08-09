# Week 3 report — Evaluation harness + baseline

## Summary

Built the inference pipeline and the eval harness (metric design included), then tested the whole thing end to end: got local GPU inference working, ran the base 3B model against the pilot data, found the pilot data wasn't hard enough to expose real weaknesses, built a harder adversarial dataset, reran, and got a baseline that actually shows where the model breaks. One small judge-calibration issue turned up along the way and was fixed. These baseline numbers are the comparison point Week 4/5 (SFT/DPO) will be measured against.

## Development

Two subsystems, built this week — inference (2 files) and the eval harness (7 files, including tests):

| Item | Path | Notes |
|------|------|-------|
| Base-model inference CLI | [`src/grounded_refusal/inference/run_inference.py`](../../src/grounded_refusal/inference/run_inference.py) | `infer_main` — validates QA data, assembles prompts, writes `model_output` |
| Inference backend | [`src/grounded_refusal/inference/hf_backend.py`](../../src/grounded_refusal/inference/hf_backend.py) | `run_batch_inference` — HF transformers/torch, lazy-imported so `--dry-run` needs no GPU deps |
| Judge output / outcome schema | [`src/grounded_refusal/eval/schema_eval.py`](../../src/grounded_refusal/eval/schema_eval.py) | `ModelBehavior`, `JudgeOutput`, `EvalResult`, `AbstentionOutcome`, `PartialOutcome` |
| Outcome logic | [`src/grounded_refusal/eval/verdict.py`](../../src/grounded_refusal/eval/verdict.py) | `derive_abstention_outcome`, `derive_partial_outcome` — two pure functions, no API calls |
| Metrics aggregation | [`src/grounded_refusal/eval/metrics.py`](../../src/grounded_refusal/eval/metrics.py) | `aggregate` — three independent metric groups, no shared denominators |
| Judge call | [`src/grounded_refusal/eval/judge.py`](../../src/grounded_refusal/eval/judge.py) | `judge_row` — GPT-4o, structured output, 7 few-shot anchors |
| CLI | [`src/grounded_refusal/eval/run_eval.py`](../../src/grounded_refusal/eval/run_eval.py) | `python -m grounded_refusal.eval.run_eval` |
| Regression tests | [`tests/test_verdict.py`](../../tests/test_verdict.py) | 20 cases, all passing |

Pipeline these pieces form together:

```text
outputs/*.jsonl              (baseline inference, from inference/run_inference.py infer_main)
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

## Eval metric design

### Why GPT-4o as the judge

Chosen for two reasons: it's clearly stronger than the Qwen2.5-0.5B/3B models being evaluated (a judge should outclass its subject, not match it), and it's a different model family entirely, avoiding the intra-family bias risk of judging Qwen outputs with another Qwen model. Structured outputs (`response_format=JudgeOutput`) were a secondary, practical factor — guaranteed-parseable output every call, matching the Pydantic-schema pattern already used throughout this codebase.

### How the judge's output becomes metrics

With GPT-4o settled as the judge, its task is kept atomic and deterministic code owns everything downstream of it. Full spec: [`docs/EVAL_METRICS.md`](../EVAL_METRICS.md); short version:

1. **GPT-4o extracts only two orthogonal signals** per row — `predicted_behavior` (`answer` / `refuse` / `partial`) and `is_faithful` (bool: are all claims grounded in the evidence?). The judge never sees the gold `answerability` label, so it can't pattern-match a label — it independently assesses behavior and grounding from `evidence` + `question` + `model_output` alone.
2. **Python derives two independent outcomes**, each a pure function of only the fields it actually needs:
   - `derive_abstention_outcome(answerability, predicted_behavior)` — a standard TP/FP/TN/FN confusion matrix for "should this row have been refused?" (SQuAD 2.0 / Abstain-QA style), strictly limited to `answerable`/`unanswerable` rows. `over_refusal_rate` is literally this matrix's false-positive rate.
   - `derive_partial_outcome(answerability, predicted_behavior)` — `match` / `under_deliver` / `over_deliver` for `partial` rows, kept fully separate from the abstention matrix rather than forced in as a 3rd class (which would break the 2×2 precision/recall/F1 semantics).
3. **Hallucination rate is a third, separate calculation** — filtered to rows where `predicted_behavior ∈ {answer, partial}` (did the model assert anything checkable?), then `is_faithful` within that filtered set. Never uses `answerability` or `evidence_challenge`.

### Why these three stay independent

Keeping the abstention confusion matrix, `hallucination_rate`, and the partial-row outcomes as three separate calculations — none sharing a denominator with another — is a deliberate choice, not an oversight. The confusion matrix follows the standard selective-QA framing (SQuAD 2.0 HasAns/NoAns, Abstain-QA), which only holds together as a clean 2×2 with valid precision/recall semantics if nothing else is folded into it. Because nothing shares a denominator, each metric stays diagnosable on its own: a training method could plausibly reduce `hallucination_rate` while making `over_refusal_rate` worse, or the reverse, and this design would show that clearly rather than blending both effects into one score that can't tell them apart. (A single combined 7-category taxonomy was tried first. It was dropped for two documented reasons — an `anomaly` bucket that conflated genuine logical impossibilities with ordinary unnamed failures, a real bug, plus sub-typing hallucination causes not being how the field standardly reports this metric — see [`docs/EVAL_METRICS.md`](../EVAL_METRICS.md#where-did-evidence_challenge-go) for the full reasoning. The denominator-independence argument above is a related benefit of the redesign, not the original stated reason for dropping it.)

## Experiments

**1. Got local GPU inference working.** This machine's GPU is an RTX 5060 Ti (Blackwell, sm_120 / compute capability 12.0) — new enough that it needed torch ≥2.6/2.7 (cu124+) to even recognize it; confirmed working on torch 2.13.0+cu130. Anonymous Hugging Face downloads are rate-limited (~2-3MB/s, ~45+ minutes for the 3B model's ~6GB of weights). A first smoke-test attempt was interrupted before finishing; the re-run confirmed both `smoke.yaml` (0.5B) and `base.yaml` (3B) produced correct, grounded output on a 2-row sanity set before committing to the full pilot run.

**2. Tested the base model against `data_v1_pilot`.**

| Model | Data | n | `abstention_recall` | `abstention_precision` | `over_refusal_rate` | `hallucination_rate` | `partial_match_rate` |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B-Instruct | `data_v1_pilot` | 50 | 0.2500 | 1.0000 | 0.0000 | 0.4667 | 0.4000 |
| Qwen2.5-3B-Instruct | `data_v1_pilot` | 50 | 0.9500 | 1.0000 | 0.0000 | 0.0645 | 1.0000 |

3B was a large step up from 0.5B across every metric — but 3B scoring this close to perfect on `data_v1_pilot` was itself a signal: the pilot set, built in Week 2 as a first pass, wasn't hard enough to show where a reasonably capable base model actually breaks.

**3. Built a harder dataset (`data_v2_pilot`).** 55 hand-built adversarial rows targeting specific failure modes the pilot set didn't cover: `known_world_conflict` (evidence that contradicts real-world facts, to test whether the model follows evidence over memory), `distractor_entity`, `partial_evidence`, plus a prompt-injection case (an instruction embedded inside the evidence itself, trying to hijack the model's answer).

**4. Reran against `data_v2_pilot`.**

| Model | Data | n | `abstention_recall` | `abstention_precision` | `over_refusal_rate` | `hallucination_rate` | `partial_match_rate` |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-3B-Instruct | `data_v2_pilot` | 55 | 0.6316 | 0.9231 | 0.0333 | 0.1905 | 0.6667 |

Same model, much worse: `abstention_recall` drops from 0.95 to 0.63, `hallucination_rate` roughly triples. This is the harness working as intended — the harder data actually surfaces grounding failures the easier pilot set hid.

Running the eval side of this hit a real constraint worth noting: this OpenAI org's `gpt-4o` limit is 30,000 tokens/minute, and judging a full pilot run at the default `--max-workers 4` reliably exhausted it (429 errors) partway through. `run_eval.py` has no checkpointing, so a crash meant re-judging (and re-paying for) every row from scratch. Worked around by always running one judge job at a time, solo, at `--max-workers 1` — this avoids the crash but isn't a real fix: there's still no backoff, retry, or checkpointing in `run_eval.py`, so a crash under any other concurrency setting (or a shared TPM budget with any other process) would still lose all progress on that run. Adding checkpointing is unstarted, real tech debt.

**5. Found and fixed a small judge-calibration issue (issue #1).** Reviewing `data_v2_pilot`'s judge output by hand turned up 4 mislabeled rows, all on `known_world_conflict` — the judge's few-shot examples had no worked case for "model faithfully follows evidence that's wrong in reality," so it sometimes penalized correct behavior (e.g. marking a response `unfaithful` for accurately restating a fictional evidence value, when it should have been faithful by definition). Fixed with one added example in `judge.py`; verified by re-judging and by manually re-checking all 19 `known_world_conflict` rows. `data_v2_pilot`'s `hallucination_rate` above (0.1905) already reflects the fix (pre-fix: 0.2857).

## How to run

**1. Generate baseline model outputs** (needs a GPU; see the compatibility/rate-limit notes above). Output path follows the `outputs/inference-<model>/base_<dataset>.jsonl` convention used throughout this report:

```bash
PYTHONPATH=src python -m grounded_refusal.inference.run_inference \
  --data data/data_v1_pilot.jsonl \
  --model-config configs/models/base.yaml \
  --output outputs/inference-qwen2.5-3b-instruct/base_v1_pilot.jsonl
```

**2. Dry-run the eval harness** (no API key needed — validates wiring: row selection, verdict mapping, aggregation, output writing):

```bash
PYTHONPATH=src python -m grounded_refusal.eval.run_eval \
  --input outputs/inference-qwen2.5-3b-instruct/base_v1_pilot.jsonl \
  --dry-run
```

**3. Judge a small sample first** (same 5-slice review set `build_preference.py` uses — one row per `negative_type`):

```bash
export OPENAI_API_KEY=...
PYTHONPATH=src python -m grounded_refusal.eval.run_eval \
  --input outputs/inference-qwen2.5-3b-instruct/base_v1_pilot.jsonl \
  --ids ex_0021 ex_0037 ex_0026 ex_0051 ex_0031 \
  --output outputs/eval-qwen2.5-3b-instruct/base_v1_pilot_eval_sample.jsonl
```

**4. Full pilot run**, one judge job at a time, `--max-workers 1`. Output path follows `outputs/eval-<model>/base_<dataset>_eval.jsonl` — this is exactly what produced the numbers in the Experiments table above:

```bash
PYTHONPATH=src python -m grounded_refusal.eval.run_eval \
  --input outputs/inference-qwen2.5-3b-instruct/base_v1_pilot.jsonl \
  --output outputs/eval-qwen2.5-3b-instruct/base_v1_pilot_eval.jsonl \
  --overwrite \
  --max-workers 1
```

Output: one `EvalResult` row per scored example in `--output`, plus up to seven metrics printed as JSON to stdout — each key present only if its underlying row count is non-zero. `--limit` caps how many selected rows are scored.

`tests/test_verdict.py` — 20 pytest cases anchored on `PREFERENCE_GENERATION_PROTOCOL.md`'s five worked examples, covering both `derive_abstention_outcome` and `derive_partial_outcome`. Pure functions, fully deterministic:

```bash
PYTHONPATH="$PWD/src" .venv/bin/python -m pytest tests/test_verdict.py -v
```

## Limitations

- **Small n, especially for slices.** `hallucination_rate` on `data_v2_pilot` is computed over 42 "attempted" rows; `partial_match_rate` over just 6 `partial` rows. A rate moving from 0.95 to 0.63 on ~19-20 unanswerable rows per set is suggestive, not statistically tight — treat these as directional signals for an 8-week MVP, not precise estimates.
- **Judge calibration is partial, not complete.** The manual review covered all 19 rows of the `known_world_conflict` category on `data_v2_pilot` — not `distractor_entity`, not `partial_evidence` beyond a few spot-checks, not `data_v1_pilot`, and not against real human labels.
- **No automated finer-than-`answerability` slicing.** The `known_world_conflict` review was done by hand; `metrics.py` doesn't compute per-`evidence_challenge` breakdowns yet.

## Not in Week 3

- Full human calibration of judge verdicts across all rows/categories (planned: full review of all pilot rows, per the same discipline already used for `rejected` quality — see Week 6)
- Multi-checkpoint comparison (base vs. SFT vs. DPO) — `EvalResult.model_name` already supports tagging results by checkpoint, but no SFT/DPO checkpoints exist yet
- Automated slice metrics finer than `answerability` (see Limitations)

## Next: Week 4

These baseline numbers (3B on `data_v1_pilot` and `data_v2_pilot`) are the comparison point for the SFT baseline: does SFT move `abstention_recall`/`hallucination_rate` on `data_v2_pilot` in the right direction, and does it introduce over-refusal that isn't visible on the easier `data_v1_pilot` set?
