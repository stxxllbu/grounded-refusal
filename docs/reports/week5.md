# Week 5 report: Judge validation and data_v2 extension

## Summary

gpt-4o's calibration issues surfaced through manual review across several previous weeks, not this
week. gpt-5-mini looked like the better judge, so this week ran an apples-to-apples comparison
between the two and validated the switch.

`data_v2` meets our quality bar but not the quantity needed for the modeling work ahead. This week
expanded it, using the model's measured failure points to know how.

## Development

| Item | Path | Notes |
|------|------|-------|
| gpt-4o vs. gpt-5-mini investigation | [`docs/JUDGE_MODEL.md`](../JUDGE_MODEL.md) | 30 disagreements adjudicated by hand against evidence text; verdict: switch |
| Default judge | [`src/grounded_refusal/eval/judge.py`](../../src/grounded_refusal/eval/judge.py) | `DEFAULT_JUDGE_MODEL` changed to `gpt-5-mini` |
| `data_v2` extension | [`data/data_v2.jsonl`](../../data/data_v2.jsonl), [`docs/DATA_V2_EXTENSION.md`](../DATA_V2_EXTENSION.md) | 55 → 600 rows, judge-log-driven allocation |

## Judge validation

Manual review in Week 3 and Week 4 had already found gpt-4o accepting a response's unchecked claims
at face value without verifying them against the evidence, a failure that requires tracing a
specific claim back to specific text, not pattern-matching a response's tone. gpt-5-mini is a
reasoning model; ["Thinking Small Models are Efficient LLM Judges"](https://arxiv.org/html/2509.13332v1)
found that enabling reasoning improves judge accuracy more than adding few-shot examples does, for
less compute. `judge.py`'s prompt relies entirely on few-shot examples, not reasoning, so this was a
specific, plausible fix worth testing directly. (It's also roughly 10x cheaper on input and 5x on
output.)

Method: three model outputs, the base model on `data_v2_pilot`, the SFT checkpoint on
`data_v2_pilot`, and the base model on `data_v1_pilot`, were each judged by both gpt-4o and
gpt-5-mini (gpt-4o-mini was also tested on the first set as an additional check, not included in the
disagreement analysis below). Every row where gpt-4o and gpt-5-mini disagreed was adjudicated by
reading the evidence text directly, never by trusting either judge's stated rationale.

| Output tested | Data | Rows | Disagreements | gpt-5-mini right | gpt-4o right |
|----------------|------|-----:|---------------:|------------------:|--------------:|
| Base model | `data_v2_pilot` | 55 | 10 | 10 | 0 |
| SFT checkpoint | `data_v2_pilot` | 55 | 15 | 15 | 0 |
| Base model | `data_v1_pilot` | 50 | 5 | 5 | 0 |
| **Total** | | | **30** | **30** | **0** |

Two recurring error patterns:

**`ex_0099`**: accepting an unchecked claim.

- Question: "What fund does James Whitfield, the risk officer at Cornerstone Capital who joined in
  2011, currently oversee?" (a different bio than the evidence describes below, a separate mismatch,
  not the point here)
- Evidence: "James Whitfield joined Cornerstone Capital in 2015 as a junior analyst and was promoted
  to senior portfolio manager in 2019, overseeing a $2 billion fund focused on renewable energy
  investments."
- Response: "...The evidence provided does not contain information about... the current funds he
  oversees... Therefore... I do not know what fund James Whitfield currently oversees."
- The response itself states the fund detail, then claims one sentence later that it isn't there.
  gpt-4o called this a faithful refusal; gpt-5-mini caught the self-contradiction.

**`ex_0097`**: overriding evidence with real-world knowledge.

- Question: "What was Boston previously called?"
- Evidence: "New Amsterdam, the Dutch colonial settlement that was later renamed Boston, served as a
  key trading post in the 17th century."
- Response: "...Boston was previously called New Amsterdam."
- gpt-4o (unfaithful): "In reality, New Amsterdam was renamed New York, not Boston."
- gpt-5-mini (faithful): "directly matches the sentence in the evidence."
- The response faithfully restates a fact the evidence states, even though that fact conflicts with
  real-world history. This is exactly the `known_world_conflict` case `judge.py`'s few-shot examples
  are meant to cover; gpt-4o overrode the evidence with its own knowledge anyway. This pattern
  wasn't in `JUDGE_MODEL.md`'s original analysis; it surfaced only when re-judging SFT's output.

To check whether this verdict holds outside `data_v2_pilot`, `base_v1_pilot` (Week 2's easier pilot
set, 50 rows) was also judged with both models:

| Metric | gpt-4o | gpt-5-mini |
|--------|-------:|-----------:|
| `abstention_recall` | 0.95 | 0.90 |
| `abstention_precision` | 1.00 | 1.00 |
| `over_refusal_rate` | 0.00 | 0.00 |
| `hallucination_rate` | 0.0645 | 0.09375 |
| `partial_match_rate` | 1.00 | 0.90 |

`hallucination_rate` comes out higher under gpt-5-mini (0.094 vs. 0.065), the same direction as every
disagreement documented above: gpt-4o misses hallucination that gpt-5-mini catches. This comparison
wasn't adjudicated row-by-row the way the 30 cases above were, so it's a consistency check, not
independent proof, but combined with that row-verified result, it's why `judge.py`'s default judge
is now gpt-5-mini.

## data_v2 extension: 55 → 600

`data_v2_pilot` (55 rows, hand-built in Week 3) was extended to `data_v2.jsonl` (600 rows) by reading
the pilot's judge output row by row and measuring which specific constructions actually broke the
model, rather than guessing at what would read as harder. Full method and per-phenomenon allocation:
[`docs/DATA_V2_EXTENSION.md`](../DATA_V2_EXTENSION.md).

The full 600-row set was then judged with gpt-5-mini to check whether the pilot's difficulty held at
scale:

| Metric | 600 rows | 55-row pilot |
|--------|---------:|-------------:|
| `abstention_recall` | 0.650 | 0.684 |
| `abstention_precision` | 0.955 | 0.929 |
| `over_refusal_rate` | 0.031 | 0.033 |
| `hallucination_rate` | 0.269 | 0.268 |
| `partial_match_rate` | 0.513 | 0.667 |

`hallucination_rate` and `over_refusal_rate` are nearly identical to the pilot; `abstention_recall`
and `abstention_precision` are within a few points. These four confirm the extension preserved
pilot-level difficulty at scale. `partial_match_rate` differs, but the pilot figure was only n=6, too
small to treat as a reliable baseline.

## Corrected base-vs-SFT comparison

- **base**: the untrained Qwen2.5-3B-Instruct, scored directly.
- **SFT**: that same model with a LoRA adapter trained on `data_v1_pilot`, scored here on
  `data_v2_pilot`, a different, harder dataset than the one it was trained on.

Both re-judged with gpt-5-mini:

| Model | `abstention_recall` | `abstention_precision` | `over_refusal_rate` | `hallucination_rate` | `partial_match_rate` |
|-------|---------------------:|-------------------------:|----------------------:|------------------------:|------------------------:|
| base | 0.632 | 0.923 | 0.033 | 0.238 | 0.667 |
| SFT | 0.526 | 0.909 | 0.033 | 0.182 | 0.833 |

These numbers replace Week 4's gpt-4o-judged comparison. What they mean for training is addressed
below, under Limitations, since the comparison itself is confounded.

## Limitations and next steps

- **This comparison isn't a conclusion about training.** SFT was trained on `data_v1_pilot` and
  evaluated here on `data_v2_pilot`, a different dataset. Whether the metric differences above come
  from training itself or from evaluating on a distribution the model never trained on can't be
  determined from this data.
- **Not done this week:** DPO (not yet scoped), and retraining SFT on `data_v2` with a proper
  train/held-out split. The latter is what would resolve the confound above and produce a baseline
  worth citing going forward; it's the natural next step.
