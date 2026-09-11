# Week 5 report: Judge validation and data_v2 extension

## Summary

Two independent problems blocked trustworthy evaluation going into this week. gpt-4o, the judge
behind every Week 3–4 result, had failed calibration checks twice: it had no coverage for evidence
that faithfully conflicts with real-world fact (Week 3), and it accepted a response's unverified
claims at face value (Week 4). Separately, `data_v2_pilot`, the only dataset hard enough to expose
real model weaknesses, had just 55 rows, too few to support the modeling work planned ahead.

This week resolved both. A blind, evidence-adjudicated comparison across three model outputs and 30
disagreement cases found gpt-5-mini correct in every case; it is now `judge.py`'s default judge.
`data_v2_pilot` was extended to `data_v2.jsonl` (600 rows) by measuring, not guessing, which specific
constructions broke the model, then verified at full scale to confirm the extension held that
difficulty.

With a validated judge in hand, the base-vs-SFT comparison Week 4 flagged as unreliable was rerun.
The result changes materially: SFT's recall regression is roughly twice what Week 4 reported, and
its hallucination direction reverses. Neither number is a verdict on whether training helps; both
are still confounded by a train/eval mismatch inherited from Week 4 (Limitations, below).

## Development

| Item | Path | Notes |
|------|------|-------|
| gpt-4o vs. gpt-5-mini investigation | [`docs/JUDGE_MODEL.md`](../JUDGE_MODEL.md) | 30 disagreements adjudicated by hand against evidence text; verdict: switch |
| Default judge | [`src/grounded_refusal/eval/judge.py`](../../src/grounded_refusal/eval/judge.py) | `DEFAULT_JUDGE_MODEL` changed to `gpt-5-mini` |
| `data_v2` extension | [`data/data_v2.jsonl`](../../data/data_v2.jsonl), [`docs/DATA_V2_EXTENSION.md`](../DATA_V2_EXTENSION.md) | 55 → 600 rows, judge-log-driven allocation |

## Judge validation

gpt-4o failed two independent calibration checks before this week, and they are failures of the same
kind: gpt-4o judges surface plausibility, not verified ground truth. In Week 3, manual review found
its few-shot prompt had no case for a model that faithfully restates evidence conflicting with
real-world fact, so it penalized correct behavior. In Week 4, manual review found it accepting a
response's unverified "the evidence doesn't specify X" claims without checking them, inflating
apparent faithfulness.

gpt-5-mini was the candidate fix, for two independent reasons. It costs roughly 10x less on input and
5x less on output than gpt-4o. More consequentially, it is a reasoning model:
["Thinking Small Models are Efficient LLM Judges"](https://arxiv.org/html/2509.13332v1) reports that
enabling reasoning is a far more efficient accuracy lever than adding few-shot examples (about 10%
accuracy gain for 1.5-2x compute, versus about 4.5% for 8x compute from more few-shot examples).
`judge.py`'s prompt relies entirely on few-shot examples. A model that reasons through a claim rather
than pattern-matching a response's tone was worth testing directly against gpt-4o's two documented
failure modes, not assumed to fix them.

**Method.** Three model outputs, the base model on `data_v2_pilot`, the SFT checkpoint on
`data_v2_pilot`, and the base model on `data_v1_pilot`, were each scored by both gpt-4o and
gpt-5-mini (gpt-4o-mini was also scored on the first set as an additional reference point, not
included in the disagreement count below). Every row where the two judges disagreed was adjudicated
by reading the evidence text directly; neither judge's stated rationale was trusted at face value.

| Output tested | Data | Rows | Disagreements | gpt-5-mini right | gpt-4o right |
|----------------|------|-----:|---------------:|------------------:|--------------:|
| Base model | `data_v2_pilot` | 55 | 10 | 10 | 0 |
| SFT checkpoint | `data_v2_pilot` | 55 | 15 | 15 | 0 |
| Base model | `data_v1_pilot` | 50 | 5 | 5 | 0 |
| **Total** | | | **30** | **30** | **0** |

30 for 30 is not explainable by chance. It confirms gpt-5-mini's calls are consistently the ones
supported by the evidence text, across three different model outputs and two distinct failure modes.
The metric-level effect varies by test set, which is itself informative:

Base model on `data_v2_pilot` (all three judges tested here):

| Metric | gpt-4o | gpt-4o-mini | gpt-5-mini |
|--------|-------:|------------:|-----------:|
| `abstention_recall` | 0.6316 | 0.6842 | 0.6842 |
| `abstention_precision` | 0.9231 | 0.9286 | 0.9286 |
| `over_refusal_rate` | 0.0333 | 0.0333 | 0.0333 |
| `hallucination_rate` | 0.1905 | 0.1951 | 0.2683 |
| `partial_match_rate` | 0.6667 | 0.8333 | 0.6667 |

Base model on `data_v1_pilot`:

| Metric | gpt-4o | gpt-5-mini |
|--------|-------:|-----------:|
| `abstention_recall` | 0.95 | 0.95 |
| `abstention_precision` | 1.00 | 1.00 |
| `over_refusal_rate` | 0.00 | 0.00 |
| `hallucination_rate` | 0.0645 | 0.0645 |
| `partial_match_rate` | 1.00 | 0.90 |

The SFT checkpoint's metric-level comparison on `data_v2_pilot` appears below, under Corrected
base-vs-SFT comparison. `data_v1_pilot` shows no hallucination-rate gap at all between judges, unlike
the other two sets. That is consistent with gpt-4o's failures running in both directions rather than
being a uniform under-detection bias: on the harder, adversarial sets, gpt-4o's misses skew toward
missing hallucination it should have caught, but on this easier set there is little hallucination
present for either judge to miss.

Two patterns account for most of the 30 disagreements. First, gpt-4o repeats a response's own
unverified claims back as true instead of checking them against the evidence, including unchecked
arithmetic and treating an unstated inference as fact (11 of 30 cases). Second, gpt-4o classifies a
response `partial` whenever it answers one thing and adds an aside about something else, without
checking whether the question actually has two distinct parts (10 of 30 cases, across all three test
sets).

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

**`ex_0049`**: misapplying `partial`.

- Question: "Who was the recipient of the Halley Prize in 2021 as stated in the evidence?"
- Evidence: "The Halley Prize, awarded annually, recognizes significant contributions to marine
  biology. In 2021, Dr. Lin was chosen by the committee for her work on reef fungi, and the award
  ceremony took place in Vancouver."
- Response: "Dr. Lin was the recipient of the Halley Prize in 2021 as stated in the evidence. The
  evidence does not provide information about the location of the award ceremony."
- The question asks one thing; the response answers it correctly, then adds a false aside (the
  evidence does state the location, Vancouver). gpt-4o called this `partial` and faithful; gpt-5-mini
  called it `answer`, unfaithful: there's no real second part to decline, and the aside itself is
  wrong.

`judge.py`'s default judge is now gpt-5-mini.

## data_v2 extension: 55 → 600

`data_v2_pilot` (55 rows, hand-built in Week 3) proved genuinely difficult: it dropped the base
model's `abstention_recall` from 95% to 63%. But 55 rows cannot support the modeling work planned
ahead, and scaling it required identifying which specific constructions break the model, not which
passages read as harder to a person. The two don't correlate: burying a false fact deeper in a
paragraph, or wrapping it in arithmetic, made `known_world_conflict` rows easier for the model, not
harder, while a flat, undisguised version broke it a third of the time. `data_v2.jsonl` (600 rows)
was built by reading the pilot's judge output row by row, measuring an empirical hit rate for each
phenomenon, and allocating new rows to the phenomena that actually broke the model. Full method and
per-phenomenon allocation: [`docs/DATA_V2_EXTENSION.md`](../DATA_V2_EXTENSION.md).

The full 600-row set was then judged with gpt-5-mini to confirm the extension didn't dilute that
difficulty:

| Metric | 600 rows | 55-row pilot |
|--------|---------:|-------------:|
| `abstention_recall` | 0.650 | 0.684 |
| `abstention_precision` | 0.955 | 0.929 |
| `over_refusal_rate` | 0.031 | 0.033 |
| `hallucination_rate` | 0.269 | 0.268 |
| `partial_match_rate` | 0.513 | 0.667 |

Four of five metrics hold within a few points of the pilot. Only `partial_match_rate` diverges, and
the pilot's figure there was computed on 6 rows, too few to have served as a real baseline in the
first place. The extension reproduces pilot-level difficulty at ten times the scale.

## Corrected base-vs-SFT comparison

- **base**: the untrained Qwen2.5-3B-Instruct, scored directly.
- **SFT**: that same model with a LoRA adapter trained on `data_v1_pilot`, scored here on
  `data_v2_pilot`, a different, harder dataset than the one it was trained on.

Both re-judged with gpt-5-mini:

| Model | `abstention_recall` | `abstention_precision` | `over_refusal_rate` | `hallucination_rate` | `partial_match_rate` |
|-------|---------------------:|-------------------------:|----------------------:|------------------------:|------------------------:|
| base | 0.684 | 0.929 | 0.033 | 0.268 | 0.667 |
| SFT | 0.579 | 0.917 | 0.033 | 0.163 | 0.833 |

Training moved every metric, but not uniformly: `abstention_recall` and `hallucination_rate` both
fall by about 0.105, while `partial_match_rate` rises by 0.167. Week 4's gpt-4o-judged numbers showed
a smaller recall drop (0.053) and, more consequentially, the opposite hallucination direction, an
increase rather than a decrease, because gpt-4o's calibration failure systematically penalized SFT's
literal-restatement style more than base's. These corrected numbers, not Week 4's, are what any
future comparison against this checkpoint should cite.

## Limitations and next steps

- **This comparison isn't a conclusion about training.** SFT was trained on `data_v1_pilot` and
  evaluated here on `data_v2_pilot`, a different dataset. Whether the metric differences above come
  from training itself or from evaluating on a distribution the model never trained on can't be
  determined from this data.
- **Not done this week:** DPO (not yet scoped), and retraining SFT on `data_v2` with a proper
  train/held-out split. The latter is what would resolve the confound above and produce a baseline
  worth citing going forward; it's the natural next step.
