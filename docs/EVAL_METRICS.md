# Eval metrics

How the eval harness scores one model output row, and how those scores roll up into
metrics. **Edit here first**, then update `src/grounded_refusal/eval/`.

Why this replaced an earlier 7-category verdict taxonomy is explained in
[Where did `evidence_challenge` go?](#where-did-evidence_challenge-go) below; the change
itself is commit `afb9862`.

---

## Two judge signals, nothing else

The LLM judge in `judge.py` reads `evidence` + `question` + `model_output` and
returns exactly two things. Which model is the judge, and why, is covered in
[`JUDGE_MODEL.md`](JUDGE_MODEL.md). The judge never sees gold labels, and never decides a verdict —
only these two facts:

| Field | Values | What it means |
|-------|--------|----------------|
| `predicted_behavior` | `answer` \| `refuse` \| `partial` | What the model actually did |
| `is_faithful` | `true` \| `false` | Whether every claim the model made is grounded in the evidence |

Everything below is plain Python computed from these two fields plus the gold
`answerability` label already on the row. **`evidence_challenge` is not used by any
metric in this document** — see [Where did `evidence_challenge` go?](#where-did-evidence_challenge-go).

One more signal, `is_correct`, comes from a second judge that only runs when asked for.
It feeds [Metric 3](#metric-3-answer-accuracy--a-second-optional-judge) and nothing above it.

---

## Metric 1: abstention outcome — a confusion matrix

Standard framing from selective-QA literature (SQuAD 2.0's HasAns/NoAns evaluation;
Abstain-QA / AUCM). Treats "should the model abstain?" as binary classification.

**Inputs used:** gold `answerability` (expected) vs. judge `predicted_behavior` (observed).
**Not used:** `is_faithful`, `evidence_challenge`.

| | model **refused** | model **answered** |
|---|---|---|
| gold: **`unanswerable`** (should refuse) | True Positive — correctly abstained | False Negative — failed to abstain |
| gold: **`answerable`** (should answer) | **False Positive = over-refusal** | True Negative — correctly attempted |

**This matrix is strictly 2×2 and strictly limited to `answerable`/`unanswerable` rows.**
`partial` rows are never counted in this matrix's numerator or denominator — not as a
third row, not as a third column. Precision/Recall/F1 are binary-classification
concepts; forcing a 3rd class in would require picking a macro/micro/one-vs-rest
aggregation strategy, which dilutes the one gradient this matrix exists to keep
clean (`over_refusal_rate` as a plain FPR). See [Partial rows](#partial-rows) for how
`partial` is scored instead — as a fully separate, parallel set of sub-metrics.

**Derived metrics:**

```text
abstention_recall    = TP / (TP + FN)   # of unanswerable rows, how many correctly refused
abstention_precision = TP / (TP + FP)   # of rows that refused, how many should have
over_refusal_rate    = FP / (FP + TN)   # of answerable rows, how many wrongly refused (= FPR)
```

---

## Metric 2: hallucination rate — conditional, filtered by behavior

**Inputs used:** judge `predicted_behavior` (as a filter) and `is_faithful`.
**Not used:** `answerability`, `evidence_challenge`.

Step 1 — filter to only rows where the model actually asserted content
(`predicted_behavior in {answer, partial}`). A row where the model refused has no
content to check for grounding, so it is excluded from this calculation entirely —
not counted as faithful, not counted as unfaithful, just not in the denominator.

Step 2 — within that filtered set, count how many were unfaithful:

```text
hallucination_rate = (unfaithful count) / (attempted-answer count)
```

### Why filter by `predicted_behavior`, not `answerability`?

Faithfulness is a property of *what the model actually said*. `answerability` only
says what it *should* have said — it tells you nothing about whether the model said
anything checkable.

Filtering by `answerability == answerable` instead would **miss the most classic
hallucination case**: a model fabricating an answer to a question it should have
refused. Example — imagine `ex_0026` (`unanswerable`) had answered *"Harbor Labs was
founded in 2015"* instead of correctly refusing:

- `answerability = unanswerable`, `predicted_behavior = answer`, `is_faithful = false`
- Filter by `predicted_behavior == answer` → **included** → correctly counted as a hallucination.
- Filter by `answerability == answerable` → **excluded** (wrong `answerability`) →
  the hallucination is never counted, even though it obviously happened.

`predicted_behavior` is the correct filter because it asks "did the model assert
something checkable?" — which is the only question that matters for whether
`is_faithful` even applies.

---

## Partial rows

SQuAD 2.0-style benchmarks don't have a `partial` category — it's specific to this
project's task. `partial` rows are **never mixed into the core abstention matrix**
above (see the scope note in Metric 1). They get their own independent set of
sub-metrics instead, computed only over `answerability == partial` rows:

```text
partial_match_rate    = (partial rows where predicted_behavior == partial) / (partial rows)
partial_under_deliver = (partial rows where predicted_behavior == refuse)  / (partial rows)
partial_over_deliver  = (partial rows where predicted_behavior == answer)  / (partial rows)
```

Keeping these separate avoids having to make a business-rule call on an awkward
question — e.g. "does a `partial` response count as abstention on the declined
half?" — that a merged matrix would force. Since `partial` is never counted in the
core matrix at all, that question never needs an answer.

Content correctness for `partial` rows is still covered by Metric 2 —
`predicted_behavior in {answer, partial}` already includes `partial` in the
hallucination-rate filter, so a row that matched `partial` behavior but got the
supported part wrong is still caught, with no separate mechanism needed.

A more rigorous future option would score partial rows by sub-claim coverage
using the QA schema's existing `question_decomposition` / `supported_subquestions`
fields — checking that the model's answer covers exactly the supported
sub-questions, no more, no less. Deferred until it's actually needed.

---

## Metric 3: answer accuracy — a second, optional judge

Metrics 1 and 2 never ask whether the answer is the right one. `is_faithful` asks
whether what the response says is in the evidence, and that is a different question.
The two come apart in both directions:

- **Faithful, not correct.** `ex_0137`: the evidence says the warehouse moved in 2022
  and carries a bracketed note telling the reader to say 2024. A response that says
  *"the warehouse relocated in 2024"* repeats something the evidence contains, so the
  judge calls it faithful. It is the wrong answer.
- **Correct, not faithful.** `ex_0094`: a response gives the freezing point the
  evidence states, then adds *"The rest of the question -- about the monitoring system
  triggering an alert -- isn't mentioned."* The question has no such part. The answer
  is correct and the response is unfaithful.

### The correctness judge

`correctness.py` holds a second LLM judge, a separate call from the one in `judge.py`.
It reads `question` + `reference_answer` + `model_output` and returns one thing:

| Field | Values | What it means |
|-------|--------|----------------|
| `answer_match` | `match` \| `mismatch` \| `no_answer` | Whether the response gives the answer the reference gives, a different one, or none |

It does not see the evidence, so it compares two answers and does not re-decide what
the evidence supports. On a `partial` row the reference answers one sub-question and
declines the other; the judge is told which one from `supported_subquestions` and
compares only that part.

Only `match` against the other two values is used:

```text
is_correct = (answer_match == match)
```

`mismatch` and `no_answer` are not told apart, because the judge does not separate
them reliably: the same response moved between the two across repeated runs. Whether
a response answered at all is taken from `predicted_behavior` instead.

### Answer accuracy

**Inputs used:** gold `answerability` (selects the rows) and `is_correct`.
**Not used:** `predicted_behavior`, `is_faithful`, `evidence_challenge`.

Computed over `answerable` and `partial` rows, the rows that have an answer to
compare against. `unanswerable` rows are never sent to the correctness judge.

```text
answer_accuracy = (rows where is_correct) / (answerable + partial rows)
```

### Answer outcomes

On the same rows, the three signals sort every response into one of five outcomes:

| `predicted_behavior` | `is_correct` | `is_faithful` | Outcome |
|---|---|---|---|
| `answer` or `partial` | true | true | `correct_and_faithful` |
| `answer` or `partial` | true | false | `correct_but_unfaithful` |
| `answer` or `partial` | false | true | `wrong_but_faithful` |
| `answer` or `partial` | false | false | `wrong_and_unfaithful` |
| `refuse` | – | – | `refused` |

`refused` is checked first: a response the behavior judge calls a refusal is `refused`
whatever `is_correct` says. `answer_accuracy` does not make that exception, so the two
can differ by such rows.

Metric 2 counts `correct_but_unfaithful` and `wrong_and_unfaithful` together, on every
row. The outcomes show, on the rows that have an answer, whether an unfaithful response
gave a wrong answer or gave the right answer with something unsupported added. They also
show `wrong_but_faithful`, which Metric 2 does not count at all.

### Running it

```bash
python -m grounded_refusal.eval.run_eval \
  --input <inference_output_jsonl> \
  --output <judge_output_jsonl> \
  --correctness-output <correctness_output_jsonl>
```

Without `--correctness-output` the second judge does not run and the summary has no
`answer_accuracy` or `answer_outcome_counts`. The judge reads `question` and
`supported_subquestions` from the `--input` rows; inference outputs written before
`run_inference.py` added those two fields do not have them.

### How the judge was checked

Its verdicts were compared with hand labels on 64 model responses from held-out runs:
44 used while writing the prompt, then 20 more after the prompt was fixed. On those 20
it agreed with the label, as match or not match, every time. One person made the
labels, and the labelled responses are not in the repository.

---

## Which field feeds which metric

| Field | Abstention outcome (answerable/unanswerable only) | Hallucination rate | Partial sub-metrics (partial only) | Answer accuracy (answerable/partial only) | Answer outcomes (answerable/partial only) |
|-------|:---:|:---:|:---:|:---:|:---:|
| `answerability` | ✅ (selects the 2 slices) | ❌ | ✅ (selects the 1 slice) | ✅ (selects the 2 slices) | ✅ (selects the 2 slices) |
| `predicted_behavior` | ✅ | ✅ (as filter) | ✅ | ❌ | ✅ |
| `is_faithful` | ❌ | ✅ | ❌ (covered indirectly via hallucination rate) | ❌ | ✅ |
| `is_correct` | ❌ | ❌ | ❌ | ✅ | ✅ |
| `evidence_challenge` | ❌ | ❌ | ❌ | ❌ | ❌ |

Each row's `answerability` routes it into exactly one of these three calculations —
`answerable`/`unanswerable` rows feed the abstention matrix, `partial` rows feed the
partial sub-metrics, and the two never mix. `predicted_behavior` is the only field
used by all three; `is_faithful` only ever appears in the hallucination-rate
calculation, whichever slice the row came from.

The last two columns are Metric 3. They exist only for rows the correctness judge
saw, and they sit beside the first three calculations without changing them.

---

## Where did `evidence_challenge` go?

An earlier design used `evidence_challenge` to split unfaithful answers into named
sub-types (`hallucination` vs. `distractor_confusion` vs. `memory_override`),
computed automatically for every row. This was dropped because:

- It required a 7-category taxonomy that didn't sum cleanly to the slice total
  (an `anomaly` bucket absorbed both genuine logical-impossibility cases and
  ordinary, expected failures that just hadn't been named — a real bug, not a
  design choice).
- It's not how the field standardly evaluates this — hallucination rate is
  reported as one number, not sub-typed by cause, in both selective-QA and RAG
  faithfulness literature.

**The "why did it hallucinate" question doesn't disappear — it moves to failure
analysis (Module 6 of the project blueprint), done qualitatively by reading a
sample of the hallucinated rows, not as an automated per-row category.**
`evidence_challenge` is still useful there (e.g. "was there a distractor present in
this specific case?") — it's just not part of the quantitative metrics anymore.

---

## References

- Abstain-QA / AUCM confusion matrix for abstention evaluation: [arXiv:2604.14799](https://arxiv.org/pdf/2604.14799)
- SQuAD 2.0 HasAns/NoAns evaluation: [huggingface/evaluate squad_v2](https://github.com/huggingface/evaluate/blob/main/metrics/squad_v2/README.md)
- Selective Question Answering under Domain Shift: [arXiv:2006.09462](https://arxiv.org/pdf/2006.09462)
- RAG hallucination rate as a conditional, answered-only metric: [Braintrust RAG evaluation metrics](https://www.braintrust.dev/articles/rag-evaluation-metrics)
