"""Regression tests for eval.verdict's two independent outcome functions.

Fixtures are the worked examples from PREFERENCE_GENERATION_PROTOCOL.md
(pref_0021/0042/0026/0051/0031): each protocol example gives a chosen
(correct) and rejected (deliberately wrong) response for one negative_type
slice. We reuse both halves here as fixed input/output pairs. Pure
functions, no LLM calls. Design: docs/EVAL_METRICS.md.
"""

from __future__ import annotations

import pytest

from grounded_refusal.data.schema_qa import Answerability
from grounded_refusal.eval.schema_eval import ModelBehavior
from grounded_refusal.eval.verdict import derive_abstention_outcome, derive_partial_outcome

ANSWERABLE = Answerability.ANSWERABLE
UNANSWERABLE = Answerability.UNANSWERABLE
PARTIAL = Answerability.PARTIAL
ANSWER = ModelBehavior.ANSWER
REFUSE = ModelBehavior.REFUSE
PART = ModelBehavior.PARTIAL

# --- derive_abstention_outcome: answerability x predicted_behavior -> TP/FP/TN/FN or None ---
ABSTENTION_CASES = [
    # pref_0021: answerable + [] -- chosen=answer (correct), rejected=refuse (over_refusal)
    pytest.param(ANSWERABLE, ANSWER, "true_negative", id="pref_0021_chosen"),
    pytest.param(ANSWERABLE, REFUSE, "false_positive", id="pref_0021_rejected"),
    # pref_0042: answerable + known_world_conflict -- both chosen and rejected ANSWER
    # (only is_faithful differs, which abstention_outcome never looks at)
    pytest.param(ANSWERABLE, ANSWER, "true_negative", id="pref_0042_chosen"),
    pytest.param(ANSWERABLE, ANSWER, "true_negative", id="pref_0042_rejected"),
    # pref_0026: unanswerable + [] -- chosen=refuse (correct), rejected=answer (hallucination)
    pytest.param(UNANSWERABLE, REFUSE, "true_positive", id="pref_0026_chosen"),
    pytest.param(UNANSWERABLE, ANSWER, "false_negative", id="pref_0026_rejected"),
    # pref_0051: unanswerable + distractor_entity -- chosen=refuse, rejected=answer
    pytest.param(UNANSWERABLE, REFUSE, "true_positive", id="pref_0051_chosen"),
    pytest.param(UNANSWERABLE, ANSWER, "false_negative", id="pref_0051_rejected"),
    # partial rows never enter this matrix, regardless of behavior
    pytest.param(PARTIAL, PART, None, id="partial_excluded_match"),
    pytest.param(PARTIAL, ANSWER, None, id="partial_excluded_answer"),
    pytest.param(PARTIAL, REFUSE, None, id="partial_excluded_refuse"),
    # PARTIAL behavior on a non-partial row: treated as "not refused"
    pytest.param(ANSWERABLE, PART, "true_negative", id="answerable_partial_behavior"),
    pytest.param(UNANSWERABLE, PART, "false_negative", id="unanswerable_partial_behavior"),
]


@pytest.mark.parametrize("answerability,predicted_behavior,expected", ABSTENTION_CASES)
def test_derive_abstention_outcome(answerability, predicted_behavior, expected):
    assert derive_abstention_outcome(answerability, predicted_behavior) == expected


# --- derive_partial_outcome: answerability x predicted_behavior -> match/under/over or None ---
PARTIAL_CASES = [
    # pref_0031: partial + partial_evidence -- chosen=partial (match), rejected=answer (over_deliver)
    pytest.param(PARTIAL, PART, "match", id="pref_0031_chosen"),
    pytest.param(PARTIAL, ANSWER, "over_deliver", id="pref_0031_rejected"),
    pytest.param(PARTIAL, REFUSE, "under_deliver", id="partial_refused_whole"),
    # non-partial rows never enter this calculation, regardless of behavior
    pytest.param(ANSWERABLE, ANSWER, None, id="answerable_excluded"),
    pytest.param(ANSWERABLE, REFUSE, None, id="answerable_excluded_refuse"),
    pytest.param(UNANSWERABLE, REFUSE, None, id="unanswerable_excluded"),
    pytest.param(UNANSWERABLE, ANSWER, None, id="unanswerable_excluded_answer"),
]


@pytest.mark.parametrize("answerability,predicted_behavior,expected", PARTIAL_CASES)
def test_derive_partial_outcome(answerability, predicted_behavior, expected):
    assert derive_partial_outcome(answerability, predicted_behavior) == expected
