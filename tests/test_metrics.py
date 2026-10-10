"""Tests for the answer-correctness block of eval.metrics: the five answer
outcomes and answer_accuracy. Pure functions, no LLM calls.
"""

from __future__ import annotations

import pytest

from grounded_refusal.data.schema_qa import Answerability
from grounded_refusal.eval.metrics import aggregate, answer_outcome
from grounded_refusal.eval.schema_eval import EvalResult, ModelBehavior

ANSWER = ModelBehavior.ANSWER
REFUSE = ModelBehavior.REFUSE
PART = ModelBehavior.PARTIAL

# --- answer_outcome: predicted_behavior x is_faithful x is_correct -> one of five outcomes ---
ANSWER_OUTCOME_CASES = [
    pytest.param(ANSWER, True, True, "correct_and_faithful", id="answer_correct_faithful"),
    pytest.param(ANSWER, False, True, "correct_but_unfaithful", id="answer_correct_unfaithful"),
    pytest.param(ANSWER, True, False, "wrong_but_faithful", id="answer_wrong_faithful"),
    pytest.param(ANSWER, False, False, "wrong_and_unfaithful", id="answer_wrong_unfaithful"),
    # PARTIAL behavior is an attempted answer, sorted the same way as ANSWER
    pytest.param(PART, True, True, "correct_and_faithful", id="partial_correct_faithful"),
    pytest.param(PART, True, False, "wrong_but_faithful", id="partial_wrong_faithful"),
    # a refusal is "refused" whatever the other two signals say
    pytest.param(REFUSE, True, False, "refused", id="refuse_faithful"),
    pytest.param(REFUSE, False, False, "refused", id="refuse_unfaithful"),
    pytest.param(REFUSE, True, True, "refused", id="refuse_but_judged_correct"),
]


@pytest.mark.parametrize("predicted_behavior,is_faithful,is_correct,expected", ANSWER_OUTCOME_CASES)
def test_answer_outcome(predicted_behavior, is_faithful, is_correct, expected):
    assert answer_outcome(predicted_behavior, is_faithful, is_correct) == expected


def _result(row_id, predicted_behavior, is_faithful, is_correct):
    return EvalResult(
        id=row_id,
        answerability=Answerability.ANSWERABLE,
        predicted_behavior=predicted_behavior,
        is_faithful=is_faithful,
        rationale="",
        is_correct=is_correct,
    )


def test_aggregate_answer_accuracy_counts_only_rows_with_is_correct():
    results = [
        _result("ex_0001", ANSWER, True, True),
        _result("ex_0002", ANSWER, False, True),
        _result("ex_0003", ANSWER, True, False),
        _result("ex_0004", REFUSE, True, False),
        _result("ex_0005", REFUSE, True, None),  # not given to the correctness judge
    ]

    summary = aggregate(results)

    assert summary["answer_accuracy"] == 2 / 4
    assert summary["answer_outcome_counts"] == {
        "correct_and_faithful": 1,
        "correct_but_unfaithful": 1,
        "wrong_but_faithful": 1,
        "wrong_and_unfaithful": 0,
        "refused": 1,
    }


def test_aggregate_has_no_answer_keys_without_is_correct():
    summary = aggregate([_result("ex_0001", ANSWER, True, None)])

    assert "answer_accuracy" not in summary
    assert "answer_outcome_counts" not in summary
