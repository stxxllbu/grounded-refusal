"""Regression tests for eval.verdict.derive_verdict.

Fixtures are the worked examples from PREFERENCE_GENERATION_PROTOCOL.md
(pref_0021/0042/0026/0051/0031): each protocol example gives a chosen
(correct) and rejected (deliberately wrong) response for one negative_type
slice. We reuse both halves here as fixed input/output pairs for the
deterministic verdict mapping. Pure function, no LLM calls.
"""

from __future__ import annotations

import pytest

from grounded_refusal.data.schema_qa import Answerability, EvidenceChallengeTag
from grounded_refusal.eval.schema_eval import ModelBehavior
from grounded_refusal.eval.verdict import derive_verdict

ANSWERABLE = Answerability.ANSWERABLE
UNANSWERABLE = Answerability.UNANSWERABLE
PARTIAL = Answerability.PARTIAL
ANSWER = ModelBehavior.ANSWER
REFUSE = ModelBehavior.REFUSE
PART = ModelBehavior.PARTIAL

WORLD_CONFLICT = [EvidenceChallengeTag.KNOWN_WORLD_CONFLICT]
DISTRACTOR = [EvidenceChallengeTag.DISTRACTOR_ENTITY]
PARTIAL_EVIDENCE = [EvidenceChallengeTag.PARTIAL_EVIDENCE]

# (answerability, evidence_challenge, predicted_behavior, is_faithful) -> expected verdict
CASES = [
    # --- rejected halves: pref_0021 / pref_0042 / pref_0026 / pref_0051 / pref_0031 ---
    pytest.param(ANSWERABLE, [], REFUSE, True, "over_refusal", id="pref_0021_rejected"),
    pytest.param(ANSWERABLE, WORLD_CONFLICT, ANSWER, False, "memory_override", id="pref_0042_rejected"),
    pytest.param(UNANSWERABLE, [], ANSWER, False, "hallucination", id="pref_0026_rejected"),
    pytest.param(UNANSWERABLE, DISTRACTOR, ANSWER, False, "distractor_confusion", id="pref_0051_rejected"),
    pytest.param(PARTIAL, PARTIAL_EVIDENCE, ANSWER, False, "over_complete", id="pref_0031_rejected"),
    # --- chosen halves: same five rows, correct behavior ---
    pytest.param(ANSWERABLE, [], ANSWER, True, "correct", id="pref_0021_chosen"),
    pytest.param(ANSWERABLE, WORLD_CONFLICT, ANSWER, True, "correct", id="pref_0042_chosen"),
    pytest.param(UNANSWERABLE, [], REFUSE, True, "correct", id="pref_0026_chosen"),
    pytest.param(UNANSWERABLE, DISTRACTOR, REFUSE, True, "correct", id="pref_0051_chosen"),
    pytest.param(PARTIAL, PARTIAL_EVIDENCE, PART, True, "correct", id="pref_0031_chosen"),
    # --- protocol FAQ's noted extensions ---
    pytest.param(ANSWERABLE, WORLD_CONFLICT, REFUSE, True, "over_refusal", id="world_conflict_refused"),
    pytest.param(PARTIAL, PARTIAL_EVIDENCE, REFUSE, True, "over_refusal", id="partial_whole_refused"),
    # --- combinations outside the pilot map: must not be forced into a category ---
    pytest.param(UNANSWERABLE, [], PART, True, "anomaly", id="unanswerable_partial_behavior"),
    pytest.param(ANSWERABLE, [], ANSWER, False, "anomaly", id="answerable_unfaithful_no_conflict_tag"),
]


@pytest.mark.parametrize(
    "answerability,evidence_challenge,predicted_behavior,is_faithful,expected", CASES
)
def test_derive_verdict(answerability, evidence_challenge, predicted_behavior, is_faithful, expected):
    assert (
        derive_verdict(answerability, evidence_challenge, predicted_behavior, is_faithful)
        == expected
    )
