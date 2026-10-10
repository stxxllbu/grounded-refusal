"""Regression test for run_eval.py's --resume with a narrower --ids than the
run that produced --output (GitHub issue #5). Uses --dry-run, so no API calls.
"""

from __future__ import annotations

import json

from grounded_refusal.eval.run_eval import main
from grounded_refusal.util.io import read_jsonl, write_jsonl


def test_resume_with_narrower_ids_scores_only_selected_rows(tmp_path, capsys):
    input_path = tmp_path / "inference.jsonl"
    output_path = tmp_path / "eval.jsonl"
    write_jsonl(
        input_path,
        [
            {"id": "ex_0001", "prompt": "p", "model_output": "o", "answerability": "answerable"},
            {"id": "ex_0002", "prompt": "p", "model_output": "o", "answerability": "unanswerable"},
        ],
    )

    assert main(["--input", str(input_path), "--output", str(output_path), "--dry-run"]) == 0
    capsys.readouterr()

    exit_code = main(
        ["--input", str(input_path), "--output", str(output_path), "--dry-run", "--resume", "--ids", "ex_0002"]
    )

    assert exit_code == 0
    # The dry-run judge always says refuse; on the unanswerable row alone that is a true positive,
    # with no answerable row left to count as over-refusal.
    assert json.loads(capsys.readouterr().out) == {"abstention_recall": 1.0, "abstention_precision": 1.0}
    assert [r["id"] for r in read_jsonl(output_path)] == ["ex_0001", "ex_0002"]


def test_correctness_output_judges_only_rows_with_a_reference_answer(tmp_path, capsys):
    input_path = tmp_path / "inference.jsonl"
    output_path = tmp_path / "eval.jsonl"
    correctness_path = tmp_path / "correctness.jsonl"
    labels = {"ex_0001": "answerable", "ex_0002": "unanswerable", "ex_0003": "partial"}
    write_jsonl(
        input_path,
        [
            {
                "id": row_id,
                "prompt": "p",
                "question": "q",
                "reference_answer": "r",
                "model_output": "o",
                "answerability": answerability,
            }
            for row_id, answerability in labels.items()
        ],
    )

    exit_code = main(
        [
            "--input", str(input_path),
            "--output", str(output_path),
            "--correctness-output", str(correctness_path),
            "--dry-run",
        ]
    )

    assert exit_code == 0
    assert [r["id"] for r in read_jsonl(correctness_path)] == ["ex_0001", "ex_0003"]
    summary = json.loads(capsys.readouterr().out)
    # The dry-run correctness judge always says no_answer, and the dry-run judge always says refuse.
    assert summary["answer_accuracy"] == 0.0
    assert summary["answer_outcome_counts"]["refused"] == 2


def test_correctness_output_rejects_input_rows_without_a_question(tmp_path, capsys):
    input_path = tmp_path / "inference.jsonl"
    write_jsonl(
        input_path,
        [{"id": "ex_0001", "prompt": "p", "reference_answer": "r", "model_output": "o", "answerability": "answerable"}],
    )

    exit_code = main(
        ["--input", str(input_path), "--correctness-output", str(tmp_path / "correctness.jsonl"), "--dry-run"]
    )

    assert exit_code == 1
    assert "question" in capsys.readouterr().err
