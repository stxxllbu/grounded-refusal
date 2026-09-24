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
