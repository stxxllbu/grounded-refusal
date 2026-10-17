"""Tables from token_attribution.py's output.

1. Check: beta times the summed delta of each training response, averaged over
   the 672 pairs, next to the rewards the DPO run logged in its last epoch.
2. Mean delta at the end-of-turn token, at sentence boundaries and at other
   tokens, and the end-of-turn token's share of the summed delta, for chosen,
   rejected (each by answerability) and the held-out reference answers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from grounded_refusal.eval.token_attribution import CHECKPOINTS, DEFAULT_OUTPUT
from grounded_refusal.util.io import read_jsonl

BETA = 0.1  # configs/train/dpo.yaml
KINDS = ("end", "boundary", "other")
GROUPS = [
    ("chosen", "train", "chosen", None),
    ("  answerable", "train", "chosen", "answerable"),
    ("  unanswerable", "train", "chosen", "unanswerable"),
    ("  partial", "train", "chosen", "partial"),
    ("rejected", "train", "rejected", None),
    ("  answerable", "train", "rejected", "answerable"),
    ("  unanswerable", "train", "rejected", "unanswerable"),
    ("  partial", "train", "rejected", "partial"),
    ("held-out reference", "heldout", "reference", None),
]


def deltas(row: dict) -> list[float]:
    return [d - s for d, s in zip(row["dpo_log_probs"], row["sft_log_probs"])]


def logged_last_epoch_rewards(seed: int) -> dict[str, float]:
    """Mean of the rewards trl logged over the last of the 3 epochs."""
    state = json.loads((Path(CHECKPOINTS[seed]["dpo"]) / "trainer_state.json").read_text())
    entries = [e for e in state["log_history"] if "rewards/chosen" in e and e["epoch"] > 2.0]
    return {key: mean(e[f"rewards/{key}"] for e in entries) for key in ("chosen", "rejected")}


def group_stats(rows: list[dict]) -> dict[str, float]:
    """Mean delta per position kind, and the end-of-turn token's share of the summed delta."""
    by_kind: dict[str, list[float]] = {kind: [] for kind in KINDS}
    for row in rows:
        for kind, delta in zip(row["kinds"], deltas(row)):
            by_kind[kind].append(delta)
    total = sum(sum(values) for values in by_kind.values())
    stats = {kind: mean(values) if values else float("nan") for kind, values in by_kind.items()}
    stats["end_share"] = sum(by_kind["end"]) / total
    stats["end_token_share"] = len(by_kind["end"]) / sum(len(values) for values in by_kind.values())
    return stats


def print_check(rows: list[dict], seeds: list[int]) -> None:
    print("Check: beta * summed delta, mean over training responses, vs rewards logged in the last epoch")
    print(f"{'seed':<6}{'chosen':>10}{'logged':>10}{'rejected':>10}{'logged':>10}")
    for seed in seeds:
        logged = logged_last_epoch_rewards(seed)
        measured = {
            response: mean(
                BETA * sum(deltas(r)) for r in rows if r["seed"] == seed and r["source"] == "train" and r["response"] == response
            )
            for response in ("chosen", "rejected")
        }
        print(
            f"{seed:<6}{measured['chosen']:>10.3f}{logged['chosen']:>10.3f}"
            f"{measured['rejected']:>10.3f}{logged['rejected']:>10.3f}"
        )
    print()


def print_tables(rows: list[dict], seeds: list[int]) -> None:
    # Keyed by position in GROUPS: the answerability labels repeat under chosen and rejected.
    stats = {
        (index, seed): group_stats(
            [
                r
                for r in rows
                if r["seed"] == seed and r["source"] == source and r["response"] == response
                and (answerability is None or r["answerability"] == answerability)
            ]
        )
        for index, (_, source, response, answerability) in enumerate(GROUPS)
        for seed in seeds
    }
    titles = {
        "end": "Mean delta at the end-of-turn token",
        "boundary": "Mean delta at sentence boundaries",
        "other": "Mean delta at other tokens",
        "end_share": "End-of-turn token's share of the summed delta",
        "end_token_share": "End-of-turn token's share of the tokens",
    }
    for key, title in titles.items():
        print(title)
        print(f"{'':<22}" + "".join(f"{'seed ' + str(s):>10}" for s in seeds) + f"{'mean':>10}")
        for index, (label, *_) in enumerate(GROUPS):
            values = [stats[(index, seed)][key] for seed in seeds]
            print(f"{label:<22}" + "".join(f"{v:>10.3f}" for v in values) + f"{mean(values):>10.3f}")
        print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tables from the per-token attribution output.")
    parser.add_argument("--input", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    rows = read_jsonl(args.input)
    seeds = sorted({r["seed"] for r in rows})
    print_check(rows, seeds)
    print_tables(rows, seeds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
