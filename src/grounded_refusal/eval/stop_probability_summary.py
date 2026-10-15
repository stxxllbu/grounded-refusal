"""Print the stop-probability tables from the rows stop_probability.py wrote.

Per-seed tables (one line per model, one column per seed, then a column across the seeds):
    mean stop probability, rows with stop probability below 0.5,
    mean log-probability per token of the reference answer.

Then two tables over all 240 rows of a model (4 seeds x 60 rows):
    each model against the SFT model of the same seed, row by row;
    the full-data DPO model on the rows where its own first sentence is the one it was given.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean

from grounded_refusal.eval.stop_probability import DEFAULT_OUTPUT, first_sentence
from grounded_refusal.util.io import read_jsonl

# seed -> held-out responses of that seed's full-data DPO model
DPO_RESPONSES = {
    42: "outputs/inference-qwen2.5-3b-instruct-dpo/20261008_223937_dpo_v3b_heldout.jsonl",
    44: "outputs/inference-qwen2.5-3b-instruct-dpo/20261009_171742_dpo_v3b_heldout.jsonl",
    45: "outputs/inference-qwen2.5-3b-instruct-dpo/20261009_181642_dpo_v3b_heldout.jsonl",
    46: "outputs/inference-qwen2.5-3b-instruct-dpo/20261009_191404_dpo_v3b_heldout.jsonl",
}


def mean_stop_prob(rows: list[dict]) -> float:
    return mean(row["stop_prob"] for row in rows)


def count_below_half(rows: list[dict]) -> int:
    return sum(row["stop_prob"] < 0.5 for row in rows)


def reference_log_prob_per_token(rows: list[dict]) -> float:
    return sum(row["reference_log_prob"] for row in rows) / sum(row["reference_num_tokens"] for row in rows)


def print_table(title: str, groups: dict, statistic, *, fmt: str, across_seeds) -> None:
    """`groups` maps (model, seed) to its rows. `across_seeds` is `mean` or `sum`, shown as the last column."""
    models = list(dict.fromkeys(model for model, _ in groups))
    seeds = list(dict.fromkeys(seed for _, seed in groups))

    print(title)
    print(f"{'':32}" + "".join(f"{f'seed {seed}':>10}" for seed in seeds) + f"{across_seeds.__name__:>10}")
    for model in models:
        values = [statistic(groups[model, seed]) for seed in seeds]
        values.append(across_seeds(values))
        print(f"{model:32}" + "".join(f"{value:>10{fmt}}" for value in values))
    print()


def print_comparison_with_sft(groups: dict) -> None:
    """For each model: on how many rows its stop probability differs from the same-seed SFT model's, and by how much."""
    sft_stop_prob = {(seed, row["id"]): row["stop_prob"] for (model, seed), rows in groups.items() if model == "sft" for row in rows}
    models = list(dict.fromkeys(model for model, _ in groups if model != "sft"))

    print("Stop probability against the same-seed SFT model (rows, of 240)")
    print(f"{'':32}{'lower':>10}{'lower by >0.5':>16}{'within 0.05':>14}")
    for model in models:
        differences = [
            row["stop_prob"] - sft_stop_prob[seed, row["id"]]
            for (group_model, seed), rows in groups.items()
            if group_model == model
            for row in rows
        ]
        lower = sum(d < 0 for d in differences)
        much_lower = sum(d < -0.5 for d in differences)
        close = sum(abs(d) < 0.05 for d in differences)
        print(f"{model:32}{lower:>10}{much_lower:>16}{close:>14}")
    print()


def print_split_by_own_first_sentence(groups: dict) -> None:
    """Full-data DPO model: mean stop probability on the rows where its own first sentence is the one it was given."""
    same: list[tuple[float, float]] = []  # (SFT stop probability, DPO stop probability)
    different: list[tuple[float, float]] = []
    for seed, responses_path in DPO_RESPONSES.items():
        own_first_sentence = {row["id"]: first_sentence(row["model_output"]) for row in read_jsonl(responses_path)}
        sft_stop_prob = {row["id"]: row["stop_prob"] for row in groups["sft", seed]}
        for row in groups["dpo", seed]:
            pair = (sft_stop_prob[row["id"]], row["stop_prob"])
            (same if own_first_sentence[row["id"]] == row["first_sentence"] else different).append(pair)

    print("Mean stop probability, by whether the DPO model's own first sentence is the one it was given")
    print(f"{'':32}{'rows':>10}{'sft':>10}{'dpo':>10}")
    for label, pairs in (("identical", same), ("different", different), ("all", same + different)):
        print(f"{label:32}{len(pairs):>10}{mean(sft for sft, _ in pairs):>10.2f}{mean(dpo for _, dpo in pairs):>10.2f}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print the stop-probability tables.")
    parser.add_argument("--input", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    groups = defaultdict(list)
    for row in read_jsonl(args.input):
        groups[row["arm"], row["seed"]].append(row)

    print_table("Mean stop probability after the first sentence", groups, mean_stop_prob, fmt=".3f", across_seeds=mean)
    print_table("Rows (of 60) with stop probability below 0.5", groups, count_below_half, fmt="d", across_seeds=sum)
    print_table("Mean log-probability per reference-answer token", groups, reference_log_prob_per_token, fmt=".3f", across_seeds=mean)
    print_comparison_with_sft(groups)
    print_split_by_own_first_sentence(groups)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
