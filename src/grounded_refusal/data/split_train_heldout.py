"""Split a QA JSONL into train / held-out sets.

Pilot rows (present in --pilot-ids-file) are pinned entirely to held-out --
they've already been read and judged repeatedly for judge validation
(docs/JUDGE_MODEL.md), so training on them would contaminate that history.
The remaining rows are stratified by (answerability, evidence_challenge) and
sampled deterministically so the overall held-out share matches
--heldout-frac.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

from grounded_refusal.data.validate_qa_jsonl_against_schema import validate_qa_jsonl_against_schema
from grounded_refusal.util.io import read_jsonl, write_jsonl


def stratum_key(row: dict) -> tuple[str, tuple[str, ...]]:
    return (row["answerability"], tuple(sorted(row.get("evidence_challenge", []))))


def apportion(counts: dict[tuple, int], total: int) -> dict[tuple, int]:
    """Largest-remainder apportionment of ``total`` items across strata sized by ``counts``."""
    grand_total = sum(counts.values())
    exact = {key: n * total / grand_total for key, n in counts.items()}
    floors = {key: int(v) for key, v in exact.items()}
    remainder = total - sum(floors.values())
    ranked = sorted(counts, key=lambda k: (-(exact[k] - floors[k]), k))
    for key in ranked[:remainder]:
        floors[key] += 1
    return floors


def split_rows(
    rows: list[dict],
    *,
    pilot_ids: set[str],
    heldout_frac: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    pilot_rows = [r for r in rows if r["id"] in pilot_ids]
    remaining_rows = [r for r in rows if r["id"] not in pilot_ids]

    heldout_target = round(len(rows) * heldout_frac)
    extra_needed = max(0, heldout_target - len(pilot_rows))

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in remaining_rows:
        grouped[stratum_key(row)].append(row)

    stratum_sizes = {key: len(group) for key, group in grouped.items()}
    heldout_counts = apportion(stratum_sizes, extra_needed)

    rng = random.Random(seed)
    train_rows: list[dict] = []
    heldout_rows: list[dict] = list(pilot_rows)
    for key in sorted(grouped):
        group = list(grouped[key])
        rng.shuffle(group)
        n_heldout = heldout_counts[key]
        heldout_rows.extend(group[:n_heldout])
        train_rows.extend(group[n_heldout:])

    return train_rows, heldout_rows


def print_stratum_summary(train_rows: list[dict], heldout_rows: list[dict]) -> None:
    all_keys = sorted({stratum_key(r) for r in train_rows + heldout_rows})
    train_counts = defaultdict(int)
    heldout_counts = defaultdict(int)
    for r in train_rows:
        train_counts[stratum_key(r)] += 1
    for r in heldout_rows:
        heldout_counts[stratum_key(r)] += 1

    print(f"{'stratum':55s} {'train':>6s} {'heldout':>8s}", file=sys.stderr)
    for key in all_keys:
        print(f"{str(key):55s} {train_counts[key]:6d} {heldout_counts[key]:8d}", file=sys.stderr)
    print(
        f"{'TOTAL':55s} {len(train_rows):6d} {len(heldout_rows):8d}",
        file=sys.stderr,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Split a QA JSONL into stratified train / held-out sets, pinning pilot rows to held-out."
    )
    parser.add_argument("--input", type=Path, default=Path("data/data_v2.jsonl"))
    parser.add_argument(
        "--pilot-ids-file",
        type=Path,
        default=Path("data/data_v2_pilot.jsonl"),
        help="Rows whose ids appear here are pinned entirely to held-out",
    )
    parser.add_argument("--heldout-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-output", type=Path, default=Path("data/data_v2_train.jsonl"))
    parser.add_argument("--heldout-output", type=Path, default=Path("data/data_v2_heldout.jsonl"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    for output in (args.train_output, args.heldout_output):
        if output.exists() and not args.overwrite:
            print(f"Output already exists: {output}. Pass --overwrite to replace it.", file=sys.stderr)
            return 1

    rows = read_jsonl(args.input)
    pilot_ids = {r["id"] for r in read_jsonl(args.pilot_ids_file)}

    train_rows, heldout_rows = split_rows(
        rows, pilot_ids=pilot_ids, heldout_frac=args.heldout_frac, seed=args.seed
    )
    print_stratum_summary(train_rows, heldout_rows)

    write_jsonl(args.train_output, train_rows)
    write_jsonl(args.heldout_output, heldout_rows)
    print(f"Wrote {len(train_rows)} rows to {args.train_output}", file=sys.stderr)
    print(f"Wrote {len(heldout_rows)} rows to {args.heldout_output}", file=sys.stderr)

    for path in (args.train_output, args.heldout_output):
        _, errors = validate_qa_jsonl_against_schema(path)
        if errors:
            print(f"Schema validation FAILED for {path}:", file=sys.stderr)
            for err in errors:
                print(f"  - {err}", file=sys.stderr)
            return 1
        print(f"OK: {path} matches QAExample schema", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
