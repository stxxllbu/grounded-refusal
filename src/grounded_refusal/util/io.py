"""Shared file helpers. Read/write only — no validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    """Write JSONL atomically so failures cannot leave a partial output file."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(out)


def append_jsonl_row(path: str | Path, row: dict) -> None:
    """Append one line without rewriting the rest of the file, so a crash after this call can't lose earlier rows."""
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def find_row(rows: list[dict], row_id: str) -> dict:
    """Return one row by its ``id`` field."""
    for row in rows:
        if row["id"] == row_id:
            return row
    raise KeyError(f"Row id not found: {row_id}")


def resolve_row_ids(
    rows: list[dict],
    *,
    all_rows: bool,
    ids: list[str] | None,
    default_ids: list[str],
) -> list[str]:
    """Choose all, explicitly requested, or default review row ids."""
    if all_rows:
        return [row["id"] for row in rows]
    if ids is not None:
        return ids
    return list(default_ids)
