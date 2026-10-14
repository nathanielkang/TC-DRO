"""Validate the published table cells stored in results/table_capsule.json.

Checks, for every metric block:
  1. the row and average dictionaries cover exactly the declared method list;
  2. every row has one cell per declared dataset;
  3. the published average equals round(mean of the seven cells, 3);
  4. the best / second-best marks implied by the cells (ties share a mark),
     printed per column so the manuscript table can be checked against them.

Usage:
    python scripts/validate_tables.py [--capsule results/table_capsule.json]
Exit code 0 on success, 1 on any mismatch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate published table cells")
    parser.add_argument(
        "--capsule",
        type=str,
        default=str(ROOT / "results" / "table_capsule.json"),
    )
    parser.add_argument(
        "--show-marks",
        action="store_true",
        help="print the best/second-best mark per column for each block",
    )
    return parser.parse_args()


def column_marks(
    values: dict[str, float], higher_is_better: bool,
) -> dict[str, str]:
    """Best rows get 'bold', next-distinct-value rows get 'underline'."""
    ordered = sorted(set(values.values()), reverse=higher_is_better)
    best = ordered[0]
    second = ordered[1] if len(ordered) > 1 else None
    marks: dict[str, str] = {}
    for method, value in values.items():
        if value == best:
            marks[method] = "bold"
        elif second is not None and value == second:
            marks[method] = "underline"
    return marks


def validate_block(
    name: str,
    block: dict[str, Any],
    methods: list[str],
    datasets: list[str],
    show_marks: bool,
) -> list[str]:
    errors: list[str] = []
    rows = block["rows"]
    avgs = block["avg"]

    if sorted(rows) != sorted(methods):
        errors.append(f"{name}: row methods {sorted(rows)} != {sorted(methods)}")
    if sorted(avgs) != sorted(methods):
        errors.append(f"{name}: avg methods {sorted(avgs)} != {sorted(methods)}")
    if errors:
        return errors

    for method in methods:
        cells = rows[method]
        if len(cells) != len(datasets):
            errors.append(
                f"{name}/{method}: {len(cells)} cells, expected {len(datasets)}"
            )
            continue
        expected = round(fmean(cells), 3)
        published = avgs[method]
        status = "ok" if expected == published else "MISMATCH"
        if status == "MISMATCH":
            errors.append(
                f"{name}/{method}: published avg {published}, "
                f"round(mean, 3) = {expected}"
            )
        print(f"  {name:<13} {method:<12} avg={published:<7} computed={expected:<7} {status}")

    if show_marks and not errors:
        higher = block["direction"] == "higher_is_better"
        columns = list(datasets) + ["avg"]
        for j, column in enumerate(columns):
            if column == "avg":
                values = {m: avgs[m] for m in methods}
            else:
                values = {m: rows[m][j] for m in methods}
            marks = column_marks(values, higher)
            bold = [m for m, k in marks.items() if k == "bold"]
            under = [m for m, k in marks.items() if k == "underline"]
            print(
                f"    {name:<13} {column:<15} bold={','.join(bold)} "
                f"underline={','.join(under)}"
            )

    return errors


def main() -> int:
    args = parse_args()
    capsule_path = Path(args.capsule)
    with open(capsule_path, encoding="utf-8") as f:
        capsule = json.load(f)

    methods = capsule["methods"]
    datasets = capsule["datasets"]
    print(f"Capsule: {capsule_path}")
    print(f"Methods: {len(methods)}  Datasets: {len(datasets)}\n")

    banned = {"calmix", "ranksim"}
    errors: list[str] = []
    for method in methods:
        if method.replace("_", "").replace("-", "").lower() in banned:
            errors.append(f"withdrawn method present in capsule: {method}")

    for name, block in capsule["blocks"].items():
        errors += validate_block(name, block, methods, datasets, args.show_marks)

    print()
    if errors:
        print(f"FAILED with {len(errors)} problem(s):")
        for err in errors:
            print(f"  - {err}")
        return 1

    n_cells = sum(len(b["rows"]) * len(datasets) for b in capsule["blocks"].values())
    print(f"PASSED: {len(capsule['blocks'])} blocks, {n_cells} cells, all averages consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
