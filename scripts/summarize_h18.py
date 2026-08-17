from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path


METRICS = (
    "region_accuracy",
    "region_balanced_accuracy",
    "region_macro_f1",
    "province_accuracy",
    "province_balanced_accuracy",
    "province_macro_f1",
    "prediction_region_consistency",
    "province_cross_region_error_rate",
)


def metric_row(path: Path) -> dict:
    metrics = json.loads(path.read_text(encoding="utf-8"))
    experiment = path.parent.name
    match = re.search(r"_seed(\d+)$", experiment)
    if match is None:
        raise ValueError(f"Cannot recover seed from experiment name: {experiment}")
    hierarchy = metrics["hierarchy"]
    return {
        "experiment": experiment,
        "seed": int(match.group(1)),
        "region_accuracy": metrics["region"]["accuracy"],
        "region_balanced_accuracy": metrics["region"]["balanced_accuracy"],
        "region_macro_f1": metrics["region"]["classification_report"]["macro avg"][
            "f1-score"
        ],
        "province_accuracy": metrics["province"]["accuracy"],
        "province_balanced_accuracy": metrics["province"]["balanced_accuracy"],
        "province_macro_f1": metrics["province"]["classification_report"][
            "macro avg"
        ]["f1-score"],
        "prediction_region_consistency": hierarchy[
            "prediction_region_consistency"
        ],
        "province_cross_region_error_rate": hierarchy[
            "province_cross_region_error_rate"
        ],
        "province_head": metrics["representation"]["province_head"],
        "metrics_file": str(path),
    }


def aggregate(rows: list[dict]) -> dict:
    result = {"experiment": "h18_soft_hierarchy", "runs": len(rows)}
    for metric in METRICS:
        values = [float(row[metric]) for row in rows]
        result[f"{metric}_mean"] = statistics.mean(values)
        result[f"{metric}_std"] = (
            statistics.stdev(values) if len(values) > 1 else 0.0
        )
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize H18 hierarchy runs.")
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--split", choices=["valid", "test"], default="valid")
    parser.add_argument("--destination", default="results_archive/h18")
    args = parser.parse_args()

    filename = f"metrics_{args.split}_best_province_accuracy.json"
    paths = sorted(Path(args.outputs).glob(f"h18_soft_hierarchy_seed*/{filename}"))
    if not paths:
        raise FileNotFoundError(f"No H18 metrics found for split {args.split}")
    rows = sorted((metric_row(path) for path in paths), key=lambda row: row["seed"])
    aggregate_row = aggregate(rows)
    destination = Path(args.destination)
    per_seed_path = destination / f"h18_{args.split}_per_seed.csv"
    aggregate_path = destination / f"h18_{args.split}_aggregate.csv"
    write_csv(per_seed_path, rows)
    write_csv(aggregate_path, [aggregate_row])

    print(f"Wrote {len(rows)} rows to {per_seed_path}")
    print(f"Wrote aggregate to {aggregate_path}")
    print(json.dumps(aggregate_row, indent=2))


if __name__ == "__main__":
    main()

