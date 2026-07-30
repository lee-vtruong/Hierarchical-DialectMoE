from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


EXPERIMENT_PATTERN = re.compile(
    r"^h11_(base|large)_vi_(acoustic|prosody)_seed(\d+)$"
)
METRIC_PATTERN = re.compile(
    r"^metrics_test_best_(province|region)_accuracy\.json$"
)


def parse_artifact(path: Path) -> tuple[str, str, int, str]:
    experiment = EXPERIMENT_PATTERN.match(path.parent.name)
    metric = METRIC_PATTERN.match(path.name)
    if not experiment or not metric:
        raise ValueError(f"Not an H11 metric artifact: {path}")
    backbone, variant, seed = experiment.groups()
    return backbone, variant, int(seed), metric.group(1)


def macro_f1(metrics: dict, task: str) -> float:
    return float(metrics[task]["classification_report"]["macro avg"]["f1-score"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize H11 checkpoint matrix.")
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--destination", default="outputs/h11_summary.csv")
    parser.add_argument(
        "--aggregate-destination", default="outputs/h11_aggregate.csv"
    )
    args = parser.parse_args()

    rows = []
    for path in sorted(
        Path(args.outputs).glob("h11_*_vi_*_seed*/metrics_test_best_*_accuracy.json")
    ):
        try:
            backbone, variant, seed, checkpoint = parse_artifact(path)
        except ValueError:
            continue
        with path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        rows.append(
            {
                "experiment": path.parent.name,
                "backbone": f"{backbone}_vi",
                "variant": variant,
                "seed": seed,
                "checkpoint": f"best_{checkpoint}_accuracy",
                "region_accuracy": metrics["region"]["accuracy"],
                "region_balanced_accuracy": metrics["region"]["balanced_accuracy"],
                "region_macro_f1": macro_f1(metrics, "region"),
                "province_accuracy": metrics["province"]["accuracy"],
                "province_balanced_accuracy": metrics["province"]["balanced_accuracy"],
                "province_macro_f1": macro_f1(metrics, "province"),
            }
        )
    if not rows:
        raise FileNotFoundError("No complete H11 metric artifacts found")

    destination = Path(args.destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metric_names = [
        "region_accuracy",
        "region_balanced_accuracy",
        "region_macro_f1",
        "province_accuracy",
        "province_balanced_accuracy",
        "province_macro_f1",
    ]
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["backbone"], row["variant"], row["checkpoint"])].append(row)
    aggregate_rows = []
    for (backbone, variant, checkpoint), values in sorted(groups.items()):
        aggregate = {
            "backbone": backbone,
            "variant": variant,
            "checkpoint": checkpoint,
            "runs": len(values),
        }
        for metric in metric_names:
            numbers = [float(row[metric]) for row in values]
            aggregate[f"{metric}_mean"] = statistics.mean(numbers)
            aggregate[f"{metric}_std"] = (
                statistics.stdev(numbers) if len(numbers) > 1 else 0.0
            )
        aggregate_rows.append(aggregate)

    aggregate_destination = Path(args.aggregate_destination)
    aggregate_destination.parent.mkdir(parents=True, exist_ok=True)
    with aggregate_destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate_rows[0]))
        writer.writeheader()
        writer.writerows(aggregate_rows)
    print(f"Wrote {len(rows)} H11 runs to {destination}")
    print(
        f"Wrote {len(aggregate_rows)} H11 groups to {aggregate_destination}"
    )


if __name__ == "__main__":
    main()
