from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--pattern", default="metrics_test_*.json")
    parser.add_argument("--destination", default="outputs/experiment_summary.csv")
    parser.add_argument(
        "--aggregate-destination", default="outputs/experiment_aggregate.csv"
    )
    args = parser.parse_args()

    rows = []
    for path in sorted(Path(args.outputs).glob(f"**/{args.pattern}")):
        with path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        rows.append(
            {
                "experiment": path.parent.name,
                "metrics_file": path.name,
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
                "router_entropy": metrics["routing"]["mean_entropy"],
                "expert_probabilities": json.dumps(
                    metrics["routing"]["mean_expert_probability"]
                ),
            }
        )
    if not rows:
        raise FileNotFoundError(
            f"No metric files matching {args.outputs}/**/{args.pattern}"
        )

    destination = Path(args.destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} experiments to {destination}")

    metric_names = [
        "region_accuracy",
        "region_balanced_accuracy",
        "region_macro_f1",
        "province_accuracy",
        "province_balanced_accuracy",
        "province_macro_f1",
        "router_entropy",
    ]
    groups: dict[str, list[dict]] = {}
    for row in rows:
        group = re.sub(r"_seed\d+$", "", row["experiment"])
        groups.setdefault(group, []).append(row)
    aggregate_rows = []
    for group, group_rows in sorted(groups.items()):
        aggregate = {"experiment": group, "runs": len(group_rows)}
        for metric in metric_names:
            values = [float(row[metric]) for row in group_rows]
            aggregate[f"{metric}_mean"] = statistics.mean(values)
            aggregate[f"{metric}_std"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        aggregate_rows.append(aggregate)

    aggregate_destination = Path(args.aggregate_destination)
    aggregate_destination.parent.mkdir(parents=True, exist_ok=True)
    with aggregate_destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate_rows[0]))
        writer.writeheader()
        writer.writerows(aggregate_rows)
    print(
        f"Wrote {len(aggregate_rows)} aggregate experiment groups to "
        f"{aggregate_destination}"
    )


if __name__ == "__main__":
    main()
