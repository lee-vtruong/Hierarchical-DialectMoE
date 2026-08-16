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
)


def metric_row(path: Path) -> dict:
    metrics = json.loads(path.read_text(encoding="utf-8"))
    representation = metrics.get("representation", {})
    experiment = path.parent.name
    match = re.search(r"_seed(\d+)$", experiment)
    if match is None:
        raise ValueError(f"Cannot recover seed from experiment name: {experiment}")
    return {
        "experiment": experiment,
        "variant": re.sub(r"_seed\d+$", "", experiment),
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
        "acoustic_pooling": representation.get("acoustic_pooling"),
        "layer_mix_enabled": representation.get("layer_mix_enabled"),
        "layer_weights_json": json.dumps(representation.get("layer_weights")),
        "metrics_file": str(path),
    }


def aggregate_rows(rows: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["variant"], []).append(row)
    aggregates = []
    for variant, group in sorted(groups.items()):
        aggregate = {"variant": variant, "runs": len(group)}
        for metric in METRICS:
            values = [float(row[metric]) for row in group]
            aggregate[f"{metric}_mean"] = statistics.mean(values)
            aggregate[f"{metric}_std"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        aggregates.append(aggregate)
    return aggregates


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize H17 pooling experiments.")
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--split", choices=["valid", "test"], default="valid")
    parser.add_argument("--destination", default="results_archive/h17")
    args = parser.parse_args()

    filename = f"metrics_{args.split}_best_province_accuracy.json"
    paths = sorted(Path(args.outputs).glob(f"h17*_seed*/{filename}"))
    if not paths:
        raise FileNotFoundError(
            f"No H17 metrics found at {args.outputs}/h17*_seed*/{filename}"
        )
    rows = [metric_row(path) for path in paths]
    rows.sort(key=lambda row: (row["variant"], row["seed"]))
    aggregates = aggregate_rows(rows)
    destination = Path(args.destination)
    per_seed_path = destination / f"h17_{args.split}_per_seed.csv"
    aggregate_path = destination / f"h17_{args.split}_aggregate.csv"
    write_csv(per_seed_path, rows)
    write_csv(aggregate_path, aggregates)

    print(f"Wrote {len(rows)} rows to {per_seed_path}")
    print(f"Wrote {len(aggregates)} rows to {aggregate_path}")
    print(f"Ranking by {args.split} province macro-F1:")
    for row in sorted(
        aggregates, key=lambda item: item["province_macro_f1_mean"], reverse=True
    ):
        print(
            f"{row['variant']}: {row['province_macro_f1_mean']:.6f} "
            f"(province accuracy {row['province_accuracy_mean']:.6f}, "
            f"runs={row['runs']})"
        )


if __name__ == "__main__":
    main()

