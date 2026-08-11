from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score


def main() -> None:
    parser = argparse.ArgumentParser(description="Select H16 checkpoint using validation province macro-F1.")
    parser.add_argument("--experiment", default="outputs/h16_vipvl_seed777_fixed20s")
    parser.add_argument("--labels", default="data/h16_chunkformer/dev/data.list")
    parser.add_argument("--destination", default="results_archive/h16")
    args = parser.parse_args()

    labels = {}
    with Path(args.labels).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            labels[row["key"]] = (int(row["region_label"]), int(row["province_label"]))
    rows = []
    for epoch in range(30):
        path = Path(args.experiment) / "validation" / f"epoch_{epoch}" / "predictions.tsv"
        if not path.is_file():
            raise FileNotFoundError(path)
        predictions = {}
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                predictions[row["key"]] = (int(row["region"]), int(row["province"]))
        if set(labels) != set(predictions):
            raise ValueError(f"Epoch {epoch}: label/prediction keys differ")
        keys = sorted(labels)
        region_true = [labels[key][0] for key in keys]
        province_true = [labels[key][1] for key in keys]
        region_pred = [predictions[key][0] for key in keys]
        province_pred = [predictions[key][1] for key in keys]
        rows.append({
            "epoch": epoch,
            "region_accuracy": accuracy_score(region_true, region_pred),
            "region_macro_f1": f1_score(region_true, region_pred, average="macro", zero_division=0),
            "province_accuracy": accuracy_score(province_true, province_pred),
            "province_macro_f1": f1_score(province_true, province_pred, average="macro", zero_division=0),
        })
    rows.sort(key=lambda row: (row["province_macro_f1"], row["province_accuracy"]), reverse=True)
    destination = Path(args.destination)
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "h16_validation_checkpoints.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    best = dict(rows[0])
    best["checkpoint"] = str(Path(args.experiment) / f"epoch_{best['epoch']}.pt")
    json_path = destination / "h16_best_validation_checkpoint.json"
    json_path.write_text(json.dumps(best, indent=2), encoding="utf-8")
    print("Top 10 validation checkpoints:")
    for row in rows[:10]:
        print(row)
    print(f"Selected: {best}")
    print(f"Wrote {csv_path} and {json_path}")


if __name__ == "__main__":
    main()

