from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


METRICS = (
    "region_accuracy",
    "region_macro_f1",
    "province_accuracy",
    "province_balanced_accuracy",
    "province_macro_f1",
)


def read_metrics(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "region_accuracy": float(payload["region"]["accuracy"]),
        "region_macro_f1": float(
            payload["region"]["classification_report"]["macro avg"]["f1-score"]
        ),
        "province_accuracy": float(payload["province"]["accuracy"]),
        "province_balanced_accuracy": float(
            payload["province"]["balanced_accuracy"]
        ),
        "province_macro_f1": float(
            payload["province"]["classification_report"]["macro avg"]["f1-score"]
        ),
    }


def compare_runs(outputs: Path, split: str, seeds: list[int]) -> list[dict]:
    filename = f"metrics_{split}_best_province_accuracy.json"
    rows = []
    for seed in seeds:
        baseline_path = (
            outputs / f"h11_large_vi_prosody_seed{seed}" / filename
        )
        candidate_path = outputs / f"h18_soft_hierarchy_seed{seed}" / filename
        if not baseline_path.is_file() or not candidate_path.is_file():
            continue
        baseline = read_metrics(baseline_path)
        candidate = read_metrics(candidate_path)
        row: dict[str, object] = {
            "seed": seed,
            "baseline_file": str(baseline_path),
            "candidate_file": str(candidate_path),
        }
        for metric in METRICS:
            row[f"baseline_{metric}"] = baseline[metric]
            row[f"candidate_{metric}"] = candidate[metric]
            row[f"difference_{metric}"] = candidate[metric] - baseline[metric]
        rows.append(row)
    return rows


def make_decision(rows: list[dict]) -> dict:
    if not rows:
        return {
            "stage": "missing",
            "passed": False,
            "reason": "No matched H11/H18 validation runs were found.",
        }
    differences = {
        metric: [float(row[f"difference_{metric}"]) for row in rows]
        for metric in METRICS
    }
    accuracy_wins = sum(value > 0 for value in differences["province_accuracy"])
    summary: dict[str, object] = {
        "runs": len(rows),
        "province_accuracy_wins": accuracy_wins,
    }
    for metric, values in differences.items():
        summary[f"difference_{metric}_mean"] = statistics.mean(values)
        summary[f"difference_{metric}_std"] = (
            statistics.stdev(values) if len(values) > 1 else 0.0
        )

    if len(rows) == 1:
        passed = (
            summary["difference_province_accuracy_mean"] > 0
            and summary["difference_province_macro_f1_mean"] >= -0.003
            and summary["difference_region_accuracy_mean"] >= -0.005
        )
        return {
            "stage": "seed42_screening",
            "passed": passed,
            "rule": (
                "province accuracy > H11; province macro-F1 drop <= 0.003; "
                "region accuracy drop <= 0.005"
            ),
            **summary,
        }

    if len(rows) >= 3:
        passed = (
            summary["difference_province_accuracy_mean"] > 0
            and summary["difference_province_macro_f1_mean"] >= 0
            and summary["difference_region_accuracy_mean"] >= -0.005
            and accuracy_wins >= 2
        )
        return {
            "stage": "multi_seed_validation",
            "passed": passed,
            "rule": (
                "mean province accuracy > H11; mean province macro-F1 >= H11; "
                "mean region accuracy drop <= 0.005; province accuracy wins >= 2/3"
            ),
            **summary,
        }

    return {
        "stage": "incomplete_multi_seed",
        "passed": False,
        "reason": "Two runs are not enough for the predefined multi-seed gate.",
        **summary,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare H18 with matched H11 Large-VI + prosody runs."
    )
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--split", choices=["valid", "test"], default="valid")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--destination", default="results_archive/h18")
    args = parser.parse_args()

    rows = compare_runs(Path(args.outputs), args.split, args.seeds)
    if not rows:
        raise FileNotFoundError("No matched H11/H18 metric files were found")
    decision = make_decision(rows)
    destination = Path(args.destination)
    rows_path = destination / f"h18_vs_h11_{args.split}_per_seed.csv"
    decision_path = destination / f"h18_vs_h11_{args.split}_decision.json"
    write_csv(rows_path, rows)
    decision_path.write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {rows_path}")
    print(f"Wrote {decision_path}")
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
