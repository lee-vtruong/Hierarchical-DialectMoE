from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# Support both `python scripts/analyze_h12.py` and module/test imports.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_predictions import compare_task, load_jsonl


CONTRASTS = {
    "base_prosody_vs_acoustic": ("base", "acoustic", "base", "prosody"),
    "large_prosody_vs_acoustic": ("large", "acoustic", "large", "prosody"),
    "large_vs_base_acoustic": ("base", "acoustic", "large", "acoustic"),
    "large_vs_base_prosody": ("base", "prosody", "large", "prosody"),
}
TASKS = ("region", "province")
METRICS = ("accuracy", "balanced_accuracy", "macro_f1")


def prediction_path(outputs: Path, backbone: str, variant: str, seed: int) -> Path:
    return (
        outputs
        / f"h11_{backbone}_vi_{variant}_seed{seed}"
        / "predictions_test_best_province_accuracy.jsonl"
    )


def aligned_rows(baseline_path: Path, candidate_path: Path) -> tuple[list[dict], list[dict]]:
    baseline = load_jsonl(baseline_path)
    candidate = load_jsonl(candidate_path)
    if set(baseline) != set(candidate):
        missing_candidate = sorted(set(baseline) - set(candidate))[:10]
        missing_baseline = sorted(set(candidate) - set(baseline))[:10]
        raise ValueError(
            "Prediction files contain different samples. "
            f"Missing from candidate: {missing_candidate}; "
            f"missing from baseline: {missing_baseline}"
        )
    keys = sorted(baseline)
    baseline_rows = [baseline[key] for key in keys]
    candidate_rows = [candidate[key] for key in keys]
    baseline_speakers = [row["speaker_id"] for row in baseline_rows]
    candidate_speakers = [row["speaker_id"] for row in candidate_rows]
    if baseline_speakers != candidate_speakers:
        raise ValueError("Speaker IDs differ between aligned prediction files")
    return baseline_rows, candidate_rows


def holm_adjust(p_values: list[float]) -> list[float]:
    count = len(p_values)
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [1.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="H12 paired speaker-bootstrap and McNemar analysis for H11."
    )
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--destination", default="outputs/h12")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=12026)
    args = parser.parse_args()

    outputs = Path(args.outputs)
    destination = Path(args.destination)
    destination.mkdir(parents=True, exist_ok=True)
    details: dict[str, dict[str, dict]] = {}
    per_seed_rows: list[dict] = []

    for contrast_index, (name, specification) in enumerate(CONTRASTS.items()):
        baseline_backbone, baseline_variant, candidate_backbone, candidate_variant = specification
        details[name] = {}
        for seed_index, seed in enumerate(args.seeds):
            baseline_path = prediction_path(
                outputs, baseline_backbone, baseline_variant, seed
            )
            candidate_path = prediction_path(
                outputs, candidate_backbone, candidate_variant, seed
            )
            if not baseline_path.is_file() or not candidate_path.is_file():
                raise FileNotFoundError(
                    f"Missing H12 prediction pair: {baseline_path}; {candidate_path}"
                )
            baseline_rows, candidate_rows = aligned_rows(
                baseline_path, candidate_path
            )
            seed_result = {
                "baseline_file": str(baseline_path),
                "candidate_file": str(candidate_path),
                "samples": len(baseline_rows),
                "speakers": len({row["speaker_id"] for row in baseline_rows}),
            }
            for task_index, task in enumerate(TASKS):
                comparison = compare_task(
                    baseline_rows,
                    candidate_rows,
                    task,
                    args.bootstrap_iterations,
                    args.bootstrap_seed
                    + contrast_index * 100
                    + seed_index * 10
                    + task_index,
                )
                seed_result[task] = comparison
                row = {
                    "contrast": name,
                    "seed": seed,
                    "task": task,
                    "samples": len(baseline_rows),
                    "speakers": seed_result["speakers"],
                }
                for metric in METRICS:
                    bootstrap = comparison["speaker_bootstrap"][metric]
                    row[f"baseline_{metric}"] = comparison["baseline"][metric]
                    row[f"candidate_{metric}"] = comparison["candidate"][metric]
                    row[f"difference_{metric}"] = comparison[
                        "difference_candidate_minus_baseline"
                    ][metric]
                    row[f"bootstrap_{metric}_ci_low"] = bootstrap["ci_95_low"]
                    row[f"bootstrap_{metric}_ci_high"] = bootstrap["ci_95_high"]
                    row[f"probability_candidate_better_{metric}"] = bootstrap[
                        "probability_candidate_better"
                    ]
                mcnemar = comparison["mcnemar_accuracy"]
                row["baseline_correct_candidate_wrong"] = mcnemar[
                    "baseline_correct_candidate_wrong"
                ]
                row["baseline_wrong_candidate_correct"] = mcnemar[
                    "baseline_wrong_candidate_correct"
                ]
                row["mcnemar_exact_p"] = mcnemar["exact_p_value"]
                per_seed_rows.append(row)
            details[name][str(seed)] = seed_result

    # Correct the 12 McNemar tests separately within region and province families.
    for task in TASKS:
        task_indices = [
            index for index, row in enumerate(per_seed_rows) if row["task"] == task
        ]
        adjusted = holm_adjust(
            [float(per_seed_rows[index]["mcnemar_exact_p"]) for index in task_indices]
        )
        for index, adjusted_p in zip(task_indices, adjusted):
            per_seed_rows[index]["mcnemar_holm_p"] = adjusted_p
            per_seed_rows[index]["mcnemar_holm_significant_0_05"] = adjusted_p < 0.05

    aggregate_rows: list[dict] = []
    for name in CONTRASTS:
        for task in TASKS:
            values = [
                row
                for row in per_seed_rows
                if row["contrast"] == name and row["task"] == task
            ]
            aggregate = {
                "contrast": name,
                "task": task,
                "runs": len(values),
            }
            for metric in METRICS:
                for field in ("baseline", "candidate", "difference"):
                    numbers = [float(row[f"{field}_{metric}"]) for row in values]
                    aggregate[f"{field}_{metric}_mean"] = sum(numbers) / len(numbers)
                aggregate[f"bootstrap_ci_excludes_zero_{metric}_runs"] = sum(
                    float(row[f"bootstrap_{metric}_ci_low"]) > 0
                    or float(row[f"bootstrap_{metric}_ci_high"]) < 0
                    for row in values
                )
                aggregate[f"candidate_better_{metric}_runs"] = sum(
                    float(row[f"difference_{metric}"]) > 0 for row in values
                )
            aggregate["mcnemar_holm_significant_runs"] = sum(
                bool(row["mcnemar_holm_significant_0_05"]) for row in values
            )
            aggregate_rows.append(aggregate)

    per_seed_path = destination / "h12_per_seed.csv"
    aggregate_path = destination / "h12_aggregate.csv"
    details_path = destination / "h12_details.json"
    summary_path = destination / "h12_summary.json"
    write_csv(per_seed_path, per_seed_rows)
    write_csv(aggregate_path, aggregate_rows)
    with details_path.open("w", encoding="utf-8") as handle:
        json.dump(details, handle, ensure_ascii=False, indent=2)
    summary = {
        "hypothesis": (
            "Large Vietnamese pretraining and explicit prosody improve paired "
            "region/province predictions on the locked H11 test set."
        ),
        "seeds": args.seeds,
        "bootstrap_iterations": args.bootstrap_iterations,
        "multiple_testing": (
            "Holm correction over 12 McNemar tests separately for each task "
            "(4 contrasts x 3 seeds)."
        ),
        "contrasts": list(CONTRASTS),
        "files": {
            "per_seed": str(per_seed_path),
            "aggregate": str(aggregate_path),
            "details": str(details_path),
        },
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(f"Wrote {len(per_seed_rows)} paired rows to {per_seed_path}")
    print(f"Wrote {len(aggregate_rows)} aggregate rows to {aggregate_path}")
    print(f"Wrote detailed comparisons to {details_path}")
    print(f"Wrote analysis manifest to {summary_path}")


if __name__ == "__main__":
    main()
