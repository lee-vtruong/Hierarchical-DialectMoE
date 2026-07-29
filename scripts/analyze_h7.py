from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def load_jsonl(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            filename = row["filename"]
            if filename in rows:
                raise ValueError(f"Duplicate filename '{filename}' in {path}")
            rows[filename] = row
    if not rows:
        raise ValueError(f"No prediction rows found in {path}")
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def expected_calibration_error(
    probabilities: np.ndarray, targets: np.ndarray, bins: int = 10
) -> tuple[float, list[dict]]:
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == targets
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    ece = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (confidence >= lower) & (
            confidence <= upper if index == bins - 1 else confidence < upper
        )
        count = int(mask.sum())
        accuracy = float(correct[mask].mean()) if count else 0.0
        mean_confidence = float(confidence[mask].mean()) if count else 0.0
        ece += count / len(targets) * abs(accuracy - mean_confidence)
        rows.append(
            {
                "bin": index + 1,
                "lower": lower,
                "upper": upper,
                "count": count,
                "accuracy": accuracy,
                "mean_confidence": mean_confidence,
                "calibration_gap": mean_confidence - accuracy,
            }
        )
    return float(ece), rows


def calibration_metrics(rows: list[dict], bins: int) -> tuple[dict, list[dict]]:
    targets = np.asarray([row["province_true_id"] for row in rows], dtype=np.int64)
    probabilities = np.asarray(
        [row["province_probabilities"] for row in rows], dtype=np.float64
    )
    if probabilities.ndim != 2 or probabilities.shape[0] != len(targets):
        raise ValueError("Invalid province probability vectors")
    if np.any(targets < 0) or np.any(targets >= probabilities.shape[1]):
        raise ValueError("Province target is outside probability-vector range")
    clipped = np.clip(probabilities, 1e-12, 1.0)
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[targets]
    ece, bin_rows = expected_calibration_error(probabilities, targets, bins)
    return (
        {
            "ece": ece,
            "nll": float(-np.log(clipped[np.arange(len(targets)), targets]).mean()),
            "brier": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
            "mean_confidence": float(probabilities.max(axis=1).mean()),
        },
        bin_rows,
    )


def analyse_seed(
    baseline_map: dict[str, dict],
    candidate_map: dict[str, dict],
    seed: int,
    bins: int,
) -> dict:
    if set(baseline_map) != set(candidate_map):
        raise ValueError(f"Seed {seed}: baseline and candidate samples differ")
    filenames = sorted(baseline_map)
    baseline = [baseline_map[name] for name in filenames]
    candidate = [candidate_map[name] for name in filenames]
    if any(
        left["province_true_id"] != right["province_true_id"]
        for left, right in zip(baseline, candidate)
    ):
        raise ValueError(f"Seed {seed}: ground-truth labels differ")

    province_counts: dict[str, Counter] = defaultdict(Counter)
    confusion_baseline: Counter = Counter()
    confusion_candidate: Counter = Counter()
    transitions: Counter = Counter()
    for left, right in zip(baseline, candidate):
        truth = left["province_true"]
        left_correct = left["province_pred_id"] == left["province_true_id"]
        right_correct = right["province_pred_id"] == right["province_true_id"]
        counts = province_counts[truth]
        counts["support"] += 1
        counts["baseline_correct"] += int(left_correct)
        counts["candidate_correct"] += int(right_correct)
        if not left_correct:
            confusion_baseline[(truth, left["province_pred"])] += 1
        if not right_correct:
            confusion_candidate[(truth, right["province_pred"])] += 1
        if not left_correct and right_correct:
            transitions["fixed"] += 1
        elif left_correct and not right_correct:
            transitions["regressed"] += 1
        elif left_correct:
            transitions["both_correct"] += 1
        else:
            transitions["both_wrong"] += 1

    province_rows = []
    for province, counts in sorted(province_counts.items()):
        support = counts["support"]
        baseline_accuracy = counts["baseline_correct"] / support
        candidate_accuracy = counts["candidate_correct"] / support
        province_rows.append(
            {
                "seed": seed,
                "province": province,
                "support": support,
                "baseline_accuracy": baseline_accuracy,
                "candidate_accuracy": candidate_accuracy,
                "improvement": candidate_accuracy - baseline_accuracy,
                "fixed": sum(
                    1
                    for left, right in zip(baseline, candidate)
                    if left["province_true"] == province
                    and left["province_pred_id"] != left["province_true_id"]
                    and right["province_pred_id"] == right["province_true_id"]
                ),
                "regressed": sum(
                    1
                    for left, right in zip(baseline, candidate)
                    if left["province_true"] == province
                    and left["province_pred_id"] == left["province_true_id"]
                    and right["province_pred_id"] != right["province_true_id"]
                ),
            }
        )

    baseline_calibration, baseline_bins = calibration_metrics(baseline, bins)
    candidate_calibration, candidate_bins = calibration_metrics(candidate, bins)
    return {
        "samples": len(filenames),
        "province_rows": province_rows,
        "confusion_baseline": confusion_baseline,
        "confusion_candidate": confusion_candidate,
        "transitions": dict(transitions),
        "baseline_calibration": baseline_calibration,
        "candidate_calibration": candidate_calibration,
        "baseline_bins": baseline_bins,
        "candidate_bins": candidate_bins,
    }


def aggregate_provinces(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["province"]].append(row)
    result = []
    for province, values in grouped.items():
        improvements = np.asarray([row["improvement"] for row in values])
        result.append(
            {
                "province": province,
                "seeds": len(values),
                "mean_support": float(np.mean([row["support"] for row in values])),
                "baseline_accuracy_mean": float(
                    np.mean([row["baseline_accuracy"] for row in values])
                ),
                "candidate_accuracy_mean": float(
                    np.mean([row["candidate_accuracy"] for row in values])
                ),
                "improvement_mean": float(improvements.mean()),
                "improvement_std": float(improvements.std(ddof=1))
                if len(improvements) > 1
                else 0.0,
                "improved_seeds": int(np.sum(improvements > 0)),
                "degraded_seeds": int(np.sum(improvements < 0)),
                "fixed_total": sum(row["fixed"] for row in values),
                "regressed_total": sum(row["regressed"] for row in values),
            }
        )
    return sorted(result, key=lambda row: row["improvement_mean"], reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="H7 multi-seed error analysis.")
    parser.add_argument("--baseline-template", required=True)
    parser.add_argument("--candidate-template", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--output-dir", default="outputs/h7")
    parser.add_argument("--calibration-bins", type=int, default=10)
    args = parser.parse_args()
    if args.calibration_bins < 2:
        raise ValueError("--calibration-bins must be at least 2")

    output_dir = Path(args.output_dir)
    all_provinces = []
    all_calibration = []
    all_bins = []
    confusion_counts: dict[str, Counter] = {
        "baseline": Counter(),
        "candidate": Counter(),
    }
    seed_summaries = []

    for seed in args.seeds:
        baseline_path = Path(args.baseline_template.format(seed=seed))
        candidate_path = Path(args.candidate_template.format(seed=seed))
        result = analyse_seed(
            load_jsonl(baseline_path),
            load_jsonl(candidate_path),
            seed,
            args.calibration_bins,
        )
        all_provinces.extend(result["province_rows"])
        confusion_counts["baseline"].update(result["confusion_baseline"])
        confusion_counts["candidate"].update(result["confusion_candidate"])
        for model in ("baseline", "candidate"):
            metrics = result[f"{model}_calibration"]
            all_calibration.append({"seed": seed, "model": model, **metrics})
            all_bins.extend(
                {"seed": seed, "model": model, **row}
                for row in result[f"{model}_bins"]
            )
        seed_summaries.append(
            {
                "seed": seed,
                "samples": result["samples"],
                **result["transitions"],
            }
        )

    aggregate_rows = aggregate_provinces(all_provinces)
    confusion_rows = []
    pairs = set(confusion_counts["baseline"]) | set(confusion_counts["candidate"])
    for truth, prediction in pairs:
        baseline_count = confusion_counts["baseline"][(truth, prediction)]
        candidate_count = confusion_counts["candidate"][(truth, prediction)]
        confusion_rows.append(
            {
                "true_province": truth,
                "predicted_province": prediction,
                "baseline_count": baseline_count,
                "candidate_count": candidate_count,
                "change_candidate_minus_baseline": candidate_count - baseline_count,
            }
        )
    confusion_rows.sort(
        key=lambda row: max(row["baseline_count"], row["candidate_count"]),
        reverse=True,
    )

    calibration_aggregate = []
    for model in ("baseline", "candidate"):
        values = [row for row in all_calibration if row["model"] == model]
        calibration_aggregate.append(
            {
                "model": model,
                **{
                    f"{metric}_mean": float(np.mean([row[metric] for row in values]))
                    for metric in ("ece", "nll", "brier", "mean_confidence")
                },
            }
        )

    province_fields = [
        "seed", "province", "support", "baseline_accuracy", "candidate_accuracy",
        "improvement", "fixed", "regressed",
    ]
    aggregate_fields = [
        "province", "seeds", "mean_support", "baseline_accuracy_mean",
        "candidate_accuracy_mean", "improvement_mean", "improvement_std",
        "improved_seeds", "degraded_seeds", "fixed_total", "regressed_total",
    ]
    write_csv(output_dir / "province_per_seed.csv", all_provinces, province_fields)
    write_csv(output_dir / "province_aggregate.csv", aggregate_rows, aggregate_fields)
    write_csv(
        output_dir / "confusion_pairs.csv",
        confusion_rows,
        [
            "true_province", "predicted_province", "baseline_count",
            "candidate_count", "change_candidate_minus_baseline",
        ],
    )
    write_csv(
        output_dir / "calibration_per_seed.csv",
        all_calibration,
        ["seed", "model", "ece", "nll", "brier", "mean_confidence"],
    )
    write_csv(
        output_dir / "calibration_bins.csv",
        all_bins,
        [
            "seed", "model", "bin", "lower", "upper", "count", "accuracy",
            "mean_confidence", "calibration_gap",
        ],
    )

    summary = {
        "seeds": args.seeds,
        "baseline_template": args.baseline_template,
        "candidate_template": args.candidate_template,
        "seed_transitions": seed_summaries,
        "calibration_aggregate": calibration_aggregate,
        "top_improved_provinces": aggregate_rows[:10],
        "top_degraded_provinces": list(reversed(aggregate_rows[-10:])),
        "top_confusion_pairs": confusion_rows[:20],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "h7_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote H7 artifacts to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
