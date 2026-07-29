from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar


def load_predictions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    targets, probabilities = [], []
    seen = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            filename = row["filename"]
            if filename in seen:
                raise ValueError(f"Duplicate filename '{filename}' in {path}")
            seen.add(filename)
            targets.append(int(row["province_true_id"]))
            probabilities.append(row["province_probabilities"])
    if not targets:
        raise ValueError(f"No prediction rows found in {path}")
    probs = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(targets, dtype=np.int64)
    if probs.ndim != 2 or np.any(labels < 0) or np.any(labels >= probs.shape[1]):
        raise ValueError(f"Invalid targets/probability vectors in {path}")
    if not np.all(np.isfinite(probs)) or np.any(probs < 0):
        raise ValueError(f"Probabilities must be finite and non-negative in {path}")
    row_sums = probs.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError(f"Probability row with zero mass in {path}")
    return labels, probs / row_sums


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    logits = np.log(np.clip(probabilities, 1e-12, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def negative_log_likelihood(
    probabilities: np.ndarray, targets: np.ndarray
) -> float:
    selected = probabilities[np.arange(len(targets)), targets]
    return float(-np.log(np.clip(selected, 1e-12, 1.0)).mean())


def fit_temperature(
    probabilities: np.ndarray,
    targets: np.ndarray,
    lower: float = 0.05,
    upper: float = 10.0,
) -> float:
    result = minimize_scalar(
        lambda log_temperature: negative_log_likelihood(
            apply_temperature(probabilities, float(np.exp(log_temperature))),
            targets,
        ),
        bounds=(float(np.log(lower)), float(np.log(upper))),
        method="bounded",
        options={"xatol": 1e-8},
    )
    if not result.success:
        raise RuntimeError(f"Temperature optimization failed: {result.message}")
    return float(np.exp(result.x))


def calibration_metrics(
    probabilities: np.ndarray, targets: np.ndarray, bins: int
) -> tuple[dict, list[dict]]:
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correctness = predictions == targets
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[targets]
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    bin_rows = []
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (confidence >= lower) & (
            confidence <= upper if index == bins - 1 else confidence < upper
        )
        support = int(mask.sum())
        accuracy = float(correctness[mask].mean()) if support else 0.0
        mean_confidence = float(confidence[mask].mean()) if support else 0.0
        gap = mean_confidence - accuracy
        ece += support / len(targets) * abs(gap)
        bin_rows.append(
            {
                "bin": index + 1,
                "lower": lower,
                "upper": upper,
                "support": support,
                "accuracy": accuracy,
                "mean_confidence": mean_confidence,
                "calibration_gap": gap,
            }
        )
    return (
        {
            "accuracy": float(correctness.mean()),
            "ece": float(ece),
            "nll": negative_log_likelihood(probabilities, targets),
            "brier": float(
                np.square(probabilities - one_hot).sum(axis=1).mean()
            ),
            "mean_confidence": float(confidence.mean()),
        },
        bin_rows,
    )


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["stage"])].append(row)
    output = []
    for (model, stage), values in sorted(grouped.items()):
        aggregate_row = {"model": model, "stage": stage, "seeds": len(values)}
        for metric in ("temperature", "accuracy", "ece", "nll", "brier", "mean_confidence"):
            numbers = np.asarray([float(row[metric]) for row in values])
            aggregate_row[f"{metric}_mean"] = float(numbers.mean())
            aggregate_row[f"{metric}_std"] = (
                float(numbers.std(ddof=1)) if len(numbers) > 1 else 0.0
            )
        output.append(aggregate_row)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="H9 validation-fitted temperature scaling."
    )
    parser.add_argument("--baseline-valid-template", required=True)
    parser.add_argument("--baseline-test-template", required=True)
    parser.add_argument("--candidate-valid-template", required=True)
    parser.add_argument("--candidate-test-template", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--bins", type=int, default=15)
    parser.add_argument("--temperature-min", type=float, default=0.05)
    parser.add_argument("--temperature-max", type=float, default=10.0)
    parser.add_argument("--output-dir", default="outputs/h9")
    args = parser.parse_args()
    if args.bins < 2:
        raise ValueError("--bins must be at least 2")
    if not 0 < args.temperature_min < args.temperature_max:
        raise ValueError("Require 0 < temperature-min < temperature-max")

    per_seed, test_bins = [], []
    file_templates = {
        "baseline": (args.baseline_valid_template, args.baseline_test_template),
        "candidate": (args.candidate_valid_template, args.candidate_test_template),
    }
    for seed in args.seeds:
        for model, (valid_template, test_template) in file_templates.items():
            valid_targets, valid_probs = load_predictions(
                Path(valid_template.format(seed=seed))
            )
            test_targets, test_probs = load_predictions(
                Path(test_template.format(seed=seed))
            )
            if valid_probs.shape[1] != test_probs.shape[1]:
                raise ValueError(
                    f"{model} seed {seed}: valid/test class counts differ"
                )
            temperature = fit_temperature(
                valid_probs,
                valid_targets,
                args.temperature_min,
                args.temperature_max,
            )
            calibrated_valid = apply_temperature(valid_probs, temperature)
            calibrated_test = apply_temperature(test_probs, temperature)
            if not np.array_equal(
                valid_probs.argmax(axis=1), calibrated_valid.argmax(axis=1)
            ) or not np.array_equal(
                test_probs.argmax(axis=1), calibrated_test.argmax(axis=1)
            ):
                raise RuntimeError(
                    f"{model} seed {seed}: temperature scaling changed argmax"
                )
            if negative_log_likelihood(
                calibrated_valid, valid_targets
            ) > negative_log_likelihood(valid_probs, valid_targets) + 1e-10:
                raise RuntimeError(
                    f"{model} seed {seed}: fitted temperature increased valid NLL"
                )
            for split, targets, probabilities in (
                ("valid", valid_targets, valid_probs),
                ("test", test_targets, test_probs),
            ):
                for calibration, calibrated_probs in (
                    ("before", probabilities),
                    ("after", apply_temperature(probabilities, temperature)),
                ):
                    metrics, bin_rows = calibration_metrics(
                        calibrated_probs, targets, args.bins
                    )
                    per_seed.append(
                        {
                            "seed": seed,
                            "model": model,
                            "stage": f"{split}_{calibration}",
                            "temperature": temperature,
                            "samples": len(targets),
                            **metrics,
                        }
                    )
                    if split == "test":
                        test_bins.extend(
                            {
                                "seed": seed,
                                "model": model,
                                "calibration": calibration,
                                **row,
                            }
                            for row in bin_rows
                        )

    aggregate_rows = aggregate(per_seed)
    output_dir = Path(args.output_dir)
    per_seed_fields = [
        "seed", "model", "stage", "temperature", "samples", "accuracy", "ece",
        "nll", "brier", "mean_confidence",
    ]
    write_csv(output_dir / "h9_per_seed.csv", per_seed, per_seed_fields)
    write_csv(
        output_dir / "h9_aggregate.csv",
        aggregate_rows,
        list(aggregate_rows[0]) if aggregate_rows else [],
    )
    write_csv(
        output_dir / "h9_test_calibration_bins.csv",
        test_bins,
        [
            "seed", "model", "calibration", "bin", "lower", "upper", "support",
            "accuracy", "mean_confidence", "calibration_gap",
        ],
    )
    summary = {
        "method": "scalar temperature scaling",
        "fit_split": "repaired validation",
        "evaluation_split": "repaired test",
        "seeds": args.seeds,
        "bins": args.bins,
        "aggregate": aggregate_rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "h9_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote H9 artifacts to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
