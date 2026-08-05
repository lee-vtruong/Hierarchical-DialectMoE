from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def prediction_path(
    outputs: Path, variant: str, seed: int, split: str
) -> Path:
    return (
        outputs
        / f"h11_large_vi_{variant}_seed{seed}"
        / f"predictions_{split}_best_province_accuracy.jsonl"
    )


def checkpoint_path(outputs: Path, variant: str, seed: int) -> Path:
    return (
        outputs
        / f"h11_large_vi_{variant}_seed{seed}"
        / "best_province_accuracy.pt"
    )


def require_files(paths: list[Path], description: str) -> None:
    missing = [str(path) for path in paths if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"Missing {description}: {missing}")


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="H13 final Large-VI error, calibration, and artifact analysis."
    )
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--destination", default="results_archive/h13")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--calibration-bins", type=int, default=15)
    parser.add_argument(
        "--skip-calibration",
        action="store_true",
        help="Run error analysis without validation-fitted temperature scaling.",
    )
    args = parser.parse_args()
    if args.calibration_bins < 2:
        raise ValueError("--calibration-bins must be at least 2")

    outputs = Path(args.outputs)
    destination = Path(args.destination)
    destination.mkdir(parents=True, exist_ok=True)
    test_files = [
        prediction_path(outputs, variant, seed, "test")
        for variant in ("acoustic", "prosody")
        for seed in args.seeds
    ]
    require_files(test_files, "H13 test predictions")

    error_dir = destination / "error_analysis"
    run(
        [
            sys.executable,
            str(ROOT / "scripts" / "analyze_h7.py"),
            "--baseline-template",
            str(
                outputs
                / "h11_large_vi_acoustic_seed{seed}"
                / "predictions_test_best_province_accuracy.jsonl"
            ),
            "--candidate-template",
            str(
                outputs
                / "h11_large_vi_prosody_seed{seed}"
                / "predictions_test_best_province_accuracy.jsonl"
            ),
            "--seeds",
            *[str(seed) for seed in args.seeds],
            "--output-dir",
            str(error_dir),
            "--calibration-bins",
            str(args.calibration_bins),
        ]
    )

    calibration_status = "skipped_by_request"
    calibration_dir = destination / "calibration"
    if not args.skip_calibration:
        valid_files = [
            prediction_path(outputs, variant, seed, "valid")
            for variant in ("acoustic", "prosody")
            for seed in args.seeds
        ]
        require_files(valid_files, "H13 validation predictions")
        run(
            [
                sys.executable,
                str(ROOT / "scripts" / "calibrate_h9.py"),
                "--baseline-valid-template",
                str(
                    outputs
                    / "h11_large_vi_acoustic_seed{seed}"
                    / "predictions_valid_best_province_accuracy.jsonl"
                ),
                "--baseline-test-template",
                str(
                    outputs
                    / "h11_large_vi_acoustic_seed{seed}"
                    / "predictions_test_best_province_accuracy.jsonl"
                ),
                "--candidate-valid-template",
                str(
                    outputs
                    / "h11_large_vi_prosody_seed{seed}"
                    / "predictions_valid_best_province_accuracy.jsonl"
                ),
                "--candidate-test-template",
                str(
                    outputs
                    / "h11_large_vi_prosody_seed{seed}"
                    / "predictions_test_best_province_accuracy.jsonl"
                ),
                "--seeds",
                *[str(seed) for seed in args.seeds],
                "--bins",
                str(args.calibration_bins),
                "--output-dir",
                str(calibration_dir),
            ]
        )
        calibration_status = "completed_validation_fitted_temperature_scaling"

    artifact_rows = []
    for variant in ("acoustic", "prosody"):
        for seed in args.seeds:
            checkpoint = checkpoint_path(outputs, variant, seed)
            test_prediction = prediction_path(outputs, variant, seed, "test")
            row = {
                "variant": variant,
                "seed": seed,
                "checkpoint": str(checkpoint),
                "checkpoint_exists": checkpoint.is_file(),
                "checkpoint_mib": (
                    checkpoint.stat().st_size / (1024**2) if checkpoint.is_file() else None
                ),
                "test_prediction": str(test_prediction),
                "test_samples": count_jsonl(test_prediction),
            }
            valid_prediction = prediction_path(outputs, variant, seed, "valid")
            row["valid_prediction"] = str(valid_prediction)
            row["valid_samples"] = (
                count_jsonl(valid_prediction) if valid_prediction.is_file() else None
            )
            artifact_rows.append(row)
    write_csv(destination / "h13_artifact_metadata.csv", artifact_rows)

    summary = {
        "hypothesis": (
            "Large-VI explicit prosody changes province-level errors and "
            "calibration relative to Large-VI acoustic-only."
        ),
        "baseline": "h11_large_vi_acoustic best_province_accuracy",
        "candidate": "h11_large_vi_prosody best_province_accuracy",
        "seeds": args.seeds,
        "test_prediction_status": "complete",
        "calibration_status": calibration_status,
        "calibration_fit_split": "repaired validation",
        "calibration_evaluation_split": "repaired test",
        "outputs": {
            "error_analysis": str(error_dir),
            "calibration": str(calibration_dir) if not args.skip_calibration else None,
            "artifact_metadata": str(destination / "h13_artifact_metadata.csv"),
        },
        "efficiency_note": (
            "Checkpoint size is descriptive storage metadata, not inference speed. "
            "Latency requires a separately controlled GPU benchmark."
        ),
    }
    with (destination / "h13_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote H13 artifacts to {destination.resolve()}")


if __name__ == "__main__":
    main()
