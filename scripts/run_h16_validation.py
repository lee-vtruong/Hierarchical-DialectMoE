from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


def discover_epochs(experiment: Path) -> list[int]:
    epochs = []
    for checkpoint in experiment.glob("epoch_*.pt"):
        suffix = checkpoint.stem.removeprefix("epoch_")
        if suffix.isdigit():
            epochs.append(int(suffix))
    epochs = sorted(set(epochs))
    if not epochs:
        raise FileNotFoundError(f"No epoch_*.pt checkpoints found in {experiment}")
    return epochs


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate H16 checkpoints on validation.")
    parser.add_argument("--chunkformer-root", default="external/chunkformer")
    parser.add_argument("--experiment", default="outputs/h16_vipvl_seed777_fixed20s")
    parser.add_argument("--data", default="data/h16_chunkformer/dev/data.list")
    parser.add_argument("--epochs", choices=["all", "even", "odd"], default="all")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    root = Path.cwd()
    chunkformer = (root / args.chunkformer_root).resolve()
    experiment = (root / args.experiment).resolve()
    data = (root / args.data).resolve()
    expected_lines = sum(1 for line in data.open(encoding="utf-8") if line.strip()) + 1
    all_epochs = discover_epochs(experiment)
    epochs = all_epochs
    if args.epochs == "even":
        epochs = [epoch for epoch in epochs if epoch % 2 == 0]
    elif args.epochs == "odd":
        epochs = [epoch for epoch in epochs if epoch % 2 == 1]

    print(
        f"Discovered {len(all_epochs)} checkpoints; processing {len(epochs)} "
        f"{args.epochs} checkpoints (epochs {epochs[0]}..{epochs[-1]}).",
        flush=True,
    )

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(chunkformer) + os.pathsep + environment.get("PYTHONPATH", "")
    for epoch in epochs:
        checkpoint = experiment / f"epoch_{epoch}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        result_dir = experiment / "validation" / f"epoch_{epoch}"
        prediction_path = result_dir / "predictions.tsv"
        if prediction_path.is_file():
            actual_lines = sum(1 for _ in prediction_path.open(encoding="utf-8"))
            if actual_lines == expected_lines:
                print(f"SKIP epoch {epoch}: complete ({actual_lines - 1} samples)", flush=True)
                continue
        result_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(chunkformer / "chunkformer" / "bin" / "classify.py"),
            "--gpu", "0",
            "--config", str(experiment / "train.yaml"),
            "--data_type", "raw",
            "--test_data", str(data),
            "--checkpoint", str(checkpoint),
            "--batch_size", str(args.batch_size),
            "--result_dir", str(result_dir),
            "--dtype", "fp16",
        ]
        print("+", " ".join(command), flush=True)
        subprocess.run(command, cwd=root, env=environment, check=True)
        actual_lines = sum(1 for _ in prediction_path.open(encoding="utf-8"))
        if actual_lines != expected_lines:
            raise RuntimeError(
                f"Epoch {epoch}: predictions has {actual_lines - 1} samples; "
                f"expected {expected_lines - 1}"
            )


if __name__ == "__main__":
    main()
