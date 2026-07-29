from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dialect_moe.config import load_config


def run(command: list[str], root: Path) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=root, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate one reproducible dialect experiment."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    config = load_config(config_path)
    output_dir = root / config["training"]["output_dir"]

    if not args.skip_train:
        train_command = [
            sys.executable,
            str(root / "scripts" / "train.py"),
            "--config",
            str(config_path),
        ]
        if args.max_samples:
            train_command.extend(["--max-samples", str(args.max_samples)])
        run(train_command, root)

    candidates = [
        output_dir / "best_province_accuracy.pt",
        output_dir / "last.pt",
        output_dir / "best.pt",
    ]
    checkpoint = next((path for path in candidates if path.exists()), None)
    if checkpoint is None:
        raise FileNotFoundError(
            f"No checkpoint found in {output_dir}. Expected one of: "
            + ", ".join(path.name for path in candidates)
        )

    evaluate_command = [
        sys.executable,
        str(root / "scripts" / "evaluate.py"),
        "--config",
        str(config_path),
        "--checkpoint",
        str(checkpoint),
        "--split",
        args.split,
    ]
    if args.max_samples:
        evaluate_command.extend(["--max-samples", str(args.max_samples)])
    run(evaluate_command, root)
    artifact_names = [
        f"metrics_{args.split}",
        f"predictions_{args.split}",
        f"region_confusion_{args.split}",
        f"province_confusion_{args.split}",
    ]
    extensions = [".json", ".jsonl", ".csv", ".csv"]
    destinations = []
    for name, extension in zip(artifact_names, extensions):
        source = output_dir / f"{name}{extension}"
        destination = output_dir / f"{name}_{checkpoint.stem}{extension}"
        shutil.copy2(source, destination)
        destinations.append(destination)
    for name in (
        f"region_to_expert_{args.split}",
        f"province_to_expert_{args.split}",
    ):
        source = output_dir / f"{name}.csv"
        if source.exists():
            destination = output_dir / f"{name}_{checkpoint.stem}.csv"
            shutil.copy2(source, destination)
            destinations.append(destination)
    print("Experiment complete. Artifacts:", flush=True)
    for destination in destinations:
        print(f"- {destination}", flush=True)


if __name__ == "__main__":
    main()
