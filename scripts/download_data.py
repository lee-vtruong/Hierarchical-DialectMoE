from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description="Download public Vietnamese dialect datasets.")
    parser.add_argument("--dataset", choices=["vimd", "vispeech", "all"], default="vimd")
    parser.add_argument("--output-dir", default="data")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    if args.dataset in {"vimd", "all"}:
        target = output / "ViMD_Dataset"
        print(f"Downloading ViMD (~74.2 GB) to {target}")
        snapshot_download(
            repo_id="nguyendv02/ViMD_Dataset",
            repo_type="dataset",
            local_dir=target,
            max_workers=8,
        )

    if args.dataset in {"vispeech", "all"}:
        import gdown

        target = output / "ViSpeech"
        target.mkdir(parents=True, exist_ok=True)
        archive = target / "vispeech_dataset.zip"
        print(f"Downloading ViSpeech to {archive}")
        gdown.download(
            id="1-BbOHf42o6eBje2WqQiiRKMtNxmZiRf9",
            output=str(archive),
            quiet=False,
            fuzzy=True,
        )


if __name__ == "__main__":
    main()

