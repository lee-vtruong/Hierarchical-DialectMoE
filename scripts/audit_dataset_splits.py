from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys

from datasets import Audio, load_dataset
import soundfile as sf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dialect_moe.config import load_config
from dialect_moe.labels import normalize_region
from dialect_moe.split_audit import (
    UtteranceRecord,
    duplicate_values_by_split,
    speaker_label_conflicts,
)
from dialect_moe.utils import save_json


def load_raw_splits(config: dict):
    data_config = config["data"]
    local_dir = Path(data_config.get("local_dir", ""))
    data_files = {
        split: [
            str(path)
            for path in sorted((local_dir / "data").glob(f"{split}-*.parquet"))
        ]
        for split in ("train", "valid", "test")
    }
    data_files = {split: files for split, files in data_files.items() if files}
    if not data_files:
        raise FileNotFoundError(f"No parquet shards found under {local_dir / 'data'}")
    datasets = load_dataset(
        "parquet", data_files=data_files, cache_dir=data_config.get("cache_dir")
    )
    return datasets.cast_column(
        data_config["audio_column"],
        Audio(sampling_rate=data_config["sample_rate"], decode=False),
    )


def audio_metadata(audio: dict, mode: str) -> tuple[float | None, str | None]:
    if mode == "none":
        return None, None
    payload = audio.get("bytes")
    path = audio.get("path")
    source = io.BytesIO(payload) if payload is not None else path
    duration = None
    if source is not None:
        info = sf.info(source)
        duration = float(info.frames / info.samplerate)
    digest = hashlib.sha256(payload).hexdigest() if mode == "sha256" and payload else None
    return duration, digest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit speaker/file/audio leakage across ViMD splits."
    )
    parser.add_argument("--config", default="configs/vimd_moe.yaml")
    parser.add_argument("--output-dir", default="outputs/h6_split_audit")
    parser.add_argument(
        "--audio-mode",
        choices=["none", "duration", "sha256"],
        default="none",
        help="duration reads audio headers; sha256 also hashes embedded bytes.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    data_config = config["data"]
    datasets = load_raw_splits(config)
    records: list[UtteranceRecord] = []
    split_summary = {}
    for split, dataset in datasets.items():
        speakers = set()
        duration_total = 0.0
        for index, row in enumerate(tqdm(dataset, desc=f"Audit {split}")):
            duration, digest = audio_metadata(row[data_config["audio_column"]], args.audio_mode)
            speaker = str(row.get(data_config.get("speaker_column", "speakerID"), ""))
            record = UtteranceRecord(
                original_split=split,
                row_index=index,
                filename=str(row.get(data_config.get("filename_column", "filename"), "")),
                speaker_id=speaker,
                region=normalize_region(row[data_config["region_column"]]),
                province=str(row[data_config["province_column"]]),
                duration_seconds=duration,
                audio_sha256=digest,
            )
            records.append(record)
            speakers.add(speaker)
            duration_total += duration or 0.0
        split_summary[split] = {
            "utterances": len(dataset),
            "speakers": len(speakers),
            "duration_seconds": duration_total if args.audio_mode != "none" else None,
        }

    overlap_attributes = ["speaker_id", "filename"]
    if args.audio_mode == "sha256":
        overlap_attributes.append("audio_sha256")
    overlaps = {
        attribute: duplicate_values_by_split(records, attribute)
        for attribute in overlap_attributes
    }
    conflicts = speaker_label_conflicts(records)
    summary = {
        "audio_mode": args.audio_mode,
        "splits": split_summary,
        "cross_split_overlap": {
            attribute: {
                "unique_values": len(values),
                "affected_utterances": sum(
                    sum(counts.values()) for counts in values.values()
                ),
            }
            for attribute, values in overlaps.items()
        },
        "speaker_label_conflicts": len(conflicts),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(summary, output_dir / "audit_summary.json")
    save_json(overlaps, output_dir / "overlap_details.json")
    save_json(conflicts, output_dir / "speaker_label_conflicts.json")
    with (output_dir / "records.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(UtteranceRecord.__annotations__))
        writer.writeheader()
        writer.writerows(record.__dict__ for record in records)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Audit artifacts: {output_dir}")


if __name__ == "__main__":
    main()
