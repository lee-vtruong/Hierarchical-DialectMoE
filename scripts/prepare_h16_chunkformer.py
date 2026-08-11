from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import sys

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dialect_moe.config import load_config
from dialect_moe.data import load_vimd
from dialect_moe.labels import normalize_region
from scripts.convert_h16_predictions import safe_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the audited ViMD split for official ChunkFormer classification."
    )
    parser.add_argument("--config", default="configs/experiments/h11_large_vi_prosody.yaml")
    parser.add_argument("--destination", default="data/h16_chunkformer")
    parser.add_argument("--max-samples", type=int)
    return parser.parse_args()


def decode_audio(audio: dict, sample_rate: int) -> np.ndarray:
    source = io.BytesIO(audio["bytes"]) if audio.get("bytes") is not None else audio["path"]
    waveform, original_rate = sf.read(source, dtype="float32", always_2d=False)
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(1)
    if int(original_rate) != sample_rate:
        waveform = librosa.resample(waveform, orig_sr=int(original_rate), target_sr=sample_rate)
    return np.asarray(waveform, dtype=np.float32)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    bundle = load_vimd(config, max_samples=args.max_samples)
    destination = Path(args.destination).resolve()
    sample_rate = int(config["data"]["sample_rate"])
    audio_column = config["data"]["audio_column"]
    filename_column = config["data"].get("filename_column", "filename")
    speaker_column = config["data"].get("speaker_column", "speakerID")
    region_column = config["data"]["region_column"]
    province_column = config["data"]["province_column"]
    metadata_path = destination / "metadata.jsonl"
    destination.mkdir(parents=True, exist_ok=True)

    metadata_rows = []
    split_map = {"train": "train", "valid": "dev", "validation": "dev", "test": "test"}
    for source_split, dataset in bundle.datasets.items():
        target_split = split_map[source_split]
        split_dir = destination / target_split
        audio_dir = split_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        tsv_path = split_dir / "data.tsv"
        with tsv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["key", "wav", "region_label", "province_label"],
                delimiter="\t",
            )
            writer.writeheader()
            for index, example in enumerate(tqdm(dataset, desc=f"Exporting {source_split}")):
                filename = str(example.get(filename_column, ""))
                key = safe_key(source_split, index, filename)
                audio_path = audio_dir / f"{key}.flac"
                if not audio_path.is_file():
                    sf.write(audio_path, decode_audio(example[audio_column], sample_rate), sample_rate, format="FLAC")
                region_id = bundle.region_vocab.encode(normalize_region(example[region_column]))
                province_id = bundle.province_vocab.encode(example[province_column])
                writer.writerow(
                    {"key": key, "wav": str(audio_path), "region_label": region_id, "province_label": province_id}
                )
                metadata_rows.append(
                    {
                        "key": key,
                        "split": target_split,
                        "filename": filename,
                        "speaker_id": str(example.get(speaker_column, "")),
                        "region_true_id": region_id,
                        "region_true": bundle.region_vocab.decode(region_id),
                        "province_true_id": province_id,
                        "province_true": bundle.province_vocab.decode(province_id),
                    }
                )

    with metadata_path.open("w", encoding="utf-8") as handle:
        for row in metadata_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    labels = {"regions": bundle.region_vocab.labels, "provinces": bundle.province_vocab.labels}
    (destination / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(metadata_rows)} rows to {destination}")


if __name__ == "__main__":
    main()
