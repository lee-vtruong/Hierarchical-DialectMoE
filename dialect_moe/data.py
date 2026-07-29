from __future__ import annotations

from dataclasses import dataclass
import csv
import io
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
import torch
from datasets import Audio, DatasetDict, concatenate_datasets, load_dataset
from transformers import AutoFeatureExtractor

from .labels import LabelVocabulary, normalize_region
from .prosody import extract_prosody, prosody_feature_names
from .spectral import SPECTRAL_FEATURE_NAMES, extract_spectral


@dataclass
class DatasetBundle:
    datasets: DatasetDict
    region_vocab: LabelVocabulary
    province_vocab: LabelVocabulary


def _apply_split_manifest(
    datasets: DatasetDict, manifest_path: str | Path
) -> DatasetDict:
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {path}")
    selections: dict[str, dict[str, list[int]]] = {}
    seen_rows: set[tuple[str, int]] = set()
    speaker_splits: dict[str, str] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            original_split = row["original_split"]
            new_split = row["new_split"]
            row_index = int(row["row_index"])
            speaker = row.get("speaker_id", "")
            if original_split not in datasets:
                raise ValueError(f"Unknown original split in manifest: {original_split}")
            if not 0 <= row_index < len(datasets[original_split]):
                raise IndexError(
                    f"Manifest row index {row_index} out of range for {original_split}"
                )
            key = (original_split, row_index)
            if key in seen_rows:
                raise ValueError(f"Duplicate manifest row: {key}")
            seen_rows.add(key)
            if speaker:
                previous = speaker_splits.setdefault(speaker, new_split)
                if previous != new_split:
                    raise ValueError(
                        f"Speaker {speaker!r} appears in both {previous} and {new_split}"
                    )
            selections.setdefault(new_split, {}).setdefault(original_split, []).append(
                row_index
            )
    expected_rows = sum(len(dataset) for dataset in datasets.values())
    if len(seen_rows) != expected_rows:
        raise ValueError(
            f"Manifest covers {len(seen_rows)} rows, expected {expected_rows}. "
            "Refuse to train on a partial or stale manifest."
        )
    rebuilt = {}
    for new_split, original_selections in selections.items():
        parts = [
            datasets[original_split].select(indices)
            for original_split, indices in sorted(original_selections.items())
            if indices
        ]
        if parts:
            rebuilt[new_split] = (
                parts[0] if len(parts) == 1 else concatenate_datasets(parts)
            )
    required = {"train", "valid", "test"}
    if not required.issubset(rebuilt):
        raise ValueError(
            f"Manifest must create train/valid/test; found {sorted(rebuilt)}"
        )
    return DatasetDict(rebuilt)


def load_vimd(config: dict[str, Any], max_samples: int | None = None) -> DatasetBundle:
    data_config = config["data"]
    local_dir = Path(data_config.get("local_dir", ""))
    if local_dir.is_dir():
        data_files = {
            split: [str(path) for path in sorted((local_dir / "data").glob(f"{split}-*.parquet"))]
            for split in ("train", "valid", "test")
        }
        data_files = {split: files for split, files in data_files.items() if files}
        if not data_files:
            raise FileNotFoundError(f"No parquet shards found under {local_dir / 'data'}")
        datasets = load_dataset(
            "parquet", data_files=data_files, cache_dir=data_config.get("cache_dir")
        )
    else:
        datasets = load_dataset(
            data_config["dataset_name"],
            cache_dir=data_config.get("cache_dir"),
        )
    # Build label vocabularies on the original Arrow tables. Calling unique()
    # after select()/concatenate_datasets() may flatten indexed audio columns;
    # large embedded-audio arrays can then overflow Arrow's 32-bit offsets.
    region_values: list[str] = []
    province_values: list[object] = []
    for dataset in datasets.values():
        region_values.extend(
            normalize_region(value)
            for value in dataset.unique(data_config["region_column"])
        )
        province_values.extend(dataset.unique(data_config["province_column"]))
    region_vocab = LabelVocabulary(region_values)
    province_vocab = LabelVocabulary(province_values)

    if data_config.get("split_manifest"):
        datasets = _apply_split_manifest(
            datasets, data_config["split_manifest"]
        )
    audio_column = data_config["audio_column"]
    # Keep encoded bytes instead of asking datasets to decode with TorchCodec.
    # This avoids a hard Torch/TorchCodec/CUDA/FFmpeg compatibility dependency.
    datasets = datasets.cast_column(
        audio_column,
        Audio(sampling_rate=data_config["sample_rate"], decode=False),
    )

    if max_samples:
        datasets = DatasetDict(
            {
                split: dataset.select(range(min(max_samples, len(dataset))))
                for split, dataset in datasets.items()
            }
        )

    return DatasetBundle(
        datasets=datasets,
        region_vocab=region_vocab,
        province_vocab=province_vocab,
    )


class DialectCollator:
    def __init__(
        self,
        backbone: str,
        data_config: dict[str, Any],
        region_vocab: LabelVocabulary,
        province_vocab: LabelVocabulary,
        use_prosody: bool = True,
        use_spectral: bool = False,
        prosody_feature_set: str = "legacy",
    ):
        self.extractor = AutoFeatureExtractor.from_pretrained(backbone)
        self.audio_column = data_config["audio_column"]
        self.region_column = data_config["region_column"]
        self.province_column = data_config["province_column"]
        self.province_name_column = data_config.get("province_name_column", "province_name")
        self.speaker_column = data_config.get("speaker_column", "speakerID")
        self.filename_column = data_config.get("filename_column", "filename")
        self.sample_rate = int(data_config["sample_rate"])
        self.max_length = int(float(data_config["max_seconds"]) * self.sample_rate)
        self.region_vocab = region_vocab
        self.province_vocab = province_vocab
        self.use_prosody = use_prosody
        self.use_spectral = use_spectral
        self.prosody_feature_set = prosody_feature_set
        self.feature_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def _decode_audio(self, audio: dict[str, Any]) -> np.ndarray:
        source: io.BytesIO | str
        if audio.get("bytes") is not None:
            source = io.BytesIO(audio["bytes"])
        elif audio.get("path"):
            source = audio["path"]
        else:
            raise ValueError("Audio sample contains neither embedded bytes nor a path")
        waveform, original_rate = sf.read(source, dtype="float32", always_2d=False)
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        if int(original_rate) != self.sample_rate:
            waveform = librosa.resample(
                waveform,
                orig_sr=int(original_rate),
                target_sr=self.sample_rate,
            )
        return np.asarray(waveform, dtype=np.float32)

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        arrays = [
            self._decode_audio(example[self.audio_column])[: self.max_length]
            for example in examples
        ]
        processed = self.extractor(
            arrays,
            sampling_rate=self.sample_rate,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
            return_tensors="pt",
        )
        prosody_rows, spectral_rows = [], []
        for example, array in zip(examples, arrays):
            cache_key = str(example.get(self.filename_column, ""))
            cached = self.feature_cache.get(cache_key) if cache_key else None
            if cached is None:
                waveform = torch.from_numpy(array)
                prosody_row = (
                    extract_prosody(
                        waveform,
                        self.sample_rate,
                        feature_set=self.prosody_feature_set,
                    )
                    if self.use_prosody
                    else torch.zeros(
                        len(prosody_feature_names(self.prosody_feature_set)),
                        dtype=torch.float32,
                    )
                )
                spectral_row = (
                    extract_spectral(waveform, self.sample_rate)
                    if self.use_spectral
                    else torch.zeros(
                        len(SPECTRAL_FEATURE_NAMES), dtype=torch.float32
                    )
                )
                if cache_key:
                    self.feature_cache[cache_key] = (prosody_row, spectral_row)
            else:
                prosody_row, spectral_row = cached
            prosody_rows.append(prosody_row)
            spectral_rows.append(spectral_row)
        prosody = torch.stack(prosody_rows)
        spectral = torch.stack(spectral_rows)
        return {
            "input_values": processed.input_values,
            "attention_mask": processed.attention_mask,
            "prosody": prosody,
            "spectral": spectral,
            "region_labels": torch.tensor(
                [self.region_vocab.encode(normalize_region(x[self.region_column])) for x in examples],
                dtype=torch.long,
            ),
            "province_labels": torch.tensor(
                [self.province_vocab.encode(x[self.province_column]) for x in examples],
                dtype=torch.long,
            ),
            "filenames": [
                str(example.get(self.filename_column, "")) for example in examples
            ],
            "speaker_ids": [
                str(example.get(self.speaker_column, "")) for example in examples
            ],
            "province_names": [
                str(example.get(self.province_name_column, "")) for example in examples
            ],
        }
