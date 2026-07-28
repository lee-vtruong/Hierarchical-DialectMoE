from __future__ import annotations

from dataclasses import dataclass
import io
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
import torch
from datasets import Audio, DatasetDict, load_dataset
from transformers import AutoFeatureExtractor

from .labels import LabelVocabulary, normalize_region
from .prosody import PROSODY_FEATURE_NAMES, extract_prosody


@dataclass
class DatasetBundle:
    datasets: DatasetDict
    region_vocab: LabelVocabulary
    province_vocab: LabelVocabulary


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

    region_values: list[str] = []
    province_values: list[object] = []
    for dataset in datasets.values():
        region_values.extend(normalize_region(value) for value in dataset.unique(data_config["region_column"]))
        province_values.extend(dataset.unique(data_config["province_column"]))

    return DatasetBundle(
        datasets=datasets,
        region_vocab=LabelVocabulary(region_values),
        province_vocab=LabelVocabulary(province_values),
    )


class DialectCollator:
    def __init__(
        self,
        backbone: str,
        data_config: dict[str, Any],
        region_vocab: LabelVocabulary,
        province_vocab: LabelVocabulary,
        use_prosody: bool = True,
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
        if self.use_prosody:
            prosody = torch.stack(
                [extract_prosody(torch.from_numpy(array), self.sample_rate) for array in arrays]
            )
        else:
            prosody = torch.zeros(
                len(arrays), len(PROSODY_FEATURE_NAMES), dtype=torch.float32
            )
        return {
            "input_values": processed.input_values,
            "attention_mask": processed.attention_mask,
            "prosody": prosody,
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
