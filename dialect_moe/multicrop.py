from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .data import DialectCollator
from .labels import normalize_region
from .multicrop_utils import crop_starts
from .prosody import extract_prosody, prosody_feature_names
from .spectral import SPECTRAL_FEATURE_NAMES, extract_spectral


class MultiCropDialectCollator(DialectCollator):
    def __init__(
        self,
        *args,
        strategy: str,
        uniform_crops: int = 3,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.strategy = strategy
        self.uniform_crops = uniform_crops

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        crop_arrays = []
        crop_sample_indices = []
        crop_counts = []
        for sample_index, example in enumerate(examples):
            waveform = self._decode_audio(example[self.audio_column])
            starts = crop_starts(
                len(waveform),
                self.max_length,
                self.strategy,
                self.uniform_crops,
            )
            crop_counts.append(len(starts))
            for start in starts:
                crop_arrays.append(waveform[start : start + self.max_length])
                crop_sample_indices.append(sample_index)

        processed = self.extractor(
            crop_arrays,
            sampling_rate=self.sample_rate,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
            return_tensors="pt",
        )
        prosody_rows, spectral_rows = [], []
        for array in crop_arrays:
            waveform = torch.from_numpy(np.asarray(array, dtype=np.float32))
            prosody_rows.append(
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
            spectral_rows.append(
                extract_spectral(waveform, self.sample_rate)
                if self.use_spectral
                else torch.zeros(len(SPECTRAL_FEATURE_NAMES), dtype=torch.float32)
            )
        return {
            "input_values": processed.input_values,
            "attention_mask": processed.attention_mask,
            "prosody": torch.stack(prosody_rows),
            "spectral": torch.stack(spectral_rows),
            "crop_sample_indices": torch.tensor(crop_sample_indices, dtype=torch.long),
            "crop_counts": crop_counts,
            "region_labels": torch.tensor(
                [
                    self.region_vocab.encode(
                        normalize_region(example[self.region_column])
                    )
                    for example in examples
                ],
                dtype=torch.long,
            ),
            "province_labels": torch.tensor(
                [
                    self.province_vocab.encode(example[self.province_column])
                    for example in examples
                ],
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
