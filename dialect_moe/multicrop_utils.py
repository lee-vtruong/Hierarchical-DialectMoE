from __future__ import annotations

import numpy as np
import torch


def crop_starts(
    num_samples: int,
    max_samples: int,
    strategy: str,
    uniform_crops: int = 3,
) -> list[int]:
    if num_samples <= 0 or max_samples <= 0:
        raise ValueError("num_samples and max_samples must be positive")
    if num_samples <= max_samples or strategy == "first":
        return [0]
    max_start = num_samples - max_samples
    if strategy == "start_end":
        return [0, max_start]
    if strategy == "uniform":
        if uniform_crops < 2:
            raise ValueError("uniform_crops must be at least 2")
        return sorted(
            {
                int(round(value))
                for value in np.linspace(0, max_start, uniform_crops)
            }
        )
    raise ValueError(f"Unknown crop strategy: {strategy}")


def aggregate_logits(
    crop_logits: torch.Tensor,
    sample_indices: torch.Tensor,
    num_samples: int,
) -> torch.Tensor:
    if crop_logits.ndim != 2:
        raise ValueError("crop_logits must have shape [crops, classes]")
    if sample_indices.ndim != 1 or len(sample_indices) != len(crop_logits):
        raise ValueError("sample_indices must align with crop_logits")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    output = crop_logits.new_zeros((num_samples, crop_logits.shape[1]))
    output.index_add_(0, sample_indices, crop_logits)
    counts = torch.bincount(sample_indices, minlength=num_samples).to(
        device=crop_logits.device, dtype=crop_logits.dtype
    )
    if torch.any(counts == 0):
        raise ValueError("Every sample must have at least one crop")
    return output / counts.unsqueeze(1)
