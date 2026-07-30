import pytest
import torch

from dialect_moe.multicrop_utils import aggregate_logits, crop_starts


def test_crop_starts():
    assert crop_starts(10, 20, "uniform", 3) == [0]
    assert crop_starts(32, 20, "first", 3) == [0]
    assert crop_starts(32, 20, "start_end", 3) == [0, 12]
    assert crop_starts(32, 20, "uniform", 3) == [0, 6, 12]


def test_crop_starts_rejects_bad_uniform_count():
    with pytest.raises(ValueError, match="at least 2"):
        crop_starts(30, 20, "uniform", 1)


def test_aggregate_logits_mean_per_sample():
    logits = torch.tensor([[1.0, 3.0], [3.0, 5.0], [10.0, 20.0]])
    indices = torch.tensor([0, 0, 1])
    result = aggregate_logits(logits, indices, num_samples=2)
    assert torch.allclose(result, torch.tensor([[2.0, 4.0], [10.0, 20.0]]))
