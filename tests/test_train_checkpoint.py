import torch

from dialect_moe.utils import inference_checkpoint


def test_inference_checkpoint_drops_training_state():
    state = {
        "epoch": 3,
        "model": {"weight": torch.ones(2)},
        "optimizer": {"state": {1: {"step": 4}}},
        "scheduler": {"last_epoch": 3},
        "best_loss": 1.2,
        "best_region_accuracy": 0.9,
        "best_province_accuracy": 0.4,
        "metrics": {"loss": 1.2},
    }

    compact = inference_checkpoint(state)

    assert compact["model"] is state["model"]
    assert "optimizer" not in compact
    assert "scheduler" not in compact
    assert compact["best_province_accuracy"] == 0.4
