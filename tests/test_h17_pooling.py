import torch
from torch import nn
from types import SimpleNamespace
import pytest

from dialect_moe.model import (
    AttentiveStatisticsPooling,
    HierarchicalDialectMoE,
    LayerMix,
)


def test_layer_mix_starts_as_uniform_average_and_learns():
    mixer = LayerMix(3)
    hidden_states = tuple(
        torch.full((2, 4, 5), float(value)) for value in (1, 2, 3)
    )
    output = mixer(hidden_states)

    assert torch.allclose(output, torch.full_like(output, 2.0))
    assert torch.allclose(mixer.weights, torch.full((3,), 1 / 3))

    output.sum().backward()
    assert mixer.layer_logits.grad is not None
    assert torch.isfinite(mixer.layer_logits.grad).all()


def test_attentive_statistics_pooling_ignores_padding():
    torch.manual_seed(17)
    pooling = AttentiveStatisticsPooling(6, 4, 5, dropout=0.0).eval()
    valid = torch.randn(2, 3, 6)
    padding_a = torch.zeros(2, 2, 6)
    padding_b = torch.full((2, 2, 6), 1000.0)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 0, 0]], dtype=torch.bool)

    output_a = pooling(torch.cat([valid, padding_a], dim=1), mask)
    output_b = pooling(torch.cat([valid, padding_b], dim=1), mask)

    assert output_a.shape == (2, 5)
    assert torch.allclose(output_a, output_b, atol=1e-6)
    assert torch.isfinite(output_a).all()


def test_attentive_statistics_pooling_backpropagates():
    pooling = AttentiveStatisticsPooling(8, 4, 8, dropout=0.0)
    sequence = torch.randn(3, 7, 8, requires_grad=True)
    output = pooling(sequence)
    output.square().mean().backward()

    assert sequence.grad is not None
    assert torch.isfinite(sequence.grad).all()


class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=12, num_hidden_layers=4)
        self.projection = nn.Linear(1, 12)

    def gradient_checkpointing_enable(self):
        pass

    def freeze_feature_encoder(self):
        pass

    def forward(self, input_values, attention_mask=None, output_hidden_states=False):
        hidden = self.projection(input_values[:, :5, None])
        states = tuple(hidden + index for index in range(5))
        return SimpleNamespace(
            last_hidden_state=states[-1],
            hidden_states=states if output_hidden_states else None,
        )


def test_h17_full_model_forward(monkeypatch):
    monkeypatch.setattr(
        "dialect_moe.model.AutoModel.from_pretrained",
        lambda *args, **kwargs: FakeBackbone(),
    )
    config = {
        "backbone": "fake",
        "use_acoustic": True,
        "use_prosody": True,
        "use_moe": False,
        "use_hierarchical_router": False,
        "freeze_feature_encoder": True,
        "gradient_checkpointing": True,
        "acoustic_dim": 8,
        "prosody_dim": 4,
        "fusion_dim": 10,
        "num_experts": 2,
        "top_k": 1,
        "expert_hidden_dim": 16,
        "dropout": 0.0,
        "layer_mix": {"enabled": True, "last_n_layers": 3},
        "acoustic_pooling": {
            "type": "attentive_statistics",
            "attention_hidden_dim": 4,
        },
    }
    model = HierarchicalDialectMoE(config, num_regions=3, num_provinces=6)
    output = model(
        input_values=torch.randn(2, 10),
        attention_mask=torch.tensor(
            [[1] * 10, [1] * 7 + [0] * 3], dtype=torch.long
        ),
        prosody=torch.randn(2, 12),
    )

    assert output.region_logits.shape == (2, 3)
    assert output.province_logits.shape == (2, 6)
    diagnostics = model.representation_diagnostics()
    assert diagnostics["acoustic_pooling"] == "attentive_statistics"
    assert diagnostics["layer_mix_last_n"] == 3
    assert sum(diagnostics["layer_weights"]) == pytest.approx(1.0)
