import pytest
import torch
from torch import nn
from types import SimpleNamespace

from dialect_moe.labels import LabelVocabulary, build_province_to_region
from dialect_moe.losses import hierarchical_loss
from dialect_moe.model import (
    DialectMoEOutput,
    HierarchicalDialectMoE,
    soft_hierarchical_province_log_probs,
)


def test_province_to_region_mapping_is_complete_and_order_independent():
    regions = LabelVocabulary(["South", "North", "Central"])
    provinces = LabelVocabulary(["79", "11", "48"])
    mapping = build_province_to_region(
        [("79", "South"), ("11", "North"), ("48", "Central")],
        regions,
        provinces,
    )

    assert mapping[provinces.encode("11")] == regions.encode("North")
    assert mapping[provinces.encode("48")] == regions.encode("Central")
    assert mapping[provinces.encode("79")] == regions.encode("South")


def test_province_to_region_mapping_rejects_conflict_and_missing_class():
    regions = LabelVocabulary(["North", "South"])
    provinces = LabelVocabulary(["11", "79"])
    with pytest.raises(ValueError, match="multiple regions"):
        build_province_to_region(
            [("11", "North"), ("11", "South"), ("79", "South")],
            regions,
            provinces,
        )
    with pytest.raises(ValueError, match="missing a region"):
        build_province_to_region(
            [("11", "North")], regions, provinces
        )


def test_soft_hierarchical_probabilities_are_normalized_and_match_region_marginal():
    torch.manual_seed(18)
    region_logits = torch.randn(4, 3, requires_grad=True)
    province_logits = torch.randn(4, 6, requires_grad=True)
    mapping = torch.tensor([0, 0, 1, 1, 2, 2])

    joint_log_probs = soft_hierarchical_province_log_probs(
        region_logits, province_logits, mapping
    )
    joint_probs = joint_log_probs.exp()
    region_probs = torch.softmax(region_logits, dim=-1)

    assert torch.allclose(joint_probs.sum(dim=-1), torch.ones(4), atol=1e-6)
    for region_id in range(3):
        assert torch.allclose(
            joint_probs[:, mapping == region_id].sum(dim=-1),
            region_probs[:, region_id],
            atol=1e-6,
        )

    loss = torch.nn.functional.nll_loss(
        joint_log_probs, torch.tensor([0, 2, 4, 5])
    )
    loss.backward()
    assert region_logits.grad is not None
    assert province_logits.grad is not None
    assert torch.isfinite(region_logits.grad).all()
    assert torch.isfinite(province_logits.grad).all()


def test_h18_loss_uses_conditional_province_term_without_double_region_ce():
    conditional = torch.log_softmax(torch.randn(2, 4), dim=-1)
    output = DialectMoEOutput(
        region_logits=torch.randn(2, 2),
        province_logits=torch.randn(2, 4),
        router_logits=torch.zeros(2, 1),
        router_probabilities=torch.ones(2, 1),
        load_balance_loss=torch.tensor(0.0),
        conditional_province_log_probs=conditional,
    )
    province_labels = torch.tensor([0, 3])
    _, parts = hierarchical_loss(
        output,
        region_labels=torch.tensor([0, 1]),
        province_labels=province_labels,
        config={
            "region_weight": 0.4,
            "province_weight": 1.0,
            "router_weight": 0.0,
            "load_balance_weight": 0.0,
        },
    )

    expected = torch.nn.functional.nll_loss(conditional, province_labels)
    assert torch.allclose(parts["province_loss"], expected)


class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8, num_hidden_layers=2)
        self.projection = nn.Linear(1, 8)

    def freeze_feature_encoder(self):
        pass

    def forward(self, input_values, attention_mask=None, output_hidden_states=False):
        hidden = self.projection(input_values[:, :5, None])
        return SimpleNamespace(last_hidden_state=hidden, hidden_states=None)


def test_h18_model_uses_joint_region_province_posterior(monkeypatch):
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
        "gradient_checkpointing": False,
        "acoustic_dim": 8,
        "prosody_dim": 4,
        "fusion_dim": 10,
        "num_experts": 2,
        "top_k": 1,
        "expert_hidden_dim": 16,
        "dropout": 0.0,
        "province_head": {"type": "soft_hierarchical"},
    }
    mapping = [0, 0, 1, 1, 2, 2]
    model = HierarchicalDialectMoE(
        config, 3, 6, province_to_region=mapping
    )
    output = model(
        torch.randn(2, 10),
        torch.ones(2, 10, dtype=torch.long),
        torch.randn(2, 12),
    )
    province_probabilities = output.province_logits.exp()
    region_probabilities = torch.softmax(output.region_logits, dim=-1)

    assert output.conditional_province_log_probs is not None
    assert torch.allclose(
        province_probabilities.sum(dim=-1), torch.ones(2), atol=1e-6
    )
    for region_id in range(3):
        mask = torch.tensor(mapping) == region_id
        assert torch.allclose(
            province_probabilities[:, mask].sum(dim=-1),
            region_probabilities[:, region_id],
            atol=1e-6,
        )
        assert torch.allclose(
            output.conditional_province_log_probs[:, mask].exp().sum(dim=-1),
            torch.ones(2),
            atol=1e-6,
        )
    assert model.representation_diagnostics()["province_head"] == (
        "soft_hierarchical"
    )
    assert "province_to_region_index" not in model.state_dict()
