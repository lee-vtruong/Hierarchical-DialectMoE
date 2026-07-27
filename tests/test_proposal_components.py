import torch

from dialect_moe.proposal_losses import full_proposal_loss
from dialect_moe.proposal_model import FeatureFusion, ProposalOutput, ProsodySequenceEncoder
from dialect_moe.taxonomy import province_to_subregion


def test_taxonomy():
    assert province_to_subregion("Thừa Thiên Huế") == "NorthCentral"
    assert province_to_subregion("CanTho") == "MekongDelta"
    assert province_to_subregion("Hà Nội") == "RedRiverDelta"


def test_all_prosody_encoders():
    sequence = torch.randn(3, 20, 8)
    mask = torch.ones(3, 20, dtype=torch.bool)
    for encoder_type in ("mlp", "cnn1d", "bilstm", "transformer"):
        encoder = ProsodySequenceEncoder(8, 16, 12, encoder_type, 2, 4, 0.0)
        hidden, pooled = encoder(sequence, mask)
        assert hidden.shape == (3, 20, 12)
        assert pooled.shape == (3, 12)


def test_all_fusions():
    acoustic = torch.randn(3, 16)
    prosody = torch.randn(3, 8)
    acoustic_sequence = torch.randn(3, 10, 16)
    prosody_sequence = torch.randn(3, 6, 8)
    prosody_mask = torch.ones(3, 6, dtype=torch.bool)
    for fusion_type in ("concat", "gated", "bilinear", "cross_attention"):
        fusion = FeatureFusion(16, 8, 32, fusion_type, 0.0)
        output = fusion(
            acoustic, prosody, acoustic_sequence, prosody_sequence, prosody_mask
        )
        assert output.shape == (3, 32)


def test_full_loss_with_ctc_and_gender():
    batch_size = 2
    output = ProposalOutput(
        region_logits=torch.randn(batch_size, 3, requires_grad=True),
        subregion_logits=torch.randn(batch_size, 4, requires_grad=True),
        province_logits=torch.randn(batch_size, 6, requires_grad=True),
        gender_logits=torch.randn(batch_size, 2, requires_grad=True),
        asr_logits=torch.randn(batch_size, 12, 10, requires_grad=True),
        asr_output_lengths=torch.tensor([12, 12]),
        router_logits=torch.randn(batch_size, 4),
        router_probabilities=torch.softmax(torch.randn(batch_size, 4), -1),
        load_balance_loss=torch.tensor(1.0),
        fused_features=torch.randn(batch_size, 16),
    )
    batch = {
        "region_labels": torch.tensor([0, 1]),
        "subregion_labels": torch.tensor([0, 2]),
        "province_labels": torch.tensor([1, 4]),
        "gender_labels": torch.tensor([0, 1]),
        "ctc_targets": torch.tensor([1, 2, 3, 2, 1]),
        "ctc_target_lengths": torch.tensor([3, 2]),
    }
    matrix = torch.tensor(
        [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32
    )
    config = {
        "region_weight": 0.3,
        "subregion_weight": 0.5,
        "province_weight": 1.0,
        "gender_weight": 0.1,
        "asr_weight": 0.5,
        "hierarchy_weight": 0.1,
        "router_weight": 0.01,
        "load_balance_weight": 0.01,
    }
    loss, parts = full_proposal_loss(output, batch, config, matrix)
    loss.backward()
    assert torch.isfinite(loss)
    assert {"asr", "gender", "hierarchy"}.issubset(parts)

