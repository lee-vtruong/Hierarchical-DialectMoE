import torch

from dialect_moe.losses import hierarchical_loss
from dialect_moe.model import DialectMoEOutput, SparseMixtureOfExperts
from dialect_moe.prosody import PROSODY_FEATURE_NAMES, extract_prosody
from dialect_moe.prosody import PITCH_ENERGY_FEATURE_NAMES
from dialect_moe.spectral import SPECTRAL_FEATURE_NAMES, extract_spectral


def test_prosody_shape_and_finite():
    sample_rate = 16000
    time = torch.arange(sample_rate, dtype=torch.float32) / sample_rate
    waveform = 0.1 * torch.sin(2 * torch.pi * 180 * time)
    features = extract_prosody(waveform, sample_rate)
    assert features.shape == (len(PROSODY_FEATURE_NAMES),)
    assert torch.isfinite(features).all()
    compact = extract_prosody(waveform, sample_rate, feature_set="pitch_energy")
    assert compact.shape == (len(PITCH_ENERGY_FEATURE_NAMES),)
    assert torch.isfinite(compact).all()


def test_spectral_shape_finite_and_volume_robust():
    sample_rate = 16000
    time = torch.arange(sample_rate, dtype=torch.float32) / sample_rate
    waveform = (
        0.1 * torch.sin(2 * torch.pi * 180 * time)
        + 0.03 * torch.sin(2 * torch.pi * 1200 * time)
    )
    features = extract_spectral(waveform, sample_rate)
    louder_features = extract_spectral(2 * waveform, sample_rate)
    assert features.shape == (len(SPECTRAL_FEATURE_NAMES),)
    assert torch.isfinite(features).all()
    assert torch.allclose(features, louder_features, atol=1e-4, rtol=1e-4)


def test_sparse_moe_shape_and_gradient():
    moe = SparseMixtureOfExperts(16, 32, num_experts=4, top_k=2, dropout=0.0)
    features = torch.randn(5, 16, requires_grad=True)
    router_logits = torch.randn(5, 4, requires_grad=True)
    output, balance_loss, probabilities = moe(features, router_logits)
    (output.mean() + balance_loss).backward()
    assert output.shape == features.shape
    assert probabilities.shape == (5, 4)
    assert features.grad is not None
    assert router_logits.grad is not None


def test_sparse_moe_mixed_precision_dtype():
    moe = SparseMixtureOfExperts(8, 16, num_experts=4, top_k=2, dropout=0.0).half()
    features = torch.randn(3, 8, dtype=torch.float16)
    # Simulate a numerically stable router softmax producing FP32 weights while
    # expert activations remain FP16 under AMP.
    router_logits = torch.randn(3, 4, dtype=torch.float32)
    output, _, _ = moe(features, router_logits)
    assert output.dtype == torch.float16
    assert output.shape == features.shape


def test_hierarchical_loss_is_finite():
    output = DialectMoEOutput(
        region_logits=torch.randn(4, 3),
        province_logits=torch.randn(4, 6),
        router_logits=torch.randn(4, 2),
        router_probabilities=torch.softmax(torch.randn(4, 2), dim=-1),
        load_balance_loss=torch.tensor(1.0),
    )
    loss, parts = hierarchical_loss(
        output,
        torch.tensor([0, 1, 2, 0]),
        torch.tensor([0, 1, 2, 3]),
        {
            "region_weight": 0.4,
            "province_weight": 1.0,
            "router_weight": 0.1,
            "load_balance_weight": 0.01,
        },
    )
    assert torch.isfinite(loss)
    assert set(parts) == {
        "loss",
        "region_loss",
        "province_loss",
        "router_entropy",
        "load_balance_loss",
    }
