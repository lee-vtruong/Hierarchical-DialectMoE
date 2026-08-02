from pathlib import Path

from dialect_moe.config import load_config


def test_experiment_config_inherits_base():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "experiments" / "acoustic_only.yaml")
    assert config["data"]["dataset_name"] == "nguyendv02/ViMD_Dataset"
    assert config["model"]["backbone"] == "facebook/wav2vec2-base"
    assert config["model"]["use_prosody"] is False
    assert config["model"]["use_moe"] is False
    assert config["training"]["batch_size"] == 4
    assert config["training"]["output_dir"] == "outputs/acoustic_only_seed42"


def test_balanced_moe_override():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "experiments" / "moe_4_balanced.yaml")
    assert config["model"]["num_experts"] == 4
    assert config["model"]["top_k"] == 2
    assert config["loss"]["router_weight"] == 0.0
    assert config["loss"]["load_balance_weight"] == 0.1


def test_legacy_acoustic_only_does_not_opt_into_dynamic_h5_fusion():
    root = Path(__file__).resolve().parents[1]
    legacy = load_config(root / "configs" / "experiments" / "acoustic_only.yaml")
    h5 = load_config(
        root / "configs" / "experiments" / "h5_acoustic_pitch_energy.yaml"
    )
    assert "use_spectral" not in legacy["model"]
    assert "prosody_feature_set" not in legacy["model"]
    assert h5["model"]["use_spectral"] is False
    assert h5["model"]["prosody_feature_set"] == "pitch_energy"


def test_h11_vietnamese_backbone_matrix():
    root = Path(__file__).resolve().parents[1]
    cases = {
        "h11_base_vi_acoustic.yaml": (
            "nguyenvulebinh/wav2vec2-base-vi",
            False,
            4,
            8,
        ),
        "h11_base_vi_prosody_seed44.yaml": (
            "nguyenvulebinh/wav2vec2-base-vi",
            True,
            4,
            8,
        ),
        "h11_large_vi_acoustic_seed43.yaml": (
            "nguyenvulebinh/wav2vec2-large-vi",
            False,
            2,
            16,
        ),
        "h11_large_vi_prosody.yaml": (
            "nguyenvulebinh/wav2vec2-large-vi",
            True,
            2,
            16,
        ),
    }
    for filename, expected in cases.items():
        config = load_config(root / "configs" / "experiments" / filename)
        backbone, prosody, batch_size, accumulation = expected
        assert config["model"]["backbone"] == backbone
        assert config["model"]["use_safetensors"] is False
        assert config["model"]["use_prosody"] is prosody
        assert config["data"]["split_manifest"].endswith(
            "vimd_speaker_disjoint_seed42.csv"
        )
        assert config["training"]["batch_size"] == batch_size
        assert config["training"]["gradient_accumulation_steps"] == accumulation
        assert config["training"]["compact_best_checkpoints"] is True
        assert config["training"]["save_best_loss_checkpoint"] is False
        assert config["training"]["save_legacy_best_checkpoint"] is False
