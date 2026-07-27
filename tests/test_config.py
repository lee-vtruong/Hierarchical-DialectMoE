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

