import json

from dialect_moe.config import load_config
from scripts.analyze_h14 import baseline_prediction, candidate_prediction


def test_h14_configs_are_locked_moe2():
    for seed in (42, 43, 44):
        suffix = "" if seed == 42 else f"_seed{seed}"
        config = load_config(
            f"configs/experiments/h14_large_vi_prosody_moe2{suffix}.yaml"
        )
        assert config["seed"] == seed
        assert config["model"]["backbone"] == "nguyenvulebinh/wav2vec2-large-vi"
        assert config["model"]["use_prosody"] is True
        assert config["model"]["use_moe"] is True
        assert config["model"]["use_hierarchical_router"] is True
        assert config["model"]["num_experts"] == 2
        assert config["model"]["top_k"] == 1
        assert config["loss"]["load_balance_weight"] == 0.001
        assert config["loss"]["router_weight"] == 0.0
        assert config["training"]["output_dir"].endswith(f"seed{seed}")


def test_h14_artifact_paths():
    outputs = __import__("pathlib").Path("outputs")
    assert "h11_large_vi_prosody_seed42" in str(baseline_prediction(outputs, 42))
    assert "h14_large_vi_prosody_moe2_seed44" in str(candidate_prediction(outputs, 44))
