from pathlib import Path

import pytest

from scripts.summarize_h11 import parse_artifact


def test_parse_h11_artifact():
    result = parse_artifact(
        Path(
            "outputs/h11_large_vi_prosody_seed44/"
            "metrics_test_best_region_accuracy.json"
        )
    )
    assert result == ("large", "prosody", 44, "region")


def test_parse_h11_rejects_unrelated_artifact():
    with pytest.raises(ValueError):
        parse_artifact(Path("outputs/acoustic_only_seed42/metrics_test.json"))
