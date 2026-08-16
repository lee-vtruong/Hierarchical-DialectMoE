import json

import pytest

from scripts.summarize_h17 import aggregate_rows, metric_row


def fake_metrics(province_accuracy: float, macro_f1: float) -> dict:
    report = {"macro avg": {"f1-score": macro_f1}}
    return {
        "region": {
            "accuracy": 0.9,
            "balanced_accuracy": 0.89,
            "classification_report": report,
        },
        "province": {
            "accuracy": province_accuracy,
            "balanced_accuracy": province_accuracy - 0.01,
            "classification_report": report,
        },
        "representation": {
            "acoustic_pooling": "attentive_statistics",
            "layer_mix_enabled": True,
            "layer_weights": [0.4, 0.6],
        },
    }


def test_metric_row_and_aggregate(tmp_path):
    rows = []
    for seed, accuracy in ((42, 0.6), (43, 0.7)):
        directory = tmp_path / f"h17b_layermix_asp_seed{seed}"
        directory.mkdir()
        path = directory / "metrics_valid_best_province_accuracy.json"
        path.write_text(
            json.dumps(fake_metrics(accuracy, accuracy - 0.02)), encoding="utf-8"
        )
        rows.append(metric_row(path))

    assert rows[0]["variant"] == "h17b_layermix_asp"
    assert rows[0]["seed"] == 42
    assert rows[0]["layer_mix_enabled"] is True
    aggregate = aggregate_rows(rows)[0]
    assert aggregate["runs"] == 2
    assert aggregate["province_accuracy_mean"] == pytest.approx(0.65)
