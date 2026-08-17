import json

import pytest

from scripts.summarize_h18 import aggregate, metric_row


def fake_metrics(accuracy: float) -> dict:
    report = {"macro avg": {"f1-score": accuracy - 0.02}}
    return {
        "region": {
            "accuracy": 0.95,
            "balanced_accuracy": 0.94,
            "classification_report": report,
        },
        "province": {
            "accuracy": accuracy,
            "balanced_accuracy": accuracy - 0.01,
            "classification_report": report,
        },
        "hierarchy": {
            "prediction_region_consistency": 0.99,
            "province_cross_region_error_rate": 0.04,
        },
        "representation": {"province_head": "soft_hierarchical"},
    }


def test_h18_metric_row_and_aggregate(tmp_path):
    rows = []
    for seed, accuracy in ((42, 0.6), (43, 0.7)):
        directory = tmp_path / f"h18_soft_hierarchy_seed{seed}"
        directory.mkdir()
        path = directory / "metrics_valid_best_province_accuracy.json"
        path.write_text(json.dumps(fake_metrics(accuracy)), encoding="utf-8")
        rows.append(metric_row(path))

    assert rows[0]["seed"] == 42
    assert rows[0]["province_head"] == "soft_hierarchical"
    summary = aggregate(rows)
    assert summary["runs"] == 2
    assert summary["province_accuracy_mean"] == pytest.approx(0.65)
    assert summary["prediction_region_consistency_mean"] == pytest.approx(0.99)

