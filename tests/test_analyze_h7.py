import json

import pytest

from scripts.analyze_h7 import (
    aggregate_provinces,
    analyse_seed,
    calibration_metrics,
)


def make_row(index, truth, prediction, probabilities):
    return {
        "filename": f"{index}.wav",
        "speaker_id": f"speaker-{index}",
        "province_true_id": truth,
        "province_true": f"p{truth}",
        "province_pred_id": prediction,
        "province_pred": f"p{prediction}",
        "province_probabilities": probabilities,
    }


def test_h7_seed_analysis_and_aggregation():
    baseline_rows = [
        make_row(0, 0, 1, [0.2, 0.8]),
        make_row(1, 1, 1, [0.1, 0.9]),
        make_row(2, 0, 0, [0.7, 0.3]),
    ]
    candidate_rows = [
        make_row(0, 0, 0, [0.8, 0.2]),
        make_row(1, 1, 0, [0.6, 0.4]),
        make_row(2, 0, 0, [0.9, 0.1]),
    ]
    baseline = {row["filename"]: row for row in baseline_rows}
    candidate = {row["filename"]: row for row in candidate_rows}
    result = analyse_seed(baseline, candidate, seed=42, bins=2)
    assert result["transitions"]["fixed"] == 1
    assert result["transitions"]["regressed"] == 1
    assert result["confusion_baseline"][("p0", "p1")] == 1
    aggregated = aggregate_provinces(result["province_rows"])
    p0 = next(row for row in aggregated if row["province"] == "p0")
    assert p0["improvement_mean"] == pytest.approx(0.5)


def test_calibration_metrics_are_finite():
    rows = [
        make_row(0, 0, 0, [0.8, 0.2]),
        make_row(1, 1, 0, [0.6, 0.4]),
    ]
    metrics, bins = calibration_metrics(rows, bins=4)
    assert 0 <= metrics["ece"] <= 1
    assert metrics["nll"] > 0
    assert len(bins) == 4
