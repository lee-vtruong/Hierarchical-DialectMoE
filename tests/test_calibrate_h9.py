import numpy as np
import pytest

from scripts.calibrate_h9 import (
    aggregate,
    apply_temperature,
    calibration_metrics,
    fit_temperature,
)


def test_temperature_preserves_argmax_and_normalization():
    probabilities = np.asarray([[0.8, 0.2], [0.1, 0.9]])
    scaled = apply_temperature(probabilities, 2.0)
    assert np.allclose(scaled.sum(axis=1), 1.0)
    assert np.array_equal(scaled.argmax(axis=1), probabilities.argmax(axis=1))


def test_fit_temperature_reduces_validation_nll_for_overconfidence():
    probabilities = np.asarray(
        [[0.99, 0.01], [0.99, 0.01], [0.99, 0.01], [0.01, 0.99]]
    )
    targets = np.asarray([0, 1, 0, 1])
    before, _ = calibration_metrics(probabilities, targets, bins=5)
    temperature = fit_temperature(probabilities, targets)
    after, _ = calibration_metrics(
        apply_temperature(probabilities, temperature), targets, bins=5
    )
    assert temperature > 1
    assert after["nll"] < before["nll"]
    assert after["accuracy"] == before["accuracy"]


def test_aggregate_multiseed():
    rows = [
        {
            "model": "baseline",
            "stage": "test_after",
            "temperature": 2,
            "accuracy": 0.5,
            "ece": 0.1,
            "nll": 1.0,
            "brier": 0.7,
            "mean_confidence": 0.6,
        },
        {
            "model": "baseline",
            "stage": "test_after",
            "temperature": 4,
            "accuracy": 0.5,
            "ece": 0.2,
            "nll": 1.2,
            "brier": 0.8,
            "mean_confidence": 0.7,
        },
    ]
    result = aggregate(rows)
    assert result[0]["temperature_mean"] == pytest.approx(3.0)
    assert result[0]["accuracy_mean"] == pytest.approx(0.5)
