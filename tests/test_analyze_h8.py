import pytest

from scripts.analyze_h8 import aggregate, analyse_seed, bucket_label


def row(filename, truth, prediction, confidence):
    other = 1 - prediction
    probabilities = [0.0, 0.0]
    probabilities[prediction] = confidence
    probabilities[other] = 1 - confidence
    return {
        "filename": filename,
        "province_true_id": truth,
        "province_true": str(truth),
        "province_pred_id": prediction,
        "province_probabilities": probabilities,
    }


def test_bucket_label_boundaries():
    edges = [0, 2, 4, 6]
    assert bucket_label(0, edges) == "[0,2)"
    assert bucket_label(2, edges) == "[2,4)"
    assert bucket_label(6, edges) == "[4,6]"


def test_analyse_seed_duration_and_confidence():
    baseline = {
        "a.wav": row("a.wav", 0, 1, 0.8),
        "b.wav": row("b.wav", 1, 1, 0.7),
    }
    candidate = {
        "a.wav": row("a.wav", 0, 0, 0.9),
        "b.wav": row("b.wav", 1, 0, 0.6),
    }
    result = analyse_seed(
        baseline,
        candidate,
        {"a.wav": 1.0, "b.wav": 3.0},
        seed=42,
        duration_edges=[0, 2, 4],
        confidence_edges=[0, 0.5, 0.75, 1],
        max_seconds=4,
        focus_provinces={"0", "1"},
    )
    assert sum(row["fixed"] for row in result["duration"]) == 1
    assert sum(row["regressed"] for row in result["duration"]) == 1
    assert len(result["focus"]) == 2
    aggregated = aggregate(
        result["duration"], ["duration_bucket"], ["improvement"]
    )
    assert len(aggregated) == 2


def test_duration_mismatch_is_rejected():
    baseline = {"a.wav": row("a.wav", 0, 0, 0.8)}
    with pytest.raises(ValueError, match="duration/prediction mismatch"):
        analyse_seed(
            baseline,
            baseline,
            {},
            42,
            [0, 2],
            [0, 1],
            2,
            set(),
        )
