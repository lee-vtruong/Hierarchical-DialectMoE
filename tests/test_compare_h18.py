import json

import pytest

from scripts.compare_h18 import compare_runs, make_decision


def payload(region: float, province: float, macro_f1: float) -> dict:
    return {
        "region": {
            "accuracy": region,
            "classification_report": {"macro avg": {"f1-score": region - 0.01}},
        },
        "province": {
            "accuracy": province,
            "balanced_accuracy": province - 0.01,
            "classification_report": {"macro avg": {"f1-score": macro_f1}},
        },
    }


def write_run(root, name, seed, value):
    directory = root / f"{name}_seed{seed}"
    directory.mkdir(parents=True)
    path = directory / "metrics_valid_best_province_accuracy.json"
    path.write_text(json.dumps(value), encoding="utf-8")


def test_compare_h18_screening_gate(tmp_path):
    write_run(tmp_path, "h11_large_vi_prosody", 42, payload(0.94, 0.60, 0.58))
    write_run(tmp_path, "h18_soft_hierarchy", 42, payload(0.939, 0.61, 0.581))

    rows = compare_runs(tmp_path, "valid", [42, 43, 44])
    decision = make_decision(rows)

    assert rows[0]["difference_province_accuracy"] == pytest.approx(0.01)
    assert decision["stage"] == "seed42_screening"
    assert decision["passed"] is True


def test_compare_h18_requires_stable_multi_seed_gain(tmp_path):
    for seed, gain in ((42, 0.01), (43, 0.02), (44, -0.005)):
        write_run(
            tmp_path,
            "h11_large_vi_prosody",
            seed,
            payload(0.94, 0.60, 0.58),
        )
        write_run(
            tmp_path,
            "h18_soft_hierarchy",
            seed,
            payload(0.939, 0.60 + gain, 0.58 + gain),
        )

    decision = make_decision(compare_runs(tmp_path, "valid", [42, 43, 44]))

    assert decision["stage"] == "multi_seed_validation"
    assert decision["province_accuracy_wins"] == 2
    assert decision["passed"] is True
