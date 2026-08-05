import csv
import json
import sys

from scripts import analyze_h12


def write_predictions(path, predictions):
    rows = []
    for index, prediction in enumerate(predictions):
        rows.append(
            {
                "filename": f"audio-{index}.wav",
                "speaker_id": f"speaker-{index // 2}",
                "region_true_id": index % 2,
                "region_pred_id": prediction % 2,
                "province_true_id": index,
                "province_pred_id": prediction,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_holm_adjust_is_monotonic_and_bounded():
    adjusted = analyze_h12.holm_adjust([0.01, 0.04, 0.02, 0.5])
    assert adjusted == [0.04, 0.08, 0.06, 0.5]
    assert all(0 <= value <= 1 for value in adjusted)


def test_h12_end_to_end(tmp_path, monkeypatch):
    outputs = tmp_path / "outputs"
    seeds = [42]
    configurations = {
        ("base", "acoustic"): [0, 0, 0, 0, 0, 0],
        ("base", "prosody"): [0, 1, 2, 0, 4, 0],
        ("large", "acoustic"): [0, 1, 2, 3, 0, 0],
        ("large", "prosody"): [0, 1, 2, 3, 4, 5],
    }
    for (backbone, variant), predictions in configurations.items():
        path = analyze_h12.prediction_path(outputs, backbone, variant, 42)
        write_predictions(path, predictions)

    destination = tmp_path / "h12"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_h12.py",
            "--outputs",
            str(outputs),
            "--destination",
            str(destination),
            "--seeds",
            "42",
            "--bootstrap-iterations",
            "20",
        ],
    )
    analyze_h12.main()

    with (destination / "h12_per_seed.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 8
    province = [
        row
        for row in rows
        if row["contrast"] == "large_prosody_vs_acoustic"
        and row["task"] == "province"
    ][0]
    assert float(province["difference_accuracy"]) > 0
    assert "mcnemar_holm_p" in province

    aggregate = destination / "h12_aggregate.csv"
    details = destination / "h12_details.json"
    summary = destination / "h12_summary.json"
    assert aggregate.is_file()
    assert details.is_file()
    assert summary.is_file()
