import json
import subprocess
import sys

from scripts import analyze_h13, benchmark_h13


def write_prediction(path, variant):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "filename": "a.wav",
            "speaker_id": "s1",
            "province_true_id": 0,
            "province_true": "p0",
            "province_pred_id": 0,
            "province_pred": "p0",
            "province_probabilities": [0.8, 0.2],
        },
        {
            "filename": "b.wav",
            "speaker_id": "s2",
            "province_true_id": 1,
            "province_true": "p1",
            "province_pred_id": 1 if variant == "prosody" else 0,
            "province_pred": "p1" if variant == "prosody" else "p0",
            "province_probabilities": (
                [0.2, 0.8] if variant == "prosody" else [0.6, 0.4]
            ),
        },
    ]
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_h13_skip_calibration_end_to_end(tmp_path):
    outputs = tmp_path / "outputs"
    for variant in ("acoustic", "prosody"):
        write_prediction(
            analyze_h13.prediction_path(outputs, variant, 42, "test"), variant
        )
    destination = tmp_path / "h13"
    subprocess.run(
        [
            sys.executable,
            str(analyze_h13.__file__),
            "--outputs",
            str(outputs),
            "--destination",
            str(destination),
            "--seeds",
            "42",
            "--calibration-bins",
            "2",
            "--skip-calibration",
        ],
        check=True,
    )
    assert (destination / "error_analysis" / "h7_summary.json").is_file()
    assert (destination / "h13_artifact_metadata.csv").is_file()
    summary = json.loads((destination / "h13_summary.json").read_text())
    assert summary["calibration_status"] == "skipped_by_request"


def test_benchmark_parameter_counts():
    model = __import__("torch").nn.Sequential(
        __import__("torch").nn.Linear(3, 4),
        __import__("torch").nn.Linear(4, 2),
    )
    model.backbone = model[0]
    counts = benchmark_h13.parameter_counts(model)
    assert counts["total_parameters"] == 26
    assert counts["backbone_parameters"] == 16
