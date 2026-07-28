import json

from scripts.compare_predictions import compare_task, load_jsonl


def test_load_and_compare_predictions(tmp_path):
    baseline_path = tmp_path / "baseline.jsonl"
    rows = [
        {
            "filename": f"{index}.wav",
            "speaker_id": f"speaker-{index // 2}",
            "region_true_id": index % 2,
            "region_pred_id": 0,
            "province_true_id": index,
            "province_pred_id": 0,
        }
        for index in range(6)
    ]
    with baseline_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    loaded = load_jsonl(baseline_path)
    assert len(loaded) == 6

    candidate = [dict(row) for row in rows]
    for row in candidate:
        row["region_pred_id"] = row["region_true_id"]
        row["province_pred_id"] = row["province_true_id"]
    result = compare_task(rows, candidate, "province", iterations=100, seed=1)
    assert result["candidate"]["accuracy"] == 1.0
    assert result["difference_candidate_minus_baseline"]["accuracy"] > 0
    assert result["mcnemar_accuracy"]["baseline_wrong_candidate_correct"] == 5

