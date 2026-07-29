import csv

import pytest

from dialect_moe.split_audit import (
    UtteranceRecord,
    assign_speakers_preserving_splits,
    assign_speakers_stratified,
    duplicate_values_by_split,
)


def make_record(split: str, index: int, speaker: str, province: str):
    return UtteranceRecord(
        original_split=split,
        row_index=index,
        filename=f"{split}_{index}.wav",
        speaker_id=speaker,
        region="North",
        province=province,
    )


def test_overlap_and_speaker_assignment_are_disjoint():
    records = [
        make_record("train", 0, "a", "01"),
        make_record("valid", 0, "a", "01"),
        make_record("train", 1, "b", "01"),
        make_record("test", 0, "c", "01"),
        make_record("train", 2, "d", "02"),
        make_record("valid", 1, "e", "02"),
        make_record("test", 1, "f", "02"),
    ]
    overlap = duplicate_values_by_split(records, "speaker_id")
    assert overlap["a"] == {"train": 1, "valid": 1}
    assignments = assign_speakers_stratified(
        records, {"train": 0.8, "valid": 0.1, "test": 0.1}, seed=42
    )
    assert set(assignments) == {"a", "b", "c", "d", "e", "f"}
    assert set(assignments.values()) == {"train", "valid", "test"}
    preserved = assign_speakers_preserving_splits(
        records, ["train", "valid", "test"]
    )
    assert preserved["a"] == "train"
    assert preserved["b"] == "train"
    assert preserved["c"] == "test"


def test_manifest_rebuilds_splits_without_speaker_overlap(tmp_path):
    datasets_module = pytest.importorskip("datasets")
    from dialect_moe.data import _apply_split_manifest

    Dataset = datasets_module.Dataset
    DatasetDict = datasets_module.DatasetDict
    datasets = DatasetDict(
        {
            "train": Dataset.from_dict(
                {"speakerID": ["a", "b"], "value": [1, 2]}
            ),
            "valid": Dataset.from_dict({"speakerID": ["a"], "value": [3]}),
            "test": Dataset.from_dict({"speakerID": ["c"], "value": [4]}),
        }
    )
    manifest = tmp_path / "manifest.csv"
    rows = [
        ("train", 0, "train", "a"),
        ("valid", 0, "train", "a"),
        ("train", 1, "valid", "b"),
        ("test", 0, "test", "c"),
    ]
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["original_split", "row_index", "new_split", "speaker_id"]
        )
        writer.writerows(rows)
    rebuilt = _apply_split_manifest(datasets, manifest)
    assert len(rebuilt["train"]) == 2
    assert len(rebuilt["valid"]) == 1
    assert len(rebuilt["test"]) == 1
    speaker_sets = {
        split: set(dataset["speakerID"]) for split, dataset in rebuilt.items()
    }
    assert speaker_sets["train"].isdisjoint(speaker_sets["valid"])
    assert speaker_sets["train"].isdisjoint(speaker_sets["test"])
    assert speaker_sets["valid"].isdisjoint(speaker_sets["test"])
