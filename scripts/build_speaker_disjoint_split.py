from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dialect_moe.split_audit import (
    UtteranceRecord,
    assign_speakers_preserving_splits,
    assign_speakers_stratified,
    speaker_label_conflicts,
    split_distribution,
)
from dialect_moe.utils import save_json


def read_records(path: Path) -> list[UtteranceRecord]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = []
        for row in csv.DictReader(handle):
            rows.append(
                UtteranceRecord(
                    original_split=row["original_split"],
                    row_index=int(row["row_index"]),
                    filename=row["filename"],
                    speaker_id=row["speaker_id"],
                    region=row["region"],
                    province=row["province"],
                    duration_seconds=(
                        float(row["duration_seconds"])
                        if row.get("duration_seconds")
                        else None
                    ),
                    audio_sha256=row.get("audio_sha256") or None,
                )
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a deterministic speaker-disjoint split manifest."
    )
    parser.add_argument(
        "--records", default="outputs/h6_split_audit/records.csv"
    )
    parser.add_argument(
        "--output", default="data/splits/vimd_speaker_disjoint_seed42.csv"
    )
    parser.add_argument(
        "--summary",
        default="outputs/h6_split_audit/speaker_disjoint_summary.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--strategy",
        choices=["preserve", "rebuild"],
        default="preserve",
        help="preserve minimally repairs original splits; rebuild reassigns all speakers.",
    )
    parser.add_argument(
        "--split-priority",
        default="train,valid,test",
        help="Priority used by preserve strategy; earlier split wins overlap.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.793)
    parser.add_argument("--valid-ratio", type=float, default=0.100)
    parser.add_argument("--test-ratio", type=float, default=0.107)
    parser.add_argument("--allow-label-conflicts", action="store_true")
    args = parser.parse_args()

    records = read_records(Path(args.records))
    conflicts = speaker_label_conflicts(records)
    if conflicts and not args.allow_label_conflicts:
        raise ValueError(
            f"Found {len(conflicts)} speakers with conflicting labels. "
            "Inspect speaker_label_conflicts.json; use --allow-label-conflicts "
            "only after deciding that majority-province stratification is acceptable."
        )
    ratios = {
        "train": args.train_ratio,
        "valid": args.valid_ratio,
        "test": args.test_ratio,
    }
    if args.strategy == "preserve":
        priority = [value.strip() for value in args.split_priority.split(",") if value.strip()]
        assignments = assign_speakers_preserving_splits(records, priority)
    else:
        priority = None
        assignments = assign_speakers_stratified(records, ratios, args.seed)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "original_split",
        "row_index",
        "new_split",
        "filename",
        "speaker_id",
        "region",
        "province",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "original_split": record.original_split,
                    "row_index": record.row_index,
                    "new_split": assignments[record.speaker_id],
                    "filename": record.filename,
                    "speaker_id": record.speaker_id,
                    "region": record.region,
                    "province": record.province,
                }
            )

    moved_utterances = sum(
        assignments[record.speaker_id] != record.original_split
        for record in records
    )
    moved_speakers = {
        record.speaker_id
        for record in records
        if assignments[record.speaker_id] != record.original_split
    }
    summary = {
        "strategy": args.strategy,
        "seed": args.seed,
        "ratios": ratios,
        "split_priority": priority,
        "label_conflict_speakers": len(conflicts),
        "moved_utterances": moved_utterances,
        "moved_speakers": len(moved_speakers),
        "distribution": split_distribution(records, assignments),
    }
    save_json(summary, args.summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Manifest: {output}")


if __name__ == "__main__":
    main()
