from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def safe_key(split: str, index: int, filename: str) -> str:
    digest = hashlib.sha1(filename.encode("utf-8")).hexdigest()[:10]
    return f"vimd_{split}_{index:06d}_{digest}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ChunkFormer TSV predictions to project JSONL.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    metadata = {}
    with Path(args.metadata).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["split"] == args.split:
                metadata[row["key"]] = row
    predictions = {}
    with Path(args.predictions).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            predictions[row["key"]] = row
    if set(metadata) != set(predictions):
        raise ValueError(
            f"Key mismatch: metadata={len(metadata)}, predictions={len(predictions)}, "
            f"missing_predictions={len(set(metadata) - set(predictions))}, "
            f"unknown_predictions={len(set(predictions) - set(metadata))}"
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for key in sorted(metadata):
            row = dict(metadata[key])
            row["region_pred_id"] = int(predictions[key]["region"])
            row["province_pred_id"] = int(predictions[key]["province"])
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(metadata)} predictions to {output}")


if __name__ == "__main__":
    main()
