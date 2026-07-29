from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dialect_moe.config import load_config


def load_jsonl(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            filename = row["filename"]
            if filename in rows:
                raise ValueError(f"Duplicate filename '{filename}' in {path}")
            rows[filename] = row
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def duration_seconds(audio: dict) -> float:
    import soundfile as sf

    payload = audio.get("bytes")
    path = audio.get("path")
    source = io.BytesIO(payload) if payload is not None else path
    if source is None:
        raise ValueError("Audio row contains neither bytes nor path")
    info = sf.info(source)
    if info.samplerate <= 0:
        raise ValueError("Invalid audio sample rate")
    return float(info.frames / info.samplerate)


def build_duration_map(config: dict, split: str) -> dict[str, float]:
    from dialect_moe.data import load_vimd
    from tqdm import tqdm

    bundle = load_vimd(config)
    data_config = config["data"]
    filename_column = data_config.get("filename_column", "filename")
    audio_column = data_config["audio_column"]
    result = {}
    for row in tqdm(bundle.datasets[split], desc=f"Reading {split} audio headers"):
        filename = str(row.get(filename_column, ""))
        if not filename:
            raise ValueError("Dataset row has an empty filename")
        if filename in result:
            raise ValueError(f"Duplicate filename in repaired {split}: {filename}")
        result[filename] = duration_seconds(row[audio_column])
    return result


def bucket_label(value: float, edges: list[float]) -> str:
    for lower, upper in zip(edges[:-1], edges[1:]):
        if lower <= value < upper:
            return f"[{lower:g},{upper:g})"
    if value == edges[-1]:
        return f"{edges[-1]:g} (capped)"
    raise ValueError(f"Value {value} is outside bucket edges")


def calibration_gap(rows: list[dict]) -> tuple[float, float]:
    if not rows:
        return 0.0, 0.0
    confidence = np.asarray(
        [max(row["province_probabilities"]) for row in rows], dtype=np.float64
    )
    accuracy = np.asarray(
        [row["province_pred_id"] == row["province_true_id"] for row in rows],
        dtype=np.float64,
    )
    return float(confidence.mean()), float(confidence.mean() - accuracy.mean())


def analyse_seed(
    baseline: dict[str, dict],
    candidate: dict[str, dict],
    durations: dict[str, float],
    seed: int,
    duration_edges: list[float],
    confidence_edges: list[float],
    max_seconds: float,
    focus_provinces: set[str],
) -> dict[str, list[dict]]:
    if set(baseline) != set(candidate):
        raise ValueError(f"Seed {seed}: prediction sample sets differ")
    missing = set(baseline) - set(durations)
    extra = set(durations) - set(baseline)
    if missing or extra:
        raise ValueError(
            f"Seed {seed}: duration/prediction mismatch; "
            f"missing durations={sorted(missing)[:5]}, extra durations={sorted(extra)[:5]}"
        )

    duration_groups: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    confidence_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    focus_groups: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for filename in sorted(baseline):
        left, right = baseline[filename], candidate[filename]
        if left["province_true_id"] != right["province_true_id"]:
            raise ValueError(f"Seed {seed}: truth mismatch for {filename}")
        effective = min(durations[filename], max_seconds)
        duration_groups[bucket_label(effective, duration_edges)].append((left, right))
        for model, row in (("baseline", left), ("candidate", right)):
            confidence = max(row["province_probabilities"])
            confidence_groups[
                (model, bucket_label(confidence, confidence_edges))
            ].append(row)
        province = str(left["province_true"])
        if province in focus_provinces:
            focus_groups[province].append((left, right))

    duration_rows = []
    for bucket, pairs in duration_groups.items():
        left_correct = np.asarray(
            [a["province_pred_id"] == a["province_true_id"] for a, _ in pairs]
        )
        right_correct = np.asarray(
            [b["province_pred_id"] == b["province_true_id"] for _, b in pairs]
        )
        duration_rows.append(
            {
                "seed": seed,
                "duration_bucket": bucket,
                "support": len(pairs),
                "baseline_accuracy": float(left_correct.mean()),
                "candidate_accuracy": float(right_correct.mean()),
                "improvement": float(right_correct.mean() - left_correct.mean()),
                "fixed": int(np.sum(~left_correct & right_correct)),
                "regressed": int(np.sum(left_correct & ~right_correct)),
            }
        )

    confidence_rows = []
    for (model, bucket), rows in confidence_groups.items():
        correct = np.asarray(
            [row["province_pred_id"] == row["province_true_id"] for row in rows]
        )
        mean_confidence, gap = calibration_gap(rows)
        confidence_rows.append(
            {
                "seed": seed,
                "model": model,
                "confidence_bucket": bucket,
                "support": len(rows),
                "accuracy": float(correct.mean()),
                "mean_confidence": mean_confidence,
                "calibration_gap": gap,
            }
        )

    focus_rows = []
    for province, pairs in focus_groups.items():
        left_correct = np.asarray(
            [a["province_pred_id"] == a["province_true_id"] for a, _ in pairs]
        )
        right_correct = np.asarray(
            [b["province_pred_id"] == b["province_true_id"] for _, b in pairs]
        )
        focus_rows.append(
            {
                "seed": seed,
                "province": province,
                "support": len(pairs),
                "mean_raw_duration_seconds": float(
                    np.mean(
                        [
                            durations[a["filename"]]
                            for a, _ in pairs
                        ]
                    )
                ),
                "baseline_accuracy": float(left_correct.mean()),
                "candidate_accuracy": float(right_correct.mean()),
                "improvement": float(right_correct.mean() - left_correct.mean()),
                "fixed": int(np.sum(~left_correct & right_correct)),
                "regressed": int(np.sum(left_correct & ~right_correct)),
            }
        )
    return {
        "duration": duration_rows,
        "confidence": confidence_rows,
        "focus": focus_rows,
    }


def aggregate(rows: list[dict], keys: list[str], metrics: list[str]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    result = []
    for key_values, values in grouped.items():
        output = dict(zip(keys, key_values))
        output["seeds"] = len(values)
        for metric in metrics:
            numbers = np.asarray([float(row[metric]) for row in values])
            output[f"{metric}_mean"] = float(numbers.mean())
            output[f"{metric}_std"] = (
                float(numbers.std(ddof=1)) if len(numbers) > 1 else 0.0
            )
        result.append(output)
    return sorted(result, key=lambda row: tuple(str(row[key]) for key in keys))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="H8 duration, confidence and focus-province analysis."
    )
    parser.add_argument(
        "--config", default="configs/experiments/h6_speaker_disjoint_acoustic.yaml"
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--baseline-template", required=True)
    parser.add_argument("--candidate-template", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--output-dir", default="outputs/h8")
    parser.add_argument(
        "--duration-edges",
        nargs="+",
        type=float,
        default=[0, 2, 4, 6, 10, 20],
    )
    parser.add_argument(
        "--confidence-edges",
        nargs="+",
        type=float,
        default=[0, 0.4, 0.6, 0.8, 1],
    )
    parser.add_argument(
        "--focus-provinces",
        nargs="+",
        default=["17", "30", "22", "38", "70", "14", "11"],
    )
    args = parser.parse_args()
    if sorted(set(args.duration_edges)) != args.duration_edges:
        raise ValueError("--duration-edges must be strictly increasing")
    if sorted(set(args.confidence_edges)) != args.confidence_edges:
        raise ValueError("--confidence-edges must be strictly increasing")

    config = load_config(args.config)
    max_seconds = float(config["data"]["max_seconds"])
    durations = build_duration_map(config, args.split)
    if args.duration_edges[-1] < max_seconds:
        raise ValueError("Last duration edge must cover data.max_seconds")

    metadata_rows = [
        {
            "filename": filename,
            "raw_duration_seconds": duration,
            "effective_duration_seconds": min(duration, max_seconds),
            "truncated": duration > max_seconds,
        }
        for filename, duration in sorted(durations.items())
    ]
    duration_rows, confidence_rows, focus_rows = [], [], []
    for seed in args.seeds:
        result = analyse_seed(
            load_jsonl(Path(args.baseline_template.format(seed=seed))),
            load_jsonl(Path(args.candidate_template.format(seed=seed))),
            durations,
            seed,
            args.duration_edges,
            args.confidence_edges,
            max_seconds,
            set(args.focus_provinces),
        )
        duration_rows.extend(result["duration"])
        confidence_rows.extend(result["confidence"])
        focus_rows.extend(result["focus"])

    duration_aggregate = aggregate(
        duration_rows,
        ["duration_bucket"],
        ["support", "baseline_accuracy", "candidate_accuracy", "improvement", "fixed", "regressed"],
    )
    confidence_aggregate = aggregate(
        confidence_rows,
        ["model", "confidence_bucket"],
        ["support", "accuracy", "mean_confidence", "calibration_gap"],
    )
    focus_aggregate = aggregate(
        focus_rows,
        ["province"],
        [
            "support", "mean_raw_duration_seconds", "baseline_accuracy",
            "candidate_accuracy", "improvement", "fixed", "regressed",
        ],
    )

    output_dir = Path(args.output_dir)
    write_csv(
        output_dir / "duration_metadata.csv",
        metadata_rows,
        ["filename", "raw_duration_seconds", "effective_duration_seconds", "truncated"],
    )
    write_csv(
        output_dir / "duration_bucket_per_seed.csv",
        duration_rows,
        [
            "seed", "duration_bucket", "support", "baseline_accuracy",
            "candidate_accuracy", "improvement", "fixed", "regressed",
        ],
    )
    write_csv(
        output_dir / "duration_bucket_aggregate.csv",
        duration_aggregate,
        list(duration_aggregate[0]) if duration_aggregate else [],
    )
    write_csv(
        output_dir / "confidence_bucket_per_seed.csv",
        confidence_rows,
        [
            "seed", "model", "confidence_bucket", "support", "accuracy",
            "mean_confidence", "calibration_gap",
        ],
    )
    write_csv(
        output_dir / "confidence_bucket_aggregate.csv",
        confidence_aggregate,
        list(confidence_aggregate[0]) if confidence_aggregate else [],
    )
    write_csv(
        output_dir / "focus_province_per_seed.csv",
        focus_rows,
        [
            "seed", "province", "support", "mean_raw_duration_seconds",
            "baseline_accuracy", "candidate_accuracy", "improvement", "fixed",
            "regressed",
        ],
    )
    write_csv(
        output_dir / "focus_province_aggregate.csv",
        focus_aggregate,
        list(focus_aggregate[0]) if focus_aggregate else [],
    )
    raw_values = np.asarray(list(durations.values()), dtype=np.float64)
    summary = {
        "samples": len(durations),
        "seeds": args.seeds,
        "duration": {
            "raw_mean_seconds": float(raw_values.mean()),
            "raw_median_seconds": float(np.median(raw_values)),
            "raw_p95_seconds": float(np.quantile(raw_values, 0.95)),
            "max_seconds": float(raw_values.max()),
            "model_max_seconds": max_seconds,
            "truncated_samples": int(np.sum(raw_values > max_seconds)),
        },
        "duration_bucket_aggregate": duration_aggregate,
        "confidence_bucket_aggregate": confidence_aggregate,
        "focus_province_aggregate": focus_aggregate,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "h8_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote H8 artifacts to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
