from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_h12 import aligned_rows, holm_adjust
from scripts.compare_predictions import compare_task


TASKS = ("region", "province")
METRICS = ("accuracy", "balanced_accuracy", "macro_f1")


def baseline_prediction(outputs: Path, seed: int) -> Path:
    return (
        outputs
        / f"h11_large_vi_prosody_seed{seed}"
        / "predictions_test_best_province_accuracy.jsonl"
    )


def candidate_prediction(outputs: Path, seed: int) -> Path:
    return (
        outputs
        / f"h14_large_vi_prosody_moe2_seed{seed}"
        / "predictions_test_best_province_accuracy.jsonl"
    )


def candidate_metrics(outputs: Path, seed: int) -> Path:
    return (
        outputs
        / f"h14_large_vi_prosody_moe2_seed{seed}"
        / "metrics_test_best_province_accuracy.json"
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="H14 paired Large-VI prosody MoE-2 confirmation analysis."
    )
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--destination", default="results_archive/h14")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=14026)
    args = parser.parse_args()

    outputs = Path(args.outputs)
    destination = Path(args.destination)
    destination.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    details: dict[str, dict] = {}
    routing_rows: list[dict] = []

    for seed_index, seed in enumerate(args.seeds):
        baseline_path = baseline_prediction(outputs, seed)
        candidate_path = candidate_prediction(outputs, seed)
        metrics_path = candidate_metrics(outputs, seed)
        for path in (baseline_path, candidate_path, metrics_path):
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"Missing H14 artifact: {path}")
        baseline_rows, candidate_rows = aligned_rows(baseline_path, candidate_path)
        details[str(seed)] = {
            "baseline_file": str(baseline_path),
            "candidate_file": str(candidate_path),
            "samples": len(baseline_rows),
            "speakers": len({row["speaker_id"] for row in baseline_rows}),
        }
        for task_index, task in enumerate(TASKS):
            comparison = compare_task(
                baseline_rows,
                candidate_rows,
                task,
                args.bootstrap_iterations,
                args.bootstrap_seed + seed_index * 10 + task_index,
            )
            details[str(seed)][task] = comparison
            row = {
                "seed": seed,
                "task": task,
                "samples": len(baseline_rows),
                "speakers": details[str(seed)]["speakers"],
            }
            for metric in METRICS:
                bootstrap = comparison["speaker_bootstrap"][metric]
                row[f"baseline_{metric}"] = comparison["baseline"][metric]
                row[f"candidate_{metric}"] = comparison["candidate"][metric]
                row[f"difference_{metric}"] = comparison[
                    "difference_candidate_minus_baseline"
                ][metric]
                row[f"bootstrap_{metric}_ci_low"] = bootstrap["ci_95_low"]
                row[f"bootstrap_{metric}_ci_high"] = bootstrap["ci_95_high"]
                row[f"probability_candidate_better_{metric}"] = bootstrap[
                    "probability_candidate_better"
                ]
            mcnemar = comparison["mcnemar_accuracy"]
            row["baseline_correct_candidate_wrong"] = mcnemar[
                "baseline_correct_candidate_wrong"
            ]
            row["baseline_wrong_candidate_correct"] = mcnemar[
                "baseline_wrong_candidate_correct"
            ]
            row["mcnemar_exact_p"] = mcnemar["exact_p_value"]
            rows.append(row)

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        routing = metrics.get("routing", {})
        routing_rows.append(
            {
                "seed": seed,
                "active_for_prediction": routing.get("active_for_prediction"),
                "mean_entropy": routing.get("mean_entropy"),
                "normalized_mean_entropy": routing.get("normalized_mean_entropy"),
                "effective_experts": routing.get("effective_experts"),
                "mean_expert_probability_json": json.dumps(
                    routing.get("mean_expert_probability", []), separators=(",", ":")
                ),
            }
        )

    for task in TASKS:
        indices = [index for index, row in enumerate(rows) if row["task"] == task]
        adjusted = holm_adjust([float(rows[index]["mcnemar_exact_p"]) for index in indices])
        for index, adjusted_p in zip(indices, adjusted):
            rows[index]["mcnemar_holm_p"] = adjusted_p
            rows[index]["mcnemar_holm_significant_0_05"] = adjusted_p < 0.05

    aggregates = []
    for task in TASKS:
        values = [row for row in rows if row["task"] == task]
        aggregate = {"task": task, "runs": len(values)}
        for metric in METRICS:
            for field in ("baseline", "candidate", "difference"):
                numbers = [float(row[f"{field}_{metric}"]) for row in values]
                aggregate[f"{field}_{metric}_mean"] = sum(numbers) / len(numbers)
            aggregate[f"candidate_better_{metric}_runs"] = sum(
                float(row[f"difference_{metric}"]) > 0 for row in values
            )
            aggregate[f"bootstrap_ci_excludes_zero_{metric}_runs"] = sum(
                float(row[f"bootstrap_{metric}_ci_low"]) > 0
                or float(row[f"bootstrap_{metric}_ci_high"]) < 0
                for row in values
            )
        aggregate["mcnemar_holm_significant_runs"] = sum(
            bool(row["mcnemar_holm_significant_0_05"]) for row in values
        )
        aggregates.append(aggregate)

    write_csv(destination / "h14_per_seed.csv", rows)
    write_csv(destination / "h14_aggregate.csv", aggregates)
    write_csv(destination / "h14_routing.csv", routing_rows)
    with (destination / "h14_details.json").open("w", encoding="utf-8") as handle:
        json.dump(details, handle, ensure_ascii=False, indent=2)
    summary = {
        "baseline": "H11 Large-VI acoustic+prosody without MoE",
        "candidate": "H14 Large-VI acoustic+prosody hierarchical MoE-2 top-1",
        "selection": "best province validation accuracy; test evaluated once",
        "load_balance_weight": 0.001,
        "router_entropy_weight": 0.0,
        "seeds": args.seeds,
        "bootstrap_iterations": args.bootstrap_iterations,
        "multiple_testing": "Holm correction over three McNemar tests per task",
        "aggregate": aggregates,
    }
    with (destination / "h14_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote H14 artifacts to {destination.resolve()}")


if __name__ == "__main__":
    main()
