from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def load_jsonl(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = row["filename"]
            if key in rows:
                raise ValueError(f"Duplicate filename '{key}' in {path}")
            rows[key] = row
    return rows


def metric_values(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def compare_task(
    baseline_rows: list[dict],
    candidate_rows: list[dict],
    task: str,
    iterations: int,
    seed: int,
) -> dict:
    true_key, pred_key = f"{task}_true_id", f"{task}_pred_id"
    y_true = np.asarray([row[true_key] for row in baseline_rows])
    baseline_pred = np.asarray([row[pred_key] for row in baseline_rows])
    candidate_pred = np.asarray([row[pred_key] for row in candidate_rows])
    candidate_true = np.asarray([row[true_key] for row in candidate_rows])
    if not np.array_equal(y_true, candidate_true):
        raise ValueError(f"Ground-truth labels differ for task '{task}'")

    baseline_correct = baseline_pred == y_true
    candidate_correct = candidate_pred == y_true
    baseline_only = int(np.sum(baseline_correct & ~candidate_correct))
    candidate_only = int(np.sum(~baseline_correct & candidate_correct))
    discordant = baseline_only + candidate_only
    mcnemar_p = (
        float(binomtest(min(baseline_only, candidate_only), discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )

    by_speaker: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(baseline_rows):
        by_speaker[row["speaker_id"]].append(index)
    speakers = sorted(by_speaker)
    rng = np.random.default_rng(seed)
    bootstrap_differences = {
        "accuracy": [],
        "balanced_accuracy": [],
        "macro_f1": [],
    }
    for _ in range(iterations):
        sampled_speakers = rng.choice(speakers, size=len(speakers), replace=True)
        sampled_indices = np.concatenate(
            [np.asarray(by_speaker[speaker], dtype=np.int64) for speaker in sampled_speakers]
        )
        baseline_metrics = metric_values(y_true[sampled_indices], baseline_pred[sampled_indices])
        candidate_metrics = metric_values(y_true[sampled_indices], candidate_pred[sampled_indices])
        for metric in bootstrap_differences:
            bootstrap_differences[metric].append(
                candidate_metrics[metric] - baseline_metrics[metric]
            )

    baseline_metrics = metric_values(y_true, baseline_pred)
    candidate_metrics = metric_values(y_true, candidate_pred)
    comparison = {
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "difference_candidate_minus_baseline": {
            metric: candidate_metrics[metric] - baseline_metrics[metric]
            for metric in baseline_metrics
        },
        "speaker_bootstrap": {},
        "mcnemar_accuracy": {
            "baseline_correct_candidate_wrong": baseline_only,
            "baseline_wrong_candidate_correct": candidate_only,
            "discordant_pairs": discordant,
            "exact_p_value": mcnemar_p,
        },
    }
    for metric, differences in bootstrap_differences.items():
        values = np.asarray(differences)
        comparison["speaker_bootstrap"][metric] = {
            "iterations": iterations,
            "mean_difference": float(values.mean()),
            "ci_95_low": float(np.quantile(values, 0.025)),
            "ci_95_high": float(np.quantile(values, 0.975)),
            "probability_candidate_better": float(np.mean(values > 0)),
        }
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired speaker-bootstrap and McNemar comparison."
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    baseline = load_jsonl(Path(args.baseline))
    candidate = load_jsonl(Path(args.candidate))
    if set(baseline) != set(candidate):
        missing_candidate = sorted(set(baseline) - set(candidate))[:10]
        missing_baseline = sorted(set(candidate) - set(baseline))[:10]
        raise ValueError(
            "Prediction files contain different samples. "
            f"Missing from candidate: {missing_candidate}; "
            f"missing from baseline: {missing_baseline}"
        )
    keys = sorted(baseline)
    baseline_rows = [baseline[key] for key in keys]
    candidate_rows = [candidate[key] for key in keys]
    result = {
        "baseline_file": str(Path(args.baseline)),
        "candidate_file": str(Path(args.candidate)),
        "samples": len(keys),
        "speakers": len({row["speaker_id"] for row in baseline_rows}),
        "region": compare_task(
            baseline_rows,
            candidate_rows,
            "region",
            args.bootstrap_iterations,
            args.seed,
        ),
        "province": compare_task(
            baseline_rows,
            candidate_rows,
            "province",
            args.bootstrap_iterations,
            args.seed + 1,
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Wrote comparison to {output}")


if __name__ == "__main__":
    main()

