from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def macro_f1(metrics: dict, task: str) -> float:
    return float(metrics[task]["classification_report"]["macro avg"]["f1-score"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize validation-only H4 load-balancing sweep."
    )
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument(
        "--pattern",
        default="h4_lb_*/metrics_valid_best_province_accuracy.json",
    )
    parser.add_argument(
        "--destination", default="outputs/h4_validation_summary.csv"
    )
    parser.add_argument(
        "--recommendation", default="outputs/h4_validation_recommendation.json"
    )
    parser.add_argument(
        "--collapse-threshold",
        type=float,
        default=0.90,
        help="Mark a run collapsed when one expert gets more than this top-1 fraction.",
    )
    args = parser.parse_args()

    rows: list[dict] = []
    root = Path(args.outputs)
    for metrics_path in sorted(root.glob(f"**/{args.pattern}")):
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        config_path = metrics_path.parent / "config.json"
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)

        routing = metrics["routing"]
        probabilities = [float(value) for value in routing["mean_expert_probability"]]
        fractions = [
            float(value) for value in routing.get("top1_assignment_fractions", [])
        ]
        num_experts = len(probabilities)
        max_entropy = math.log(num_experts) if num_experts > 1 else 0.0
        mean_entropy = float(routing["mean_entropy"])
        normalized_soft_entropy = float(
            routing.get(
                "normalized_mean_entropy",
                mean_entropy / max_entropy if max_entropy else 0.0,
            )
        )
        top1_entropy = -sum(
            fraction * math.log(max(fraction, 1e-8)) for fraction in fractions
        )
        normalized_top1_entropy = float(
            routing.get(
                "normalized_top1_assignment_entropy",
                top1_entropy / max_entropy if max_entropy else 0.0,
            )
        )
        max_fraction = max(fractions) if fractions else 0.0
        active_experts = sum(fraction > 0 for fraction in fractions)
        collapsed = active_experts < 2 or max_fraction > args.collapse_threshold
        near_uniform_soft = (
            normalized_soft_entropy > 0.99
            and max(probabilities) - min(probabilities) < 0.02
        )

        rows.append(
            {
                "experiment": metrics_path.parent.name,
                "load_balance_weight": float(config["loss"]["load_balance_weight"]),
                "region_accuracy": float(metrics["region"]["accuracy"]),
                "region_balanced_accuracy": float(
                    metrics["region"]["balanced_accuracy"]
                ),
                "region_macro_f1": macro_f1(metrics, "region"),
                "province_accuracy": float(metrics["province"]["accuracy"]),
                "province_balanced_accuracy": float(
                    metrics["province"]["balanced_accuracy"]
                ),
                "province_macro_f1": macro_f1(metrics, "province"),
                "soft_router_entropy": mean_entropy,
                "normalized_soft_router_entropy": normalized_soft_entropy,
                "effective_experts": float(
                    routing.get("effective_experts", math.exp(mean_entropy))
                ),
                "top1_assignment_entropy": top1_entropy,
                "normalized_top1_assignment_entropy": normalized_top1_entropy,
                "active_experts_top1": active_experts,
                "max_top1_assignment_fraction": max_fraction,
                "region_expert_nmi": float(routing.get("region_expert_nmi", 0.0)),
                "province_expert_nmi": float(
                    routing.get("province_expert_nmi", 0.0)
                ),
                "collapsed": collapsed,
                "near_uniform_soft_router": near_uniform_soft,
                "mean_expert_probabilities": json.dumps(probabilities),
                "top1_assignment_fractions": json.dumps(fractions),
            }
        )

    if not rows:
        raise FileNotFoundError(f"No H4 validation metrics matching {root}/**/{args.pattern}")

    destination = Path(args.destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    eligible = [row for row in rows if not row["collapsed"]]
    if not eligible:
        eligible = rows
    selected = max(
        eligible,
        key=lambda row: (
            row["province_macro_f1"],
            row["province_balanced_accuracy"],
            row["province_accuracy"],
        ),
    )
    recommendation = {
        "selection_split": "valid",
        "selection_rule": (
            "Highest province macro-F1 among non-collapsed runs; ties use "
            "province balanced accuracy then province accuracy."
        ),
        "collapse_threshold": args.collapse_threshold,
        "selected_experiment": selected["experiment"],
        "selected_load_balance_weight": selected["load_balance_weight"],
        "selected_metrics": {
            key: value
            for key, value in selected.items()
            if key
            in {
                "province_accuracy",
                "province_balanced_accuracy",
                "province_macro_f1",
                "normalized_soft_router_entropy",
                "normalized_top1_assignment_entropy",
                "max_top1_assignment_fraction",
                "region_expert_nmi",
                "province_expert_nmi",
                "collapsed",
                "near_uniform_soft_router",
            }
        },
        "warning": (
            "This recommendation is validation-only. Run seeds 42/43/44 for the "
            "selected setting before one final test evaluation."
        ),
    }
    recommendation_path = Path(args.recommendation)
    recommendation_path.parent.mkdir(parents=True, exist_ok=True)
    with recommendation_path.open("w", encoding="utf-8") as handle:
        json.dump(recommendation, handle, ensure_ascii=False, indent=2)

    print(f"Wrote {len(rows)} H4 validation runs to {destination}")
    print(
        "Selected:",
        selected["experiment"],
        f"(load_balance_weight={selected['load_balance_weight']})",
    )
    print(f"Recommendation: {recommendation_path}")


if __name__ == "__main__":
    main()
