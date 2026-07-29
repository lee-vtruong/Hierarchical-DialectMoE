from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    normalized_mutual_info_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dialect_moe.config import load_config
from dialect_moe.data import DialectCollator, load_vimd
from dialect_moe.model import HierarchicalDialectMoE
from dialect_moe.utils import move_to_device, save_json


def save_confusion_csv(
    targets: list[int],
    predictions: list[int],
    labels: list[str],
    path: Path,
) -> None:
    matrix = confusion_matrix(targets, predictions, labels=list(range(len(labels))))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true/pred", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row.tolist()])


def save_expert_matrix_csv(
    matrix: np.ndarray, row_labels: list[str], path: Path
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label/expert", *[f"expert_{i}" for i in range(matrix.shape[1])]])
        for label, row in zip(row_labels, matrix):
            writer.writerow([label, *row.tolist()])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/vimd_moe.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    bundle = load_vimd(config, max_samples=args.max_samples)
    collator = DialectCollator(
        config["model"]["backbone"],
        config["data"],
        bundle.region_vocab,
        bundle.province_vocab,
        use_prosody=bool(config["model"].get("use_prosody", True)),
        use_spectral=bool(config["model"].get("use_spectral", False)),
        prosody_feature_set=config["model"].get("prosody_feature_set", "legacy"),
    )
    loader = DataLoader(
        bundle.datasets[args.split],
        batch_size=int(config["training"]["batch_size"]),
        collate_fn=collator,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=True,
        persistent_workers=int(config["data"]["num_workers"]) > 0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HierarchicalDialectMoE(
        config["model"], len(bundle.region_vocab), len(bundle.province_vocab)
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    targets_region, predictions_region = [], []
    targets_province, predictions_province = [], []
    router_probabilities = []
    province_rankings = []
    prediction_rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {args.split}"):
            batch = move_to_device(batch, device)
            output = model(
                batch["input_values"],
                batch["attention_mask"],
                batch["prosody"],
                batch["spectral"],
            )
            region_targets = batch["region_labels"].cpu().tolist()
            province_targets = batch["province_labels"].cpu().tolist()
            region_probabilities = torch.softmax(output.region_logits, dim=-1).cpu()
            province_probabilities = torch.softmax(output.province_logits, dim=-1).cpu()
            region_predictions = region_probabilities.argmax(-1).tolist()
            province_predictions = province_probabilities.argmax(-1).tolist()
            rankings = province_probabilities.argsort(dim=-1, descending=True)
            router_batch = output.router_probabilities.cpu()

            targets_region.extend(region_targets)
            predictions_region.extend(region_predictions)
            targets_province.extend(province_targets)
            predictions_province.extend(province_predictions)
            province_rankings.append(rankings.numpy())
            router_probabilities.append(router_batch.numpy())

            for index in range(len(region_targets)):
                true_region = region_targets[index]
                pred_region = region_predictions[index]
                true_province = province_targets[index]
                pred_province = province_predictions[index]
                prediction_rows.append(
                    {
                        "filename": batch["filenames"][index],
                        "speaker_id": batch["speaker_ids"][index],
                        "province_name": batch["province_names"][index],
                        "region_true_id": true_region,
                        "region_true": bundle.region_vocab.decode(true_region),
                        "region_pred_id": pred_region,
                        "region_pred": bundle.region_vocab.decode(pred_region),
                        "province_true_id": true_province,
                        "province_true": bundle.province_vocab.decode(true_province),
                        "province_pred_id": pred_province,
                        "province_pred": bundle.province_vocab.decode(pred_province),
                        "region_probabilities": region_probabilities[index].tolist(),
                        "province_probabilities": province_probabilities[index].tolist(),
                        "expert_top1": (
                            int(router_batch[index].argmax().item())
                            if model.use_moe
                            else None
                        ),
                        "expert_probabilities": (
                            router_batch[index].tolist() if model.use_moe else None
                        ),
                    }
                )

    def scores(targets: list[int], predictions: list[int], names: list[str]) -> dict:
        return {
            "accuracy": accuracy_score(targets, predictions),
            "balanced_accuracy": balanced_accuracy_score(targets, predictions),
            "classification_report": classification_report(
                targets,
                predictions,
                labels=list(range(len(names))),
                target_names=names,
                output_dict=True,
                zero_division=0,
            ),
        }

    routing = np.concatenate(router_probabilities)
    rankings = np.concatenate(province_rankings)
    province_targets_array = np.asarray(targets_province)
    reciprocal_ranks = []
    for target, ranking in zip(province_targets_array, rankings):
        rank = int(np.where(ranking == target)[0][0]) + 1
        reciprocal_ranks.append(1.0 / rank)
    routing_metrics = {
        "mean_expert_probability": routing.mean(axis=0).tolist(),
        "mean_entropy": float(
            -(routing * np.log(np.clip(routing, 1e-8, 1.0))).sum(axis=1).mean()
        ),
        "active_for_prediction": bool(model.use_moe),
    }
    num_router_outputs = routing.shape[1]
    max_entropy = float(np.log(num_router_outputs)) if num_router_outputs > 1 else 0.0
    routing_metrics["normalized_mean_entropy"] = (
        routing_metrics["mean_entropy"] / max_entropy if max_entropy > 0 else 0.0
    )
    routing_metrics["effective_experts"] = float(
        np.exp(routing_metrics["mean_entropy"])
    )
    region_expert_matrix = province_expert_matrix = None
    if model.use_moe:
        assignments = routing.argmax(axis=1)
        num_experts = routing.shape[1]
        assignment_counts = np.bincount(assignments, minlength=num_experts)
        routing_metrics["top1_assignment_counts"] = assignment_counts.tolist()
        routing_metrics["top1_assignment_fractions"] = (
            assignment_counts / assignment_counts.sum()
        ).tolist()
        assignment_fractions = assignment_counts / assignment_counts.sum()
        top1_entropy = float(
            -(
                assignment_fractions
                * np.log(np.clip(assignment_fractions, 1e-8, 1.0))
            ).sum()
        )
        routing_metrics["top1_assignment_entropy"] = top1_entropy
        routing_metrics["normalized_top1_assignment_entropy"] = (
            top1_entropy / max_entropy if max_entropy > 0 else 0.0
        )
        routing_metrics["active_experts_top1"] = int((assignment_counts > 0).sum())
        routing_metrics["max_top1_assignment_fraction"] = float(
            assignment_fractions.max()
        )
        routing_metrics["min_top1_assignment_fraction"] = float(
            assignment_fractions.min()
        )
        routing_metrics["region_expert_nmi"] = float(
            normalized_mutual_info_score(targets_region, assignments)
        )
        routing_metrics["province_expert_nmi"] = float(
            normalized_mutual_info_score(targets_province, assignments)
        )
        region_expert_matrix = np.zeros(
            (len(bundle.region_vocab), num_experts), dtype=np.int64
        )
        province_expert_matrix = np.zeros(
            (len(bundle.province_vocab), num_experts), dtype=np.int64
        )
        np.add.at(region_expert_matrix, (np.asarray(targets_region), assignments), 1)
        np.add.at(
            province_expert_matrix, (np.asarray(targets_province), assignments), 1
        )
        routing_metrics["region_to_expert_counts"] = region_expert_matrix.tolist()
        routing_metrics["province_to_expert_counts"] = province_expert_matrix.tolist()

    metrics = {
        "region": scores(targets_region, predictions_region, bundle.region_vocab.labels),
        "province": scores(
            targets_province, predictions_province, bundle.province_vocab.labels
        ),
        "routing": routing_metrics,
        "province_ranking": {
            "top_1_accuracy": float(
                np.mean(rankings[:, :1] == province_targets_array[:, None])
            ),
            "top_3_accuracy": float(
                np.mean(np.any(rankings[:, :3] == province_targets_array[:, None], axis=1))
            ),
            "top_5_accuracy": float(
                np.mean(np.any(rankings[:, :5] == province_targets_array[:, None], axis=1))
            ),
            "mrr": float(np.mean(reciprocal_ranks)),
        },
    }
    output_dir = Path(args.checkpoint).parent
    output_path = output_dir / f"metrics_{args.split}.json"
    save_json(metrics, output_path)
    predictions_path = output_dir / f"predictions_{args.split}.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for row in prediction_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    save_confusion_csv(
        targets_region,
        predictions_region,
        bundle.region_vocab.labels,
        output_dir / f"region_confusion_{args.split}.csv",
    )
    save_confusion_csv(
        targets_province,
        predictions_province,
        bundle.province_vocab.labels,
        output_dir / f"province_confusion_{args.split}.csv",
    )
    if region_expert_matrix is not None and province_expert_matrix is not None:
        save_expert_matrix_csv(
            region_expert_matrix,
            bundle.region_vocab.labels,
            output_dir / f"region_to_expert_{args.split}.csv",
        )
        save_expert_matrix_csv(
            province_expert_matrix,
            bundle.province_vocab.labels,
            output_dir / f"province_to_expert_{args.split}.csv",
        )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Predictions: {predictions_path}")


if __name__ == "__main__":
    main()
