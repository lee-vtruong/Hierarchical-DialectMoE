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
    )
    loader = DataLoader(
        bundle.datasets[args.split],
        batch_size=int(config["training"]["batch_size"]),
        collate_fn=collator,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=True,
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
            output = model(batch["input_values"], batch["attention_mask"], batch["prosody"])
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
    metrics = {
        "region": scores(targets_region, predictions_region, bundle.region_vocab.labels),
        "province": scores(
            targets_province, predictions_province, bundle.province_vocab.labels
        ),
        "routing": {
            "mean_expert_probability": routing.mean(axis=0).tolist(),
            "mean_entropy": float(
                -(routing * np.log(np.clip(routing, 1e-8, 1.0))).sum(axis=1).mean()
            ),
        },
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
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Predictions: {predictions_path}")


if __name__ == "__main__":
    main()
