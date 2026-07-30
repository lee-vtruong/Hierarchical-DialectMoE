from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dialect_moe.multicrop_utils import aggregate_logits


def task_metrics(targets: list[int], predictions: list[int]) -> dict:
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "macro_f1": float(
            f1_score(targets, predictions, average="macro", zero_division=0)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="H10 multi-crop inference with mean-logit aggregation."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--strategy", choices=["first", "start_end", "uniform"], required=True
    )
    parser.add_argument("--uniform-crops", type=int, default=3)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()

    from dialect_moe.config import load_config
    from dialect_moe.data import load_vimd
    from dialect_moe.model import HierarchicalDialectMoE
    from dialect_moe.multicrop import MultiCropDialectCollator
    from dialect_moe.utils import move_to_device, save_json

    config = load_config(args.config)
    bundle = load_vimd(config)
    collator = MultiCropDialectCollator(
        config["model"]["backbone"],
        config["data"],
        bundle.region_vocab,
        bundle.province_vocab,
        use_prosody=bool(config["model"].get("use_prosody", True)),
        use_spectral=bool(config["model"].get("use_spectral", False)),
        prosody_feature_set=config["model"].get("prosody_feature_set", "legacy"),
        strategy=args.strategy,
        uniform_crops=args.uniform_crops,
    )
    num_workers = (
        int(config["data"]["num_workers"])
        if args.num_workers is None
        else args.num_workers
    )
    loader = DataLoader(
        bundle.datasets[args.split],
        batch_size=args.batch_size or int(config["training"]["batch_size"]),
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HierarchicalDialectMoE(
        config["model"], len(bundle.region_vocab), len(bundle.province_vocab)
    ).to(device)
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()

    region_targets, region_predictions = [], []
    province_targets, province_predictions = [], []
    prediction_rows = []
    all_crop_counts = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"H10 {args.strategy} {args.split}"):
            batch = move_to_device(batch, device)
            output = model(
                batch["input_values"],
                batch["attention_mask"],
                batch["prosody"],
                batch["spectral"],
            )
            sample_indices = batch["crop_sample_indices"]
            sample_count = len(batch["region_labels"])
            region_logits = aggregate_logits(
                output.region_logits, sample_indices, sample_count
            )
            province_logits = aggregate_logits(
                output.province_logits, sample_indices, sample_count
            )
            region_probs = torch.softmax(region_logits, dim=-1).cpu()
            province_probs = torch.softmax(province_logits, dim=-1).cpu()
            region_pred = region_probs.argmax(-1).tolist()
            province_pred = province_probs.argmax(-1).tolist()
            true_region = batch["region_labels"].cpu().tolist()
            true_province = batch["province_labels"].cpu().tolist()
            region_targets.extend(true_region)
            province_targets.extend(true_province)
            region_predictions.extend(region_pred)
            province_predictions.extend(province_pred)
            all_crop_counts.extend(batch["crop_counts"])
            for index in range(sample_count):
                prediction_rows.append(
                    {
                        "filename": batch["filenames"][index],
                        "speaker_id": batch["speaker_ids"][index],
                        "province_name": batch["province_names"][index],
                        "region_true_id": true_region[index],
                        "region_true": bundle.region_vocab.decode(true_region[index]),
                        "region_pred_id": region_pred[index],
                        "region_pred": bundle.region_vocab.decode(region_pred[index]),
                        "province_true_id": true_province[index],
                        "province_true": bundle.province_vocab.decode(
                            true_province[index]
                        ),
                        "province_pred_id": province_pred[index],
                        "province_pred": bundle.province_vocab.decode(
                            province_pred[index]
                        ),
                        "region_probabilities": region_probs[index].tolist(),
                        "province_probabilities": province_probs[index].tolist(),
                        "crop_count": batch["crop_counts"][index],
                        "crop_strategy": args.strategy,
                    }
                )

    metrics = {
        "strategy": args.strategy,
        "uniform_crops": args.uniform_crops,
        "samples": len(prediction_rows),
        "mean_crops_per_sample": float(np.mean(all_crop_counts)),
        "multi_crop_samples": int(np.sum(np.asarray(all_crop_counts) > 1)),
        "region": task_metrics(region_targets, region_predictions),
        "province": task_metrics(province_targets, province_predictions),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.split}_{args.strategy}"
    save_json(metrics, output_dir / f"metrics_{stem}.json")
    with (output_dir / f"predictions_{stem}.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in prediction_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Predictions: {output_dir / f'predictions_{stem}.jsonl'}")


if __name__ == "__main__":
    main()
