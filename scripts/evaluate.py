from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dialect_moe.config import load_config
from dialect_moe.data import DialectCollator, load_vimd
from dialect_moe.model import HierarchicalDialectMoE
from dialect_moe.utils import move_to_device, save_json


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
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {args.split}"):
            batch = move_to_device(batch, device)
            output = model(batch["input_values"], batch["attention_mask"], batch["prosody"])
            targets_region.extend(batch["region_labels"].cpu().tolist())
            predictions_region.extend(output.region_logits.argmax(-1).cpu().tolist())
            targets_province.extend(batch["province_labels"].cpu().tolist())
            predictions_province.extend(output.province_logits.argmax(-1).cpu().tolist())
            router_probabilities.append(output.router_probabilities.cpu().numpy())

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
    }
    output_path = Path(args.checkpoint).parent / f"metrics_{args.split}.json"
    save_json(metrics, output_path)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
