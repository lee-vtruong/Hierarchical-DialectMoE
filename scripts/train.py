from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dialect_moe.config import load_config
from dialect_moe.data import DialectCollator, load_vimd
from dialect_moe.losses import hierarchical_loss
from dialect_moe.model import HierarchicalDialectMoE
from dialect_moe.utils import move_to_device, save_json, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/vimd_moe.yaml")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, loss_config) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "region_correct": 0, "province_correct": 0, "samples": 0}
    for batch in tqdm(loader, desc="Validation", leave=False):
        batch = move_to_device(batch, device)
        output = model(batch["input_values"], batch["attention_mask"], batch["prosody"])
        loss, _ = hierarchical_loss(
            output, batch["region_labels"], batch["province_labels"], loss_config
        )
        size = batch["region_labels"].shape[0]
        totals["loss"] += loss.item() * size
        totals["region_correct"] += (
            output.region_logits.argmax(-1) == batch["region_labels"]
        ).sum().item()
        totals["province_correct"] += (
            output.province_logits.argmax(-1) == batch["province_labels"]
        ).sum().item()
        totals["samples"] += size
    count = max(totals["samples"], 1)
    return {
        "loss": totals["loss"] / count,
        "region_accuracy": totals["region_correct"] / count,
        "province_accuracy": totals["province_correct"] / count,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config["seed"]))
    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_vimd(config, max_samples=args.max_samples)
    save_json(
        {
            "regions": bundle.region_vocab.labels,
            "provinces": bundle.province_vocab.labels,
        },
        output_dir / "labels.json",
    )
    save_json(config, output_dir / "config.json")

    collator = DialectCollator(
        config["model"]["backbone"],
        config["data"],
        bundle.region_vocab,
        bundle.province_vocab,
    )
    train_loader = DataLoader(
        bundle.datasets["train"],
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        collate_fn=collator,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=True,
    )
    validation_split = "valid" if "valid" in bundle.datasets else "validation"
    validation_loader = DataLoader(
        bundle.datasets[validation_split],
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        collate_fn=collator,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HierarchicalDialectMoE(
        config["model"], len(bundle.region_vocab), len(bundle.province_vocab)
    ).to(device)
    backbone_parameters, head_parameters = [], []
    for name, parameter in model.named_parameters():
        (backbone_parameters if name.startswith("backbone.") else head_parameters).append(parameter)
    optimizer = AdamW(
        [
            {"params": backbone_parameters, "lr": float(config["training"]["learning_rate"])},
            {"params": head_parameters, "lr": float(config["training"]["head_learning_rate"])},
        ],
        weight_decay=float(config["training"]["weight_decay"]),
    )

    accumulation = int(config["training"]["gradient_accumulation_steps"])
    total_steps = math.ceil(len(train_loader) / accumulation) * int(config["training"]["epochs"])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * float(config["training"]["warmup_ratio"])),
        num_training_steps=total_steps,
    )
    use_amp = device.type == "cuda" and config["training"]["mixed_precision"] == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_epoch, best_loss, patience = 0, float("inf"), 0
    best_region_accuracy, best_province_accuracy = 0.0, 0.0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint.get("best_loss", best_loss)
        best_region_accuracy = checkpoint.get(
            "best_region_accuracy", best_region_accuracy
        )
        best_province_accuracy = checkpoint.get(
            "best_province_accuracy", best_province_accuracy
        )

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, int(config["training"]["epochs"])):
        model.train()
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for step, batch in enumerate(progress):
            batch = move_to_device(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                output = model(batch["input_values"], batch["attention_mask"], batch["prosody"])
                loss, parts = hierarchical_loss(
                    output,
                    batch["region_labels"],
                    batch["province_labels"],
                    config["loss"],
                )
                scaled_loss = loss / accumulation
            scaler.scale(scaled_loss).backward()

            should_step = (step + 1) % accumulation == 0 or step + 1 == len(train_loader)
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(config["training"]["max_grad_norm"])
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            progress.set_postfix(loss=f"{parts['loss'].item():.4f}")

        metrics = evaluate(model, validation_loader, device, config["loss"])
        print({"epoch": epoch + 1, **metrics})
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_loss": min(best_loss, metrics["loss"]),
            "best_region_accuracy": max(
                best_region_accuracy, metrics["region_accuracy"]
            ),
            "best_province_accuracy": max(
                best_province_accuracy, metrics["province_accuracy"]
            ),
            "metrics": metrics,
        }
        torch.save(state, output_dir / "last.pt")
        if metrics["loss"] < best_loss:
            best_loss, patience = metrics["loss"], 0
            torch.save(state, output_dir / "best_loss.pt")
            # Backward-compatible name used by the evaluation instructions.
            torch.save(state, output_dir / "best.pt")
        else:
            patience += 1
        if metrics["region_accuracy"] > best_region_accuracy:
            best_region_accuracy = metrics["region_accuracy"]
            torch.save(state, output_dir / "best_region_accuracy.pt")
        if metrics["province_accuracy"] > best_province_accuracy:
            best_province_accuracy = metrics["province_accuracy"]
            torch.save(state, output_dir / "best_province_accuracy.pt")
        if patience >= int(config["training"]["early_stopping_patience"]):
            print("Early stopping.")
            break


if __name__ == "__main__":
    main()
