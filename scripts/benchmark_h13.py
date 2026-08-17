from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def parameter_counts(model: torch.nn.Module) -> dict[str, int]:
    return {
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "backbone_parameters": sum(
            parameter.numel()
            for parameter in model.backbone.parameters()
        )
        if model.backbone is not None
        else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Controlled H13 GPU forward-pass benchmark."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup-repeats", type=int, default=1)
    parser.add_argument("--timed-repeats", type=int, default=3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.max_samples < 1 or args.batch_size < 1:
        raise ValueError("--max-samples and --batch-size must be positive")
    if args.warmup_repeats < 0 or args.timed_repeats < 1:
        raise ValueError("Require warmup >= 0 and timed repeats >= 1")
    if not torch.cuda.is_available():
        raise RuntimeError("H13 benchmark requires an allocated CUDA GPU")

    # Keep --help and lightweight imports usable without the audio stack.
    from dialect_moe.config import load_config
    from dialect_moe.data import DialectCollator, load_vimd
    from dialect_moe.model import HierarchicalDialectMoE
    from dialect_moe.utils import move_to_device

    config = load_config(args.config)
    config["data"]["num_workers"] = 0
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
        batch_size=args.batch_size,
        collate_fn=collator,
        num_workers=0,
        pin_memory=False,
        shuffle=False,
    )
    # Precompute CPU features so disk access and feature extraction are excluded.
    batches = list(loader)
    if not batches:
        raise ValueError("Benchmark split produced no batches")

    device = torch.device("cuda")
    model = HierarchicalDialectMoE(
        config["model"],
        len(bundle.region_vocab),
        len(bundle.province_vocab),
        province_to_region=bundle.province_to_region,
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    def forward_once() -> int:
        sample_count = 0
        with torch.inference_mode():
            for cpu_batch in batches:
                batch = move_to_device(cpu_batch, device)
                model(
                    batch["input_values"],
                    batch["attention_mask"],
                    batch["prosody"],
                    batch["spectral"],
                )
                sample_count += int(batch["input_values"].shape[0])
        torch.cuda.synchronize()
        return sample_count

    for _ in range(args.warmup_repeats):
        forward_once()
    torch.cuda.reset_peak_memory_stats()
    elapsed = []
    samples = 0
    for _ in range(args.timed_repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        samples = forward_once()
        elapsed.append(time.perf_counter() - start)

    mean_seconds = sum(elapsed) / len(elapsed)
    result = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "backbone": config["model"]["backbone"],
        "use_prosody": bool(config["model"].get("use_prosody", True)),
        "split": args.split,
        "samples": samples,
        "batch_size": args.batch_size,
        "warmup_repeats": args.warmup_repeats,
        "timed_repeats": args.timed_repeats,
        "elapsed_seconds": elapsed,
        "mean_seconds": mean_seconds,
        "samples_per_second": samples / mean_seconds,
        "milliseconds_per_sample": mean_seconds * 1000.0 / samples,
        "peak_cuda_memory_mib": torch.cuda.max_memory_allocated() / (1024**2),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        **parameter_counts(model),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Wrote benchmark to {output}")


if __name__ == "__main__":
    main()
