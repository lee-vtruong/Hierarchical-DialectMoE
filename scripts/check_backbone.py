from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dialect_moe.config import load_config
from dialect_moe.model import HierarchicalDialectMoE


def version_tuple(version: str) -> tuple[int, ...]:
    numeric = version.split("+", 1)[0]
    return tuple(int(part) for part in numeric.split(".") if part.isdigit())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and validate a configured speech backbone."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--num-regions", type=int, default=3)
    parser.add_argument("--num-provinces", type=int, default=63)
    args = parser.parse_args()

    config = load_config(args.config)
    model_config = config["model"]
    if not model_config.get("use_safetensors", True) and version_tuple(
        torch.__version__
    ) < (2, 6):
        raise RuntimeError(
            "This backbone only publishes pytorch_model.bin. "
            f"PyTorch >= 2.6 is required; found {torch.__version__}."
        )
    model = HierarchicalDialectMoE(
        model_config, args.num_regions, args.num_provinces
    )
    trainable = sum(parameter.numel() for parameter in model.parameters())
    backbone_parameters = (
        sum(parameter.numel() for parameter in model.backbone.parameters())
        if model.backbone is not None
        else 0
    )
    result = {
        "config": args.config,
        "backbone": model_config["backbone"],
        "torch": torch.__version__,
        "hidden_size": int(model.backbone.config.hidden_size),
        "backbone_parameters": backbone_parameters,
        "total_parameters": trainable,
        "use_safetensors": bool(model_config.get("use_safetensors", True)),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
