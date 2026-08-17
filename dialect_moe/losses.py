from __future__ import annotations

import torch
from torch.nn import functional as F

from .model import DialectMoEOutput


def hierarchical_loss(
    output: DialectMoEOutput,
    region_labels: torch.Tensor,
    province_labels: torch.Tensor,
    config: dict,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    region_loss = F.cross_entropy(output.region_logits, region_labels)
    province_loss = (
        F.nll_loss(output.conditional_province_log_probs, province_labels)
        if output.conditional_province_log_probs is not None
        else F.cross_entropy(output.province_logits, province_labels)
    )

    # Lower router entropy encourages confident specialization. Load balancing
    # prevents the confidence objective from collapsing all samples to one expert.
    probabilities = output.router_probabilities.clamp_min(1e-8)
    router_entropy = -(probabilities * probabilities.log()).sum(dim=-1).mean()
    total = (
        float(config["region_weight"]) * region_loss
        + float(config["province_weight"]) * province_loss
        + float(config["router_weight"]) * router_entropy
        + float(config["load_balance_weight"]) * output.load_balance_loss
    )
    return total, {
        "loss": total.detach(),
        "region_loss": region_loss.detach(),
        "province_loss": province_loss.detach(),
        "router_entropy": router_entropy.detach(),
        "load_balance_loss": output.load_balance_loss.detach(),
    }
