from __future__ import annotations

import torch
from torch.nn import functional as F

from .proposal_model import ProposalOutput


def full_proposal_loss(
    output: ProposalOutput,
    batch: dict[str, torch.Tensor],
    loss_config: dict,
    subregion_region_matrix: torch.Tensor,
    ctc_blank_id: int = 0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    losses: dict[str, torch.Tensor] = {}
    losses["region"] = F.cross_entropy(output.region_logits, batch["region_labels"])
    losses["subregion"] = F.cross_entropy(
        output.subregion_logits, batch["subregion_labels"]
    )
    losses["province"] = F.cross_entropy(
        output.province_logits, batch["province_labels"]
    )

    region_probabilities = torch.softmax(output.region_logits, dim=-1)
    subregion_probabilities = torch.softmax(output.subregion_logits, dim=-1)
    implied_region = subregion_probabilities @ subregion_region_matrix.to(
        subregion_probabilities
    )
    losses["hierarchy"] = F.kl_div(
        region_probabilities.clamp_min(1e-8).log(),
        implied_region.clamp_min(1e-8),
        reduction="batchmean",
    )

    router_probabilities = output.router_probabilities.clamp_min(1e-8)
    losses["router"] = -(
        router_probabilities * router_probabilities.log()
    ).sum(dim=-1).mean()
    losses["load_balance"] = output.load_balance_loss

    if output.gender_logits is not None and "gender_labels" in batch:
        losses["gender"] = F.cross_entropy(
            output.gender_logits, batch["gender_labels"]
        )

    if (
        output.asr_logits is not None
        and output.asr_output_lengths is not None
        and "ctc_targets" in batch
    ):
        losses["asr"] = F.ctc_loss(
            output.asr_logits.log_softmax(-1).transpose(0, 1),
            batch["ctc_targets"],
            output.asr_output_lengths,
            batch["ctc_target_lengths"],
            blank=ctc_blank_id,
            zero_infinity=True,
        )

    weights = {
        "region": "region_weight",
        "subregion": "subregion_weight",
        "province": "province_weight",
        "gender": "gender_weight",
        "asr": "asr_weight",
        "hierarchy": "hierarchy_weight",
        "router": "router_weight",
        "load_balance": "load_balance_weight",
    }
    total = sum(
        float(loss_config.get(weights[name], 0.0)) * value
        for name, value in losses.items()
    )
    return total, {name: value.detach() for name, value in {"total": total, **losses}.items()}

