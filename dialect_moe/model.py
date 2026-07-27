from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoModel

from .prosody import PROSODY_FEATURE_NAMES


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class SparseMixtureOfExperts(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_experts: int,
        top_k: int,
        dropout: float,
    ):
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be between 1 and num_experts")
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, input_dim),
                )
                for _ in range(num_experts)
            ]
        )

    def forward(
        self, features: torch.Tensor, router_logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        top_logits, top_indices = router_logits.topk(self.top_k, dim=-1)
        top_weights = torch.softmax(top_logits, dim=-1)
        output = torch.zeros_like(features)

        for expert_index, expert in enumerate(self.experts):
            batch_indices, slot_indices = torch.where(top_indices == expert_index)
            if batch_indices.numel() == 0:
                continue
            expert_output = expert(features[batch_indices])
            weights = top_weights[batch_indices, slot_indices, None]
            output.index_add_(0, batch_indices, expert_output * weights)

        router_probabilities = torch.softmax(router_logits, dim=-1)
        importance = router_probabilities.mean(dim=0)
        assignments = torch.nn.functional.one_hot(
            top_indices[:, 0], num_classes=self.num_experts
        ).float().mean(dim=0)
        load_balance_loss = self.num_experts * torch.sum(importance * assignments)
        return output + features, load_balance_loss, router_probabilities


@dataclass
class DialectMoEOutput:
    region_logits: torch.Tensor
    province_logits: torch.Tensor
    router_logits: torch.Tensor
    router_probabilities: torch.Tensor
    load_balance_loss: torch.Tensor


class HierarchicalDialectMoE(nn.Module):
    def __init__(self, model_config: dict, num_regions: int, num_provinces: int):
        super().__init__()
        # Force the non-pickle checkpoint format. Recent Transformers versions
        # reject torch.load on PyTorch < 2.6 because of CVE-2025-32434.
        self.backbone = AutoModel.from_pretrained(
            model_config["backbone"],
            use_safetensors=True,
        )
        backbone_dim = int(self.backbone.config.hidden_size)

        if model_config.get("gradient_checkpointing", False):
            self.backbone.gradient_checkpointing_enable()
        if model_config.get("freeze_feature_encoder", True):
            freeze = getattr(self.backbone, "freeze_feature_encoder", None)
            if freeze is not None:
                freeze()
        if model_config.get("freeze_backbone", False):
            self.backbone.requires_grad_(False)

        dropout = float(model_config["dropout"])
        acoustic_dim = int(model_config["acoustic_dim"])
        prosody_dim = int(model_config["prosody_dim"])
        fusion_dim = int(model_config["fusion_dim"])
        num_experts = int(model_config["num_experts"])

        self.acoustic_projection = MLP(backbone_dim, acoustic_dim, acoustic_dim, dropout)
        self.prosody_encoder = MLP(
            len(PROSODY_FEATURE_NAMES), prosody_dim, prosody_dim, dropout
        )
        self.fusion_projection = nn.Linear(acoustic_dim + prosody_dim, fusion_dim)
        self.fusion_gate = nn.Sequential(
            nn.Linear(acoustic_dim + prosody_dim, fusion_dim),
            nn.Sigmoid(),
        )

        self.region_head = nn.Linear(fusion_dim, num_regions)
        self.region_embedding = nn.Linear(num_regions, fusion_dim, bias=False)
        self.router = MLP(fusion_dim * 2, fusion_dim, num_experts, dropout)
        self.moe = SparseMixtureOfExperts(
            input_dim=fusion_dim,
            hidden_dim=int(model_config["expert_hidden_dim"]),
            num_experts=num_experts,
            top_k=int(model_config["top_k"]),
            dropout=dropout,
        )
        self.province_head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_provinces),
        )

    @staticmethod
    def _masked_mean(
        hidden_states: torch.Tensor, attention_mask: torch.Tensor | None
    ) -> torch.Tensor:
        if attention_mask is None:
            return hidden_states.mean(dim=1)
        mask = torch.nn.functional.interpolate(
            attention_mask[:, None].float(),
            size=hidden_states.shape[1],
            mode="nearest",
        ).squeeze(1)
        return (hidden_states * mask[..., None]).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None,
        prosody: torch.Tensor,
    ) -> DialectMoEOutput:
        encoded = self.backbone(
            input_values=input_values,
            attention_mask=attention_mask,
        ).last_hidden_state
        acoustic = self.acoustic_projection(self._masked_mean(encoded, attention_mask))
        prosodic = self.prosody_encoder(prosody)
        joined = torch.cat([acoustic, prosodic], dim=-1)
        fused = self.fusion_projection(joined) * self.fusion_gate(joined)

        region_logits = self.region_head(fused)
        region_context = self.region_embedding(torch.softmax(region_logits, dim=-1))
        router_logits = self.router(torch.cat([fused, region_context], dim=-1))
        expert_features, balance_loss, router_probabilities = self.moe(fused, router_logits)
        province_logits = self.province_head(expert_features)
        return DialectMoEOutput(
            region_logits=region_logits,
            province_logits=province_logits,
            router_logits=router_logits,
            router_probabilities=router_probabilities,
            load_balance_loss=balance_loss,
        )
