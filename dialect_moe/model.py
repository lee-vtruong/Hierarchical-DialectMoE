from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoModel

from .prosody import TEMPORAL_PROSODY_FEATURE_NAMES, prosody_feature_names
from .spectral import SPECTRAL_FEATURE_NAMES


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
            weights = top_weights[batch_indices, slot_indices, None].to(
                dtype=expert_output.dtype
            )
            output.index_add_(
                0,
                batch_indices,
                (expert_output * weights).to(dtype=output.dtype),
            )

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
        self.use_acoustic = bool(model_config.get("use_acoustic", True))
        self.use_prosody = bool(model_config.get("use_prosody", True))
        self.use_spectral = bool(model_config.get("use_spectral", False))
        self.use_temporal_prosody = bool(model_config.get("temporal_prosody", {}).get("enabled", False))
        self.prosody_feature_set = model_config.get(
            "prosody_feature_set", "legacy"
        )
        # Pre-H5 checkpoints always fused acoustic_dim + prosody_dim, even when
        # one branch was disabled and represented by zeros. Only configs that
        # explicitly opt into H5 feature keys use dynamic branch dimensions.
        self.dynamic_feature_fusion = (
            "use_spectral" in model_config
            or "prosody_feature_set" in model_config
        )
        # Prefer the non-pickle checkpoint format. A few explicitly configured
        # legacy research backbones only publish pytorch_model.bin; those require
        # PyTorch >= 2.6 and opt out via use_safetensors: false.
        use_safetensors = bool(model_config.get("use_safetensors", True))
        self.backbone = (
            AutoModel.from_pretrained(
                model_config["backbone"],
                use_safetensors=use_safetensors,
            )
            if self.use_acoustic
            else None
        )
        self.use_hierarchical_router = bool(
            model_config.get("use_hierarchical_router", True)
        )
        self.use_moe = bool(model_config.get("use_moe", True))
        self.router_input = model_config.get("router_input", "acoustic_prosody")
        backbone_dim = (
            int(self.backbone.config.hidden_size)
            if self.backbone is not None
            else int(model_config.get("acoustic_dim", 256))
        )

        if self.backbone is not None and model_config.get("gradient_checkpointing", False):
            self.backbone.gradient_checkpointing_enable()
        if self.backbone is not None and model_config.get("freeze_feature_encoder", True):
            freeze = getattr(self.backbone, "freeze_feature_encoder", None)
            if freeze is not None:
                freeze()
        if self.backbone is not None and model_config.get("freeze_backbone", False):
            self.backbone.requires_grad_(False)

        dropout = float(model_config["dropout"])
        acoustic_dim = int(model_config["acoustic_dim"])
        prosody_dim = int(model_config["prosody_dim"])
        spectral_dim = int(model_config.get("spectral_dim", 128))
        self.acoustic_dim = acoustic_dim
        self.prosody_dim = prosody_dim
        self.spectral_dim = spectral_dim
        fusion_dim = int(model_config["fusion_dim"])
        num_experts = int(model_config["num_experts"])

        self.acoustic_projection = MLP(backbone_dim, acoustic_dim, acoustic_dim, dropout)
        if self.use_temporal_prosody:
            temporal_config = model_config["temporal_prosody"]
            temporal_dim = int(temporal_config.get("hidden_dim", acoustic_dim))
            self.temporal_projection = nn.Sequential(
                nn.Linear(len(TEMPORAL_PROSODY_FEATURE_NAMES), temporal_dim),
                nn.LayerNorm(temporal_dim), nn.GELU(),
            )
            self.temporal_key = nn.Linear(temporal_dim, acoustic_dim)
            self.temporal_attention = nn.MultiheadAttention(
                acoustic_dim, int(temporal_config.get("num_heads", 4)),
                dropout=dropout, batch_first=True,
            )
            self.temporal_gate = nn.Sequential(nn.Linear(acoustic_dim * 2, acoustic_dim), nn.Sigmoid())
            self.temporal_norm = nn.LayerNorm(acoustic_dim)
        self.prosody_encoder = MLP(
            len(prosody_feature_names(self.prosody_feature_set)),
            prosody_dim,
            prosody_dim,
            dropout,
        )
        if self.use_spectral:
            self.spectral_encoder = MLP(
                len(SPECTRAL_FEATURE_NAMES), spectral_dim, spectral_dim, dropout
            )
        joined_dim = (
            (
                (acoustic_dim if self.use_acoustic else 0)
                + (prosody_dim if self.use_prosody else 0)
                + (spectral_dim if self.use_spectral else 0)
            )
            if self.dynamic_feature_fusion
            else acoustic_dim + prosody_dim
        )
        if joined_dim == 0:
            raise ValueError("At least one of acoustic, prosody, or spectral must be enabled")
        self.fusion_projection = nn.Linear(joined_dim, fusion_dim)
        self.fusion_gate = nn.Sequential(
            nn.Linear(joined_dim, fusion_dim),
            nn.Sigmoid(),
        )

        self.region_head = nn.Linear(fusion_dim, num_regions)
        self.region_embedding = nn.Linear(num_regions, fusion_dim, bias=False)
        # These projections are instantiated only for new H3 configurations.
        # Existing checkpoints omit router_input and remain state-dict compatible.
        if "router_input" in model_config:
            self.router_acoustic_projection = nn.Linear(acoustic_dim, fusion_dim)
            self.router_prosody_projection = nn.Linear(prosody_dim, fusion_dim)
            self.router_joint_projection = nn.Linear(
                acoustic_dim + prosody_dim, fusion_dim
            )
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
        spectral: torch.Tensor | None = None,
        temporal_prosody: torch.Tensor | None = None,
        temporal_prosody_mask: torch.Tensor | None = None,
    ) -> DialectMoEOutput:
        if self.use_acoustic:
            if self.backbone is None:
                raise RuntimeError("Acoustic input is enabled but backbone is missing")
            encoded = self.backbone(
                input_values=input_values,
                attention_mask=attention_mask,
            ).last_hidden_state
            acoustic_sequence = self.acoustic_projection(encoded)
            acoustic = self._masked_mean(acoustic_sequence, attention_mask)
            if self.use_temporal_prosody:
                if temporal_prosody is None:
                    raise ValueError("temporal_prosody is required by this config")
                temporal = self.temporal_key(self.temporal_projection(temporal_prosody))
                attended, _ = self.temporal_attention(
                    acoustic_sequence, temporal, temporal,
                    key_padding_mask=None if temporal_prosody_mask is None else ~temporal_prosody_mask.bool(),
                    need_weights=False,
                )
                temporal_acoustic = self._masked_mean(attended, attention_mask)
                gate = self.temporal_gate(torch.cat([acoustic, temporal_acoustic], -1))
                acoustic = self.temporal_norm(acoustic + gate * temporal_acoustic)
        else:
            acoustic = prosody.new_zeros(prosody.shape[0], self.acoustic_dim)
        if self.use_prosody:
            prosodic = self.prosody_encoder(prosody)
        else:
            prosodic = acoustic.new_zeros(acoustic.shape[0], self.prosody_dim)
        if spectral is None:
            spectral = prosody.new_zeros(
                prosody.shape[0], len(SPECTRAL_FEATURE_NAMES)
            )
        if self.use_spectral:
            spectral_features = self.spectral_encoder(spectral)
        else:
            spectral_features = acoustic.new_zeros(
                acoustic.shape[0], self.spectral_dim
            )
        if self.dynamic_feature_fusion:
            joined_parts = []
            if self.use_acoustic:
                joined_parts.append(acoustic)
            if self.use_prosody:
                joined_parts.append(prosodic)
            if self.use_spectral:
                joined_parts.append(spectral_features)
        else:
            joined_parts = [acoustic, prosodic]
        joined = torch.cat(joined_parts, dim=-1)
        fused = self.fusion_projection(joined) * self.fusion_gate(joined)

        region_logits = self.region_head(fused)
        if self.use_hierarchical_router:
            region_context = self.region_embedding(torch.softmax(region_logits, dim=-1))
        else:
            region_context = fused.new_zeros(fused.shape)
        if self.router_input == "acoustic":
            router_features = self.router_acoustic_projection(acoustic)
        elif self.router_input == "prosody":
            router_features = self.router_prosody_projection(prosodic)
        elif self.router_input == "acoustic_prosody":
            router_features = (
                self.router_joint_projection(joined)
                if hasattr(self, "router_joint_projection")
                else fused
            )
        else:
            raise ValueError(
                "router_input must be one of: acoustic, prosody, acoustic_prosody"
            )
        router_logits = self.router(
            torch.cat([router_features, region_context], dim=-1)
        )
        if self.use_moe:
            expert_features, balance_loss, router_probabilities = self.moe(
                fused, router_logits
            )
        else:
            expert_features = fused
            balance_loss = fused.new_zeros(())
            router_probabilities = torch.softmax(router_logits, dim=-1)
        province_logits = self.province_head(expert_features)
        return DialectMoEOutput(
            region_logits=region_logits,
            province_logits=province_logits,
            router_logits=router_logits,
            router_probabilities=router_probabilities,
            load_balance_loss=balance_loss,
        )
