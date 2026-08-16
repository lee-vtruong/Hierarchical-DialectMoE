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


class LayerMix(nn.Module):
    """Learn a normalized scalar mixture over the last SSL encoder layers."""

    def __init__(self, num_layers: int):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        self.num_layers = num_layers
        # Zero initialization makes the initial interface an exact uniform mix.
        self.layer_logits = nn.Parameter(torch.zeros(num_layers))

    @property
    def weights(self) -> torch.Tensor:
        return torch.softmax(self.layer_logits, dim=0)

    def forward(self, hidden_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
        if len(hidden_states) < self.num_layers:
            raise ValueError(
                f"Backbone returned {len(hidden_states)} hidden states, but "
                f"LayerMix requires {self.num_layers}"
            )
        selected = torch.stack(hidden_states[-self.num_layers :], dim=1)
        weights = self.weights.to(dtype=selected.dtype).view(1, -1, 1, 1)
        return (selected * weights).sum(dim=1)


class AttentiveStatisticsPooling(nn.Module):
    """Pool a sequence using learned attention-weighted mean and deviation."""

    def __init__(
        self,
        input_dim: int,
        attention_hidden_dim: int,
        output_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_hidden_dim),
            nn.Tanh(),
            nn.Linear(attention_hidden_dim, 1),
        )
        self.output = nn.Sequential(
            nn.Linear(input_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        sequence: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scores = self.attention(sequence).squeeze(-1)
        if mask is not None:
            if mask.shape != scores.shape:
                raise ValueError(
                    f"Pooling mask shape {tuple(mask.shape)} does not match "
                    f"sequence scores {tuple(scores.shape)}"
                )
            scores = scores.masked_fill(~mask.bool(), torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        mean = (weights * sequence).sum(dim=1)
        variance = (weights * (sequence - mean[:, None, :]).square()).sum(dim=1)
        deviation = variance.clamp_min(1e-5).sqrt()
        return self.output(torch.cat([mean, deviation], dim=-1))


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

        layer_mix_config = model_config.get("layer_mix", {})
        if not isinstance(layer_mix_config, dict):
            raise ValueError("model.layer_mix must be a mapping")
        self.use_layer_mix = bool(layer_mix_config.get("enabled", False))
        self.layer_mix_last_n = int(layer_mix_config.get("last_n_layers", 8))
        if self.use_layer_mix:
            if self.backbone is None:
                raise ValueError("LayerMix requires model.use_acoustic=true")
            available_layers = int(self.backbone.config.num_hidden_layers) + 1
            if self.layer_mix_last_n > available_layers:
                raise ValueError(
                    f"last_n_layers={self.layer_mix_last_n} exceeds the "
                    f"{available_layers} hidden states exposed by the backbone"
                )
            self.layer_mixer = LayerMix(self.layer_mix_last_n)

        pooling_config = model_config.get("acoustic_pooling", "mean")
        if isinstance(pooling_config, str):
            pooling_type = pooling_config
            pooling_config = {"type": pooling_type}
        elif isinstance(pooling_config, dict):
            pooling_type = pooling_config.get("type", "mean")
        else:
            raise ValueError("model.acoustic_pooling must be a string or mapping")
        if pooling_type not in {"mean", "attentive_statistics"}:
            raise ValueError(
                "model.acoustic_pooling.type must be mean or attentive_statistics"
            )
        self.acoustic_pooling_type = pooling_type

        self.acoustic_projection = MLP(backbone_dim, acoustic_dim, acoustic_dim, dropout)
        if self.acoustic_pooling_type == "attentive_statistics":
            self.attentive_statistics_pooling = AttentiveStatisticsPooling(
                input_dim=acoustic_dim,
                attention_hidden_dim=int(
                    pooling_config.get("attention_hidden_dim", 128)
                ),
                output_dim=acoustic_dim,
                dropout=dropout,
            )
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
    def _frame_mask(
        attention_mask: torch.Tensor | None,
        sequence_length: int,
    ) -> torch.Tensor | None:
        if attention_mask is None:
            return None
        return torch.nn.functional.interpolate(
            attention_mask[:, None].float(),
            size=sequence_length,
            mode="nearest",
        ).squeeze(1).bool()

    @staticmethod
    def _masked_mean(
        hidden_states: torch.Tensor, attention_mask: torch.Tensor | None
    ) -> torch.Tensor:
        if attention_mask is None:
            return hidden_states.mean(dim=1)
        mask = HierarchicalDialectMoE._frame_mask(
            attention_mask, hidden_states.shape[1]
        ).float()
        return (hidden_states * mask[..., None]).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)

    def _pool_acoustic(
        self,
        sequence: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.acoustic_pooling_type == "mean":
            return self._masked_mean(sequence, attention_mask)
        frame_mask = self._frame_mask(attention_mask, sequence.shape[1])
        return self.attentive_statistics_pooling(sequence, frame_mask)

    def representation_diagnostics(self) -> dict:
        diagnostics = {
            "acoustic_pooling": self.acoustic_pooling_type,
            "layer_mix_enabled": self.use_layer_mix,
            "layer_mix_last_n": self.layer_mix_last_n if self.use_layer_mix else 1,
            "layer_weights": None,
        }
        if self.use_layer_mix:
            diagnostics["layer_weights"] = (
                self.layer_mixer.weights.detach().float().cpu().tolist()
            )
        return diagnostics

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
            backbone_output = self.backbone(
                input_values=input_values,
                attention_mask=attention_mask,
                output_hidden_states=self.use_layer_mix,
            )
            encoded = (
                self.layer_mixer(backbone_output.hidden_states)
                if self.use_layer_mix
                else backbone_output.last_hidden_state
            )
            acoustic_sequence = self.acoustic_projection(encoded)
            acoustic = self._pool_acoustic(acoustic_sequence, attention_mask)
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
