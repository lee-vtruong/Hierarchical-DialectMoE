from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoModel

from .model import MLP, SparseMixtureOfExperts


class ProsodySequenceEncoder(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        output_dim: int,
        encoder_type: str,
        layers: int,
        heads: int,
        dropout: float,
    ):
        super().__init__()
        self.encoder_type = encoder_type
        self.input_projection = nn.Linear(feature_dim, hidden_dim)
        if encoder_type == "mlp":
            self.encoder = MLP(hidden_dim, hidden_dim, hidden_dim, dropout)
        elif encoder_type == "cnn1d":
            self.encoder = nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, 5, padding=2),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
                nn.GELU(),
            )
        elif encoder_type == "bilstm":
            self.encoder = nn.LSTM(
                hidden_dim,
                hidden_dim // 2,
                num_layers=layers,
                dropout=dropout if layers > 1 else 0.0,
                batch_first=True,
                bidirectional=True,
            )
        elif encoder_type == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        else:
            raise ValueError(f"Unsupported prosody encoder: {encoder_type}")
        self.output_projection = nn.Linear(hidden_dim, output_dim)

    def forward(
        self, sequence: torch.Tensor, mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.input_projection(sequence)
        if self.encoder_type == "cnn1d":
            hidden = self.encoder(hidden.transpose(1, 2)).transpose(1, 2)
        elif self.encoder_type == "bilstm":
            hidden, _ = self.encoder(hidden)
        elif self.encoder_type == "transformer":
            hidden = self.encoder(hidden, src_key_padding_mask=None if mask is None else ~mask.bool())
        else:
            hidden = self.encoder(hidden)
        hidden = self.output_projection(hidden)
        if mask is None:
            pooled = hidden.mean(dim=1)
        else:
            weights = mask.float()
            pooled = (hidden * weights[..., None]).sum(dim=1) / weights.sum(
                dim=1, keepdim=True
            ).clamp_min(1)
        return hidden, pooled


class FeatureFusion(nn.Module):
    def __init__(
        self,
        acoustic_dim: int,
        prosody_dim: int,
        output_dim: int,
        fusion_type: str,
        dropout: float,
    ):
        super().__init__()
        self.fusion_type = fusion_type
        joined_dim = acoustic_dim + prosody_dim
        if fusion_type == "concat":
            self.fusion = nn.Sequential(
                nn.Linear(joined_dim, output_dim), nn.GELU(), nn.Dropout(dropout)
            )
        elif fusion_type == "gated":
            self.value = nn.Linear(joined_dim, output_dim)
            self.gate = nn.Sequential(nn.Linear(joined_dim, output_dim), nn.Sigmoid())
        elif fusion_type == "cross_attention":
            self.acoustic_projection = nn.Linear(acoustic_dim, output_dim)
            self.prosody_projection = nn.Linear(prosody_dim, output_dim)
            self.attention = nn.MultiheadAttention(
                output_dim, num_heads=4, dropout=dropout, batch_first=True
            )
            self.norm = nn.LayerNorm(output_dim)
        elif fusion_type == "bilinear":
            self.bilinear = nn.Bilinear(acoustic_dim, prosody_dim, output_dim)
            self.residual = nn.Linear(joined_dim, output_dim)
        else:
            raise ValueError(f"Unsupported fusion: {fusion_type}")

    def forward(
        self,
        acoustic_pooled: torch.Tensor,
        prosody_pooled: torch.Tensor,
        acoustic_sequence: torch.Tensor | None = None,
        prosody_sequence: torch.Tensor | None = None,
        prosody_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        joined = torch.cat([acoustic_pooled, prosody_pooled], dim=-1)
        if self.fusion_type == "concat":
            return self.fusion(joined)
        if self.fusion_type == "gated":
            return self.value(joined) * self.gate(joined)
        if self.fusion_type == "bilinear":
            return self.bilinear(acoustic_pooled, prosody_pooled) + self.residual(joined)
        query = self.acoustic_projection(acoustic_sequence)
        key_value = self.prosody_projection(prosody_sequence)
        attended, _ = self.attention(
            query,
            key_value,
            key_value,
            key_padding_mask=None if prosody_mask is None else ~prosody_mask.bool(),
            need_weights=False,
        )
        return self.norm(query + attended).mean(dim=1)


@dataclass
class ProposalOutput:
    region_logits: torch.Tensor
    subregion_logits: torch.Tensor
    province_logits: torch.Tensor
    gender_logits: torch.Tensor | None
    asr_logits: torch.Tensor | None
    asr_output_lengths: torch.Tensor | None
    router_logits: torch.Tensor
    router_probabilities: torch.Tensor
    load_balance_loss: torch.Tensor
    fused_features: torch.Tensor


class FullProposalDialectMoE(nn.Module):
    """Stages 1-6 of the proposal in one configurable architecture."""

    def __init__(
        self,
        config: dict,
        num_regions: int,
        num_subregions: int,
        num_provinces: int,
        prosody_feature_dim: int,
        vocabulary_size: int = 0,
        num_genders: int = 2,
    ):
        super().__init__()
        model_config = config["model"]
        dropout = float(model_config["dropout"])
        self.tasks = model_config["tasks"]
        self.router_config = model_config["router"]
        self.backbone = AutoModel.from_pretrained(model_config["backbone"])
        backbone_dim = int(self.backbone.config.hidden_size)
        if model_config.get("gradient_checkpointing"):
            self.backbone.gradient_checkpointing_enable()
        if model_config.get("freeze_feature_encoder"):
            freeze = getattr(self.backbone, "freeze_feature_encoder", None)
            if freeze:
                freeze()
        if model_config.get("freeze_backbone"):
            self.backbone.requires_grad_(False)

        acoustic_dim = int(model_config["acoustic_dim"])
        prosody_dim = int(model_config["prosody_dim"])
        fusion_dim = int(model_config["fusion_dim"])
        prosody_config = model_config["prosody"]
        self.acoustic_projection = nn.Linear(backbone_dim, acoustic_dim)
        self.prosody_encoder = ProsodySequenceEncoder(
            prosody_feature_dim,
            int(prosody_config["sequence_hidden_dim"]),
            prosody_dim,
            prosody_config["encoder"],
            int(prosody_config["sequence_layers"]),
            int(prosody_config["sequence_heads"]),
            dropout,
        )
        self.fusion = FeatureFusion(
            acoustic_dim,
            prosody_dim,
            fusion_dim,
            model_config["fusion"]["type"],
            dropout,
        )

        self.region_head = nn.Linear(fusion_dim, num_regions)
        self.region_context = nn.Linear(num_regions, fusion_dim, bias=False)
        self.subregion_head = MLP(fusion_dim * 2, fusion_dim, num_subregions, dropout)
        self.subregion_context = nn.Linear(num_subregions, fusion_dim, bias=False)
        self.expert_router = MLP(fusion_dim * 3, fusion_dim, model_config["moe"]["num_experts"], dropout)
        self.moe = SparseMixtureOfExperts(
            fusion_dim,
            int(model_config["moe"]["expert_hidden_dim"]),
            int(model_config["moe"]["num_experts"]),
            int(model_config["moe"]["top_k"]),
            dropout,
        )
        self.province_head = nn.Linear(fusion_dim, num_provinces)
        self.gender_head = nn.Linear(fusion_dim, num_genders) if self.tasks.get("gender") else None
        self.asr_head = (
            nn.Linear(backbone_dim, vocabulary_size)
            if self.tasks.get("asr") and vocabulary_size > 0
            else None
        )

    @staticmethod
    def _pool(sequence: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            return sequence.mean(dim=1)
        resized = torch.nn.functional.interpolate(
            mask[:, None].float(), size=sequence.shape[1], mode="nearest"
        ).squeeze(1)
        return (sequence * resized[..., None]).sum(1) / resized.sum(1, keepdim=True).clamp_min(1)

    def _route_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        mode = self.router_config["mode"]
        temperature = float(self.router_config.get("temperature", 1.0))
        if mode == "soft":
            return torch.softmax(logits / temperature, dim=-1)
        if mode == "hard":
            return torch.nn.functional.gumbel_softmax(
                logits, tau=temperature, hard=True, dim=-1
            )
        if mode == "random":
            indices = torch.randint(logits.shape[-1], (logits.shape[0],), device=logits.device)
            return torch.nn.functional.one_hot(indices, logits.shape[-1]).float()
        raise ValueError(f"Unsupported router mode: {mode}")

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None,
        prosody_sequence: torch.Tensor,
        prosody_mask: torch.Tensor | None,
    ) -> ProposalOutput:
        backbone_output = self.backbone(
            input_values=input_values, attention_mask=attention_mask
        ).last_hidden_state
        acoustic_sequence = self.acoustic_projection(backbone_output)
        acoustic_pooled = self._pool(acoustic_sequence, attention_mask)
        prosody_hidden, prosody_pooled = self.prosody_encoder(prosody_sequence, prosody_mask)
        fused = self.fusion(
            acoustic_pooled,
            prosody_pooled,
            acoustic_sequence,
            prosody_hidden,
            prosody_mask,
        )

        region_logits = self.region_head(fused)
        region_probability = self._route_probabilities(region_logits)
        region_context = self.region_context(region_probability)
        subregion_logits = self.subregion_head(torch.cat([fused, region_context], dim=-1))
        subregion_probability = self._route_probabilities(subregion_logits)
        subregion_context = self.subregion_context(subregion_probability)
        router_logits = self.expert_router(
            torch.cat([fused, region_context, subregion_context], dim=-1)
        )
        if self.router_config["mode"] == "random":
            router_logits = torch.rand_like(router_logits)
        expert_features, balance_loss, router_probabilities = self.moe(fused, router_logits)

        asr_logits = self.asr_head(backbone_output) if self.asr_head is not None else None
        if asr_logits is not None:
            if attention_mask is None:
                asr_lengths = torch.full(
                    (input_values.shape[0],), asr_logits.shape[1], device=input_values.device
                )
            else:
                ratio = asr_logits.shape[1] / attention_mask.shape[1]
                asr_lengths = (attention_mask.sum(-1) * ratio).floor().long().clamp_min(1)
        else:
            asr_lengths = None
        return ProposalOutput(
            region_logits=region_logits,
            subregion_logits=subregion_logits,
            province_logits=self.province_head(expert_features),
            gender_logits=None if self.gender_head is None else self.gender_head(expert_features),
            asr_logits=asr_logits,
            asr_output_lengths=asr_lengths,
            router_logits=router_logits,
            router_probabilities=router_probabilities,
            load_balance_loss=balance_loss,
            fused_features=fused,
        )

