"""QGNN attention routes LLM future states across interacting targets."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path("/home/js_cn/sensing")
BASE = ROOT / "diagnostics/core_message_seed2026_v2"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BASE))

from future_token_model import FutureTokenCoreGraphLLM


class AttentionRoutedFutureLLM(FutureTokenCoreGraphLLM):
    """Fuse temporal LLM states only through the two graph attention maps."""

    def __init__(self, graph):
        super().__init__(graph)
        self._captured_graph_attention = []
        self._attention_hooks = []
        for layer in self.graph_backbone.graph_layers:
            self._attention_hooks.append(layer.register_forward_hook(self._capture_attention))
        fusion_dim = 128
        self.routed_temporal_projection = nn.Sequential(
            nn.LayerNorm(self.d_llm), nn.Linear(self.d_llm, fusion_dim), nn.GELU()
        )
        # Inputs contain only graph-routed neighbor differences and their
        # multiplicative coupling with the target's LLM state. There is no
        # independent LLM-only or graph-only coordinate bypass.
        self.routed_interaction_head = nn.Sequential(
            nn.LayerNorm(4 * fusion_dim),
            nn.Linear(4 * fusion_dim, 256),
            nn.GELU(),
            nn.Linear(256, 2),
        )
        nn.init.zeros_(self.routed_interaction_head[-1].weight)
        nn.init.zeros_(self.routed_interaction_head[-1].bias)
        self.routed_scale_m = 0.75
        self.attention_ablation_mode = "learned"
        self.last_routed_residual_mean = None

    def _capture_attention(self, module, inputs, output):
        self._captured_graph_attention.append(output[1])

    def _neighbor_attention(self, attention, adjacency):
        batch, targets, _ = attention.shape
        eye = torch.eye(targets, dtype=torch.bool, device=attention.device)[None]
        neighbor_mask = adjacency & ~eye
        if self.attention_ablation_mode == "uniform":
            weights = neighbor_mask.to(attention.dtype)
        elif self.attention_ablation_mode == "off":
            weights = torch.zeros_like(attention)
        else:
            weights = attention * neighbor_mask
        return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)

    def forward(self, history, target_mask):
        self._captured_graph_attention = []
        output = super().forward(history, target_mask)
        assert len(self._captured_graph_attention) == 2
        batch, _, targets, _ = history.shape
        horizon = self.config.prediction_length
        hidden = output["future_hidden_features"].view(batch, targets, horizon, self.d_llm)
        projected = self.routed_temporal_projection(hidden)
        interactions = []
        route_active = None
        for attention in self._captured_graph_attention:
            weights = self._neighbor_attention(attention, output["adjacency"])
            neighbor = torch.einsum("bij,bjhd->bihd", weights, projected)
            difference = neighbor - projected
            has_neighbor = (weights.sum(dim=-1) > 0)[:, :, None, None]
            route_active = has_neighbor if route_active is None else (route_active | has_neighbor)
            difference = difference * has_neighbor
            interactions.extend([difference, projected * difference])
        joint = torch.cat(interactions, dim=-1)
        residual = torch.tanh(self.routed_interaction_head(joint)) * self.routed_scale_m
        residual = residual * route_active
        residual = residual.permute(0, 2, 1, 3) * target_mask[:, None, :, None]
        output["future_position"] = output["future_position"] + residual
        output["displacement"] = output["future_position"] - history[:, -1, None, :, :2]
        output["routed_residual"] = residual
        self.last_routed_residual_mean = residual.detach().norm(dim=-1).mean()
        return output
