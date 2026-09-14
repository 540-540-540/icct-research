"""Time-aligned physical risk + graph attention + LLM future-state fusion."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path("/home/js_cn/sensing")
BASE = ROOT / "diagnostics/core_message_seed2026_v2"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(BASE))

from future_token_model import FutureTokenCoreGraphLLM
from target_interaction_graph import build_edge_features


class TemporalRiskRoutedFutureLLM(FutureTokenCoreGraphLLM):
    """Route neighbor LLM states with time-specific graph relation weights."""

    def __init__(self, graph):
        super().__init__(graph)
        self._captured_graph_attention = []
        self._attention_hooks = [
            layer.register_forward_hook(self._capture_attention)
            for layer in self.graph_backbone.graph_layers
        ]
        width = 128
        self.temporal_route_projection = nn.Sequential(
            nn.LayerNorm(self.d_llm), nn.Linear(self.d_llm, width), nn.GELU()
        )
        self.temporal_route_head = nn.Sequential(
            nn.LayerNorm(4 * width),
            nn.Linear(4 * width, 256), nn.GELU(), nn.Linear(256, 2),
        )
        nn.init.zeros_(self.temporal_route_head[-1].weight)
        nn.init.zeros_(self.temporal_route_head[-1].bias)
        self.temporal_route_scale_m = 0.75
        self.attention_ablation_mode = "learned"
        self.last_temporal_route_residual_mean = None
        self.last_temporal_route_active_fraction = None

    def _capture_attention(self, module, inputs, output):
        self._captured_graph_attention.append(output[1])

    def _time_risk(self, history, adjacency):
        edge, _ = build_edge_features(history[:, -1, :, :2], history[:, -1, :, 2:4])
        targets = edge.shape[1]
        eye = torch.eye(targets, dtype=torch.bool, device=history.device)[None]
        mask = adjacency & ~eye
        distance_m = 45.0 * edge[..., 4]
        closing_scaled = edge[..., 5]
        ttc_seconds = 20.0 * edge[..., 6]
        times = torch.arange(
            1, self.config.prediction_length + 1, device=history.device, dtype=history.dtype
        ) * self.config.dt
        time_kernel = torch.exp(
            -torch.abs(ttc_seconds[:, None, :, :] - times[None, :, None, None]) / 0.75
        )
        closing_gate = torch.sigmoid(10.0 * (closing_scaled - 0.02))[:, None, :, :]
        proximity = torch.exp(-distance_m / 20.0)[:, None, :, :]
        risk = (closing_gate * time_kernel + 0.12 * proximity) * mask[:, None, :, :]
        return risk, mask

    def _route_weights(self, attention, risk, mask):
        if self.attention_ablation_mode == "off":
            base = torch.zeros_like(attention)
        elif self.attention_ablation_mode == "uniform":
            base = mask.to(attention.dtype)
        else:
            base = attention * mask
        weights = base[:, None, :, :] * risk
        return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)

    def forward(self, history, target_mask):
        self._captured_graph_attention = []
        output = super().forward(history, target_mask)
        assert len(self._captured_graph_attention) == 2
        batch, _, targets, _ = history.shape
        horizon = self.config.prediction_length
        hidden = output["future_hidden_features"].view(batch, targets, horizon, self.d_llm)
        projected = self.temporal_route_projection(hidden)
        risk, neighbor_mask = self._time_risk(history, output["adjacency"])
        interactions = []
        route_active = None
        for attention in self._captured_graph_attention:
            weights = self._route_weights(attention, risk, neighbor_mask)
            neighbor = torch.einsum("btij,bjtd->bitd", weights, projected)
            difference = neighbor - projected
            active = (weights.sum(dim=-1) > 0).permute(0, 2, 1)[:, :, :, None]
            difference = difference * active
            route_active = active if route_active is None else (route_active | active)
            interactions.extend([difference, projected * difference])
        joint = torch.cat(interactions, dim=-1)
        residual = torch.tanh(self.temporal_route_head(joint)) * self.temporal_route_scale_m
        residual = residual * route_active
        residual = residual.permute(0, 2, 1, 3) * target_mask[:, None, :, None]
        output["future_position"] = output["future_position"] + residual
        output["displacement"] = output["future_position"] - history[:, -1, None, :, :2]
        output["temporal_route_residual"] = residual
        self.last_temporal_route_residual_mean = residual.detach().norm(dim=-1).mean()
        self.last_temporal_route_active_fraction = route_active.detach().float().mean()
        return output
