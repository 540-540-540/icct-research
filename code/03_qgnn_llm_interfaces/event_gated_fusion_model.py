"""Interaction-event-gated fusion of QGNN relations and LLM future states."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path("/home/js_cn/sensing")
BASE = ROOT / "diagnostics/core_message_seed2026_v2"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(BASE))

from future_token_model import FutureTokenCoreGraphLLM


class EventGatedFutureLLM(FutureTokenCoreGraphLLM):
    """QGNN answers who interacts; LLM predicts when; event gates correction."""

    def __init__(self, graph):
        super().__init__(graph)
        self._captured_attention = []
        self._hooks = [layer.register_forward_hook(self._capture) for layer in self.graph_backbone.graph_layers]
        width = 128
        self.event_route_projection = nn.Sequential(
            nn.LayerNorm(self.d_llm), nn.Linear(self.d_llm, width), nn.GELU()
        )
        self.event_classifier = nn.Sequential(
            nn.LayerNorm(4 * width), nn.Linear(4 * width, 192), nn.GELU(), nn.Linear(192, 1)
        )
        nn.init.constant_(self.event_classifier[-1].bias, -1.5)
        self.event_coordinate_head = nn.Sequential(
            nn.LayerNorm(4 * width), nn.Linear(4 * width, 256), nn.GELU(), nn.Linear(256, 2)
        )
        nn.init.zeros_(self.event_coordinate_head[-1].weight)
        nn.init.zeros_(self.event_coordinate_head[-1].bias)
        self.event_coordinate_scale_m = 0.75
        self.attention_ablation_mode = "learned"
        self.event_gate_ablation = False
        self.last_event_residual_mean = None

    def _capture(self, module, inputs, output):
        self._captured_attention.append(output[1])

    def _weights(self, attention, adjacency):
        targets = attention.shape[1]
        eye = torch.eye(targets, dtype=torch.bool, device=attention.device)[None]
        mask = adjacency & ~eye
        if self.attention_ablation_mode == "off": base = torch.zeros_like(attention)
        elif self.attention_ablation_mode == "uniform": base = mask.to(attention.dtype)
        else: base = attention * mask
        return base / base.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)

    def forward(self, history, target_mask):
        self._captured_attention = []
        output = super().forward(history, target_mask)
        assert len(self._captured_attention) == 2
        batch, _, targets, _ = history.shape; horizon = self.config.prediction_length
        hidden = output["future_hidden_features"].view(batch, targets, horizon, self.d_llm)
        own = self.event_route_projection(hidden)
        interactions = []; active_any = None
        for attention in self._captured_attention:
            weights = self._weights(attention, output["adjacency"])
            neighbor = torch.einsum("bij,bjtd->bitd", weights, own)
            active = (weights.sum(-1) > 0)[:, :, None, None]
            difference = (neighbor - own) * active
            active_any = active if active_any is None else (active_any | active)
            interactions.extend([difference, own * difference])
        joint = torch.cat(interactions, dim=-1)
        logits = self.event_classifier(joint).squeeze(-1)  # [B,N,H]
        probability = torch.sigmoid(logits)
        gate = torch.ones_like(probability) if self.event_gate_ablation else probability
        residual = torch.tanh(self.event_coordinate_head(joint)) * self.event_coordinate_scale_m
        residual = residual * gate[..., None] * active_any
        residual = residual.permute(0, 2, 1, 3) * target_mask[:, None, :, None]
        output["future_position"] = output["future_position"] + residual
        output["displacement"] = output["future_position"] - history[:, -1, None, :, :2]
        output["interaction_logits"] = logits.permute(0, 2, 1)
        output["interaction_probability"] = probability.permute(0, 2, 1)
        output["event_residual"] = residual
        self.last_event_residual_mean = residual.detach().norm(dim=-1).mean()
        return output

    @staticmethod
    def interaction_labels(history, future, target_mask, distance_threshold_m=12.0):
        positions = future[..., :2]
        previous = torch.cat([history[:, -1:, :, :2], positions[:, :-1]], dim=1)
        distance = torch.linalg.vector_norm(positions[:, :, :, None, :] - positions[:, :, None, :, :], dim=-1)
        previous_distance = torch.linalg.vector_norm(previous[:, :, :, None, :] - previous[:, :, None, :, :], dim=-1)
        targets = positions.shape[2]
        valid = target_mask[:, None, :, None] & target_mask[:, None, None, :]
        valid = valid & ~torch.eye(targets, dtype=torch.bool, device=positions.device)[None, None]
        event_edges = valid & (distance <= distance_threshold_m) & (distance < previous_distance)
        return event_edges.any(dim=-1)

