"""Independent and target-interaction graph trajectory forecasters.

The graph is intentionally implemented with dense PyTorch operations: a scene has
at most a handful of targets, so installing torch-geometric would add complexity
without improving the experiment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn


@dataclass
class ForecasterConfig:
    history_length: int = 20
    prediction_length: int = 20
    dt: float = 0.1
    hidden_dim: int = 128
    graph_heads: int = 4
    graph_layers: int = 2
    graph_radius_m: float = 45.0
    dropout: float = 0.10
    position_scale: float = 20.0
    velocity_scale: float = 15.0
    residual_scale: float = 8.0


def build_edge_features(position: torch.Tensor, velocity: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return directed edge features i<-j and pairwise distances.

    Args:
        position: [B, M, 2]
        velocity: [B, M, 2]
    """
    relative_position = position[:, None, :, :] - position[:, :, None, :]
    relative_velocity = velocity[:, None, :, :] - velocity[:, :, None, :]
    distance = torch.linalg.vector_norm(relative_position, dim=-1, keepdim=True)
    direction = relative_position / distance.clamp_min(1.0)
    closing_speed = -(relative_velocity * direction).sum(dim=-1, keepdim=True)
    time_to_collision = distance / closing_speed.clamp_min(0.25)
    time_to_collision = torch.where(closing_speed > 0.0, time_to_collision, torch.full_like(time_to_collision, 20.0))
    features = torch.cat(
        [
            relative_position / 30.0,
            relative_velocity / 15.0,
            distance / 45.0,
            closing_speed / 15.0,
            time_to_collision.clamp_max(20.0) / 20.0,
        ],
        dim=-1,
    )
    return features, distance.squeeze(-1)


class DenseEdgeGraphAttention(nn.Module):
    """Multi-head graph attention using node and relative-motion edge features."""

    def __init__(self, hidden_dim: int, heads: int, dropout: float):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by graph_heads")
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.edge_bias = nn.Sequential(nn.Linear(7, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, heads))
        self.edge_value = nn.Sequential(nn.Linear(7, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        # Zero gates make a newly attached graph an exact identity mapping.  This
        # lets the graph stage start from the independently trained predictor and
        # learn only interaction corrections instead of relearning kinematics.
        self.message_gate = nn.Parameter(torch.zeros(1))
        self.feed_forward_gate = nn.Parameter(torch.zeros(1))
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        nodes: torch.Tensor,
        edge_features: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, targets, hidden = nodes.shape
        normalized = self.norm1(nodes)
        query = self.query(normalized).view(batch, targets, self.heads, self.head_dim)
        key = self.key(normalized).view(batch, targets, self.heads, self.head_dim)
        value = self.value(normalized).view(batch, targets, self.heads, self.head_dim)
        # i receives from j: [B, i, j, heads]
        scores = torch.einsum("bihd,bjhd->bijh", query, key) / (self.head_dim ** 0.5)
        scores = scores + self.edge_bias(edge_features)
        scores = scores.masked_fill(~adjacency[..., None], -1.0e4)
        attention = torch.softmax(scores, dim=2)
        attention = self.dropout(attention)

        edge_value = self.edge_value(edge_features).view(batch, targets, targets, self.heads, self.head_dim)
        messages = value[:, None, :, :, :] + edge_value
        aggregated = (attention[..., None] * messages).sum(dim=2).reshape(batch, targets, hidden)
        nodes = nodes + torch.tanh(self.message_gate) * self.dropout(self.output(aggregated))
        nodes = nodes + torch.tanh(self.feed_forward_gate) * self.dropout(self.feed_forward(self.norm2(nodes)))
        return nodes, attention.mean(dim=-1)


class MultiTargetForecaster(nn.Module):
    """Shared per-target encoder with an optional interaction graph."""

    def __init__(self, config: ForecasterConfig, use_graph: bool):
        super().__init__()
        self.config = config
        self.use_graph = use_graph
        self.history_encoder = nn.GRU(input_size=6, hidden_size=config.hidden_dim, batch_first=True)
        self.node_projection = nn.Sequential(
            nn.Linear(config.hidden_dim + 4, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
        )
        self.graph_layers = nn.ModuleList(
            [DenseEdgeGraphAttention(config.hidden_dim, config.graph_heads, config.dropout) for _ in range(config.graph_layers)]
            if use_graph
            else []
        )
        self.time_embedding = nn.Embedding(config.prediction_length, 24)
        self.decoder = nn.Sequential(
            nn.Linear(config.hidden_dim + 24 + 2, config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(config.hidden_dim // 2, 2),
        )

    def forward(self, history: torch.Tensor, target_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict future displacement from each target's final observed position.

        Args:
            history: [B, H, M, 4] with x, y, vx, vy in SI units.
            target_mask: [B, M]
        """
        batch, history_length, targets, _ = history.shape
        last_position = history[:, -1, :, :2]
        last_velocity = history[:, -1, :, 2:4]
        relative_position = (history[..., :2] - last_position[:, None, :, :]) / self.config.position_scale
        velocity = history[..., 2:4] / self.config.velocity_scale
        if history_length > 1:
            acceleration = torch.diff(velocity, dim=1, prepend=velocity[:, :1]) / self.config.dt
        else:
            acceleration = torch.zeros_like(velocity)
        encoder_input = torch.cat([relative_position, velocity, acceleration], dim=-1)
        encoder_input = encoder_input.permute(0, 2, 1, 3).reshape(batch * targets, history_length, 6)
        _, hidden = self.history_encoder(encoder_input)
        nodes = hidden[-1].view(batch, targets, self.config.hidden_dim)
        state_context = torch.cat(
            [last_position / self.config.position_scale, last_velocity / self.config.velocity_scale], dim=-1
        )
        nodes = self.node_projection(torch.cat([nodes, state_context], dim=-1))
        nodes = nodes * target_mask[..., None]

        edge_features, distances = build_edge_features(last_position, last_velocity)
        valid_pairs = target_mask[:, :, None] & target_mask[:, None, :]
        adjacency = valid_pairs & (distances <= self.config.graph_radius_m)
        eye = torch.eye(targets, dtype=torch.bool, device=history.device)[None]
        adjacency = adjacency | (eye & valid_pairs)
        attention = torch.zeros_like(distances)
        for graph_layer in self.graph_layers:
            nodes, attention = graph_layer(nodes, edge_features, adjacency)
            nodes = nodes * target_mask[..., None]

        steps = torch.arange(self.config.prediction_length, device=history.device)
        times = (steps.to(history.dtype) + 1.0) * self.config.dt
        constant_velocity = last_velocity[:, :, None, :] * times[None, None, :, None]
        time_features = self.time_embedding(steps)[None, None].expand(batch, targets, -1, -1)
        decoder_input = torch.cat(
            [
                nodes[:, :, None, :].expand(-1, -1, self.config.prediction_length, -1),
                time_features,
                constant_velocity / self.config.position_scale,
            ],
            dim=-1,
        )
        residual = self.decoder(decoder_input) * self.config.residual_scale
        displacement = constant_velocity + residual
        displacement = displacement.permute(0, 2, 1, 3) * target_mask[:, None, :, None]
        return {
            "displacement": displacement,
            "future_position": last_position[:, None, :, :] + displacement,
            "node_features": nodes,
            "attention": attention,
            "adjacency": adjacency,
        }


class IndependentGRUForecaster(MultiTargetForecaster):
    def __init__(self, config: ForecasterConfig):
        super().__init__(config=config, use_graph=False)


class TargetInteractionGNN(MultiTargetForecaster):
    def __init__(self, config: ForecasterConfig):
        super().__init__(config=config, use_graph=True)
