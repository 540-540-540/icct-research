"""Dense uncertainty-aware bipartite graph for measurement-track association."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn

from measurement_track_dataset import EDGE_DIM, MEASUREMENT_DIM, TRACK_DIM


@dataclass
class AssociationGraphConfig:
    max_tracks: int = 8
    max_measurements: int = 12
    hidden_dim: int = 96
    graph_layers: int = 2
    dropout: float = 0.10


class BipartiteMessageLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.edge_message = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_score = nn.Linear(hidden_dim, 1)
        self.track_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.measurement_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.track_norm = nn.LayerNorm(hidden_dim)
        self.measurement_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tracks: torch.Tensor,
        measurements: torch.Tensor,
        edges: torch.Tensor,
        candidate_mask: torch.Tensor,
    ):
        track_pair = tracks[:, :, None, :].expand(-1, -1, measurements.shape[1], -1)
        measurement_pair = measurements[:, None, :, :].expand(-1, tracks.shape[1], -1, -1)
        message = self.edge_message(torch.cat([track_pair, measurement_pair, edges], dim=-1))
        score = self.edge_score(message).squeeze(-1).masked_fill(~candidate_mask, -1.0e4)

        track_attention = torch.softmax(score, dim=2) * candidate_mask
        track_attention = track_attention / track_attention.sum(dim=2, keepdim=True).clamp_min(1.0e-8)
        track_message = (track_attention[..., None] * message).sum(dim=2)

        measurement_attention = torch.softmax(score, dim=1) * candidate_mask
        measurement_attention = measurement_attention / measurement_attention.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
        measurement_message = (measurement_attention[..., None] * message).sum(dim=1)

        tracks = self.track_norm(
            tracks + self.dropout(self.track_update(torch.cat([tracks, track_message], dim=-1)))
        )
        measurements = self.measurement_norm(
            measurements
            + self.dropout(self.measurement_update(torch.cat([measurements, measurement_message], dim=-1)))
        )
        return tracks, measurements


class MeasurementTrackAssociationNet(nn.Module):
    """Edge MLP when graph_layers=0, bidirectional bipartite GNN otherwise."""

    def __init__(self, config: AssociationGraphConfig):
        super().__init__()
        self.config = config
        hidden = config.hidden_dim
        self.track_encoder = nn.Sequential(
            nn.Linear(TRACK_DIM, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(config.dropout)
        )
        self.measurement_encoder = nn.Sequential(
            nn.Linear(MEASUREMENT_DIM, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(config.dropout)
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(EDGE_DIM, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(config.dropout)
        )
        self.layers = nn.ModuleList(
            [BipartiteMessageLayer(hidden, config.dropout) for _ in range(config.graph_layers)]
        )
        self.edge_head = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, 1),
        )
        self.dustbin_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.SiLU(), nn.Linear(hidden // 2, 1)
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        track_mask = batch["track_mask"].bool()
        measurement_mask = batch["measurement_mask"].bool()
        candidate_mask = batch["candidate_mask"].bool()
        tracks = self.track_encoder(batch["track_features"])
        measurements = self.measurement_encoder(batch["measurement_features"])
        edges = self.edge_encoder(batch["edge_features"])
        tracks = tracks * track_mask[..., None]
        measurements = measurements * measurement_mask[..., None]
        for layer in self.layers:
            tracks, measurements = layer(tracks, measurements, edges, candidate_mask)
            tracks = tracks * track_mask[..., None]
            measurements = measurements * measurement_mask[..., None]

        track_pair = tracks[:, :, None, :].expand(-1, -1, measurements.shape[1], -1)
        measurement_pair = measurements[:, None, :, :].expand(-1, tracks.shape[1], -1, -1)
        logits = self.edge_head(torch.cat([track_pair, measurement_pair, edges], dim=-1)).squeeze(-1)
        logits = logits.masked_fill(~candidate_mask, -1.0e4)
        dustbin = self.dustbin_head(tracks)
        logits = torch.cat([logits, dustbin], dim=-1)
        probabilities = torch.softmax(logits, dim=-1)
        entropy = -(probabilities * probabilities.clamp_min(1.0e-9).log()).sum(dim=-1)
        entropy = entropy / torch.log(
            torch.tensor(float(logits.shape[-1]), device=logits.device, dtype=logits.dtype)
        )
        return {
            "logits": logits,
            "probabilities": probabilities,
            "association_entropy": entropy * track_mask,
        }

