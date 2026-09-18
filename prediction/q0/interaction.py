"""Frozen NoGraph base plus trainable cross-vehicle interaction residuals."""
from __future__ import annotations

import torch
from torch import nn

from .features import normalize_state_and_edges
from .graph import ResidualEdgeMPNNLayer, RootedTripletLayer
from .metrics import trajectory_metrics
from .model import UnifiedForecastModel


class PairwiseInteractionBranch(nn.Module):
    def __init__(self, hidden_dim: int = 64, out_dim: int = 64, layers: int = 3) -> None:
        super().__init__()
        self.layers = nn.ModuleList([ResidualEdgeMPNNLayer(hidden_dim) for _ in range(layers)])
        self.gate = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.output = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        local_h: torch.Tensor,
        edge: torch.Tensor,
        pair_mask: torch.Tensor,
        vehicle_mask: torch.Tensor,
    ) -> torch.Tensor:
        h = local_h
        for layer in self.layers:
            h = layer(h, edge, pair_mask, vehicle_mask)
        interaction = h - local_h
        gate = torch.sigmoid(self.gate(torch.cat([local_h, interaction], dim=-1)))
        out = gate * self.output(interaction)
        return torch.where(vehicle_mask[:, None, :, None], out, torch.zeros_like(out))


class PairTripletInteractionBranch(nn.Module):
    """Pairwise branch plus an explicit rooted-triplet residual."""

    def __init__(self, hidden_dim: int = 64, out_dim: int = 64, layers: int = 3) -> None:
        super().__init__()
        self.pair = PairwiseInteractionBranch(hidden_dim, out_dim, layers)
        self.triplet = RootedTripletLayer(hidden_dim)
        self.triplet_gate = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.triplet_output = nn.Linear(hidden_dim, out_dim)
        # The model starts with a real pairwise branch; the higher-order increment
        # itself starts at zero so G2 asks whether triplets add value beyond pairwise.
        nn.init.zeros_(self.triplet_output.weight)
        nn.init.zeros_(self.triplet_output.bias)

    def forward(
        self,
        local_h: torch.Tensor,
        edge: torch.Tensor,
        pair_mask: torch.Tensor,
        vehicle_mask: torch.Tensor,
    ) -> torch.Tensor:
        pair_feature = self.pair(local_h, edge, pair_mask, vehicle_mask)
        trip_h = self.triplet(local_h, edge, vehicle_mask)
        trip_interaction = trip_h - local_h
        gate = torch.sigmoid(
            self.triplet_gate(torch.cat([local_h, trip_interaction], dim=-1))
        )
        trip_feature = gate * self.triplet_output(trip_interaction)
        return torch.where(
            vehicle_mask[:, None, :, None],
            pair_feature + trip_feature,
            torch.zeros_like(pair_feature),
        )


class FrozenBaseInteractionModel(nn.Module):
    """Adds a trainable graph interaction prompt to a frozen, trained NoGraph+LLM."""

    def __init__(
        self,
        base: UnifiedForecastModel,
        branch: nn.Module,
        *,
        interaction_dim: int = 64,
        llm_dim: int = 768,
    ) -> None:
        super().__init__()
        self.base = base
        self.branch = branch
        self.interaction_projection = nn.Linear(interaction_dim, llm_dim, bias=False)
        # Exact epoch-0 equality to the frozen base.
        nn.init.zeros_(self.interaction_projection.weight)
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    def forward(self, history_state: torch.Tensor, vehicle_mask: torch.Tensor):
        stats = self.base.normalization
        standardized_state, standardized_edge, pair_mask = normalize_state_and_edges(
            history_state, vehicle_mask, stats
        )

        # This is per-vehicle processing only; it is legal shared preprocessing.
        with torch.no_grad():
            local_h = self.base.graph.encoder(standardized_state, vehicle_mask)
            base_graph = self.base.graph(
                standardized_state, standardized_edge, pair_mask, vehicle_mask
            )

        interaction = self.branch(local_h, standardized_edge, pair_mask, vehicle_mask)
        token_delta = self.interaction_projection(interaction)
        token_delta = torch.where(
            vehicle_mask[:, None, :, None], token_delta, torch.zeros_like(token_delta)
        )
        output = self.base.temporal(
            history_state,
            standardized_state,
            vehicle_mask,
            base_graph,
            token_delta=token_delta,
        )
        output["interaction_features"] = interaction
        output["token_delta"] = token_delta
        return output

    def parameter_summary(self) -> dict[str, int]:
        branch = sum(p.numel() for p in self.branch.parameters() if p.requires_grad)
        projection = sum(
            p.numel() for p in self.interaction_projection.parameters() if p.requires_grad
        )
        trainable = branch + projection
        frozen = sum(p.numel() for p in self.base.parameters())
        return {
            "interaction_branch": branch,
            "interaction_projection": projection,
            "trainable_total": trainable,
            "frozen_base": frozen,
            "total": trainable + frozen,
        }


def build_interaction_model(
    base: UnifiedForecastModel,
    kind: str,
    *,
    hidden_dim: int = 64,
    interaction_dim: int = 64,
    layers: int = 3,
) -> FrozenBaseInteractionModel:
    key = kind.lower()
    if key in {"local", "local_residual"}:
        branch = LocalResidualBranch(hidden_dim, interaction_dim, blocks=9)
    elif key in {"pairwise", "pair"}:
        branch = PairwiseInteractionBranch(hidden_dim, interaction_dim, layers)
    elif key in {"pair_triplet", "triplet"}:
        branch = PairTripletInteractionBranch(hidden_dim, interaction_dim, layers)
    else:
        raise ValueError(f"Unknown interaction branch: {kind}")
    return FrozenBaseInteractionModel(
        base, branch, interaction_dim=interaction_dim, llm_dim=768
    )



class _LocalResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.SiLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class LocalResidualBranch(nn.Module):
    """Capacity control: trainable residual that never sees another vehicle."""

    def __init__(self, hidden_dim: int = 64, out_dim: int = 64, blocks: int = 9) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_LocalResidualBlock(hidden_dim) for _ in range(blocks)])
        self.gate = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.output = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        local_h: torch.Tensor,
        edge: torch.Tensor,
        pair_mask: torch.Tensor,
        vehicle_mask: torch.Tensor,
    ) -> torch.Tensor:
        del edge, pair_mask
        h = local_h
        for block in self.blocks:
            h = block(h)
        residual = h - local_h
        gate = torch.sigmoid(self.gate(torch.cat([local_h, residual], dim=-1)))
        out = gate * self.output(residual)
        return torch.where(vehicle_mask[:, None, :, None], out, torch.zeros_like(out))

