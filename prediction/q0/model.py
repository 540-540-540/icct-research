"""Unified model wrapper for Q0 classical diagnostics and later QGNN insertion."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
from torch import nn

from .contracts import GRAPH_DIM_DEFAULT, NormalizationStats, validate_model_inputs
from .features import normalize_state_and_edges
from .graph import EdgeGATv2, EdgeResidualMPNN, GatedEdgeResidualMPNN, GatedPairTriplet, NoGraphCore
from .temporal import SharedTrajectoryPredictor


class UnifiedForecastModel(nn.Module):
    def __init__(
        self,
        graph: nn.Module,
        normalization: NormalizationStats,
        *,
        graph_dim: int = GRAPH_DIM_DEFAULT,
        checkpoint: str | None = None,
        lora_rank: int = 8,
    ) -> None:
        super().__init__()
        self.graph = graph
        self.normalization = normalization
        self.temporal = SharedTrajectoryPredictor(
            graph_dim, checkpoint=checkpoint, lora_rank=lora_rank
        )

    def forward(
        self,
        history_state: torch.Tensor,
        vehicle_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        validate_model_inputs(history_state, vehicle_mask)
        standardized_state, standardized_edges, pair_mask = normalize_state_and_edges(
            history_state, vehicle_mask, self.normalization
        )
        graph_features = self.graph(
            standardized_state, standardized_edges, pair_mask, vehicle_mask
        )
        output = self.temporal(
            history_state, standardized_state, vehicle_mask, graph_features
        )
        output["graph_features"] = graph_features
        return output

    def parameter_summary(self) -> dict[str, Any]:
        groups: dict[str, int] = defaultdict(int)
        frozen = 0
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                frozen += parameter.numel()
                continue
            if name.startswith("graph."):
                group = "graph"
            elif "lora_" in name:
                group = "lora"
            elif name.startswith("temporal.adapter."):
                group = "adapter"
            elif name.startswith("temporal.head."):
                group = "head"
            else:
                group = "other_trainable"
            groups[group] += parameter.numel()
        groups["trainable_total"] = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        groups["frozen_total"] = frozen
        groups["total"] = sum(p.numel() for p in self.parameters())
        return dict(groups)


def build_q0_model(
    name: str,
    normalization: NormalizationStats,
    *,
    graph_dim: int = GRAPH_DIM_DEFAULT,
    hidden_dim: int = 64,
    graph_layers: int = 3,
    checkpoint: str | None = None,
    lora_rank: int = 8,
    init_seed: int = 2026,
) -> UnifiedForecastModel:
    key = name.lower()
    # Graph and shared downstream use independent deterministic RNG streams.
    # This keeps adapter/LoRA/head initialization paired across graph architectures.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(init_seed) + 100_003)
        if key in {"nograph", "no_graph"}:
            graph = NoGraphCore(hidden_dim=hidden_dim, graph_dim=graph_dim)
        elif key in {"mpnn", "edge_mpnn"}:
            graph = EdgeResidualMPNN(
                hidden_dim=hidden_dim, graph_dim=graph_dim, layers=graph_layers
            )
        elif key in {"gatv2", "edge_gatv2"}:
            graph = EdgeGATv2(
                hidden_dim=hidden_dim, graph_dim=graph_dim, layers=graph_layers, heads=4
            )
        elif key in {"gated_mpnn", "gated_edge_mpnn"}:
            graph = GatedEdgeResidualMPNN(
                hidden_dim=hidden_dim, graph_dim=graph_dim, layers=graph_layers
            )
        elif key in {"pair_triplet", "triplet"}:
            graph = GatedPairTriplet(
                hidden_dim=hidden_dim, graph_dim=graph_dim, layers=graph_layers
            )
        else:
            raise ValueError(f"Unknown Q0 model: {name}")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(init_seed) + 200_003)
        model = UnifiedForecastModel(
            graph,
            normalization,
            graph_dim=graph_dim,
            checkpoint=checkpoint,
            lora_rank=lora_rank,
        )
    return model

