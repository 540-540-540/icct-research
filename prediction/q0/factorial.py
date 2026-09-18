"""Factorial model wrapper for comparing Graph marginal gain with and without GPT-2."""
from __future__ import annotations

from collections import defaultdict
import torch
from torch import nn

from .contracts import NormalizationStats, validate_model_inputs
from .features import normalize_state_and_edges
from .graph import NoGraphCore, EdgeResidualMPNN
from .simple_temporal import SimpleGRUTrajectoryPredictor
from .temporal import SharedTrajectoryPredictor


class FactorialForecastModel(nn.Module):
    def __init__(self, graph: nn.Module, temporal: nn.Module, normalization: NormalizationStats):
        super().__init__()
        self.graph=graph
        self.temporal=temporal
        self.normalization=normalization

    def forward(self, history_state: torch.Tensor, vehicle_mask: torch.Tensor):
        validate_model_inputs(history_state, vehicle_mask)
        state,edge,pair=normalize_state_and_edges(history_state,vehicle_mask,self.normalization)
        gf=self.graph(state,edge,pair,vehicle_mask)
        out=self.temporal(history_state,state,vehicle_mask,gf)
        out["graph_features"]=gf
        return out

    def parameter_summary(self):
        groups=defaultdict(int); frozen=0
        for name,p in self.named_parameters():
            if not p.requires_grad:
                frozen += p.numel(); continue
            if name.startswith("graph."): g="graph"
            elif "lora_" in name: g="lora"
            elif name.startswith("temporal."): g="temporal"
            else: g="other"
            groups[g]+=p.numel()
        groups["trainable_total"]=sum(p.numel() for p in self.parameters() if p.requires_grad)
        groups["frozen_total"]=frozen
        groups["total"]=sum(p.numel() for p in self.parameters())
        return dict(groups)


def build_factorial_model(
    graph_kind: str,
    downstream: str,
    normalization: NormalizationStats,
    *,
    graph_dim: int = 64,
    hidden_dim: int = 64,
    graph_layers: int = 3,
    init_seed: int = 2026,
):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(init_seed + 100003)
        if graph_kind=="nograph":
            graph=NoGraphCore(hidden_dim,graph_dim)
        elif graph_kind=="mpnn":
            graph=EdgeResidualMPNN(hidden_dim,graph_dim,graph_layers)
        else:
            raise ValueError(graph_kind)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(init_seed + 200003)
        if downstream=="gpt2":
            temporal=SharedTrajectoryPredictor(graph_dim)
        elif downstream=="simple":
            temporal=SimpleGRUTrajectoryPredictor(graph_dim,hidden_dim=192,layers=2)
        else:
            raise ValueError(downstream)
    return FactorialForecastModel(graph,temporal,normalization)

