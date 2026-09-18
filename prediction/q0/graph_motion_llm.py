"""Graph + motion-token GPT-2 co-core diagnostic model."""
from __future__ import annotations

from collections import defaultdict
import torch
from torch import nn

from .contracts import NormalizationStats, validate_model_inputs
from .features import normalize_state_and_edges
from .graph import EdgeResidualMPNN, GatedPairTriplet, NoGraphCore, PhysicsRoutedEdgeMPNN
from .motion_token_llm import MotionTokenGPT2Core


class GraphMotionLLM(nn.Module):
    def __init__(self, graph, llm, normalization: NormalizationStats):
        super().__init__()
        self.graph = graph
        self.llm = llm
        self.normalization = normalization

    def forward(self, history_state, vehicle_mask):
        validate_model_inputs(history_state, vehicle_mask)
        state, edge, pair = normalize_state_and_edges(history_state, vehicle_mask, self.normalization)
        graph_features = self.graph(state, edge, pair, vehicle_mask)
        out = self.llm(history_state, vehicle_mask, graph_features)
        out["graph_features"] = graph_features
        return out

    def parameter_summary(self):
        groups=defaultdict(int); frozen=0
        for name,p in self.named_parameters():
            if not p.requires_grad:
                frozen += p.numel(); continue
            if name.startswith("graph."): group="graph"
            elif "lora_" in name: group="lora"
            elif name.startswith("llm."): group="llm_non_lora"
            else: group="other"
            groups[group]+=p.numel()
        groups["trainable_total"]=sum(p.numel() for p in self.parameters() if p.requires_grad)
        groups["frozen_total"]=frozen
        groups["total"]=sum(p.numel() for p in self.parameters())
        return dict(groups)


def build_graph_motion_llm(
    graph_kind: str,
    normalization: NormalizationStats,
    *,
    graph_dim: int = 64,
    hidden_dim: int = 64,
    graph_layers: int = 3,
    llm_layers: int = 4,
    lora_rank: int = 8,
    init_seed: int = 2026,
    tokenizer_config=None,
):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(init_seed + 100_003)
        if graph_kind=="nograph":
            graph=NoGraphCore(hidden_dim,graph_dim)
        elif graph_kind=="mpnn":
            graph=EdgeResidualMPNN(hidden_dim,graph_dim,graph_layers)
        elif graph_kind=="routed_mpnn":
            graph=PhysicsRoutedEdgeMPNN(hidden_dim,graph_dim,graph_layers)
        elif graph_kind=="pair_triplet":
            graph=GatedPairTriplet(hidden_dim,graph_dim,graph_layers)
        else:
            raise ValueError(graph_kind)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(init_seed + 300_003)
        llm=MotionTokenGPT2Core(
            graph_dim=graph_dim,llm_layers=llm_layers,lora_rank=lora_rank,
            tokenizer_config=tokenizer_config,
        )
    return GraphMotionLLM(graph,llm,normalization)

