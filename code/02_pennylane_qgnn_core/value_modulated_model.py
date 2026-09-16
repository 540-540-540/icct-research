"""Pairwise channel-wise value modulation driven by the retained hybrid core."""
import math
import torch
from torch import nn

from model import ClassicalCore
from pennylane_core import PennyLaneStatevectorPQC
from quantum_core import QuantumCircuitConfig
from target_interaction_graph import ForecasterConfig, TargetInteractionGNN


class ValueModulatedCoreMessageLayer(nn.Module):
    """Use core measurements for attention, scalar gating, and message channels."""

    def __init__(self, config, kind):
        super().__init__()
        h = config.hidden_dim
        self.heads = config.graph_heads
        self.head_dim = h // self.heads
        self.input_norm = nn.LayerNorm(2 * h + 7)
        self.encoder = nn.Linear(2 * h + 7, 6)
        # Match the retained core initialization without advancing shared RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(2028)
            if kind == "quantum":
                self.core = PennyLaneStatevectorPQC(QuantumCircuitConfig())
            elif kind == "classical":
                self.core = ClassicalCore()
            else:
                raise ValueError(kind)
        self.latent_norm = nn.LayerNorm(12)
        self.score = nn.Linear(12, self.heads, bias=False)
        self.gate = nn.Linear(12, self.heads, bias=False)
        self.value_norm = nn.LayerNorm(h)
        self.value = nn.Linear(h, h, bias=False)
        self.edge_value = nn.Sequential(nn.Linear(7, h), nn.SiLU(), nn.Linear(h, h, bias=False))
        self.output_norm = nn.LayerNorm(h)
        self.ffn = nn.Sequential(
            nn.Linear(h, 2 * h), nn.SiLU(), nn.Dropout(config.dropout), nn.Linear(2 * h, h)
        )
        self.final_norm = nn.LayerNorm(h)
        self.dropout = nn.Dropout(config.dropout)

        # Zero initialization makes the new path exactly neutral at epoch zero:
        # 2*sigmoid(0)=1. This separates learnable value modulation from a changed
        # starting prediction while giving every message channel its own control.
        self.value_channel = nn.Linear(12, h, bias=False)
        nn.init.zeros_(self.value_channel.weight)

    def forward(self, nodes, edge_features, adjacency):
        b, n, h = nodes.shape
        pairs = torch.cat(
            [
                nodes[:, :, None, :].expand(-1, -1, n, -1),
                nodes[:, None, :, :].expand(-1, n, -1, -1),
                edge_features,
            ],
            -1,
        )
        selected = adjacency.reshape(-1).nonzero().squeeze(-1)
        x = pairs.reshape(-1, 2 * h + 7).index_select(0, selected)
        angles = math.pi * torch.tanh(self.encoder(self.input_norm(x)))
        z = self.latent_norm(self.core(angles))

        value = self.value(self.value_norm(nodes)).view(b, n, self.heads, self.head_dim)
        edge_value = self.edge_value(edge_features).view(b, n, n, self.heads, self.head_dim)
        base_msg = value[:, None, :, :, :] + edge_value

        score = nodes.new_zeros(b * n * n, self.heads).index_copy(
            0, selected, self.score(z)
        ).view(b, n, n, self.heads)
        gate = nodes.new_zeros(b * n * n, self.heads).index_copy(
            0, selected, self.gate(z)
        ).view(b, n, n, self.heads)
        gate = 2.0 * torch.sigmoid(gate)

        channel = nodes.new_zeros(b * n * n, h).index_copy(
            0, selected, self.value_channel(z)
        ).view(b, n, n, self.heads, self.head_dim)
        channel = 2.0 * torch.sigmoid(channel)
        message = channel * base_msg

        attention = torch.softmax(score.masked_fill(~adjacency[..., None], -1e4), dim=2)
        aggregate = (
            self.dropout(attention)[..., None] * gate[..., None] * message
        ).sum(2).reshape(b, n, h)
        nodes = self.output_norm(nodes + self.dropout(aggregate))
        nodes = self.final_norm(nodes + self.dropout(self.ffn(nodes)))
        return nodes, attention.mean(-1)


def build_value_graph(arm, seed=2026):
    if arm not in ("classical_value", "quantum_value"):
        raise ValueError(arm)
    torch.manual_seed(seed)
    config = ForecasterConfig()
    model = TargetInteractionGNN(config)
    torch.manual_seed(seed + 1)
    kind = arm[:-6]
    model.graph_layers[1] = ValueModulatedCoreMessageLayer(config, kind)
    return model
