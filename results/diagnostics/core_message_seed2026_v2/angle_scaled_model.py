"""Six-qubit PennyLane core with a fixed reduced data-encoding angle range."""
import math

import torch

from model import CoreMessageLayer
from target_interaction_graph import ForecasterConfig, TargetInteractionGNN


class AngleScaledCoreMessageLayer(CoreMessageLayer):
    def __init__(self, config, angle_scale):
        super().__init__(config, "quantum", quantum_backend="pennylane")
        self.angle_scale = float(angle_scale)

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
        angles = self.angle_scale * math.pi * torch.tanh(self.encoder(self.input_norm(x)))
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
        attention = torch.softmax(score.masked_fill(~adjacency[..., None], -1e4), dim=2)
        aggregate = (
            self.dropout(attention)[..., None] * gate[..., None] * base_msg
        ).sum(2).reshape(b, n, h)
        nodes = self.output_norm(nodes + self.dropout(aggregate))
        nodes = self.final_norm(nodes + self.dropout(self.ffn(nodes)))
        return nodes, attention.mean(-1)


def build_angle_scaled_graph(seed=2026, angle_scale=0.5):
    torch.manual_seed(seed)
    config = ForecasterConfig()
    model = TargetInteractionGNN(config)
    torch.manual_seed(seed + 1)
    model.graph_layers[1] = AngleScaledCoreMessageLayer(config, angle_scale)
    return model
