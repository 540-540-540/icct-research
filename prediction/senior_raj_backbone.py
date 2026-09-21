"""Raj interaction replacement with the senior forecaster's shared self path."""
from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from prediction.qgnn_paper_native.raj_paper import RajWeightedMultiJQGNNCore
from target_interaction_graph import MultiTargetForecaster, build_edge_features


class SeniorRajQGNN(MultiTargetForecaster):
    """Senior self path with optional local GNN and mandatory Raj interaction."""

    def __init__(
        self,
        config,
        quantum_scale: float = 0.05,
        projection_hidden: int = 128,
        projection_depth: int = 1,
        classical_layers: int = 0,
    ):
        if classical_layers < 0 or classical_layers > config.graph_layers:
            raise ValueError("classical_layers must be between zero and config.graph_layers")
        super().__init__(config=config, use_graph=classical_layers > 0)
        self.graph_layers = nn.ModuleList(list(self.graph_layers)[:classical_layers])
        self.core = RajWeightedMultiJQGNNCore(rounds=3)
        if projection_hidden <= 0 or projection_depth <= 0:
            raise ValueError("projection_hidden and projection_depth must be positive")
        projection = [nn.Linear(64, projection_hidden), nn.LayerNorm(projection_hidden), nn.SiLU()]
        for _ in range(projection_depth - 1):
            projection.extend(
                [nn.Linear(projection_hidden, projection_hidden), nn.LayerNorm(projection_hidden), nn.SiLU()]
            )
        projection.append(nn.Linear(projection_hidden, config.hidden_dim))
        self.raj_projection = nn.Sequential(*projection)
        if quantum_scale < 0:
            raise ValueError("quantum_scale must be non-negative")
        self.register_buffer("quantum_scale", torch.tensor(float(quantum_scale)))

    def set_quantum_scale(self, value: float) -> None:
        self.quantum_scale.fill_(float(value))

    def forward(self, history: torch.Tensor, target_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch, history_length, targets, channels = history.shape
        if history_length != self.config.history_length or channels != 4:
            raise ValueError("Expected history [B,%d,N,4]" % self.config.history_length)
        if target_mask.shape != (batch, targets) or target_mask.dtype != torch.bool:
            raise ValueError("Expected boolean target mask [B,N]")

        last_position = history[:, -1, :, :2]
        last_velocity = history[:, -1, :, 2:4]
        relative_position = (history[..., :2] - last_position[:, None]) / self.config.position_scale
        velocity = history[..., 2:4] / self.config.velocity_scale
        acceleration = torch.diff(velocity, dim=1, prepend=velocity[:, :1]) / self.config.dt
        encoder_input = torch.cat((relative_position, velocity, acceleration), dim=-1)
        encoder_input = encoder_input.permute(0, 2, 1, 3).reshape(batch * targets, history_length, 6)
        _, hidden = self.history_encoder(encoder_input)
        nodes = hidden[-1].view(batch, targets, self.config.hidden_dim)
        state_context = torch.cat(
            (last_position / self.config.position_scale, last_velocity / self.config.velocity_scale), dim=-1
        )
        nodes = self.node_projection(torch.cat((nodes, state_context), dim=-1))
        nodes = nodes * target_mask[..., None]

        edge_features, distances = build_edge_features(last_position, last_velocity)
        pair_mask = target_mask[:, :, None] & target_mask[:, None, :]
        adjacency = pair_mask & (distances <= self.config.graph_radius_m)
        eye = torch.eye(targets, dtype=torch.bool, device=history.device)[None]
        adjacency = adjacency | (eye & pair_mask)
        attention = adjacency.to(history.dtype) / adjacency.sum(-1, keepdim=True).clamp_min(1)
        for graph_layer in self.graph_layers:
            nodes, attention = graph_layer(nodes, edge_features, adjacency)
            nodes = nodes * target_mask[..., None]

        # Complex Raj operations stay fp32 when this frozen backbone is later
        # called from the senior GPT-2 AMP training region.
        with torch.autocast(device_type=history.device.type, enabled=False):
            raj = self.core(history.float(), target_mask)
            interaction = self.raj_projection(raj)
        nodes = (nodes + self.quantum_scale.to(nodes.dtype) * interaction) * target_mask[..., None]

        steps = torch.arange(self.config.prediction_length, device=history.device)
        times = (steps.to(history.dtype) + 1.0) * self.config.dt
        constant_velocity = last_velocity[:, :, None, :] * times[None, None, :, None]
        time_features = self.time_embedding(steps)[None, None].expand(batch, targets, -1, -1)
        decoder_input = torch.cat(
            (
                nodes[:, :, None].expand(-1, -1, self.config.prediction_length, -1),
                time_features,
                constant_velocity / self.config.position_scale,
            ),
            dim=-1,
        )
        residual = self.decoder(decoder_input) * self.config.residual_scale
        displacement = (constant_velocity + residual).permute(0, 2, 1, 3)
        displacement = displacement * target_mask[:, None, :, None]
        return {
            "displacement": displacement,
            "future_position": last_position[:, None] + displacement,
            "node_features": nodes,
            "attention": attention,
            "adjacency": adjacency,
        }


def _self_check() -> None:
    from target_interaction_graph import ForecasterConfig, IndependentGRUForecaster

    torch.manual_seed(7)
    config = ForecasterConfig(dropout=0.0)
    base = IndependentGRUForecaster(config).eval()
    model = SeniorRajQGNN(config, quantum_scale=0.0).eval()
    missing, unexpected = model.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected
    assert missing and all(key.startswith(("core.", "raj_projection.", "quantum_scale")) for key in missing)
    history = torch.randn(2, 20, 5, 4)
    mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]], dtype=torch.bool)
    base_output = base(history, mask)
    output = model(history, mask)
    assert torch.equal(output["future_position"], base_output["future_position"])
    model.set_quantum_scale(0.1)
    quantum_output = model(history, mask)
    assert not torch.equal(quantum_output["future_position"], base_output["future_position"])
    assert output["node_features"].shape == (2, 5, 128)
    assert torch.isfinite(output["future_position"]).all()
    permutation = torch.tensor([2, 0, 3, 1, 4])
    permuted = model(history[:, :, permutation], mask[:, permutation])["future_position"]
    assert torch.allclose(permuted, quantum_output["future_position"][:, :, permutation], atol=2e-4, rtol=2e-4)
    print("warm-started senior Raj backbone self-check passed")


if __name__ == "__main__":
    _self_check()
