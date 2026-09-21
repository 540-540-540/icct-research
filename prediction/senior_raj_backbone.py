"""Raj QGNN drop-in backbone for the reproduced senior Graph-LLM stack."""
from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from prediction.qgnn_paper_native.raj_paper import RajWeightedMultiJQGNNCore


class SeniorRajQGNN(nn.Module):
    """Match TargetInteractionGNN's output contract with a Raj multi-j core."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.core = RajWeightedMultiJQGNNCore(rounds=3)
        self.node_projection = nn.Sequential(
            nn.Linear(64, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
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
        batch, history_length, targets, channels = history.shape
        if history_length != self.config.history_length or channels != 4:
            raise ValueError("Expected history [B,%d,N,4]" % self.config.history_length)
        if target_mask.shape != (batch, targets) or target_mask.dtype != torch.bool:
            raise ValueError("Expected boolean target mask [B,N]")

        # Raj's complex matrix exponential must stay in fp32 even when the
        # frozen backbone is called inside the senior stack's AMP region.
        with torch.autocast(device_type=history.device.type, enabled=False):
            readout = self.core(history.float(), target_mask)
            nodes = self.node_projection(readout) * target_mask[..., None]

        last_position = history[:, -1, :, :2]
        last_velocity = history[:, -1, :, 2:4]
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
        pair_mask = target_mask[:, :, None] & target_mask[:, None, :]
        attention = pair_mask.to(history.dtype) / pair_mask.sum(-1, keepdim=True).clamp_min(1)
        return {
            "displacement": displacement,
            "future_position": last_position[:, None] + displacement,
            "node_features": nodes,
            "attention": attention,
            "adjacency": pair_mask,
        }


def _self_check() -> None:
    from target_interaction_graph import ForecasterConfig

    torch.manual_seed(7)
    model = SeniorRajQGNN(ForecasterConfig(dropout=0.0)).eval()
    history = torch.randn(2, 20, 5, 4)
    mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]], dtype=torch.bool)
    output = model(history, mask)
    assert output["node_features"].shape == (2, 5, 128)
    assert output["future_position"].shape == (2, 20, 5, 2)
    assert torch.isfinite(output["future_position"]).all()
    assert torch.count_nonzero(output["displacement"] * (~mask[:, None, :, None])) == 0
    permutation = torch.tensor([2, 0, 3, 1, 4])
    permuted = model(history[:, :, permutation], mask[:, permutation])["future_position"]
    assert torch.allclose(permuted, output["future_position"][:, :, permutation], atol=2e-4, rtol=2e-4)
    print("senior Raj backbone self-check passed")


if __name__ == "__main__":
    _self_check()
