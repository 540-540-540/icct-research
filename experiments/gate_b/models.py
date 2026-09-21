from __future__ import annotations

import torch
from torch import nn

from experiments.gate_a.models import NodeTCN
from prediction.qgnn_paper_native.raj_paper import RajWeightedMultiJQGNNCore
from prediction.qgnn_paper_native.raj_subset import RajMultiJJohnsonCore


class GateBRajModel(nn.Module):
    """Frozen Raj/Johnson interaction core with the Gate A trajectory contract."""

    def __init__(self, kind: str, width: int = 128, dt_s: float = 0.1, rounds: int = 3):
        super().__init__()
        if kind not in {"raj_weighted_multij_quantum", "raj_multij_johnson"}:
            raise ValueError(kind)
        self.kind, self.dt_s = kind, float(dt_s)
        self.encoder = NodeTCN(width)
        self.interaction = nn.Sequential(nn.Linear(64, width), nn.GELU(), nn.Linear(width, width))
        self.time = nn.Embedding(40, 24)
        self.decoder = nn.Sequential(
            nn.Linear(2 * width + 24 + 2, 192), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(192, 96), nn.GELU(), nn.Linear(96, 2),
        )
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)
        if kind == "raj_weighted_multij_quantum":
            self.core = RajWeightedMultiJQGNNCore(rounds=rounds)
        else:
            self.core = RajMultiJJohnsonCore(rounds=rounds, hidden=64)

    def forward(self, history: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        if history.ndim != 4 or history.shape[1] != 20 or history.shape[2] < 8 or history.shape[3] != 4:
            raise ValueError("Gate B expects history [B,20,N>=8,4]")
        if node_mask.shape != (history.shape[0], history.shape[2]) or node_mask.dtype != torch.bool:
            raise ValueError("Gate B expects boolean node_mask [B,N]")
        history, node_mask = history[:, :, :8], node_mask[:, :8]
        if not bool(node_mask[:, 0].all()):
            raise ValueError("target slot 0 must be active")
        history = torch.where(node_mask[:, None, :, None], history, 0.0)
        target = self.encoder(history)[:, 0]
        interaction = self.interaction(self.core(history, node_mask)[:, 0])
        target = torch.cat((target, interaction), -1)
        steps = torch.arange(40, device=history.device)
        times = (steps.to(history.dtype) + 1) * self.dt_s
        cv = history[:, -1, 0, 2:4][:, None] * times[None, :, None]
        x = torch.cat((target[:, None].expand(-1, 40, -1),
                       self.time(steps)[None].expand(len(history), -1, -1), cv / 20.0), -1)
        return cv + self.decoder(x) * 10.0

    def parameter_audit(self) -> dict[str, int]:
        groups = {
            "temporal_encoder": self.encoder,
            "interaction_core": self.core,
            "interaction_projection": self.interaction,
            "time_embedding": self.time,
            "decoder": self.decoder,
        }
        result = {name: sum(p.numel() for p in module.parameters()) for name, module in groups.items()}
        result["total"] = sum(p.numel() for p in self.parameters())
        return result
