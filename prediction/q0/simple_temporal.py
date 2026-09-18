"""Simple non-LLM downstream used only for the LLM marginal-effect factorial diagnostic."""
from __future__ import annotations

import torch
from torch import nn

from .contracts import HISTORY_LENGTH, PREDICTION_LENGTH
from .temporal import constant_velocity_baseline


class SimpleTokenAdapter(nn.Module):
    def __init__(self, graph_dim: int, hidden_dim: int = 192) -> None:
        super().__init__()
        self.state_projection = nn.Linear(4, hidden_dim)
        self.graph_projection = nn.Linear(graph_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, state: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
        if state.shape[:-1] != graph.shape[:-1]:
            raise ValueError("state/graph grids must match")
        return self.norm(self.state_projection(state) + self.graph_projection(graph))


class SimpleGRUTrajectoryPredictor(nn.Module):
    """Per-vehicle GRU decoder with no cross-vehicle attention and the same CV residual target."""

    def __init__(self, graph_dim: int, hidden_dim: int = 192, layers: int = 2) -> None:
        super().__init__()
        self.adapter = SimpleTokenAdapter(graph_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, PREDICTION_LENGTH * 2),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self,
        state_hat: torch.Tensor,
        standardized_state: torch.Tensor,
        vehicle_mask: torch.Tensor,
        graph_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        b,t,n,_ = state_hat.shape
        if t != HISTORY_LENGTH:
            raise ValueError("Unexpected history length")
        tokens = self.adapter(standardized_state, graph_features)
        x = tokens.permute(0,2,1,3).reshape(b*n,t,-1)
        active = vehicle_mask.reshape(-1).nonzero(as_tuple=True)[0]
        correction = x.new_zeros((b*n,PREDICTION_LENGTH,2))
        if active.numel():
            output,_ = self.gru(x[active])
            delta = self.head(output[:,-1]).reshape(-1,PREDICTION_LENGTH,2)
            correction = correction.index_copy(0, active, delta)
        correction = correction.reshape(b,n,PREDICTION_LENGTH,2).permute(0,2,1,3)
        pred = constant_velocity_baseline(state_hat) + correction
        pred = torch.where(vehicle_mask[:,None,:,None], pred, torch.zeros_like(pred))
        return {"prediction":pred,"origin_eligible":vehicle_mask,"correction":correction}

