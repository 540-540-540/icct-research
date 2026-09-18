"""Physical features shared by every Q0 graph method."""
from __future__ import annotations

import torch

from .contracts import CANONICAL_DT, EDGE_DIM, NormalizationStats, validate_model_inputs


def physical_edge_features(
    history_state: torch.Tensor,
    vehicle_mask: torch.Tensor,
    *,
    cpa_horizon_seconds: float | None = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return directed i<-j edge features [B,T,N,N,8] and validity mask [B,N,N]."""
    validate_model_inputs(history_state, vehicle_mask)
    if cpa_horizon_seconds is None:
        cpa_horizon_seconds = 20 * CANONICAL_DT

    position = history_state[..., :2]
    velocity = history_state[..., 2:4]

    # Axis 2 is receiver i, axis 3 is sender j.
    rel_pos = position.unsqueeze(2) - position.unsqueeze(3)
    rel_vel = velocity.unsqueeze(2) - velocity.unsqueeze(3)

    distance = torch.linalg.vector_norm(rel_pos, dim=-1)
    dot = (rel_pos * rel_vel).sum(dim=-1)
    speed_sq = rel_vel.square().sum(dim=-1)

    closing = -dot / distance.clamp_min(eps)
    tau = (-dot / speed_sq.clamp_min(eps)).clamp(0.0, float(cpa_horizon_seconds))
    cpa_vector = rel_pos + tau[..., None] * rel_vel
    d_cpa = torch.linalg.vector_norm(cpa_vector, dim=-1)

    edge = torch.cat(
        [
            rel_pos,
            rel_vel,
            distance[..., None],
            closing[..., None],
            tau[..., None],
            d_cpa[..., None],
        ],
        dim=-1,
    )
    if edge.shape[-1] != EDGE_DIM:
        raise AssertionError("Unexpected edge feature dimension")

    n = history_state.shape[2]
    not_self = ~torch.eye(n, dtype=torch.bool, device=history_state.device)
    pair_mask = vehicle_mask[:, :, None] & vehicle_mask[:, None, :] & not_self[None]
    edge = torch.where(pair_mask[:, None, :, :, None], edge, torch.zeros_like(edge))
    return edge, pair_mask


def normalize_state_and_edges(
    history_state: torch.Tensor,
    vehicle_mask: torch.Tensor,
    stats: NormalizationStats,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    state_mean, state_std, edge_mean, edge_std = stats.tensors(
        device=history_state.device, dtype=history_state.dtype
    )
    state = (history_state - state_mean) / state_std
    state = torch.where(vehicle_mask[:, None, :, None], state, torch.zeros_like(state))

    edge, pair_mask = physical_edge_features(history_state, vehicle_mask)
    edge = (edge - edge_mean) / edge_std
    edge = torch.where(pair_mask[:, None, :, :, None], edge, torch.zeros_like(edge))
    return state, edge, pair_mask

