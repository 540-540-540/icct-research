"""Losses and scene-macro metrics for Q0."""
from __future__ import annotations

import torch

from .contracts import validate_future


def trajectory_metrics(
    prediction: torch.Tensor,
    future_state: torch.Tensor,
    vehicle_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    validate_future(future_state, vehicle_mask)
    future_position = future_state[..., :2]
    if prediction.shape != future_position.shape:
        raise ValueError("prediction must match future positions [B,20,8,2]")
    if not torch.isfinite(prediction).all():
        raise FloatingPointError("prediction contains non-finite values")

    valid = vehicle_mask[:, None, :].expand(prediction.shape[:3])
    distance = torch.linalg.vector_norm(
        torch.where(valid[..., None], prediction - future_position, 0.0), dim=-1
    )

    vehicle_count = vehicle_mask.sum(dim=1).clamp_min(1)
    scene_ade = distance.sum(dim=(1, 2)) / (vehicle_count * prediction.shape[1])
    scene_fde = distance[:, -1].sum(dim=1) / vehicle_count
    loss = (scene_ade + 0.5 * scene_fde).mean()
    return {
        "loss": loss,
        "scene_ade": scene_ade,
        "scene_fde": scene_fde,
        "ADE": scene_ade.mean(),
        "FDE": scene_fde.mean(),
        "J": scene_ade.mean() + 0.5 * scene_fde.mean(),
    }

