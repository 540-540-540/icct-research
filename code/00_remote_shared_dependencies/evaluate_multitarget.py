"""Metrics for oracle-ID multi-target trajectory prediction."""
from __future__ import annotations

from typing import Dict

import torch


@torch.no_grad()
def trajectory_metric_sums(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
    history_last_position: torch.Tensor,
    interaction_radius_m: float = 15.0,
    collision_distance_m: float = 2.0,
) -> Dict[str, torch.Tensor]:
    """Return additive metric components.

    ``prediction`` and ``target`` are [B, T, M, 2].  Collision rate is a scene-
    time pair rate, not a target rate, and ``excess_collision_rate`` measures only
    predicted collisions that are absent from ground truth.
    """
    errors = torch.linalg.vector_norm(prediction - target, dim=-1)
    valid = target_mask[:, None, :].expand_as(errors)
    final_valid = target_mask

    last_relative = history_last_position[:, :, None, :] - history_last_position[:, None, :, :]
    last_distance = torch.linalg.vector_norm(last_relative, dim=-1)
    pair_valid = target_mask[:, :, None] & target_mask[:, None, :]
    eye = torch.eye(target_mask.shape[1], dtype=torch.bool, device=target_mask.device)[None]
    close_neighbor = ((last_distance < interaction_radius_m) & pair_valid & ~eye).any(dim=-1)
    interaction_valid = valid & close_neighbor[:, None, :]

    def collision_flags(positions: torch.Tensor) -> torch.Tensor:
        relative = positions[:, :, :, None, :] - positions[:, :, None, :, :]
        distance = torch.linalg.vector_norm(relative, dim=-1)
        valid_pairs = target_mask[:, None, :, None] & target_mask[:, None, None, :]
        upper = torch.triu(torch.ones(target_mask.shape[1], target_mask.shape[1], dtype=torch.bool, device=positions.device), diagonal=1)
        return ((distance < collision_distance_m) & valid_pairs & upper[None, None]).any(dim=(-1, -2))

    predicted_collision = collision_flags(prediction)
    target_collision = collision_flags(target)
    scene_time_count = torch.tensor(prediction.shape[0] * prediction.shape[1], device=prediction.device)

    return {
        "error_sum": errors[valid].sum(),
        "point_count": valid.sum(),
        "final_error_sum": errors[:, -1][final_valid].sum(),
        "target_count": final_valid.sum(),
        "interaction_error_sum": errors[interaction_valid].sum(),
        "interaction_point_count": interaction_valid.sum(),
        "interaction_final_error_sum": errors[:, -1][close_neighbor & final_valid].sum(),
        "interaction_target_count": (close_neighbor & final_valid).sum(),
        "predicted_collision_sum": predicted_collision.sum(),
        "target_collision_sum": target_collision.sum(),
        "excess_collision_sum": (predicted_collision & ~target_collision).sum(),
        "scene_time_count": scene_time_count,
    }


def finalize_metric_sums(sums: Dict[str, float]) -> Dict[str, float]:
    epsilon = 1.0e-9
    return {
        "ade_m": sums["error_sum"] / max(sums["point_count"], epsilon),
        "fde_m": sums["final_error_sum"] / max(sums["target_count"], epsilon),
        "interaction_ade_m": sums["interaction_error_sum"] / max(sums["interaction_point_count"], epsilon),
        "interaction_fde_m": sums["interaction_final_error_sum"] / max(sums["interaction_target_count"], epsilon),
        "collision_rate": sums["predicted_collision_sum"] / max(sums["scene_time_count"], epsilon),
        "ground_truth_collision_rate": sums["target_collision_sum"] / max(sums["scene_time_count"], epsilon),
        "excess_collision_rate": sums["excess_collision_sum"] / max(sums["scene_time_count"], epsilon),
    }
