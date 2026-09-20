"""Frozen contracts for the Automatum Q0 unified forecasting harness."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch

SOURCE_HZ = 29.97
STRIDE = 3
CANONICAL_DT = STRIDE / SOURCE_HZ
HISTORY_LENGTH = 20
PREDICTION_LENGTH = 20
MAX_VEHICLES = 8
STATE_DIM = 4
EDGE_DIM = 8
SNR_LEVELS_DB = (-10.0, -5.0, 0.0, 5.0, 10.0)
GRAPH_DIM_DEFAULT = 64


@dataclass(frozen=True)
class NormalizationStats:
    state_mean: tuple[float, float, float, float]
    state_std: tuple[float, float, float, float]
    edge_mean: tuple[float, ...]
    edge_std: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.state_mean) != STATE_DIM or len(self.state_std) != STATE_DIM:
            raise ValueError("State normalization must have four channels")
        if len(self.edge_mean) != EDGE_DIM or len(self.edge_std) != EDGE_DIM:
            raise ValueError("Edge normalization must have eight channels")
        if min(self.state_std) <= 0 or min(self.edge_std) <= 0:
            raise ValueError("Normalization std values must be positive")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state_mean": list(self.state_mean),
            "state_std": list(self.state_std),
            "edge_mean": list(self.edge_mean),
            "edge_std": list(self.edge_std),
            "dt_seconds": CANONICAL_DT,
            "history_length": HISTORY_LENGTH,
            "prediction_length": PREDICTION_LENGTH,
            "edge_fields": [
                "dx", "dy", "dvx", "dvy",
                "distance", "closing_rate", "t_cpa", "d_cpa",
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NormalizationStats":
        if abs(float(data["dt_seconds"]) - CANONICAL_DT) > 1e-12:
            raise ValueError("Normalization uses a non-canonical dt")
        return cls(
            tuple(float(x) for x in data["state_mean"]),
            tuple(float(x) for x in data["state_std"]),
            tuple(float(x) for x in data["edge_mean"]),
            tuple(float(x) for x in data["edge_std"]),
        )

    def tensors(self, *, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
        return (
            torch.tensor(self.state_mean, device=device, dtype=dtype),
            torch.tensor(self.state_std, device=device, dtype=dtype),
            torch.tensor(self.edge_mean, device=device, dtype=dtype),
            torch.tensor(self.edge_std, device=device, dtype=dtype),
        )


def validate_model_inputs(history_state: torch.Tensor, vehicle_mask: torch.Tensor) -> None:
    if history_state.ndim != 4 or history_state.shape[1:] != (
        HISTORY_LENGTH, MAX_VEHICLES, STATE_DIM
    ):
        raise ValueError(
            f"history_state must be [B,{HISTORY_LENGTH},{MAX_VEHICLES},{STATE_DIM}], "
            f"got {tuple(history_state.shape)}"
        )
    if vehicle_mask.shape != (history_state.shape[0], MAX_VEHICLES):
        raise ValueError("vehicle_mask must be [B,8]")
    if vehicle_mask.dtype != torch.bool:
        raise TypeError("vehicle_mask must be bool")
    if not torch.isfinite(history_state).all():
        raise FloatingPointError("history_state contains non-finite values")

    padding = ~vehicle_mask[:, None, :, None]
    padded_values = torch.where(padding, history_state, torch.zeros_like(history_state))
    if padded_values.abs().max().item() != 0.0:
        raise ValueError("Padded vehicle slots must be exactly zero")


def validate_future(future_state: torch.Tensor, vehicle_mask: torch.Tensor) -> None:
    if future_state.ndim != 4 or future_state.shape[1:] != (
        PREDICTION_LENGTH, MAX_VEHICLES, STATE_DIM
    ):
        raise ValueError("future_state must be [B,20,8,4]")
    if vehicle_mask.shape != (future_state.shape[0], MAX_VEHICLES):
        raise ValueError("vehicle_mask mismatch")
    valid = vehicle_mask[:, None, :, None].expand_as(future_state)
    if not torch.isfinite(future_state[valid]).all():
        raise FloatingPointError("Valid future supervision contains non-finite values")

