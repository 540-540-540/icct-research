"""Train-split-only normalization utilities for Q0."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from frontend.automatum_prediction_dataset import AutomatumPredictionDataset
from .contracts import EDGE_DIM, STATE_DIM, NormalizationStats
from .features import physical_edge_features


@dataclass
class _Moments:
    dim: int

    def __post_init__(self) -> None:
        self.count = 0
        self.total = torch.zeros(self.dim, dtype=torch.float64)
        self.total_sq = torch.zeros(self.dim, dtype=torch.float64)

    def add(self, values: torch.Tensor) -> None:
        if values.numel() == 0:
            return
        values = values.detach().to(device="cpu", dtype=torch.float64).reshape(-1, self.dim)
        self.count += values.shape[0]
        self.total += values.sum(dim=0)
        self.total_sq += values.square().sum(dim=0)

    def finish(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        if self.count < 2:
            raise ValueError("Insufficient samples for normalization")
        mean = self.total / self.count
        var = (self.total_sq / self.count - mean.square()).clamp_min(1e-12)
        std = var.sqrt().clamp_min(1e-6)
        return tuple(mean.tolist()), tuple(std.tolist())


def compute_train_normalization(
    snr_db: float,
    *,
    root_dir: str | Path | None = None,
    batch_size: int = 64,
    workers: int = 0,
) -> NormalizationStats:
    dataset = AutomatumPredictionDataset(
        "train", snr_db=snr_db, root_dir=root_dir, return_tensors=True
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    state_moments = _Moments(STATE_DIM)
    edge_moments = _Moments(EDGE_DIM)

    with torch.no_grad():
        for batch in loader:
            history = batch["history_state"].float()
            mask = batch["vehicle_mask"].bool()
            state_valid = mask[:, None, :].expand(history.shape[:3])
            state_moments.add(history[state_valid])

            edge, pair_mask = physical_edge_features(history, mask)
            edge_valid = pair_mask[:, None, :, :].expand(edge.shape[:4])
            edge_moments.add(edge[edge_valid])

    state_mean, state_std = state_moments.finish()
    edge_mean, edge_std = edge_moments.finish()
    return NormalizationStats(state_mean, state_std, edge_mean, edge_std)


def normalization_path(root: str | Path, snr_db: float) -> Path:
    label = f"{float(snr_db):+g}".replace("+", "p").replace("-", "m")
    return Path(root) / "reports/q0" / f"normalization_snr_{label}.json"


def save_normalization(path: str | Path, stats: NormalizationStats, snr_db: float) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = stats.to_dict() | {"snr_db": float(snr_db), "source_split": "train"}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_normalization(path: str | Path) -> NormalizationStats:
    return NormalizationStats.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

