from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def pair_metrics(state: np.ndarray, cfg: dict) -> dict[str, np.ndarray]:
    """History-bound interaction geometry from the final observed state only."""
    p, v = state[:, :2], state[:, 2:]
    r = p[None] - p[:, None]
    u = v[None] - v[:, None]
    d = np.linalg.norm(r, axis=-1)
    a = np.sum(u * u, axis=-1)
    b = np.sum(r * u, axis=-1)
    closing = -b / np.maximum(d, 1e-6)
    tcpa = np.divide(-b, a, out=np.full_like(a, np.inf), where=a > 1e-8)
    safe_tcpa = np.where(np.isfinite(tcpa), tcpa, 0.0)
    dcpa = np.linalg.norm(r + u * safe_tcpa[..., None], axis=-1)
    cpa = ((closing >= cfg["closing_mps"]) & (tcpa > 0)
           & (tcpa <= cfg["tcpa_s"]) & (dcpa <= cfg["dcpa_m"]))
    speed = np.linalg.norm(v, axis=-1)
    heading = v / np.maximum(speed[:, None], 1e-6)
    longitudinal = np.sum(r * heading[:, None], axis=-1)
    lateral = np.abs(r[..., 0] * heading[:, None, 1] - r[..., 1] * heading[:, None, 0])
    aligned = heading @ heading.T >= np.cos(np.radians(cfg["following_heading_deg"]))
    following = ((speed[:, None] >= cfg["minimum_speed_mps"])
                 & (speed[None] >= cfg["minimum_speed_mps"]) & aligned
                 & (longitudinal > 0) & (lateral <= cfg["following_lateral_m"])
                 & (longitudinal / np.maximum(speed[:, None], cfg["minimum_speed_mps"])
                    <= cfg["following_headway_s"]))
    eye = np.eye(len(state), dtype=bool)
    near = (d <= cfg["radius_m"]) & ~eye
    edge = near & (cpa | following)
    c0 = d * d - 25.0
    disc = b * b - a * c0
    ttc = np.full_like(d, np.inf)
    valid = (a > 1e-8) & (b < 0) & (disc >= 0)
    ttc[valid] = (-b[valid] - np.sqrt(disc[valid])) / a[valid]
    ttc[c0 <= 0] = 0.0
    np.fill_diagonal(ttc, np.inf)
    return {"edge": edge, "near": near, "cpa": cpa & near, "following": following & near,
            "distance": d, "closing": closing, "tcpa": tcpa, "dcpa": dcpa, "ttc": ttc}


def ego_transform(history: np.ndarray, future: np.ndarray, future_mask: np.ndarray) -> tuple:
    """Transform using target history only; target is node zero."""
    origin = history[-1, 0, :2].copy()
    velocity = history[-1, 0, 2:4]
    if np.linalg.norm(velocity) >= 0.2:
        direction = velocity
    else:
        displacement = history[-1, 0, :2] - history[0, 0, :2]
        direction = displacement if np.linalg.norm(displacement) > 1e-8 else np.array([1.0, 0.0])
    direction = direction / max(float(np.linalg.norm(direction)), 1e-8)
    rotation = np.array([[direction[0], direction[1]], [-direction[1], direction[0]]])
    out = history.copy()
    out[..., :2] = (out[..., :2] - origin) @ rotation.T
    out[..., 2:4] = out[..., 2:4] @ rotation.T
    label = np.zeros_like(future)
    label[future_mask] = (future[future_mask] - origin) @ rotation.T
    return out, label, origin, rotation


class GateADataset(Dataset):
    def __init__(self, root: str | Path, split: str, limit: int = 0, seed: int = 0):
        if split not in {"train", "val"}:
            raise ValueError("Only train/validation are legal; formal test is sealed")
        root = Path(root)
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest["status"] != "COMPLETE" or manifest["test_materialized"]:
            raise RuntimeError("Benchmark is incomplete or violates sealed-test contract")
        with np.load(root / f"{split}.npz", allow_pickle=False) as z:
            self.arrays = {k: z[k] for k in z.files}
        n = len(self.arrays["history_state"])
        _, inverse, counts = np.unique(self.arrays["time_ms"], return_inverse=True, return_counts=True)
        self.arrays["target_weight"] = (n / (len(counts) * counts[inverse])).astype(np.float32)
        self.indices = np.arange(n)
        if limit and limit < n:
            self.indices = np.random.default_rng(seed).choice(n, limit, replace=False)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = int(self.indices[index])
        return {k: torch.as_tensor(v[i]) for k, v in self.arrays.items()}
