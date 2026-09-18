"""Model-Agnostic Automatum Prediction Dataset Loader.

Combines:
  1. samples.npz: Ground-truth sample definitions, window metadata, and future GT [x, y, vx, vy]
  2. isac/sensing_cache.npz: Frozen Route-B ISAC sensing history [x_hat, y_hat, vx_hat, vy_hat]

Output contract per sample:
  history_state:     [20, 8, 4] float32 (from Frozen ISAC output: x_hat, y_hat, vx_hat, vy_hat)
  future_state:      [20, 8, 4] float32 (from Automatum GT: x, y, vx, vy)
  vehicle_mask:      [8] bool (True for active vehicles, False for padding)
  history_timestamp: [20] float64 (frame * 3 / 29.97)
  scene_id:          int / uint8
  start_frame:       int
  vehicle_ids:       [8] int32 (IDs for active vehicles, -1 for padding)

Formal SNR levels: [-10, -5, 0, 5, 10] dB.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    Dataset = object
    TORCH_AVAILABLE = False


SOURCE_FPS = 29.97
STRIDE = 3
CANONICAL_DT = STRIDE / SOURCE_FPS  # 0.1001001001001001 s


class AutomatumPredictionDataset(Dataset):
    """Model-agnostic dataset combining Automatum ground-truth samples with frozen ISAC sensing cache."""

    def __init__(
        self,
        split: str = "train",
        snr_db: float = 0.0,
        root_dir: Optional[Union[str, Path]] = None,
        return_tensors: bool = True,
    ) -> None:
        super().__init__()
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be one of 'train', 'val', 'test', got '{split}'")

        if root_dir is None:
            self.root = Path(__file__).resolve().parents[1]
        else:
            self.root = Path(root_dir).resolve()

        self.split = split
        self.return_tensors = return_tensors and TORCH_AVAILABLE

        # Load samples.npz
        samples_path = self.root / f"data/automatum_t_crossing/splits/{split}/samples.npz"
        if not samples_path.exists():
            raise FileNotFoundError(f"samples file not found: {samples_path}")
        samples_data = np.load(samples_path)

        self.scene_id = samples_data["scene_id"]
        self.start_frame = samples_data["start_frame"]
        self.vehicle_ids = samples_data["vehicle_ids"]
        self.vehicle_mask = samples_data["vehicle_mask"]
        self.future = samples_data["future"]
        self.num_samples = int(self.scene_id.shape[0])

        # Load sensing_cache.npz
        cache_path = self.root / f"data/automatum_t_crossing/isac/{split}/sensing_cache.npz"
        if not cache_path.exists():
            raise FileNotFoundError(f"sensing cache not found: {cache_path}")
        cache_data = np.load(cache_path)

        self.snr_levels_db = cache_data["snr_levels_db"]  # [-10, -5, 0, 5, 10]
        snr_matches = np.where(np.isclose(self.snr_levels_db, float(snr_db)))[0]
        if len(snr_matches) == 0:
            raise ValueError(
                f"snr_db={snr_db} not found in cache snr_levels_db={self.snr_levels_db.tolist()}"
            )
        self.snr_idx = int(snr_matches[0])
        self.snr_db = float(self.snr_levels_db[self.snr_idx])

        # state_hat shape: [5, M, 4]
        self.state_hat = cache_data["state_hat"]
        cache_scene = cache_data["scene_id"]
        cache_frame = cache_data["frame"]
        cache_veh = cache_data["vehicle_id"]
        M = cache_scene.shape[0]

        # In-memory fast mapping: (scene_id, frame, vehicle_id) -> cache_row_idx
        # Precompute index grid: [num_samples, 20, 8]
        lookup = {
            (int(cache_scene[i]), int(cache_frame[i]), int(cache_veh[i])): i
            for i in range(M)
        }
        self._history_indices = np.full((self.num_samples, 20, 8), -1, dtype=np.int32)
        for idx in range(self.num_samples):
            sc = int(self.scene_id[idx])
            st = int(self.start_frame[idx])
            for slot in range(8):
                if not self.vehicle_mask[idx, slot]:
                    continue
                vid = int(self.vehicle_ids[idx, slot])
                for t in range(20):
                    fr = st + t
                    self._history_indices[idx, t, slot] = lookup[(sc, fr, vid)]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if index < 0 or index >= self.num_samples:
            raise IndexError(f"index {index} out of range [0, {self.num_samples})")

        # 1. History state from ISAC cache: [20, 8, 4] float32
        idx_grid = self._history_indices[index]  # (20, 8)
        mask_slot = self.vehicle_mask[index]     # (8,)
        history = self.state_hat[self.snr_idx, idx_grid]  # (20, 8, 4)
        # Apply padding mask: padding slots must be strictly 0.0
        history = np.where(mask_slot[None, :, None], history, 0.0).astype(np.float32)

        # 2. Future state from Automatum GT: [20, 8, 4] float32
        future = self.future[index].astype(np.float32)

        # 3. Vehicle mask: [8] bool
        vehicle_mask = mask_slot.astype(np.bool_)

        # 4. History timestamps: [20] float64
        st_frame = int(self.start_frame[index])
        frames = st_frame + np.arange(20, dtype=np.int64)
        history_timestamp = frames * CANONICAL_DT

        # 5. Metadata
        scene_id = int(self.scene_id[index])
        start_frame = st_frame
        vehicle_ids = self.vehicle_ids[index].astype(np.int32)

        if self.return_tensors:
            return {
                "history_state": torch.from_numpy(history),
                "future_state": torch.from_numpy(future),
                "vehicle_mask": torch.from_numpy(vehicle_mask),
                "history_timestamp": torch.from_numpy(history_timestamp),
                "scene_id": torch.tensor(scene_id, dtype=torch.uint8),
                "start_frame": torch.tensor(start_frame, dtype=torch.int32),
                "vehicle_ids": torch.from_numpy(vehicle_ids),
            }
        else:
            return {
                "history_state": history,
                "future_state": future,
                "vehicle_mask": vehicle_mask,
                "history_timestamp": history_timestamp,
                "scene_id": scene_id,
                "start_frame": start_frame,
                "vehicle_ids": vehicle_ids,
            }
