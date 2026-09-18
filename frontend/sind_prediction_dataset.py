"""Model-facing SinD high-interaction prediction dataset.

history_state is from frozen Route-B sensing cache.
future_state is GT. IDs and scene metadata are metadata only.
Source timestamps are preserved; no fixed dt is reconstructed from frame_id.
"""
from __future__ import annotations
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

FORMAL_SNR = (-10.0, -5.0, 0.0, 5.0, 10.0)

class SinDPredictionDataset(Dataset):
    def __init__(self, split: str = "train", snr_db: float = 0.0,
                 root_dir: Optional[Union[str, Path]] = None,
                 return_tensors: bool = True):
        super().__init__()
        if split not in ("train", "val", "test"):
            raise ValueError(split)
        self.root = Path(root_dir).resolve() if root_dir is not None else Path(__file__).resolve().parents[1]
        self.split = split
        self.return_tensors = bool(return_tensors and TORCH_AVAILABLE)
        z = np.load(self.root / f"data/sind/splits/{split}/samples.npz", allow_pickle=False)
        self.scene_id = z["scene_id"]
        self.start_frame = z["start_frame"]
        self.vehicle_ids = z["vehicle_ids"]
        self.vehicle_mask = z["vehicle_mask"]
        self.future = z["future"]
        self.history_timestamp = z["history_timestamp"]
        self.original_num_vehicles = z["original_num_vehicles"]
        self.selection_mode = z["selection_mode"]
        self.focal_vehicle_id = z["focal_vehicle_id"]
        self.num_samples = len(self.scene_id)

        c = np.load(self.root / f"data/sind/isac/{split}/sensing_cache.npz", allow_pickle=False)
        self.snr_levels_db = c["snr_levels_db"]
        hit = np.where(np.isclose(self.snr_levels_db, float(snr_db)))[0]
        if len(hit) != 1:
            raise ValueError(f"snr {snr_db} unavailable: {self.snr_levels_db.tolist()}")
        self.snr_idx = int(hit[0])
        self.snr_db = float(self.snr_levels_db[self.snr_idx])
        self.state_hat = c["state_hat"]
        lookup = {
            (int(sc), int(fr), int(vi)): i
            for i, (sc, fr, vi) in enumerate(zip(c["scene_id"], c["frame"], c["vehicle_id"]))
        }
        self._history_indices = np.full((self.num_samples, 20, 8), -1, dtype=np.int32)
        for i in range(self.num_samples):
            sc = int(self.scene_id[i])
            st = int(self.start_frame[i])
            for slot in range(8):
                if not self.vehicle_mask[i, slot]:
                    continue
                vid = int(self.vehicle_ids[i, slot])
                for t in range(20):
                    key = (sc, st + t, vid)
                    if key not in lookup:
                        raise KeyError(f"missing sensing key {key}")
                    self._history_indices[i, t, slot] = lookup[key]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if index < 0 or index >= self.num_samples:
            raise IndexError(index)
        mask = self.vehicle_mask[index].astype(np.bool_)
        idx = self._history_indices[index]
        safe = np.where(idx >= 0, idx, 0)
        hist = self.state_hat[self.snr_idx, safe]
        hist = np.where(mask[None, :, None], hist, 0.0).astype(np.float32)
        future = self.future[index].astype(np.float32)
        ts = self.history_timestamp[index].astype(np.float64)
        result = {
            "history_state": hist,
            "future_state": future,
            "vehicle_mask": mask,
            "history_timestamp": ts,
            "scene_id": int(self.scene_id[index]),
            "start_frame": int(self.start_frame[index]),
            "vehicle_ids": self.vehicle_ids[index].astype(np.int32),
            "original_num_vehicles": int(self.original_num_vehicles[index]),
            "selection_mode": int(self.selection_mode[index]),
            "focal_vehicle_id": int(self.focal_vehicle_id[index]),
        }
        if not self.return_tensors:
            return result
        return {
            "history_state": torch.from_numpy(result["history_state"]),
            "future_state": torch.from_numpy(result["future_state"]),
            "vehicle_mask": torch.from_numpy(result["vehicle_mask"]),
            "history_timestamp": torch.from_numpy(result["history_timestamp"]),
            "scene_id": torch.tensor(result["scene_id"], dtype=torch.uint8),
            "start_frame": torch.tensor(result["start_frame"], dtype=torch.int32),
            "vehicle_ids": torch.from_numpy(result["vehicle_ids"]),
            "original_num_vehicles": torch.tensor(result["original_num_vehicles"], dtype=torch.int16),
            "selection_mode": torch.tensor(result["selection_mode"], dtype=torch.uint8),
            "focal_vehicle_id": torch.tensor(result["focal_vehicle_id"], dtype=torch.int32),
        }
