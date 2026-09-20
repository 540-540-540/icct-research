"""Per-target SinD views with shared sensing/history pools. Test is unavailable."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class SinDTargetPredictionDataset(Dataset):
    def __init__(self, split="train", snr_db=0.0, root_dir=None, return_tensors=True):
        if split not in ("train", "val"):
            raise ValueError("SinD target views only permit train/val; test remains closed")
        if float(snr_db) != 0.0:
            raise ValueError("target_views_v1 contains only the matched 0 dB protocol")
        self.root = Path(root_dir) if root_dir is not None else Path(__file__).resolve().parents[1]
        self.split, self.snr_db, self.return_tensors = split, 0.0, return_tensors
        folder = self.root / "data/sind/target_views_v1" / split
        if not (folder / ".BUILD_COMPLETE").is_file():
            raise RuntimeError(f"incomplete/missing target views: {folder}; build them first")
        marker = json.loads((folder / ".BUILD_COMPLETE").read_text())
        self.manifest = json.loads((folder / "manifest.json").read_text())
        if (self.manifest.get("status") != "COMPLETE"
                or self.manifest.get("test_used") is not False
                or self.manifest.get("revision") != "SIND-TARGET-VIEWS-V1-FIXED-ORIGIN"
                or self.manifest.get("split") != split
                or self.manifest.get("snr_db") != 0.0
                or marker.get("status") != "COMPLETE"
                or marker.get("revision") != self.manifest["revision"]):
            raise RuntimeError("invalid target-view completion manifest")
        for name, size in self.manifest["output_bytes"].items():
            if not (folder / name).is_file() or (folder / name).stat().st_size != size:
                raise RuntimeError(f"incomplete/corrupted target artifact: {folder / name}")
        with np.load(folder / "views.npz", allow_pickle=False) as arrays:
            for name in arrays.files:
                setattr(self, name, arrays[name])
        with np.load(folder / "state_pool.npz", allow_pickle=False) as pool:
            self.state_hat, self.state_gt = pool["state_hat"], pool["state_gt"]
        self.num_samples = len(self.origin_index)
        self.num_origins = len(self.origin_scene)
        self.scene_id = self.origin_scene[self.origin_index]
        self.start_frame = self.origin_start[self.origin_index]
        self.vehicle_mask = self.vehicle_ids >= 0
        if self.num_samples != self.manifest["targets"] or not self.vehicle_mask[:, 0].all():
            raise RuntimeError("invalid target count or missing center target")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        if not 0 <= index < self.num_samples:
            raise IndexError(index)
        origin = int(self.origin_index[index])
        mask = self.vehicle_mask[index].copy()
        rows = self.history_rows[index]
        indices = self.history_indices[np.maximum(rows, 0)].T
        history = self.state_hat[indices].copy()
        history[:, ~mask] = 0.0
        future = np.zeros((20, 8, 4), np.float32)
        future[:, 0] = self.state_gt[self.future_indices[index]]
        target_mask = np.zeros(8, bool)
        target_mask[0] = True
        count = int(self.origin_target_count[origin])
        result = {
            "history_state": history, "future_state": future,
            "vehicle_mask": mask, "target_mask": target_mask,
            "history_timestamp": self.history_timestamp[origin].copy(),
            "scene_id": np.int32(self.origin_scene[origin]),
            "start_frame": np.int32(self.origin_start[origin]),
            "vehicle_ids": self.vehicle_ids[index].copy(),
            "target_vehicle_id": np.int32(self.vehicle_ids[index, 0]),
            "focal_vehicle_id": np.int32(self.vehicle_ids[index, 0]),
            "origin_index": np.int32(origin), "origin_target_count": np.int32(count),
            "target_weight": np.float32(self.num_samples / (self.num_origins * count)),
            "num_vehicles": np.int32(mask.sum()),
            "original_num_vehicles": np.int32(self.origin_context_count[origin]),
        }
        return {name: torch.as_tensor(value) for name, value in result.items()} if self.return_tensors else result
