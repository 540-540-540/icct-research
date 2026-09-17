"""Automatum T-Crossing scene-level source (A domain only).

Reads one frozen split ``trajectories.csv`` and returns **all** vehicles present in a
canonical frame of one scene. It never reads ``samples.npz``, never applies the prediction
cohort, never interpolates or smooths, never looks at the future, and never rotates the
world-frame ``vx/vy`` again.

Time rule (frozen by the canonical preprocessing report):
    source_frame = round(timestamp * 29.97)
    canonical_frame = source_frame // 3
    dt = 3 / 29.97 = 0.100100100100... s
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

COLUMNS = ("scene_id", "vehicle_id", "timestamp", "x", "y", "vx", "vy")


@dataclass(frozen=True)
class SceneFrame:
    scene_id: int
    frame: int
    timestamp: float
    vehicle_ids: np.ndarray
    states: np.ndarray

    def __len__(self) -> int:
        return int(self.states.shape[0])


class AutomatumSceneSource:
    """Deterministic frame-indexed view over one split's trajectories.csv."""

    def __init__(self, csv_path, source_hz: float = 29.97, stride: int = 3):
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"missing split trajectories: {self.csv_path}")
        if not (25.0 < float(source_hz) < 35.0) or int(stride) < 1:
            raise ValueError("source_hz/stride must reproduce the frozen 29.97/3 canonical grid")
        self.source_hz = float(source_hz)
        self.stride = int(stride)
        self.dt = self.stride / self.source_hz

        table = np.genfromtxt(self.csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
        if table.dtype.names != COLUMNS:
            raise ValueError(f"trajectories.csv must have columns {COLUMNS}, got {table.dtype.names}")
        scene = table["scene_id"].astype(np.int64)
        vehicle = table["vehicle_id"].astype(np.int64)
        timestamp = table["timestamp"].astype(np.float64)
        states = np.stack([table[name].astype(np.float64) for name in ("x", "y", "vx", "vy")], axis=1)
        if not (np.isfinite(timestamp).all() and np.isfinite(states).all()):
            raise ValueError("trajectories.csv contains non-finite values")

        source_frame = np.rint(timestamp * self.source_hz).astype(np.int64)
        if np.any(source_frame % self.stride != 0):
            raise ValueError("trajectories.csv violates the frozen source_frame % 3 == 0 phase rule")
        if np.any(np.abs(timestamp - source_frame / self.source_hz) > 1e-9):
            raise ValueError("timestamp is not the true source time source_frame / 29.97")
        frame = source_frame // self.stride

        order = np.lexsort((vehicle, frame, scene))
        scene, vehicle, timestamp, states, frame = (array[order] for array in
                                                    (scene, vehicle, timestamp, states, frame))
        self._scene = scene
        self._vehicle = vehicle
        self._timestamp = timestamp
        self._states = states
        self._frame = frame

        self._ranges: dict[int, tuple[int, int]] = {}
        self._frames: dict[int, np.ndarray] = {}
        self._starts: dict[int, np.ndarray] = {}
        self._counts: dict[int, np.ndarray] = {}
        for scene_id in self.scenes:
            rows = np.flatnonzero(scene == scene_id)
            if rows.size == 0:
                raise ValueError(f"scene {scene_id} has no rows")
            unique, starts, counts = np.unique(frame[rows], return_index=True, return_counts=True)
            self._ranges[int(scene_id)] = (int(rows[0]), int(rows[-1]) + 1)
            self._frames[int(scene_id)] = unique
            self._starts[int(scene_id)] = rows[starts]
            self._counts[int(scene_id)] = counts
            if np.unique(timestamp[rows]).size != unique.size:
                raise ValueError(f"scene {scene_id} maps two timestamps to one canonical frame")

    @property
    def scenes(self) -> list[int]:
        return sorted(int(value) for value in np.unique(self._scene))

    def frame_bounds(self, scene_id: int) -> tuple[int, int]:
        frames = self._frames.get(int(scene_id))
        if frames is None:
            raise KeyError(f"unknown scene {scene_id}")
        return int(frames[0]), int(frames[-1])

    def frames(self, scene_id: int) -> np.ndarray:
        frames = self._frames.get(int(scene_id))
        if frames is None:
            raise KeyError(f"unknown scene {scene_id}")
        return frames.copy()

    def at_frame(self, scene_id: int, frame: int) -> SceneFrame:
        frames = self._frames.get(int(scene_id))
        if frames is None:
            raise KeyError(f"unknown scene {scene_id}")
        index = int(np.searchsorted(frames, int(frame)))
        if index >= frames.size or int(frames[index]) != int(frame):
            raise KeyError(f"scene {scene_id} has no vehicle rows in canonical frame {frame}")
        start = int(self._starts[int(scene_id)][index])
        stop = start + int(self._counts[int(scene_id)][index])
        return SceneFrame(scene_id=int(scene_id), frame=int(frame),
                          timestamp=float(self._timestamp[start]),
                          vehicle_ids=self._vehicle[start:stop].copy(),
                          states=self._states[start:stop].copy())

    def frame_features(self, scene_id: int, junction_center) -> dict:
        """Per-frame occupancy/geometry statistics used by the deterministic smoke selector."""
        frames = self._frames[int(scene_id)]
        counts = self._counts[int(scene_id)].astype(np.int64)
        min_pair = np.full(frames.size, np.inf)
        mean_center = np.zeros(frames.size)
        max_center = np.zeros(frames.size)
        starts = self._starts[int(scene_id)]
        for index in range(frames.size):
            start = int(starts[index])
            positions = self._states[start:start + int(counts[index]), :2]
            if positions.shape[0] >= 2:
                delta = positions[:, None, :] - positions[None, :, :]
                distance = np.linalg.norm(delta, axis=-1)
                np.fill_diagonal(distance, np.inf)
                min_pair[index] = float(np.min(distance))
            distance_center = np.linalg.norm(positions - np.asarray(junction_center), axis=-1)
            mean_center[index] = float(distance_center.mean())
            max_center[index] = float(distance_center.max())
        return {"frames": frames, "counts": counts, "min_pair_m": min_pair,
                "mean_junction_m": mean_center, "max_junction_m": max_center}


def selfcheck() -> dict:
    root = Path(__file__).resolve().parents[1]
    source = AutomatumSceneSource(root / "data/automatum_t_crossing/splits/train/trajectories.csv")
    assert source.scenes == [0, 1]
    assert abs(source.dt - 3 / 29.97) < 1e-15 and source.dt != 0.1
    for scene_id in source.scenes:
        first, last = source.frame_bounds(scene_id)
        frame = source.at_frame(scene_id, first)
        assert len(frame) >= 1 and frame.states.shape == (len(frame), 4)
        assert np.isfinite(frame.states).all()
        second = source.at_frame(scene_id, first + 1)
        assert abs((second.timestamp - frame.timestamp) - source.dt) < 1e-9
        table = np.genfromtxt(source.csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
        rows = (table["scene_id"] == scene_id) & (table["timestamp"] == frame.timestamp)
        assert int(rows.sum()) == len(frame)
        assert np.allclose(np.stack([table["x"][rows], table["y"][rows],
                                     table["vx"][rows], table["vy"][rows]], axis=1), frame.states)
    return {"scenes": source.scenes, "dt": source.dt, "splits": "train"}


if __name__ == "__main__":
    print(selfcheck())