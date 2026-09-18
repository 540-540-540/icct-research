"""Shared read-only helpers for the Route-B final execution (geometry fix, recalibration,
acceptance and freeze). All searches use train only; val verifies; test is never read."""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CONFIG_PATH = ROOT / "configs/automatum_controlled_isac.json"
OUT_DIR = ROOT / "reports/isac_final"
DT = 3.0 / 29.97
LEVELS = (-10.0, -5.0, 0.0, 5.0, 10.0)
SCENES = (0, 1)
HEIGHT = 5.0
RANGE_MIN, RANGE_MAX = 5.0, 150.0


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def scene_geometry(config: dict) -> dict:
    return {
        int(scene["scene_id"]): {
            "stations": np.asarray(scene["stations_xy_m"], dtype=float),
            "boresights_deg": np.asarray(scene["boresights_deg"], dtype=float),
        }
        for scene in config["scenes"]
    }


def config_with_geometry(config: dict, scene0_geometry: dict | None = None) -> dict:
    """Deep copy of the config with a candidate scene-0 geometry (scene 1 untouched)."""
    payload = json.loads(json.dumps(config))
    if scene0_geometry is not None:
        for scene in payload["scenes"]:
            if int(scene["scene_id"]) == 0:
                scene["stations_xy_m"] = [[float(v) for v in row]
                                          for row in scene0_geometry["stations_xy_m"]]
                scene["boresights_deg"] = [float(v) for v in scene0_geometry["boresights_deg"]]
        payload["visibility"]["fov_half_angle_deg"] = float(
            scene0_geometry.get("fov_half_angle_deg", payload["visibility"]["fov_half_angle_deg"]))
        payload["scene0_geometry_status"] = scene0_geometry.get("status", "CANDIDATE")
    return payload


def unique_history(split: str) -> dict:
    samples = np.load(ROOT / "data/automatum_t_crossing/splits" / split / "samples.npz")
    keys: set[tuple[int, int, int]] = set()
    windows = []
    total_rows = 0
    for index in range(samples["scene_id"].size):
        scene_id = int(samples["scene_id"][index])
        start = int(samples["start_frame"][index])
        ids = samples["vehicle_ids"][index][samples["vehicle_mask"][index]].tolist()
        windows.append({"scene_id": scene_id, "start_frame": start,
                        "vehicle_ids": [int(v) for v in ids]})
        for offset in range(20):
            for vehicle in ids:
                keys.add((scene_id, start + offset, int(vehicle)))
                total_rows += 1
    return {"keys": sorted(keys), "windows": windows, "history_rows": total_rows,
            "unique_states": len(keys)}


def state_lookup(split: str) -> dict:
    path = ROOT / load_config()["dataset"][f"{split}_trajectories"]
    table = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    scene = table["scene_id"].astype(int)
    vehicle = table["vehicle_id"].astype(int)
    frame = np.rint(table["timestamp"].astype(float) * 29.97).astype(np.int64) // 3
    states = np.stack([table["x"], table["y"], table["vx"], table["vy"]], axis=1).astype(np.float64)
    return {(int(s), int(f), int(v)): states[i] for i, (s, f, v) in
            enumerate(zip(scene, frame, vehicle))}


def keys_positions(keys, lookup) -> np.ndarray:
    return np.asarray([lookup[key][:2] for key in keys], dtype=np.float64)


def visibility(states_xy: np.ndarray, stations: np.ndarray, boresights_deg: np.ndarray,
               fov_half_deg: float, height: float = HEIGHT,
               range_min: float = RANGE_MIN, range_max: float = RANGE_MAX) -> np.ndarray:
    """(N, 3) boolean geometric visibility, identical to measurement.py conventions."""
    delta = states_xy[:, None, :] - stations[None, :, :]
    rho = np.linalg.norm(delta, axis=2)
    r = np.sqrt(rho ** 2 + height ** 2)
    bearing = (np.arctan2(delta[:, :, 1], delta[:, :, 0])
               - np.radians(boresights_deg)[None, :] + np.pi) % (2 * np.pi) - np.pi
    return (r >= range_min) & (r <= range_max) & (np.abs(bearing) <= math.radians(fov_half_deg))


def coverage_counts(visible: np.ndarray) -> dict:
    count = visible.sum(axis=1)
    return {"n": int(count.size), "n_bs_0": int((count == 0).sum()),
            "n_bs_1": int((count == 1).sum()), "n_bs_2": int((count == 2).sum()),
            "n_bs_3": int((count == 3).sum())}


def truth_polar(states_xy: np.ndarray, states_v: np.ndarray, station: np.ndarray,
                boresight_rad: float, height: float = HEIGHT) -> tuple:
    delta = states_xy - station[None, :]
    rho = np.linalg.norm(delta, axis=1)
    r = np.sqrt(rho ** 2 + height ** 2)
    bearing = (np.arctan2(delta[:, 1], delta[:, 0]) - boresight_rad + np.pi) % (2 * np.pi) - np.pi
    radial = ((station[None, :] - states_xy) * states_v).sum(axis=1) / np.maximum(r, 1e-9)
    return r, bearing, radial, rho


def base_noise_arrays(keys, channels=("range", "bearing", "radial_velocity")) -> dict:
    from frontend.controlled_isac.automatum_measurement import base_noise

    arrays = {channel: np.empty((len(keys), 3), dtype=np.float64) for channel in channels}
    for index, (scene_id, frame_id, vehicle) in enumerate(keys):
        for bs_id in range(3):
            for channel in channels:
                arrays[channel][index, bs_id] = base_noise(scene_id, frame_id, bs_id, vehicle, channel)
    return arrays


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["empty"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def distribution(values, percentiles=(1, 5, 10, 25, 50, 75, 90, 95, 99)) -> dict:
    array = np.asarray([value for value in values if value is not None and np.isfinite(value)],
                       dtype=np.float64)
    if array.size == 0:
        return {"n": 0}
    result = {"n": int(array.size), "mean": float(array.mean()), "std": float(array.std()),
              "max": float(array.max())}
    for percentile in percentiles:
        result[f"p{percentile:02d}"] = float(np.percentile(array, percentile))
    return result