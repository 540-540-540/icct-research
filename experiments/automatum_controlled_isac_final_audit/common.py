"""Read-only shared helpers for the ISAC Route-B final audit.

Nothing here modifies sensing parameters, geometry, data or frontend numerical paths.
Prediction-history unique states are defined as the de-duplicated set of
``(scene_id, frame, vehicle_id)`` covered by the 20-frame history of the frozen samples.npz.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CONFIG_PATH = ROOT / "configs/automatum_controlled_isac.json"
OUT_DIR = ROOT / "reports/isac_final_audit"
DT = 3.0 / 29.97
LEVELS = (-10.0, -5.0, 0.0, 5.0, 10.0)
EXPECTED_CANDIDATE_B = {"range": 0.10, "bearing": 0.004, "radial_velocity": 0.16}


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def unique_history(split: str) -> dict:
    """De-duplicated prediction-history states plus the sample-window bookkeeping."""
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
            frame = start + offset
            for vehicle in ids:
                keys.add((scene_id, frame, int(vehicle)))
                total_rows += 1
    return {"keys": sorted(keys), "windows": windows, "history_rows": total_rows,
            "unique_states": len(keys), "duplication_factor": total_rows / len(keys)}


def state_lookup(split: str) -> tuple[dict, dict]:
    """(scene, frame, vehicle) -> state vector, and (scene, frame) -> (ids, positions, velocities)."""
    path = ROOT / load_config()["dataset"][f"{split}_trajectories"]
    table = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    scene = table["scene_id"].astype(int)
    vehicle = table["vehicle_id"].astype(int)
    frame = np.rint(table["timestamp"].astype(float) * 29.97).astype(np.int64) // 3
    states = np.stack([table["x"], table["y"], table["vx"], table["vy"]], axis=1).astype(np.float64)
    lookup = {(int(s), int(f), int(v)): states[i] for i, (s, f, v) in
              enumerate(zip(scene, frame, vehicle))}
    frames: dict[tuple[int, int], dict] = {}
    for index in range(scene.size):
        entry = frames.setdefault((int(scene[index]), int(frame[index])),
                                  {"ids": [], "positions": [], "states": []})
        entry["ids"].append(int(vehicle[index]))
        entry["positions"].append(states[index, :2])
        entry["states"].append(states[index])
    for entry in frames.values():
        entry["ids"] = np.asarray(entry["ids"], dtype=np.int64)
        entry["positions"] = np.asarray(entry["positions"], dtype=np.float64)
        entry["states"] = np.asarray(entry["states"], dtype=np.float64)
    return lookup, frames


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


def rate(values, threshold: float) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    return float(np.mean(array <= threshold)) if array.size else 0.0


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


def load_calibration(config: dict) -> dict:
    from frontend.controlled_isac.automatum_measurement import calibration_from_config

    return calibration_from_config(config)


def calibration_is_candidate_b(calibration: dict) -> bool:
    return all(abs(float(calibration["a"][channel]) - value) < 1e-12
               for channel, value in EXPECTED_CANDIDATE_B.items())