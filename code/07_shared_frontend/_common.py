"""Shared helpers for REBUILD-03A diagnostics (C-domain scripts only)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]


def read_json(path) -> dict:
    return json.loads(Path(path).read_text())


def _sanitize(value):
    if isinstance(value, dict):
        return {key: _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if callable(getattr(value, "item", None)) and not isinstance(value, (int, float, bool, str, bytes)):
        try:
            value = value.item()
        except Exception:  # noqa: BLE001
            pass
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (int, bool, str)) or value is None:
        return value
    return str(value)


def write_json(path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_sanitize(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def load_frontend_config(root: Path = ROOT) -> dict:
    return read_json(Path(root) / "configs/shared_frontend.json")


def load_geometry_config(root: Path = ROOT) -> dict:
    from frontend.sensing.waveform import load_geometry

    return load_geometry(root)


def build_objects(config: dict):
    from frontend.sensing.waveform import ArrayConfig, PaperWaveform

    return PaperWaveform(**config["waveform"]), ArrayConfig(**config["array"])


def synthesize(positions, velocities, keys, stations, boresights, waveform, array, config, snr_ref_db,
               episode: int, frame: int, device: str, noise: bool = True, height_m: float | None = None) -> dict:
    from frontend.sensing.simulator import synthesize_shared

    if height_m is None:
        height_m = float(config["visibility"]["height_difference_m"])
    rcs = [float(config["power"]["fixed_rcs_m2"])] * len(positions)
    return synthesize_shared(positions, velocities, keys, stations, boresights, waveform, array, rcs,
                             snr_ref_db, episode, frame, height_m=float(height_m), noise=noise, device=device)


def assert_height_alignment(config: dict, geometry: dict) -> None:
    config_height = float(config["visibility"]["height_difference_m"])
    geometry_height = float(geometry["height_difference_m"])
    if config_height != geometry_height:
        raise ValueError(f"height mismatch: config {config_height} vs A01 geometry {geometry_height}")


def load_lut(root: Path = ROOT, path: str | None = None) -> dict | None:
    lut_path = Path(root) / (path or "reports/f01e/covariance_calibration.json")
    if not lut_path.exists():
        return None
    return read_json(lut_path)


def load_production_lut(config: dict, root: Path = ROOT) -> dict | None:
    """Load the covariance LUT referenced by the production config."""
    return load_lut(root, config.get("detector", {}).get("covariance_lut"))


def load_resource(config: dict):
    from frontend.sensing.waveform import SensingResource

    return SensingResource.from_config(config)


def place_target(station, boresight: float, r_m: float, bearing_deg: float, velocity, height: float = 5.0):
    """C-domain helper: build one truth target at a requested polar location."""
    rho = math.sqrt(r_m ** 2 - height ** 2)
    phi = float(boresight) + math.radians(bearing_deg)
    position = [float(station[0]) + rho * math.cos(phi), float(station[1]) + rho * math.sin(phi)]
    return position, [float(velocity[0]), float(velocity[1])]


def truth_polar(position, station, boresight: float, height: float = 5.0):
    from frontend.sensing import coords

    return coords.cartesian_to_polar(torch.tensor(position, dtype=torch.float64),
                                     torch.tensor(station, dtype=torch.float64), boresight, height)


def nearest_truth_detection(detections, r_gt: float, u_gt: float, r_gate: float = 8.0, u_gate: float = 0.12):
    best, best_cost = None, None
    for detection in detections:
        if abs(detection["r_m"] - r_gt) > r_gate or abs(detection["u"] - u_gt) > u_gate:
            continue
        cost = abs(detection["r_m"] - r_gt) + 50.0 * abs(detection["u"] - u_gt)
        if best_cost is None or cost < best_cost:
            best, best_cost = detection, cost
    return best