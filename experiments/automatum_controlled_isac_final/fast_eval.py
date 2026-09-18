"""Vectorized replica of the Route-B chain for search phases.

Same formulas as ``frontend/controlled_isac/automatum_measurement.py`` (truth polar, Gaussian
noise, polar-to-xy, covariance), ``automatum_frontend.py`` (inverse-variance position fusion,
3D-LOS direction override) and ``velocity_fusion.recover_velocity`` (weak-prior MAP velocity).
The self-check verifies this replica against the real frontend on sampled states.
"""
from __future__ import annotations

import math

import numpy as np

from experiments.automatum_controlled_isac_final import common

FLOOR_KEYS = {"range": "range", "bearing": "bearing", "radial_velocity": "radial_velocity"}
PRIOR_SIGMA = 200.0
COVARIANCE_FLOOR_M2 = 1e-8


def prepare(config: dict, geometry_scene0: dict | None, split: str = "train") -> dict:
    effective = common.config_with_geometry(config, geometry_scene0)
    geometry = common.scene_geometry(effective)
    fov = float(effective["visibility"]["fov_half_angle_deg"])
    history = common.unique_history(split)
    lookup = common.state_lookup(split)
    keys = history["keys"]
    states = np.asarray([lookup[key] for key in keys], dtype=np.float64)
    n = len(keys)
    r = np.zeros((n, 3))
    bearing = np.zeros((n, 3))
    radial = np.zeros((n, 3))
    rho = np.zeros((n, 3))
    for scene_id in common.SCENES:
        rows = np.asarray([i for i, key in enumerate(keys) if key[0] == scene_id], dtype=np.int64)
        if rows.size == 0:
            continue
        stations = geometry[scene_id]["stations"]
        boresights = geometry[scene_id]["boresights_deg"]
        for bs_id in range(3):
            rr, bb, vv, pp = common.truth_polar(states[rows, :2], states[rows, 2:],
                                                stations[bs_id], math.radians(boresights[bs_id]))
            r[rows, bs_id] = rr
            bearing[rows, bs_id] = bb
            radial[rows, bs_id] = vv
            rho[rows, bs_id] = pp
    scene_index = np.asarray([key[0] for key in keys], dtype=np.int64)
    visible = np.zeros((n, 3), dtype=bool)
    for scene_id in common.SCENES:
        rows = np.flatnonzero(scene_index == scene_id)
        if rows.size == 0:
            continue
        visible[rows] = common.visibility(states[rows, :2], geometry[scene_id]["stations"],
                                          geometry[scene_id]["boresights_deg"], fov)
    stations = np.stack([geometry[scene_id]["stations"] for scene_id in common.SCENES])  # (2,3,2)
    boresights = np.stack([np.radians(geometry[scene_id]["boresights_deg"])
                           for scene_id in common.SCENES])  # (2,3)
    noise = common.base_noise_arrays(keys)
    return {"keys": keys, "states": states, "visible": visible, "r": r, "bearing": bearing,
            "radial": radial, "rho": rho, "stations": stations, "boresights": boresights,
            "scene_index": scene_index, "noise": noise, "fov_half_angle_deg": fov}


def sigma_for(channel: str, snr_db: float, calibration: dict) -> float:
    a = float(calibration["a"][channel])
    floor = float(calibration["floors"][channel])
    return math.sqrt(floor ** 2 + (a * 10 ** (-float(snr_db) / 20.0)) ** 2)


def evaluate(prepared: dict, calibration: dict, levels=common.LEVELS) -> dict:
    states = prepared["states"]
    visible = prepared["visible"]
    n = states.shape[0]
    output = {}
    for level in levels:
        sigma_r = sigma_for("range", level, calibration)
        sigma_b = sigma_for("bearing", level, calibration)
        sigma_v = sigma_for("radial_velocity", level, calibration)
        r_hat = prepared["r"] + sigma_r * prepared["noise"]["range"]
        r_hat = np.where(r_hat <= common.HEIGHT, common.HEIGHT + 1e-6, r_hat)
        bearing_hat = (prepared["bearing"] + sigma_b * prepared["noise"]["bearing"] + np.pi) \
            % (2 * np.pi) - np.pi
        radial_hat = prepared["radial"] + sigma_v * prepared["noise"]["radial_velocity"]
        rho_hat = np.sqrt(np.maximum(r_hat ** 2 - common.HEIGHT ** 2, 1e-9))
        sigma_rho = sigma_r * r_hat / np.maximum(rho_hat, 1e-9)

        x_hat = np.zeros((n, 3))
        y_hat = np.zeros((n, 3))
        inv11 = np.zeros((n, 3))
        inv12 = np.zeros((n, 3))
        inv22 = np.zeros((n, 3))
        for scene_id in common.SCENES:
            rows = np.flatnonzero(prepared["scene_index"] == scene_id)
            if rows.size == 0:
                continue
            for bs_id in range(3):
                station = prepared["stations"][scene_id, bs_id]
                phi = prepared["boresights"][scene_id, bs_id] + bearing_hat[rows, bs_id]
                cos_phi, sin_phi = np.cos(phi), np.sin(phi)
                rho_row = rho_hat[rows, bs_id]
                x_hat[rows, bs_id] = station[0] + rho_row * cos_phi
                y_hat[rows, bs_id] = station[1] + rho_row * sin_phi
                sig_rho = sigma_rho[rows, bs_id]
                c11 = cos_phi ** 2 * sig_rho ** 2 + rho_row ** 2 * sin_phi ** 2 * sigma_b ** 2 \
                    + COVARIANCE_FLOOR_M2
                c12 = cos_phi * sin_phi * (sig_rho ** 2 - rho_row ** 2 * sigma_b ** 2)
                c22 = sin_phi ** 2 * sig_rho ** 2 + rho_row ** 2 * cos_phi ** 2 * sigma_b ** 2 \
                    + COVARIANCE_FLOOR_M2
                det = c11 * c22 - c12 ** 2
                inv11[rows, bs_id] = c22 / det
                inv12[rows, bs_id] = -c12 / det
                inv22[rows, bs_id] = c11 / det

        mask = visible.astype(np.float64)
        info11 = (mask * inv11).sum(axis=1)
        info12 = (mask * inv12).sum(axis=1)
        info22 = (mask * inv22).sum(axis=1)
        rhs1 = (mask * (inv11 * x_hat + inv12 * y_hat)).sum(axis=1)
        rhs2 = (mask * (inv12 * x_hat + inv22 * y_hat)).sum(axis=1)
        det_info = info11 * info22 - info12 ** 2
        position_x = (info22 * rhs1 - info12 * rhs2) / det_info
        position_y = (info11 * rhs2 - info12 * rhs1) / det_info

        delta_x = np.zeros((n, 3))
        delta_y = np.zeros((n, 3))
        for scene_id in common.SCENES:
            rows = np.flatnonzero(prepared["scene_index"] == scene_id)
            if rows.size == 0:
                continue
            for bs_id in range(3):
                station = prepared["stations"][scene_id, bs_id]
                delta_x[rows, bs_id] = position_x[rows] - station[0]
                delta_y[rows, bs_id] = position_y[rows] - station[1]
        r3 = np.sqrt(delta_x ** 2 + delta_y ** 2 + common.HEIGHT ** 2)
        direction_x = delta_x / r3
        direction_y = delta_y / r3
        weight = 1.0 / max(sigma_v ** 2, 1e-12)
        normal11 = (mask * weight * direction_x ** 2).sum(axis=1)
        normal12 = (mask * weight * direction_x * direction_y).sum(axis=1)
        normal22 = (mask * weight * direction_y ** 2).sum(axis=1)
        rhs_v1 = -(mask * weight * direction_x * radial_hat).sum(axis=1)
        rhs_v2 = -(mask * weight * direction_y * radial_hat).sum(axis=1)
        prior_info = 1.0 / PRIOR_SIGMA ** 2
        a11 = normal11 + prior_info
        a12 = normal12
        a22 = normal22 + prior_info
        det_v = a11 * a22 - a12 ** 2
        vx = (a22 * rhs_v1 - a12 * rhs_v2) / det_v
        vy = (a11 * rhs_v2 - a12 * rhs_v1) / det_v

        position_error = np.hypot(position_x - states[:, 0], position_y - states[:, 1])
        velocity_error = np.hypot(vx - states[:, 2], vy - states[:, 3])
        output[float(level)] = {
            "position_error": position_error, "velocity_error": velocity_error,
            "vx_error": vx - states[:, 2], "vy_error": vy - states[:, 3],
            "position_x": position_x, "position_y": position_y, "vx": vx, "vy": vy,
            "n_bs": mask.sum(axis=1).astype(int),
        }
    return output


BANDS = {10.0: {"position": (0.10, 0.30), "velocity": (0.10, 0.30)},
         0.0: {"position": (0.30, 0.70), "velocity": (0.30, 0.80)},
         -10.0: {"position": (0.80, 1.50), "velocity": (1.00, 2.00)}}
SOFT_REFERENCES = {10.0: {"position": 0.15, "velocity": 0.175},
                   0.0: {"position": 0.45, "velocity": 0.50},
                   -10.0: {"position": 1.15, "velocity": 1.50}}


def pooled_rmse(output: dict, level: float, metric: str, subset=None) -> float:
    errors = output[level][f"{metric}_error"]
    if subset is not None:
        errors = errors[subset]
    return float(np.sqrt((errors ** 2).mean()))


def objective(output: dict, subset=None) -> dict:
    losses, band_report, soft_loss = 0.0, {}, 0.0
    values = {}
    for level in (10.0, 0.0, -10.0):
        for metric in ("position", "velocity"):
            value = pooled_rmse(output, level, metric, subset)
            values[f"{level:+g}_{metric}"] = value
            low, high = BANDS[level][metric]
            if value < low:
                loss = (low - value) / (high - low)
            elif value > high:
                loss = (value - high) / (high - low)
            else:
                loss = 0.0
            losses += loss
            band_report[f"{level:+g}_{metric}"] = {
                "value": value, "band": [low, high], "loss": loss}
            reference = SOFT_REFERENCES[level][metric]
            soft_loss += ((value - reference) / reference) ** 2
    monotonicity_violations = 0
    ordered = sorted(common.LEVELS, reverse=True)
    for metric in ("position", "velocity"):
        series = [pooled_rmse(output, level, metric, subset) for level in ordered]
        monotonicity_violations += sum(1 for i in range(len(series) - 1) if series[i] >= series[i + 1])
    return {"band_loss": float(losses), "soft_reference_loss": float(soft_loss),
            "monotonicity_violations": int(monotonicity_violations),
            "score": float(losses + 10.0 * monotonicity_violations),
            "values": values, "bands": band_report,
            "rmse_by_level": {str(level): {
                "position": pooled_rmse(output, level, "position", subset),
                "velocity": pooled_rmse(output, level, "velocity", subset)}
                for level in common.LEVELS}}