"""C-domain evaluator for AUTOMATUM-ISAC-SMOKE-01.

This module is the only place where anonymous detector measurements meet ground truth.
It is never imported by ``frontend.*``. Truth polar quantities are recomputed independently
from the source states (same physical visibility model as the simulator, no simulator
internals), and the anonymous-to-GT matching result is never fed back to the detector.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linear_sum_assignment

BIG = 1e9


def truth_targets(states: np.ndarray, station, boresight: float, visibility: dict) -> dict:
    states = np.asarray(states, dtype=np.float64).reshape(-1, 4)
    station = np.asarray(station, dtype=np.float64).reshape(2)
    height = float(visibility["height_difference_m"])
    delta = states[:, :2] - station
    rho = np.linalg.norm(delta, axis=1)
    r_m = np.sqrt(rho ** 2 + height ** 2)
    radial = ((station[None, :] - states[:, :2]) * states[:, 2:]).sum(axis=1) / r_m
    bearing = (np.arctan2(delta[:, 1], delta[:, 0]) - float(boresight) + np.pi) % (2 * np.pi) - np.pi
    visible = ((r_m >= float(visibility["range_min_m"])) & (r_m <= float(visibility["range_max_m"]))
               & (np.abs(bearing) <= math.radians(float(visibility["fov_half_angle_deg"]))))
    return {"x_m": states[:, 0].copy(), "y_m": states[:, 1].copy(), "r_m": r_m,
            "radial_velocity_mps": radial, "bearing_rad": bearing,
            "u": np.sin(bearing), "visible": visible}


def match_to_truth(measurements: list[dict], truth: dict, gates: dict) -> dict:
    """One-to-one gated Hungarian matching in (range, u, radial velocity)."""
    visible = np.flatnonzero(truth["visible"])
    if len(measurements) == 0 or visible.size == 0:
        return {"pairs": [], "unmatched_measurements": list(range(len(measurements))),
                "unmatched_targets": visible.tolist()}
    r_gate = float(gates["range_m"])
    u_gate = float(gates["u"])
    v_gate = float(gates["radial_velocity_mps"])
    cost = np.full((len(measurements), visible.size), BIG, dtype=np.float64)
    for row, measurement in enumerate(measurements):
        for column, target in enumerate(visible):
            dr = measurement["range_hat"] - truth["r_m"][target]
            du = measurement["u_hat"] - truth["u"][target]
            dv = measurement["radial_velocity_hat"] - truth["radial_velocity_mps"][target]
            if abs(dr) <= r_gate and abs(du) <= u_gate and abs(dv) <= v_gate:
                cost[row, column] = math.sqrt((dr / r_gate) ** 2 + (du / u_gate) ** 2 + (dv / v_gate) ** 2)
    rows, columns = linear_sum_assignment(cost)
    pairs, matched_measurements, matched_targets = [], set(), set()
    for row, column in zip(rows.tolist(), columns.tolist()):
        if cost[row, column] >= BIG / 2:
            continue
        target = int(visible[column])
        pairs.append((row, target))
        matched_measurements.add(row)
        matched_targets.add(target)
    return {"pairs": pairs,
            "unmatched_measurements": [i for i in range(len(measurements)) if i not in matched_measurements],
            "unmatched_targets": [int(t) for t in visible if int(t) not in matched_targets]}


def pair_errors(measurement: dict, truth: dict, target: int, crowded: bool = False) -> dict:
    dr = measurement["range_hat"] - truth["r_m"][target]
    dv = measurement["radial_velocity_hat"] - truth["radial_velocity_mps"][target]
    db = (measurement["bearing_hat"] - truth["bearing_rad"][target] + math.pi) % (2 * math.pi) - math.pi
    dx = measurement["x_hat_bs"] - truth["x_m"][target]
    dy = measurement["y_hat_bs"] - truth["y_m"][target]
    return {"range_error_m": float(dr), "radial_velocity_error_mps": float(dv),
            "bearing_error_rad": float(db), "bearing_error_deg": float(math.degrees(db)),
            "position_error_m": float(math.hypot(dx, dy)),
            "angle_rank": int(measurement["angle_rank"]), "crowded": bool(crowded)}


def rd_crowded_mask(truth: dict, range_window_m: float = 8.0, velocity_window_mps: float = 8.0) -> np.ndarray:
    """Per-target flag: another visible target can win the local-maximum competition.

    The detector keeps a candidate only if it is the maximum of an 11x11 (radius 5) window,
    so any stronger target within +-5 range bins (4.0 m) and +-5 Doppler bins (4.9 m/s) of it
    disqualifies it. The windows here are deliberately wider so that near-miss cases are still
    called crowded instead of polluting the isolated statistics.
    """
    visible = np.flatnonzero(truth["visible"])
    flags = np.zeros(truth["r_m"].shape, dtype=bool)
    for a in visible:
        for b in visible:
            if a == b:
                continue
            if (abs(truth["r_m"][a] - truth["r_m"][b]) <= range_window_m
                    and abs(truth["radial_velocity_mps"][a] - truth["radial_velocity_mps"][b])
                    <= velocity_window_mps):
                flags[a] = True
                break
    return flags


def summarize(rows: list[dict]) -> dict:
    """MAE / RMSE / median / P95 for each error channel of one group."""

    def channel(key: str) -> dict:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        if values.size == 0:
            return {"mae": None, "rmse": None, "median": None, "p95": None}
        return {"mae": float(np.abs(values).mean()), "rmse": float(np.sqrt((values ** 2).mean())),
                "median": float(np.median(np.abs(values))), "p95": float(np.percentile(np.abs(values), 95))}

    return {"matched_pairs": len(rows),
            "range": channel("range_error_m"),
            "radial_velocity": channel("radial_velocity_error_mps"),
            "bearing": channel("bearing_error_deg"),
            "position": channel("position_error_m")}