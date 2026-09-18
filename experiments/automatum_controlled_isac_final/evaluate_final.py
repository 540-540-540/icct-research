"""Phase C: final acceptance of the frozen geometry + calibration with the real frontend.

Runs train and val prediction-history unique states at all five SNR levels, reports the full
metric set (overall / scene / 2-BS / 3-BS / speed bins, tails, stability) and draws the
GT-vs-sensing trajectory figures. Test is never read.
"""
from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.automatum_controlled_isac_final import common, fast_eval  # noqa: E402
from frontend.controlled_isac.automatum_frontend import sense_frame, sense_vehicle  # noqa: E402
from frontend.controlled_isac.automatum_measurement import (calibration_from_config,  # noqa: E402
                                                            setup_from_config)

SPEED_BINS = ((0.0, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, float("inf")))
STABILITY_LEVELS = (-10.0, 0.0, 10.0)
NORMAL_SPEED = 2.0


def load_selected() -> tuple[dict, dict]:
    geometry = json.loads((common.OUT_DIR / "geometry_search.json").read_text()) \
        ["selected_scene0_geometry"]
    calibration = json.loads((common.OUT_DIR / "calibration_search.json").read_text())["selected"]
    if geometry["status"] != "FROZEN" or calibration["status"] != "FROZEN":
        raise SystemExit("geometry/calibration not frozen; refusing final acceptance")
    return geometry, calibration


def frozen_config(config: dict, geometry: dict, calibration: dict) -> dict:
    effective = common.config_with_geometry(config, geometry)
    effective["measurement"]["a"] = {key: float(value) for key, value in calibration["a"].items()}
    return effective


def run_split(effective: dict, split: str) -> tuple[list[tuple], dict]:
    setup = setup_from_config(effective)
    calibration = calibration_from_config(effective)
    history = common.unique_history(split)
    lookup = common.state_lookup(split)
    rows, stability = [], {}
    for scene_id, frame_id, vehicle in history["keys"]:
        state = lookup[(scene_id, frame_id, vehicle)]
        speed = float(np.hypot(state[2], state[3]))
        for level in common.LEVELS:
            estimate = sense_vehicle(scene_id, frame_id, vehicle, state[:2], state[2:], level,
                                     calibration, setup)
            dx = estimate["x_hat"] - float(state[0])
            dy = estimate["y_hat"] - float(state[1])
            dvx = estimate["vx_hat"] - float(state[2])
            dvy = estimate["vy_hat"] - float(state[3])
            rows.append((scene_id, frame_id, vehicle, level, estimate["n_bs"],
                         float(np.hypot(dx, dy)), float(np.hypot(dvx, dvy)), dx, dy,
                         float(state[2]), float(state[3]), speed))
            if level in STABILITY_LEVELS:
                stability[(scene_id, frame_id, vehicle, level)] = (
                    estimate["x_hat"], estimate["y_hat"], estimate["vx_hat"], estimate["vy_hat"],
                    float(state[0]), float(state[1]), float(state[2]), float(state[3]))
    return rows, stability


def stats(values) -> dict:
    array = np.asarray([value for value in values if value is not None and np.isfinite(value)],
                       dtype=np.float64)
    if array.size == 0:
        return {"n": 0}
    return {"n": int(array.size), "mae": float(np.abs(array).mean()),
            "rmse": float(np.sqrt((array ** 2).mean())), "median": float(np.median(np.abs(array))),
            "p90": float(np.percentile(np.abs(array), 90)),
            "p95": float(np.percentile(np.abs(array), 95)),
            "p99": float(np.percentile(np.abs(array), 99))}


def metric_rows(split: str, rows: list[tuple]) -> list[dict]:
    output = []
    scopes = {"overall": (0, 1), "scene_0": (0,), "scene_1": (1,)}
    subsets = {"all": lambda row: True, "n_bs_2": lambda row: row[4] == 2,
               "n_bs_3": lambda row: row[4] == 3}
    for scope, scene_filter in scopes.items():
        for subset_name, predicate in subsets.items():
            for level in common.LEVELS:
                selected = [row for row in rows
                            if row[0] in scene_filter and row[3] == level and predicate(row)]
                entry = {"split": split, "scope": scope, "subset": subset_name, "snr_db": level}
                position = stats([row[5] for row in selected])
                velocity = stats([row[6] for row in selected])
                for key, value in position.items():
                    entry[f"position_{key}"] = value
                for key, value in velocity.items():
                    entry[f"velocity_{key}"] = value
                entry["vx_rmse"] = float(np.sqrt(np.mean([row[7] ** 2 for row in selected]))) \
                    if selected else None
                entry["vy_rmse"] = float(np.sqrt(np.mean([row[8] ** 2 for row in selected]))) \
                    if selected else None
                output.append(entry)
    return output


def speed_bin_rows(split: str, rows: list[tuple]) -> list[dict]:
    output = []
    for low, high in SPEED_BINS:
        label = f"[{low:g},{high:g})" if math.isfinite(high) else f"[{low:g},inf)"
        for level in common.LEVELS:
            selected = [row for row in rows if row[3] == level and low <= row[11] < high]
            entry = {"split": split, "speed_bin_mps": label, "snr_db": level, "n": len(selected)}
            if selected:
                entry["position_rmse_m"] = float(np.sqrt(np.mean([row[5] ** 2
                                                                  for row in selected])))
                entry["velocity_rmse_mps"] = float(np.sqrt(np.mean([row[6] ** 2
                                                                    for row in selected])))
                entry["velocity_median_mps"] = float(np.median([row[6] for row in selected]))
                if low >= NORMAL_SPEED:
                    entry["velocity_error_over_speed_median"] = float(
                        np.median([row[6] / row[11] for row in selected]))
            output.append(entry)
    return output


def stability_rows(split: str, stability: dict) -> list[dict]:
    output = []
    for level in STABILITY_LEVELS:
        for speed_class, predicate in (("low_speed_lt2", lambda s: s < NORMAL_SPEED),
                                       ("normal_speed_ge2", lambda s: s >= NORMAL_SPEED)):
            groups = {}
            for (scene_id, frame_id, vehicle, row_level), value in stability.items():
                if row_level != level:
                    continue
                groups.setdefault((scene_id, vehicle), {})[frame_id] = value
            steps, second_differences, angles, speed_ratios = [], [], [], []
            flips = anomalies = 0
            for (scene_id, vehicle), frames in groups.items():
                for frame_id in sorted(frames):
                    entry = frames[frame_id]
                    speed = math.hypot(entry[6], entry[7])
                    if not predicate(speed):
                        continue
                    if speed >= NORMAL_SPEED:
                        hat_speed = math.hypot(entry[2], entry[3])
                        cos_angle = (entry[2] * entry[6] + entry[3] * entry[7]) / max(
                            hat_speed * speed, 1e-12)
                        angles.append(math.degrees(math.acos(min(max(cos_angle, -1.0), 1.0))))
                        flips += int(cos_angle < 0.0)
                        ratio = hat_speed / speed
                        speed_ratios.append(ratio)
                        anomalies += int(ratio > 1.5 or ratio < 0.5)
                    if frame_id + 1 in frames:
                        nxt = frames[frame_id + 1]
                        steps.append(math.hypot((nxt[0] - entry[0]) - (nxt[4] - entry[4]),
                                                (nxt[1] - entry[1]) - (nxt[5] - entry[5])))
                    if frame_id + 1 in frames and frame_id + 2 in frames:
                        second = frames[frame_id + 1]
                        third = frames[frame_id + 2]
                        truth = [(entry[4], entry[5]), (second[4], second[5]), (third[4], third[5])]
                        hat = [(entry[0], entry[1]), (second[0], second[1]), (third[0], third[1])]
                        truth_curvature = math.hypot(truth[2][0] - 2 * truth[1][0] + truth[0][0],
                                                     truth[2][1] - 2 * truth[1][1] + truth[0][1])
                        hat_curvature = math.hypot(hat[2][0] - 2 * hat[1][0] + hat[0][0],
                                                   hat[2][1] - 2 * hat[1][1] + hat[0][1])
                        second_differences.append(abs(hat_curvature - truth_curvature))
            output.append({
                "split": split, "snr_db": level, "speed_class": speed_class,
                "frame_step_error": stats(steps),
                "second_difference_error": stats(second_differences),
                "direction_angle_deg": stats(angles),
                "direction_flip_fraction": (flips / len(angles)) if angles else None,
                "speed_ratio_median": float(np.median(speed_ratios)) if speed_ratios else None,
                "speed_anomaly_fraction": (anomalies / len(speed_ratios)) if speed_ratios else None,
            })
    return output


def trajectory_figure(effective: dict, split: str, scene_id: int) -> str | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from frontend.automatum_scene_source import AutomatumSceneSource

    samples = np.load(ROOT / "data/automatum_t_crossing/splits" / split / "samples.npz")
    window = None
    for index in range(samples["scene_id"].size):
        if int(samples["scene_id"][index]) != scene_id:
            continue
        if int(samples["num_vehicles"][index]) >= 4:
            window = (int(samples["start_frame"][index]),
                      [int(v) for v in samples["vehicle_ids"][index][samples["vehicle_mask"][index]]])
            break
    if window is None:
        return None
    start, vehicle_ids = window
    source = AutomatumSceneSource(ROOT / effective["dataset"][f"{split}_trajectories"])
    setup = setup_from_config(effective)
    calibration = calibration_from_config(effective)
    frames = [source.at_frame(scene_id, start + offset) for offset in range(20)]
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 9.6))
    panels = axes.ravel()
    panels[0].set_title("GT")
    for panel, level in zip(panels[1:], (10.0, 0.0, -10.0)):
        panel.set_title(f"{level:+.0f} dB")
    for vehicle_id in vehicle_ids:
        truth_x = [float(frame.states[frame.vehicle_ids.tolist().index(vehicle_id), 0])
                   for frame in frames]
        truth_y = [float(frame.states[frame.vehicle_ids.tolist().index(vehicle_id), 1])
                   for frame in frames]
        panels[0].plot(truth_x, truth_y, "-", color="black", linewidth=2.0,
                       label="GT" if vehicle_id == vehicle_ids[0] else None)
        for panel, level in zip(panels[1:], (10.0, 0.0, -10.0)):
            estimates = []
            for frame in frames:
                states = sense_frame(frame, scene_id, level, calibration, setup)
                estimates.append(next(item for item in states if item["vehicle_id"] == vehicle_id))
            panel.plot([item["x_hat"] for item in estimates], [item["y_hat"] for item in estimates],
                       "--", color="tab:red", linewidth=1.3,
                       label="sensing" if vehicle_id == vehicle_ids[0] else None)
            panel.plot(truth_x, truth_y, "-", color="black", linewidth=1.6, alpha=0.5)
    for panel in panels:
        panel.set_xlabel("x (m)")
        panel.set_ylabel("y (m)")
        panel.set_aspect("equal", adjustable="datalim")
        panel.grid(True, alpha=0.3)
        panel.legend(fontsize=7)
    figure.suptitle(f"Automatum Route-B final: {split} scene {scene_id} frames "
                    f"{start}-{start + 19} (same vehicles/window); GT black, sensing red dashed")
    figure.tight_layout()
    plots = common.OUT_DIR / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    path = plots / f"trajectories_{split}_scene{scene_id}.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return str(path.relative_to(ROOT)).replace("\\", "/")


def main() -> int:
    started = time.time()
    config = common.load_config()
    geometry, calibration = load_selected()
    effective = frozen_config(config, geometry, calibration)

    metrics_rows, speed_rows, stability = [], [], []
    instability = {}
    for split in ("train", "val"):
        rows, stability_data = run_split(effective, split)
        instability[split] = stability_data
        metrics_rows.extend(metric_rows(split, rows))
        speed_rows.extend(speed_bin_rows(split, rows))
        stability.extend(stability_rows(split, stability_data))
        print(f"{split}: {len(rows)} state-level rows")
    common.write_csv(common.OUT_DIR / "final_metrics.csv",
                     [row for row in metrics_rows if row["subset"] == "all"])
    common.write_csv(common.OUT_DIR / "final_metrics_by_bs.csv",
                     [row for row in metrics_rows if row["subset"] != "all"])
    common.write_csv(common.OUT_DIR / "final_speed_bins.csv", speed_rows)
    common.write_csv(common.OUT_DIR / "final_stability.csv", stability)

    plots = []
    for scene_id in (0, 1):
        for split in ("train", "val"):
            path = trajectory_figure(effective, split, scene_id)
            if path:
                plots.append(path)

    subset = None
    bands = {}
    for level in (10.0, 0.0, -10.0):
        for metric, band in fast_eval.BANDS[level].items():
            value = next(row[f"{metric}_rmse"] for row in metrics_rows
                         if row["split"] == "train" and row["scope"] == "overall"
                         and row["subset"] == "all" and row["snr_db"] == level)
            low, high = band
            bands[f"{level:+g}_{metric}"] = {"value": value, "band": [low, high],
                                             "pass": low <= value <= high}
    monotonic = {}
    for metric in ("position", "velocity"):
        series = [next(row[f"{metric}_rmse"] for row in metrics_rows
                       if row["split"] == "train" and row["scope"] == "overall"
                       and row["subset"] == "all" and row["snr_db"] == level)
                  for level in sorted(common.LEVELS, reverse=True)]
        monotonic[metric] = {"values_desc_snr": series,
                             "strictly_increasing_as_snr_drops": all(
                                 series[i] < series[i + 1] for i in range(len(series) - 1))}
    payload = {
        "baseline_head": "eaadbe6d826f9f6e83baf3caf3c01bf75ac093d1",
        "geometry": geometry,
        "calibration": calibration,
        "bands": bands,
        "monotonicity": monotonic,
        "stability": stability,
        "trajectory_plots": plots,
        "test_used": False,
    }
    common.write_json(common.OUT_DIR / "final_metrics.json", payload)
    print(json.dumps({"bands": {key: value["pass"] for key, value in bands.items()},
                      "monotonic": {key: value["strictly_increasing_as_snr_drops"]
                                    for key, value in monotonic.items()},
                      "runtime_s": time.time() - started}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())