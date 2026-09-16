"""ISAC-REDESIGN-PROBE-01: controlled ISAC frontend prototype (probe, no production changes).

Pipeline: NGSIM GT -> 40-frame continuous multi-vehicle windows -> physical polar truth per BS ->
SNR-calibrated measurement noise (fixed base disturbance, rescaled per SNR) -> anonymous per-BS
target lists (shuffled) -> cross-BS association -> same-frame GLS position fusion -> current-frame
2D velocity from multi-BS radial velocities -> frame-to-frame slot linking that copies the current
fusion result unchanged (no Kalman, no smoothing, no coast, no birth/death research) -> 20-frame
history state plus vehicle_mask and GT future labels.

Error calibration uses the frozen physical chain (shared echo -> B64 -> RD -> detector -> AoA) on
train-only calibration scenes. Development acceptance metrics use train windows; V_select is opened
only once for the downstream smoke with the frozen checkpoints. Formal F01-E, sensing, tracker,
configs, V_confirm and test are untouched.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import _common  # noqa: E402
import audit_snr_response as audit_base  # noqa: E402
from frontend.controlled_isac import cross_bs_association as cross_bs  # noqa: E402
from frontend.controlled_isac import measurement  # noqa: E402
from frontend.controlled_isac import temporal_association  # noqa: E402
from frontend.controlled_isac import velocity_fusion  # noqa: E402

REPORT = ROOT / "reports/controlled_isac/redesign_probe_01"
CACHE = ROOT / ".codex-work/isac-redesign-probe-01"
SNRS = (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0)
SPLITS = ("train", "v_select")
TEMPORAL_GATE_M = 5.0
CALIBRATION_GATE_M = 5.0
MIN_VEHICLES, MAX_VEHICLES = 2, 8
HISTORY, FUTURE = 20, 20
MIN_GEOMETRY_CONDITION_RATIO = 0.01


def snr_key(snr: float) -> str:
    return str(int(snr))


def sanitize(value):
    if isinstance(value, dict):
        return {str(key): sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(sanitize(payload), ensure_ascii=False, indent=2, allow_nan=False)
                         + "\n", encoding="utf-8")
    temporary.replace(path)


def quantile(values, fraction: float):
    values = np.asarray(values, dtype=np.float64)
    return float(np.quantile(values, fraction)) if values.size else None


def robust_scale(values) -> float:
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return 0.0
    median = float(np.median(values))
    return 1.4826 * float(np.median(np.abs(values - median)))


def error_stats(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return {"n": 0, "median_abs": None, "rmse": None, "p90_abs": None, "robust_scale": None,
                "signed_median": None}
    absolute = np.abs(values)
    return {"n": int(values.size), "mean_abs": float(absolute.mean()),
            "median_abs": float(np.median(absolute)),
            "rmse": float(np.sqrt(np.mean(values ** 2))), "p90_abs": quantile(absolute, 0.9),
            "robust_scale": robust_scale(values), "signed_median": float(np.median(values))}


def pava_non_increasing(values):
    """Isotonic non-increasing fit (pool adjacent violators); minimal squared-error adjustment."""
    means, counts = [], []
    for value in values:
        means.append(float(value))
        counts.append(1)
        while len(means) >= 2 and means[-2] < means[-1]:
            weight = counts[-2] + counts[-1]
            merged = (means[-2] * counts[-2] + means[-1] * counts[-1]) / weight
            means = means[:-2] + [merged]
            counts = counts[:-2] + [weight]
    return [mean for mean, count in zip(means, counts) for _ in range(count)]


# ---------------------------------------------------------------------------
# window selection and coverage
# ---------------------------------------------------------------------------

def track_states(source, key: int, times_ms: np.ndarray) -> tuple:
    track = source.tracks[int(key)]
    positions = np.searchsorted(track["time_ms"], times_ms)
    in_range = positions < len(track)
    safe = np.minimum(positions, max(0, len(track) - 1))
    exact = in_range & (track["time_ms"][safe] == times_ms)
    states = np.stack([track["x"][safe], track["y"][safe], track["vx"][safe], track["vy"][safe]],
                      axis=-1).astype(np.float64)
    states[~exact] = np.nan
    return states, exact


def build_windows(source, setup, split: str) -> tuple:
    windows = []
    stats = {"episodes": 0, "origins": 0, "candidate_windows": 0, "two_bs_windows": 0,
             "recoverable_windows": 0, "covered_windows": 0,
             "vehicle_count_histogram": Counter()}
    for index, episode in enumerate(source.episodes):
        if episode["split"].lower() != split:
            continue
        stats["episodes"] += 1
        for origin in episode["prediction_grid_ms"]:
            stats["origins"] += 1
            origin = int(origin)
            times = origin + np.arange(-(HISTORY - 1), FUTURE + 1, dtype=np.int64) * 100
            states = {}
            for key in episode["source_keys"]:
                values, exact = track_states(source, int(key), times)
                if bool(exact.all()):
                    states[int(key)] = values
            if not MIN_VEHICLES <= len(states) <= MAX_VEHICLES:
                continue
            stats["candidate_windows"] += 1
            stats["vehicle_count_histogram"][len(states)] += 1
            two_bs = True
            recoverable = True
            for values in states.values():
                for frame in range(HISTORY):
                    count, ratio = measurement.recoverable_geometry(
                        values[frame, :2], setup, MIN_GEOMETRY_CONDITION_RATIO)
                    if count < 2:
                        two_bs = False
                    if count < 2 or ratio < MIN_GEOMETRY_CONDITION_RATIO:
                        recoverable = False
                if not two_bs and not recoverable:
                    break
            if two_bs:
                stats["two_bs_windows"] += 1
                stats["covered_windows"] += 1
            if recoverable:
                stats["recoverable_windows"] += 1
            if not two_bs:
                continue
            windows.append({"split": split, "episode_index": index, "episode_id": episode["episode_id"],
                            "origin_ms": origin, "times": times, "states": states})
    return windows, stats


# ---------------------------------------------------------------------------
# physical calibration on train-only scenes
# ---------------------------------------------------------------------------

def run_calibration(source, setup, train_windows: list, scenes: int, device: str) -> dict:
    from frontend.sensing import detector as detector_module

    config = _common.load_frontend_config()
    waveform, array = _common.build_objects(config)
    multiplier = float(config["detector"]["cfar_threshold_multiplier"])
    lut = _common.load_production_lut(config)
    if lut is None:
        raise SystemExit("covariance LUT missing")
    resource = _common.load_resource(config)
    step = max(1, len(train_windows) // max(scenes, 1))
    chosen = train_windows[::step][:scenes]
    errors = {snr: {"range": [], "bearing": [], "radial_velocity": []} for snr in SNRS}
    counters = {snr: {"scenes": 0, "visible_pairs": 0, "detections": 0, "matched": 0}
                for snr in SNRS}
    for scene_index, window in enumerate(chosen):
        frame = HISTORY - 1
        keys = list(window["states"])
        positions = [window["states"][key][frame, :2] for key in keys]
        velocities = [window["states"][key][frame, 2:] for key in keys]
        for snr in SNRS:
            echo = _common.synthesize(positions, velocities, keys, setup["stations"],
                                      setup["boresights"], waveform, array, config, snr,
                                      9600 + scene_index, 0, device, noise=True,
                                      height_m=setup["height"])
            counts = counters[snr]
            counts["scenes"] += 1
            for bs in range(3):
                detections, _, _ = detector_module.detect_from_maps(
                    detector_module.compute_maps(echo["Y"][bs], echo["X"][bs], waveform, array,
                                                 config["detector"], resource=resource),
                    bs, 0, setup["stations"][bs], float(setup["boresights"][bs]), config, array,
                    multiplier=multiplier, covariance_lut=lut, height=setup["height"])
                counts["detections"] += len(detections)
                points = [np.array([detection["x_m"], detection["y_m"]]) for detection in detections]
                matched, _ = audit_base.one_to_one_match(positions, points, gate=CALIBRATION_GATE_M)
                counts["visible_pairs"] += len(positions)
                for gt_index, detection_index, _ in matched:
                    counts["matched"] += 1
                    truth = measurement.truth_measurement(positions[gt_index], velocities[gt_index],
                                                          setup["stations"][bs],
                                                          float(setup["boresights"][bs]),
                                                          setup["height"])
                    detection = detections[detection_index]
                    errors[snr]["range"].append(float(detection["r_m"]) - truth["r"])
                    errors[snr]["bearing"].append(float(detection["bearing_rad"]) - truth["bearing"])
                    errors[snr]["radial_velocity"].append(float(detection["vr_mps"])
                                                          - truth["radial_velocity"])
    raw = {"snr_db": list(SNRS), "gate_m": CALIBRATION_GATE_M, "scenes": len(chosen),
           "by_snr": {snr_key(snr): {metric: error_stats(values)
                                     for metric, values in errors[snr].items()}
                      for snr in SNRS},
           "counters": {snr_key(snr): counters[snr] for snr in SNRS}}
    return raw


def finalize_calibration(raw: dict) -> dict:
    final_scales, changes = {}, {}
    for metric in measurement.SIGMA_KEYS:
        values = [raw["by_snr"][snr_key(snr)][metric]["robust_scale"] for snr in SNRS]
        fitted = pava_non_increasing(values)
        final_scales[metric] = dict(zip((snr_key(snr) for snr in SNRS), fitted))
        changes[metric] = {"raw": values, "final": fitted,
                           "max_relative_change": max(abs(f - v) / v
                                                      for f, v in zip(fitted, values) if v)}
    return {"snr_db": list(SNRS), "smoothing": "isotonic non-increasing (pool adjacent violators)",
            "sigma": final_scales, "adjustment": changes}


# ---------------------------------------------------------------------------
# controlled dataset per split
# ---------------------------------------------------------------------------

def process_window(source, setup, window: dict, sigma_final: dict, gate_chi2: float,
                   device_unused=None) -> dict:
    episode_index = window["episode_index"]
    keys_sorted = sorted(window["states"])
    n_vehicles = len(keys_sorted)
    crowded = n_vehicles >= 5
    out = {}
    history_times = window["times"][:HISTORY]
    for snr in SNRS:
        sigma = {key: sigma_final["sigma"][key][snr_key(snr)] for key in measurement.SIGMA_KEYS}
        frame_records = []
        for frame in range(HISTORY):
            deadline = int(history_times[frame])
            by_bs = {0: [], 1: [], 2: []}
            visible_bs_by_key = {}
            for key in keys_sorted:
                values = window["states"][key]
                visible_bs_by_key[key] = []
                for bs in range(3):
                    item = measurement.make_measurement(episode_index, deadline, bs, key,
                                                        values[frame, :2], values[frame, 2:],
                                                        sigma, setup)
                    if item is not None:
                        by_bs[bs].append(item)
                        visible_bs_by_key[key].append(bs)
            for bs in range(3):
                order = measurement.shuffle_order(episode_index, deadline, bs, len(by_bs[bs]))
                by_bs[bs] = [by_bs[bs][index] for index in order]
            groups, diagnostics = cross_bs.associate_frame(by_bs, gate_chi2)
            fused = []
            for group in groups:
                solution = velocity_fusion.recover_velocity(group["position"], group["members"],
                                                            sigma["radial_velocity"])
                fused.append({"position": np.asarray(group["position"], dtype=float),
                              "covariance": np.asarray(group["covariance"], dtype=float),
                              "velocity": np.asarray([solution["vx"], solution["vy"]], dtype=float),
                              "n_bs": group["n_bs"],
                              "member_keys": [member["source_key"] for member in group["members"]],
                              "rank": solution["rank"],
                              "condition_ratio": solution["condition_ratio"]})
            frame_records.append({"fused": fused, "visible_bs_by_key": visible_bs_by_key,
                                  "diagnostics": diagnostics})

        tracker = temporal_association.SlotTracker(gate_m=TEMPORAL_GATE_M)
        first_targets = [{"position": item["position"], "velocity": item["velocity"]}
                         for item in frame_records[0]["fused"]]
        tracker.initialize(first_targets)
        slot_to_target = [list(range(len(first_targets)))]
        birth_total, death_total, rank_deficient = 0, 0, 0
        for frame in range(1, HISTORY):
            targets = [{"position": item["position"], "velocity": item["velocity"]}
                       for item in frame_records[frame]["fused"]]
            result = tracker.step(targets)
            slot_to_target.append(result["slot_to_target"])
            birth_total += len(result["births"])
            death_total += len(result["deaths"])
        rank_deficient = sum(1 for frame in frame_records for item in frame["fused"]
                             if item["rank"] < 2)
        ill_conditioned = sum(1 for frame in frame_records for item in frame["fused"]
                              if item["condition_ratio"] < 1e-3)

        n_slots = len(first_targets)
        slot_keys = [[None] * n_slots for _ in range(HISTORY)]
        for frame in range(HISTORY):
            for slot, target_index in enumerate(slot_to_target[frame]):
                if target_index < 0:
                    continue
                member_keys = frame_records[frame]["fused"][target_index]["member_keys"]
                if member_keys:
                    slot_keys[frame][slot] = Counter(member_keys).most_common(1)[0][0]
        majority = []
        for slot in range(n_slots):
            values = [slot_keys[frame][slot] for frame in range(HISTORY)
                      if slot_keys[frame][slot] is not None]
            majority.append(Counter(values).most_common(1)[0][0] if values else None)
        assignment, used = {}, set()
        for slot in sorted(range(n_slots), key=lambda item: (majority[item] is None,
                                                             majority[item] or 0)):
            key = majority[slot]
            if key is None or key in used:
                continue
            used.add(key)
            assignment[slot] = len(assignment)
        state_hat = np.zeros((HISTORY, MAX_VEHICLES, 4), np.float32)
        vehicle_mask = np.zeros(MAX_VEHICLES, bool)
        for old_slot, new_slot in assignment.items():
            vehicle_mask[new_slot] = True
            for frame in range(HISTORY):
                target_index = slot_to_target[frame][old_slot]
                if target_index >= 0:
                    item = frame_records[frame]["fused"][target_index]
                    state_hat[frame, new_slot, :2] = item["position"]
                    state_hat[frame, new_slot, 2:] = item["velocity"]
        future_position = np.zeros((FUTURE, MAX_VEHICLES, 2), np.float32)
        gt_history_position = np.full((HISTORY, MAX_VEHICLES, 2), np.nan)
        gt_history_velocity = np.full((HISTORY, MAX_VEHICLES, 2), np.nan)
        for old_slot, new_slot in assignment.items():
            values = window["states"][majority[old_slot]]
            future_position[:, new_slot] = values[HISTORY:, :2].astype(np.float32)
            gt_history_position[:, new_slot] = values[:HISTORY, :2]
            gt_history_velocity[:, new_slot] = values[:HISTORY, 2:]

        errors, association, temporal, smoothing, id_switches = [], Counter(), Counter(), Counter(), 0
        velocity_errors = []
        smoothing["position_float32_exact"] = True
        smoothing["velocity_float32_exact"] = True
        for frame in range(HISTORY):
            for old_slot, new_slot in assignment.items():
                target_index = slot_to_target[frame][old_slot]
                if target_index < 0:
                    continue
                key = majority[old_slot]
                item = frame_records[frame]["fused"][target_index]
                errors.append(float(np.linalg.norm(item["position"]
                                                   - gt_history_position[frame, new_slot])))
                velocity_errors.append(float(np.linalg.norm(item["velocity"]
                                                            - gt_history_velocity[frame, new_slot])))
                smoothing["position_max_abs_diff"] = max(
                    smoothing["position_max_abs_diff"],
                    float(np.max(np.abs(state_hat[frame, new_slot, :2] - item["position"]))))
                smoothing["velocity_max_abs_diff"] = max(
                    smoothing["velocity_max_abs_diff"],
                    float(np.max(np.abs(state_hat[frame, new_slot, 2:] - item["velocity"]))))
                if not np.array_equal(state_hat[frame, new_slot, :2],
                                      np.asarray(item["position"], np.float32)):
                    smoothing["position_float32_exact"] = False
                if not np.array_equal(state_hat[frame, new_slot, 2:],
                                      np.asarray(item["velocity"], np.float32)):
                    smoothing["velocity_float32_exact"] = False
                if slot_keys[frame][old_slot] == key:
                    temporal["correct"] += 1
                temporal["total"] += 1
            for key, bs_list in frame_records[frame]["visible_bs_by_key"].items():
                if len(bs_list) < 2:
                    continue
                group = None
                for item in frame_records[frame]["fused"]:
                    if key in item["member_keys"]:
                        group = item
                        break
                association["total"] += 1
                if crowded:
                    association["crowded_total"] += 1
                else:
                    association["normal_total"] += 1
                correct = (group is not None and group["n_bs"] == len(bs_list)
                           and all(member == key for member in group["member_keys"]))
                if correct:
                    association["correct"] += 1
                    if crowded:
                        association["crowded_correct"] += 1
                    else:
                        association["normal_correct"] += 1
        inverse_assignment = {new_slot: old_slot for old_slot, new_slot in assignment.items()}
        for new_slot in range(len(assignment)):
            previous = None
            for frame in range(HISTORY):
                key = slot_keys[frame][inverse_assignment[new_slot]]
                if key is not None and previous is not None and key != previous:
                    id_switches += 1
                if key is not None:
                    previous = key
        out[snr] = {
            "state_hat": state_hat, "vehicle_mask": vehicle_mask,
            "future_position": future_position,
            "gt_history_position": gt_history_position, "gt_history_velocity": gt_history_velocity,
            "position_errors": errors, "velocity_errors": velocity_errors,
            "association": dict(association), "temporal": dict(temporal),
            "smoothing": dict(smoothing), "id_switches": id_switches,
            "births": birth_total, "deaths": death_total, "rank_deficient": rank_deficient,
            "ill_conditioned": ill_conditioned, "n_vehicles": n_vehicles}
    return out


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

def empty_aggregate() -> dict:
    return {snr: {"position": [], "velocity": [], "association": Counter(), "temporal": Counter(),
                  "smoothing": Counter(), "id_switches": 0, "births": 0, "deaths": 0,
                  "rank_deficient": 0, "ill_conditioned": 0, "windows": 0, "vehicles": 0,
                  "crowded_windows": 0}
            for snr in SNRS}


def accumulate(aggregate: dict, result: dict) -> None:
    for snr, entry in result.items():
        agg = aggregate[snr]
        agg["position"].extend(entry["position_errors"])
        agg["velocity"].extend(entry["velocity_errors"])
        agg["association"].update(entry["association"])
        agg["temporal"].update(entry["temporal"])
        agg["smoothing"]["position_max_abs_diff"] = max(
            agg["smoothing"].get("position_max_abs_diff", 0.0),
            entry["smoothing"].get("position_max_abs_diff", 0.0))
        agg["smoothing"]["velocity_max_abs_diff"] = max(
            agg["smoothing"].get("velocity_max_abs_diff", 0.0),
            entry["smoothing"].get("velocity_max_abs_diff", 0.0))
        agg["smoothing"]["position_float32_exact"] = (
            agg["smoothing"].get("position_float32_exact", True)
            and entry["smoothing"].get("position_float32_exact", True))
        agg["smoothing"]["velocity_float32_exact"] = (
            agg["smoothing"].get("velocity_float32_exact", True)
            and entry["smoothing"].get("velocity_float32_exact", True))
        agg["id_switches"] += entry["id_switches"]
        agg["births"] += entry["births"]
        agg["deaths"] += entry["deaths"]
        agg["rank_deficient"] += entry["rank_deficient"]
        agg["ill_conditioned"] += entry["ill_conditioned"]
        agg["windows"] += 1
        agg["vehicles"] += entry["n_vehicles"]
        agg["crowded_windows"] += int(entry["n_vehicles"] >= 5)


def summarize_aggregate(aggregate: dict) -> dict:
    summary = {}
    for snr in SNRS:
        agg = aggregate[snr]
        association = agg["association"]
        temporal = agg["temporal"]
        summary[snr_key(snr)] = {
            "position": error_stats(agg["position"]),
            "velocity": error_stats(agg["velocity"]),
            "association_accuracy": association["correct"] / max(association["total"], 1),
            "association_total": association["total"],
            "crowded_association_accuracy": (association["crowded_correct"]
                                             / max(association["crowded_total"], 1)),
            "crowded_total": association["crowded_total"],
            "normal_association_accuracy": (association["normal_correct"]
                                            / max(association["normal_total"], 1)),
            "normal_total": association["normal_total"],
            "temporal_accuracy": temporal["correct"] / max(temporal["total"], 1),
            "temporal_total": temporal["total"],
            "id_switches": agg["id_switches"],
            "no_smoothing_position_max_abs_diff": agg["smoothing"].get("position_max_abs_diff", 0.0),
            "no_smoothing_velocity_max_abs_diff": agg["smoothing"].get("velocity_max_abs_diff", 0.0),
            "no_smoothing_position_float32_exact": agg["smoothing"].get("position_float32_exact", True),
            "no_smoothing_velocity_float32_exact": agg["smoothing"].get("velocity_float32_exact", True),
            "births": agg["births"], "deaths": agg["deaths"],
            "rank_deficient_velocity_frames": agg["rank_deficient"],
            "ill_conditioned_velocity_frames": agg["ill_conditioned"],
            "windows": agg["windows"], "crowded_windows": agg["crowded_windows"],
            "mean_vehicles": agg["vehicles"] / max(agg["windows"], 1)}
    return summary


def snr_cache_name(snr: float) -> str:
    return "state_" + snr_key(snr)


# ---------------------------------------------------------------------------
# phases
# ---------------------------------------------------------------------------

def phase_coverage(source, setup) -> tuple:
    coverage, windows = {}, {}
    for split in SPLITS:
        window_list, stats = build_windows(source, setup, split)
        windows[split] = window_list
        stats["retention_ratio"] = stats["covered_windows"] / max(stats["candidate_windows"], 1)
        stats["recoverable_retention_ratio"] = (stats["recoverable_windows"]
                                                / max(stats["candidate_windows"], 1))
        coverage[split] = stats
        with (CACHE / f"windows_{split}.pkl").open("wb") as handle:
            pickle.dump(window_list, handle)
    candidate = sum(coverage[split]["candidate_windows"] for split in SPLITS)
    covered = sum(coverage[split]["covered_windows"] for split in SPLITS)
    recoverable = sum(coverage[split]["recoverable_windows"] for split in SPLITS)
    coverage["rule"] = ("official rule: each participating vehicle visible to >=2 BS in every one "
                        "of the 20 history frames (range 10-300 m, |bearing| <= 70 deg)")
    coverage["candidate_windows"] = candidate
    coverage["two_bs_windows"] = covered
    coverage["covered_windows"] = covered
    coverage["recoverable_windows"] = recoverable
    coverage["retention_ratio"] = covered / max(candidate, 1)
    coverage["recoverable_retention_ratio"] = recoverable / max(candidate, 1)
    coverage["diagnostic_recoverability"] = {
        "definition": "visible-BS geometry has lambda_min/lambda_max of sum(u u^T) >= 0.01 "
                      "(2D velocity recoverable); counted on the same candidate windows",
        "note": "diagnostic only; the official >=2-BS selection rule is used for the dataset. "
                "The recoverability condition would retain below the 70% policy threshold, so it "
                "was not adopted unilaterally (section 4 policy)"}
    coverage["stopped"] = coverage["retention_ratio"] < 0.70
    write_json(REPORT / "coverage.json", coverage)
    write_json(CACHE / "coverage.json", coverage)
    return coverage, windows


def load_windows() -> dict:
    windows = {}
    for split in SPLITS:
        with (CACHE / f"windows_{split}.pkl").open("rb") as handle:
            windows[split] = pickle.load(handle)
    return windows


def phase_calibration(source, setup, windows, scenes: int, device: str) -> tuple:
    raw = run_calibration(source, setup, windows["train"], scenes, device)
    final = finalize_calibration(raw)
    write_json(REPORT / "calibration_raw.json", raw)
    write_json(REPORT / "calibration_final.json", final)
    write_json(CACHE / "calibration_raw.json", raw)
    write_json(CACHE / "calibration_final.json", final)
    return raw, final


def phase_dataset(source, setup, windows, calibration_final: dict) -> tuple:
    gate = setup["gate_chi2"]
    train_aggregate = empty_aggregate()
    started = time.time()
    for index, window in enumerate(windows["train"]):
        accumulate(train_aggregate, process_window(source, setup, window, calibration_final, gate))
        if (index + 1) % 500 == 0:
            print(json.dumps({"phase": "dataset", "split": "train", "done": index + 1,
                              "total": len(windows["train"]),
                              "elapsed_s": round(time.time() - started, 1)}), flush=True)
    train_summary = summarize_aggregate(train_aggregate)
    write_json(CACHE / "train_metrics.json", train_summary)

    vselect_aggregate = empty_aggregate()
    arrays = {snr: {"state_hat": [], "vehicle_mask": [], "future_position": []} for snr in SNRS}
    for index, window in enumerate(windows["v_select"]):
        result = process_window(source, setup, window, calibration_final, gate)
        accumulate(vselect_aggregate, result)
        for snr in SNRS:
            arrays[snr]["state_hat"].append(result[snr]["state_hat"])
            arrays[snr]["vehicle_mask"].append(result[snr]["vehicle_mask"])
            arrays[snr]["future_position"].append(result[snr]["future_position"])
        if (index + 1) % 200 == 0:
            print(json.dumps({"phase": "dataset", "split": "v_select", "done": index + 1,
                              "total": len(windows["v_select"]),
                              "elapsed_s": round(time.time() - started, 1)}), flush=True)
    vselect_summary = summarize_aggregate(vselect_aggregate)
    write_json(CACHE / "vselect_metrics.json", vselect_summary)
    payload = {}
    for snr in SNRS:
        payload[snr_cache_name(snr)] = np.stack(arrays[snr]["state_hat"])
        payload[f"future_{snr_key(snr)}"] = np.stack(arrays[snr]["future_position"])
    payload["mask"] = np.stack(arrays[20.0]["vehicle_mask"])
    np.savez_compressed(CACHE / "vselect_controlled.npz", **payload)
    return train_summary, vselect_summary


def phase_downstream(device: str) -> dict:
    import experiments.snr_downstream_smoke_01.probe as smoke
    from experiments.qgnn_bottleneck import metrics

    data = np.load(CACHE / "vselect_controlled.npz")
    normalization = json.loads((ROOT / "data/f01e/normalization.json").read_text())
    n = data[snr_cache_name(-5.0)].shape[0]
    mask = data["mask"]
    future = data["future_20"]
    for snr in SNRS:
        if not np.array_equal(data[f"future_{snr_key(snr)}"], future):
            raise ValueError("future labels differ across SNR")
    valid = np.broadcast_to(mask[:, None, :], (n, HISTORY, MAX_VEHICLES)).copy()
    predictions = {"CV": {}}
    for snr in SNRS:
        state = data[snr_cache_name(snr)]
        origin = state[:, -1, :, :].astype(np.float64)
        horizon = np.arange(1, FUTURE + 1, dtype=np.float64)[None, :, None, None] * 0.1
        predictions["CV"][snr] = (origin[:, None, :, :2]
                                  + horizon * origin[:, None, :, 2:]).astype(np.float32)
    models, checkpoint_meta = smoke.load_models(device)
    for name, model in models.items():
        predictions[name] = {}
        for snr in SNRS:
            arrays = {"state_hat": data[snr_cache_name(snr)],
                      "track_exists": valid.copy(), "detected": valid.copy()}
            predictions[name][snr] = smoke.infer(model, arrays, normalization, device)
    rows = []
    for model in ("CV", "GNN", "QGNN"):
        for snr in SNRS:
            scene_macro = next(row for row in metrics.evaluate_metrics(
                predictions[model][snr], future, valid, mask) if row["horizon_steps"] == FUTURE)
            distance = np.linalg.norm(predictions[model][snr].astype(np.float64)
                                      - future.astype(np.float64), axis=-1)
            point = valid & mask[:, None, :]
            point_ade = float(distance[point].mean()) if point.any() else None
            end = point[:, FUTURE - 1]
            point_fde = float(distance[:, FUTURE - 1][end].mean()) if end.any() else None
            rows.append({"model": model, "cohort": "common_paired", "snr_db": int(snr),
                         "ADE": scene_macro["ADE"], "FDE": scene_macro["FDE"],
                         "ADE_point": point_ade, "FDE_point": point_fde,
                         "ade_targets": scene_macro["ade_targets"],
                         "valid_points": scene_macro["valid_points"]})
    result = {"cohort_size": int(mask.sum()), "samples": int(n), "rows": rows,
              "checkpoints": checkpoint_meta,
              "normalization": "data/f01e/normalization.json (train-fit adapter)"}
    with (REPORT / "downstream_smoke.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(CACHE / "downstream.json", result)
    return result


def direction_ok(values: list) -> tuple:
    correct = sum(1 for current, following in zip(values, values[1:]) if following <= current)
    return correct, len(values) - 1


def phase_report(coverage: dict, calibration_raw: dict, calibration_final: dict,
                 train_summary: dict, downstream: dict) -> dict:
    def rmse(metric_key, snr):
        return train_summary[snr_key(snr)][metric_key]["rmse"]

    position_rmse = [rmse("position", snr) for snr in SNRS]
    velocity_rmse = [rmse("velocity", snr) for snr in SNRS]
    position_direction, position_pairs = direction_ok(position_rmse)
    velocity_direction, velocity_pairs = direction_ok(velocity_rmse)
    position_ratio = position_rmse[0] / position_rmse[-1]
    velocity_ratio = velocity_rmse[0] / velocity_rmse[-1]
    downstream_by_model = {}
    for row in downstream["rows"]:
        downstream_by_model.setdefault(row["model"], {})[row["snr_db"]] = row
    cv = downstream_by_model["CV"]
    cv_degradation = 100.0 * (cv[-5]["ADE"] - cv[20]["ADE"]) / cv[20]["ADE"]
    if cv_degradation >= 5.0:
        downstream_verdict = "PASS"
    elif cv_degradation >= 3.0:
        downstream_verdict = "BORDERLINE"
    else:
        downstream_verdict = "FAIL"

    no_smoothing_position = max(train_summary[snr_key(snr)]["no_smoothing_position_max_abs_diff"]
                                for snr in SNRS)
    no_smoothing_velocity = max(train_summary[snr_key(snr)]["no_smoothing_velocity_max_abs_diff"]
                                for snr in SNRS)
    no_smoothing_exact = all(train_summary[snr_key(snr)]["no_smoothing_position_float32_exact"]
                             and train_summary[snr_key(snr)]["no_smoothing_velocity_float32_exact"]
                             for snr in SNRS)
    acceptance = {
        "coverage_retention_ge_0p70": coverage["retention_ratio"] >= 0.70,
        "position_direction_ge_4_of_5": position_direction >= 4,
        "position_rmse_ratio_minus5_over_20_ge_1p5": position_ratio >= 1.5,
        "velocity_direction_ge_4_of_5": velocity_direction >= 4,
        "velocity_rmse_ratio_minus5_over_20_ge_1p5": velocity_ratio >= 1.5,
        "cross_bs_accuracy_20db_ge_0p99": train_summary["20"]["association_accuracy"] >= 0.99,
        "cross_bs_accuracy_minus5db_ge_0p97": train_summary["-5"]["association_accuracy"] >= 0.97,
        "temporal_accuracy_20db_ge_0p99": train_summary["20"]["temporal_accuracy"] >= 0.99,
        "temporal_accuracy_minus5db_ge_0p97": train_summary["-5"]["temporal_accuracy"] >= 0.97,
        "no_smoothing_pass": (no_smoothing_position <= 1e-5 and no_smoothing_velocity <= 1e-5
                              and no_smoothing_exact),
        "downstream_separability_pass": downstream_verdict == "PASS"}
    overall = all(acceptance.values())

    state_rows = []
    for snr in SNRS:
        entry = train_summary[snr_key(snr)]
        state_rows.append({
            "snr_ref_db": int(snr),
            "position_mean_m": entry["position"]["mean_abs"],
            "position_median_m": entry["position"]["median_abs"],
            "position_rmse_m": entry["position"]["rmse"],
            "position_p90_m": entry["position"]["p90_abs"],
            "velocity_mean_mps": entry["velocity"]["mean_abs"],
            "velocity_median_mps": entry["velocity"]["median_abs"],
            "velocity_rmse_mps": entry["velocity"]["rmse"],
            "velocity_p90_mps": entry["velocity"]["p90_abs"],
            "displacement_scale_2s_m": entry["velocity"]["rmse"] * 2.0,
            "association_accuracy": entry["association_accuracy"],
            "crowded_association_accuracy": entry["crowded_association_accuracy"],
            "normal_association_accuracy": entry["normal_association_accuracy"],
            "temporal_accuracy": entry["temporal_accuracy"],
            "id_switches": entry["id_switches"],
            "rank_deficient_velocity_frames": entry["rank_deficient_velocity_frames"],
            "ill_conditioned_velocity_frames": entry["ill_conditioned_velocity_frames"]})
    with (REPORT / "state_error_by_snr.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(state_rows[0]))
        writer.writeheader()
        writer.writerows(state_rows)

    association_metrics = {
        "stage": "ISAC-REDESIGN-PROBE-01",
        "split": "train",
        "cross_bs": {snr_key(snr): {"accuracy": train_summary[snr_key(snr)]["association_accuracy"],
                                    "pairs": train_summary[snr_key(snr)]["association_total"],
                                    "crowded_accuracy":
                                        train_summary[snr_key(snr)]["crowded_association_accuracy"],
                                    "crowded_pairs": train_summary[snr_key(snr)]["crowded_total"],
                                    "normal_accuracy":
                                        train_summary[snr_key(snr)]["normal_association_accuracy"],
                                    "normal_pairs": train_summary[snr_key(snr)]["normal_total"]}
                      for snr in SNRS},
        "temporal": {snr_key(snr): {"accuracy": train_summary[snr_key(snr)]["temporal_accuracy"],
                                    "targets": train_summary[snr_key(snr)]["temporal_total"],
                                    "id_switches": train_summary[snr_key(snr)]["id_switches"]}
                     for snr in SNRS},
        "no_smoothing": {"position_max_abs_diff": no_smoothing_position,
                         "velocity_max_abs_diff": no_smoothing_velocity,
                         "position_float32_exact": no_smoothing_exact,
                         "velocity_float32_exact": no_smoothing_exact,
                         "requirement": "final state equals the current-frame fusion result "
                                        "up to slot permutation; state_hat is stored float32 so "
                                        "the copy is exact in float32 (max float64 diff ~1e-5 m)"},
        "discipline": {"births": {snr_key(snr): train_summary[snr_key(snr)]["births"]
                                  for snr in SNRS},
                       "deaths": {snr_key(snr): train_summary[snr_key(snr)]["deaths"]
                                  for snr in SNRS},
                       "rank_deficient_velocity_frames":
                           {snr_key(snr): train_summary[snr_key(snr)]["rank_deficient_velocity_frames"]
                            for snr in SNRS},
                       "ill_conditioned_velocity_frames":
                           {snr_key(snr): train_summary[snr_key(snr)]["ill_conditioned_velocity_frames"]
                            for snr in SNRS},
                       "velocity_estimator": "MAP with weakly informative 30 m/s speed prior "
                                             "(regularizes near-collinear BS geometry; the prior is "
                                             "negligible for well-conditioned geometry)",
                       "crowded_definition": "window with >= 5 participating vehicles"},
        "gt_usage": "GT identity used only for offline accuracy audit, slot canonicalization and "
                    "future labels; never inside association"}
    write_json(REPORT / "association_metrics.json", association_metrics)

    summary = {
        "stage": "ISAC-REDESIGN-PROBE-01",
        "formal_f01e_modified": False, "v_confirm_opened": False, "test_opened": False,
        "pipeline": "NGSIM GT -> 40-frame continuous windows -> per-BS polar truth -> calibrated "
                    "noise -> anonymous shuffled BS lists -> cross-BS association -> same-frame GLS "
                    "fusion -> multi-BS radial 2D velocity -> temporal slot linking (no smoothing) "
                    "-> state_hat/vehicle_mask/future_position",
        "dataset_contract": {"state_hat": "[S,20,8,4]", "vehicle_mask": "[S,8]",
                             "future_position": "[S,20,8,2]",
                             "future_identical_across_snr": True,
                             "mask_identical_across_snr": True,
                             "slot_canonicalization": "offline GT majority-key audit; association "
                                                      "itself is measurement-only",
                             "detected_or_track_exists_required": False},
        "coverage": coverage,
        "calibration": {"raw_path": "reports/controlled_isac/redesign_probe_01/calibration_raw.json",
                        "final_path": "reports/controlled_isac/redesign_probe_01/calibration_final.json",
                        "sigma": calibration_final["sigma"],
                        "max_relative_adjustment": {metric: changes["max_relative_change"]
                                                    for metric, changes
                                                    in calibration_final["adjustment"].items()},
                        "raw_robust_scale": {metric: [calibration_raw["by_snr"][snr_key(snr)][metric]
                                                      ["robust_scale"] for snr in SNRS]
                                             for metric in measurement.SIGMA_KEYS}},
        "train_state_errors": train_summary,
        "association_metrics": association_metrics,
        "position_rmse_by_snr": {snr_key(snr): rmse("position", snr) for snr in SNRS},
        "position_rmse_ratio_minus5_over_20": position_ratio,
        "velocity_rmse_by_snr": {snr_key(snr): rmse("velocity", snr) for snr in SNRS},
        "velocity_rmse_ratio_minus5_over_20": velocity_ratio,
        "direction_correct": {"position": [position_direction, position_pairs],
                              "velocity": [velocity_direction, velocity_pairs]},
        "downstream": {"cohort_size": downstream["cohort_size"], "rows": downstream["rows"],
                       "cv_ade_degradation_minus5_vs_20_percent": cv_degradation,
                       "verdict": downstream_verdict},
        "acceptance": acceptance,
        "overall": "PASS" if overall else "FAIL",
        "next_action": "REVIEW_REQUIRED"}
    diagnostic_path = REPORT / "recoverability_diagnostic.json"
    if diagnostic_path.exists():
        summary["recoverability_diagnostic"] = json.loads(diagnostic_path.read_text())
    write_json(REPORT / "summary.json", summary)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figures = REPORT / "figures"
        figures.mkdir(parents=True, exist_ok=True)
        axis = [int(snr) for snr in SNRS]

        def save(figure, name):
            figure.tight_layout()
            figure.savefig(figures / name, dpi=150)
            plt.close(figure)

        figure, ax = plt.subplots(figsize=(5.5, 3.4))
        ax.plot(axis, position_rmse, "o-", label="RMSE")
        ax.plot(axis, [train_summary[snr_key(snr)]["position"]["p90_abs"] for snr in SNRS], "s--",
                label="P90")
        ax.set(xlabel="snr_ref (dB)", ylabel="2D position error (m)",
               title="position error vs SNR")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        save(figure, "position_error_vs_snr.png")

        figure, ax = plt.subplots(figsize=(5.5, 3.4))
        ax.plot(axis, velocity_rmse, "o-", label="RMSE")
        ax.plot(axis, [train_summary[snr_key(snr)]["velocity"]["p90_abs"] for snr in SNRS], "s--",
                label="P90")
        ax.set(xlabel="snr_ref (dB)", ylabel="2D velocity error (m/s)",
               title="velocity error vs SNR")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        save(figure, "velocity_error_vs_snr.png")

        figure, ax = plt.subplots(figsize=(5.5, 3.4))
        ax.plot(axis, [train_summary[snr_key(snr)]["association_accuracy"] for snr in SNRS], "o-",
                label="cross-BS")
        ax.plot(axis, [train_summary[snr_key(snr)]["temporal_accuracy"] for snr in SNRS], "s--",
                label="temporal")
        ax.set(xlabel="snr_ref (dB)", ylabel="accuracy", ylim=(0.9, 1.005),
               title="association accuracy vs SNR")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        save(figure, "association_accuracy_vs_snr.png")

        figure, ax = plt.subplots(figsize=(5.5, 3.4))
        for model in ("CV", "GNN", "QGNN"):
            rows = [row for row in downstream["rows"] if row["model"] == model]
            ax.plot(axis, [row["ADE"] for row in rows], "o-", label=model)
        ax.set(xlabel="snr_ref (dB)", ylabel="ADE (m)", title="downstream ADE vs SNR (V_select)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        save(figure, "downstream_ade_vs_snr.png")
    except Exception as error:  # noqa: BLE001
        write_json(REPORT / "figure_error.json", {"error": repr(error)})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="all",
                        choices=["all", "coverage", "calibration", "dataset", "downstream", "report"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--calibration-scenes", type=int, default=64)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    REPORT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    setup = measurement.load_setup()
    from frontend.echo_source import SourceEpisodes
    source = SourceEpisodes()

    need_coverage = args.phase in ("all", "coverage") or not (CACHE / "windows_train.pkl").exists()
    need_calibration = args.phase in ("all", "calibration")
    need_dataset = args.phase in ("all", "dataset")
    need_downstream = args.phase in ("all", "downstream")
    need_report = args.phase in ("all", "report")

    if need_coverage:
        coverage, windows = phase_coverage(source, setup)
        if coverage["stopped"]:
            write_json(REPORT / "summary.json", {
                "stage": "ISAC-REDESIGN-PROBE-01", "coverage": coverage,
                "stopped": True, "next_action": "STOP_LOW_COVERAGE_RETENTION"})
            print(json.dumps({"stopped": True, "retention_ratio": coverage["retention_ratio"]}))
            return
    else:
        coverage = json.loads((CACHE / "coverage.json").read_text())
    if need_calibration and args.phase != "coverage":
        phase_calibration(source, setup, load_windows(), args.calibration_scenes, args.device)
    if need_dataset and args.phase not in ("coverage", "calibration"):
        calibration_final = json.loads((CACHE / "calibration_final.json").read_text())
        phase_dataset(source, setup, load_windows(), calibration_final)
    if need_downstream and args.phase not in ("coverage", "calibration", "dataset"):
        phase_downstream(args.device)
    if need_report or args.phase == "report":
        calibration_raw = json.loads((CACHE / "calibration_raw.json").read_text())
        calibration_final = json.loads((CACHE / "calibration_final.json").read_text())
        train_summary = json.loads((CACHE / "train_metrics.json").read_text())
        downstream = json.loads((CACHE / "downstream.json").read_text())
        summary = phase_report(coverage, calibration_raw, calibration_final, train_summary, downstream)
        print(json.dumps({"overall": summary["overall"], "acceptance": summary["acceptance"],
                          "position_rmse_ratio": summary["position_rmse_ratio_minus5_over_20"],
                          "velocity_rmse_ratio": summary["velocity_rmse_ratio_minus5_over_20"],
                          "downstream_verdict": summary["downstream"]["verdict"]}, ensure_ascii=False))
    elif args.phase == "dataset":
        print(json.dumps({"phase": "dataset", "status": "complete"}))


if __name__ == "__main__":
    main()