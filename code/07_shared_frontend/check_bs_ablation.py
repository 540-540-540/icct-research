"""T6-A/T6-B 1/2/3-BS sanity ablations plus weak-measurement check (SENS-REBUILD-03B sections 15, 30, 31)."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import _common  # noqa: E402

T6A_DIR = ROOT / "reports/f01e/t6a_bs_ablation"
T6B_DIR = ROOT / "reports/f01e/t6b_bs_tracker"
FRAMES = 16
SCENES = [
    {"targets": [[-20.0, -5.0], [0.0, 5.0], [20.0, -2.0]], "velocity": [[8.0, 1.0], [9.0, 0.5], [7.0, -1.0]]},
    {"targets": [[-10.0, 8.0], [15.0, -6.0], [35.0, 3.0]], "velocity": [[10.0, -0.5], [8.5, 1.0], [9.5, 0.0]]},
    {"targets": [[-30.0, 0.0], [5.0, -8.0], [25.0, 8.0]], "velocity": [[7.5, 0.5], [10.5, 0.2], [8.0, -0.8]]},
]
SUBSETS = {"1bs": [(0,), (1,), (2,)], "2bs": [(0, 1), (0, 2), (1, 2)], "3bs": [(0, 1, 2)]}
CANONICAL = {"1bs": (0,), "2bs": (0, 1), "3bs": (0, 1, 2)}
SNR_POINTS = {"high": 25.0, "medium": 5.0, "low": -10.0}


def scene_states(scene: dict, frame: int):
    positions = np.asarray(scene["targets"], dtype=float) + np.asarray(scene["velocity"], dtype=float) * (0.1 * frame)
    velocities = np.asarray(scene["velocity"], dtype=float)
    return positions, velocities


def nearest_observation(observation, position, gate=5.0):
    best, best_distance = None, gate
    for entry in observation:
        distance = math.hypot(entry["x_m"] - position[0], entry["y_m"] - position[1])
        if distance < best_distance:
            best, best_distance = entry, distance
    return best, best_distance


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("T6 ablations require CUDA per frozen design")
    from frontend.fusion.association import fuse_frame
    from frontend.sensing import detector as detector_module
    from frontend.tracking.cv_kf import CvKalmanTracker

    device = "cuda:0"
    config = _common.load_frontend_config()
    geometry = _common.load_geometry_config()
    _common.assert_height_alignment(config, geometry)
    waveform, array = _common.build_objects(config)
    stations, boresights = geometry["stations"], geometry["boresights"]
    height = float(geometry["height_difference_m"])
    multiplier = float(config["detector"]["cfar_threshold_multiplier"])
    lut = _common.load_lut(ROOT)
    if lut is None:
        raise SystemExit("covariance LUT missing")

    runs = {}
    truth_by_key = {}
    for snr_name, snr_ref in SNR_POINTS.items():
        frames = []
        for scene_index, scene in enumerate(SCENES):
            for frame in range(FRAMES):
                positions, velocities = scene_states(scene, frame)
                keys = [1000 * (scene_index + 1) + index for index in range(len(positions))]
                echo = _common.synthesize(positions.tolist(), velocities.tolist(), keys, stations, boresights,
                                          waveform, array, config, snr_ref, 9500 + scene_index, frame, device,
                                          height_m=height)
                detections_by_bs = {}
                for bs in range(3):
                    maps = detector_module.compute_maps(echo["Y"][bs], echo["X"][bs], waveform, array,
                                                        config["detector"])
                    detections, _, _ = detector_module.detect_from_maps(
                        maps, bs, frame, stations[bs], float(boresights[bs]), config, array, multiplier=multiplier,
                        covariance_lut=lut, height=height)
                    detections_by_bs[bs] = detections
                frames.append({"scene": scene_index, "frame": frame, "truth": positions,
                               "detections": detections_by_bs,
                               "time_ns": frame * 100_000_000})
        runs[snr_name] = frames

    def observations_for(frame, bs_set):
        detections = {bs: (frame["detections"][bs] if bs in bs_set else []) for bs in (0, 1, 2)}
        observations, _ = fuse_frame(detections, config)
        return observations

    t6a = {}
    for group, subsets in SUBSETS.items():
        entries = []
        for subset in subsets:
            matched, false_alarms, errors, targets_seen = 0, 0, [], 0
            for frame in runs["high"]:
                observations = observations_for(frame, subset)
                used = set()
                for position in frame["truth"]:
                    targets_seen += 1
                    best, best_distance = None, 5.0
                    for index, observation in enumerate(observations):
                        if index in used:
                            continue
                        distance = math.hypot(observation["x_m"] - position[0], observation["y_m"] - position[1])
                        if distance < best_distance:
                            best, best_distance = index, distance
                    if best is not None:
                        matched += 1
                        used.add(best)
                        errors.append(best_distance)
                false_alarms += len(observations) - len(used)
            entries.append({"subset": list(subset), "recall": matched / targets_seen,
                            "false_alarms_per_frame": false_alarms / len(runs["high"]),
                            "position_rmse_m": float(np.sqrt(np.mean(np.square(errors)))) if errors else None})
        t6a[group] = {"runs": entries,
                      "mean_recall": float(np.mean([entry["recall"] for entry in entries])),
                      "mean_false_alarms_per_frame": float(np.mean([entry["false_alarms_per_frame"]
                                                                    for entry in entries])),
                      "mean_position_rmse_m": float(np.mean([entry["position_rmse_m"] for entry in entries]))}

    t6b, weak = {}, {}
    for group, subset in CANONICAL.items():
        local = copy.deepcopy(config)
        local["tracker"]["confirm_requires_nbs2"] = False
        matched_positions, continuity_hits, continuity_total, switches = [], 0, 0, 0
        coast_frames = 0
        for scene_index in range(len(SCENES)):
            tracker = CvKalmanTracker(local)
            previous_key = {}
            for frame in [entry for entry in runs["high"] if entry["scene"] == scene_index]:
                observations = observations_for(frame, subset)
                records = tracker.step(observations, frame["time_ns"])
                for target_index, position in enumerate(frame["truth"]):
                    best, best_distance = None, 5.0
                    for record in records:
                        distance = math.hypot(record["state_hat"][0] - position[0],
                                              record["state_hat"][1] - position[1])
                        if distance < best_distance:
                            best, best_distance = record, distance
                    if frame["frame"] >= 3:
                        continuity_total += 1
                    if best is not None:
                        matched_positions.append(best_distance)
                        if frame["frame"] >= 3:
                            continuity_hits += 1
                            key = best["track_key"]
                            if target_index in previous_key and previous_key[target_index] != key:
                                switches += 1
                            previous_key[target_index] = key
                            if not best["detected"]:
                                coast_frames += 1
        t6b[group] = {"canonical_subset": list(subset), "track_position_rmse_m":
                      float(np.sqrt(np.mean(np.square(matched_positions)))) if matched_positions else None,
                      "continuity_after_warmup": continuity_hits / max(continuity_total, 1),
                      "id_switches": switches, "coastal_records": coast_frames,
                      "confirm_requires_nbs2": False}

    for snr_name in ("medium", "low"):
        shots, coast, detected_frames = [], 0, 0
        for scene_index in range(len(SCENES)):
            tracker = CvKalmanTracker(copy.deepcopy(config))
            for frame in [entry for entry in runs[snr_name] if entry["scene"] == scene_index]:
                observations = observations_for(frame, CANONICAL["3bs"])
                shots.append(len(observations))
                records = tracker.step(observations, frame["time_ns"])
                detected_frames += sum(1 for record in records if record["detected"])
                coast += sum(1 for record in records if not record["detected"])
        detections_per_bs = {bs: float(np.mean([len(frame["detections"][bs]) for frame in runs[snr_name]]))
                             for bs in (0, 1, 2)}
        weak[snr_name] = {"snr_ref_db": SNR_POINTS[snr_name],
                          "mean_detections_per_bs_frame": detections_per_bs,
                          "mean_observations_per_frame": float(np.mean(shots)) if shots else 0.0,
                          "records_detected": detected_frames, "records_coasting": coast}

    t6a_summary = {"test": "T6-A 1/2/3-BS sensing/fusion sanity", "snr_ref_db": SNR_POINTS["high"],
                   "scenes": len(SCENES), "frames_per_scene": FRAMES, "results": t6a,
                   "note": "averages over all single/pair subsets for 1/2 BS; no formal SNR conclusion"}
    t6b_summary = {"test": "T6-B 1/2/3-BS tracker sanity (ablation mode)", "snr_ref_db": SNR_POINTS["high"],
                   "confirm_requires_nbs2": False,
                   "note": "ablation mode only; mainline tracker keeps confirm_requires_nbs2=true",
                   "results": t6b, "weak_measurement": weak,
                   "source_hashes": {"frontend/fusion/association.py":
                                     hashlib.sha256((ROOT / "frontend/fusion/association.py").read_bytes()).hexdigest(),
                                     "frontend/tracking/cv_kf.py":
                                     hashlib.sha256((ROOT / "frontend/tracking/cv_kf.py").read_bytes()).hexdigest()}}
    T6A_DIR.mkdir(parents=True, exist_ok=True)
    T6B_DIR.mkdir(parents=True, exist_ok=True)
    _common.write_json(T6A_DIR / "summary.json", t6a_summary)
    _common.write_json(T6B_DIR / "summary.json", t6b_summary)
    print(json.dumps({"T6A": t6a, "T6B": t6b, "weak": weak}, ensure_ascii=False))


if __name__ == "__main__":
    main()