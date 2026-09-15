"""P4 tracker unit tests: lifecycle, coast, slots, T7 dropout and synthetic accuracy."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import _common  # noqa: E402

OUT = ROOT / "reports/f01e/p4_tracker"
DT_NS = 100_000_000
SIGMA = 0.05


def observation(x: float, y: float, frame: int, n_bs: int = 2, sigma: float = SIGMA) -> dict:
    return {"time_ns": frame * DT_NS, "x_m": float(x), "y_m": float(y),
            "C_xy": [[sigma ** 2, 0.0], [0.0, sigma ** 2]], "bs_mask": 0b011 if n_bs == 2 else 0b111,
            "n_bs": n_bs, "quality_db": 25.0}


def cv_point(frame: int, start=np.array([0.0, 0.0]), velocity=np.array([10.0, 2.0])) -> np.ndarray:
    return start + velocity * frame * 0.1


def invariants(records: list[dict]) -> bool:
    return all(not (record["detected"] and not record["track_exists"]) for record in records)


def main() -> None:
    from frontend.tracking.cv_kf import CvKalmanTracker

    config = _common.load_frontend_config()
    if config["tracker"].get("q_a_m2_s3") is None:
        raise SystemExit("tracker q_a not calibrated; run calibrate_tracker first")
    checks, details = {}, {}

    # 1. lifecycle + T7 dropout (frames 10-13 missing), mainline confirmation rule
    tracker = CvKalmanTracker(copy.deepcopy(config))
    frame_records = {}
    for frame in range(20):
        observations = [] if 10 <= frame <= 13 else [observation(*cv_point(frame), frame, n_bs=2)]
        records = tracker.step(observations, frame * DT_NS)
        frame_records[frame] = records
        assert invariants(records), "detected without track_exists"
    first_output_frame = min(frame for frame, records in frame_records.items() if records)
    coast_ok = all(any(record["slot"] == 0 for record in frame_records[frame]) for frame in range(10, 14))
    coast_semantics = all(record["track_exists"] and not record["detected"]
                          for frame in range(10, 14) for record in frame_records[frame])
    resume_ok = any(record["detected"] for record in frame_records[14])
    coast_error = max(abs(frame_records[frame][0]["state_hat"][0] - cv_point(frame)[0])
                      for frame in range(10, 14) if frame_records[frame])
    checks["lifecycle_and_dropout"] = {"passed": first_output_frame == 1 and coast_ok and coast_semantics
                                       and resume_ok and coast_error < 0.5,
                                       "detail": {"first_output_frame": first_output_frame,
                                                  "coast_max_x_error_m": coast_error,
                                                  "coast_window": [10, 13], "dropout": [10, 13]}}
    details["dropout_frames"] = [10, 13]

    # 2. long gap -> delete -> cooldown -> new key; slot reuse after cooldown
    cycle_before_gap = tracker.cycle
    for frame in range(20, 27):
        tracker.step([], frame * DT_NS)
    after_delete = tracker.step([], 27 * DT_NS)
    alive_keys_after_gap = {record["track_key"] for record in after_delete}
    deletion_cycle = cycle_before_gap + 6
    expected_free_at = deletion_cycle + config["tracker"]["slot_cooldown_cycles"]
    freed_slot_free_at = tracker.slot_free_at[0]
    new_observations = [observation(*cv_point(frame, start=np.array([50.0, 50.0])), frame, n_bs=2)
                        for frame in range(28, 33)]
    new_keys = []
    for offset, obs in enumerate(new_observations):
        new_keys.extend(record["track_key"] for record in tracker.step([obs], obs["time_ns"]))
    new_key_set = set(new_keys)
    checks["delete_cooldown_new_identity"] = {
        "passed": not alive_keys_after_gap and new_key_set and not (new_key_set & {0})
        and freed_slot_free_at == expected_free_at,
        "detail": {"alive_after_delete": sorted(alive_keys_after_gap), "new_keys": sorted(new_key_set),
                   "freed_slot_free_at": freed_slot_free_at, "expected_free_at": expected_free_at,
                   "deletion_cycle": deletion_cycle,
                   "cooldown": config["tracker"]["slot_cooldown_cycles"]}}

    # 3. single-BS observations never confirm a new track
    solo = CvKalmanTracker(copy.deepcopy(config))
    solo_records = [solo.step([observation(*cv_point(frame), frame, n_bs=1)], frame * DT_NS) for frame in range(6)]
    checks["single_bs_no_confirmation"] = {"passed": all(not records for records in solo_records),
                                           "detail": {"records": [len(records) for records in solo_records]}}

    # 4. fixed 8-slot policy with more than 8 confirmed tracks
    many = CvKalmanTracker(copy.deepcopy(config))
    positions = [np.array([10.0 * index, 0.0]) for index in range(10)]
    for frame in range(4):
        observations = [observation(*(start + np.array([2.0, 0.0]) * frame * 0.1), frame, n_bs=2)
                        for start in positions]
        records = many.step(observations, frame * DT_NS)
    confirmed = [key for key, track in many.tracks.items() if track["confirmed"]]
    slotless_keys = sorted(key for key in confirmed if many.tracks[key]["slot"] is None)
    victim = next(key for key in confirmed if many.tracks[key]["slot"] == 0)
    unique_slots = sorted(record["slot"] for record in records)
    over_eight_ok = (len(confirmed) > 8 and len(records) == 8 and unique_slots == list(range(8))
                     and len(slotless_keys) == len(confirmed) - 8)
    for frame in range(4, 11):
        observations = [observation(*(start + np.array([2.0, 0.0]) * frame * 0.1), frame, n_bs=2)
                        for index, start in enumerate(positions) if index != victim]
        records = many.step(observations, frame * DT_NS)
    victim_alive = victim in many.tracks
    for frame in range(11, 23):
        observations = [observation(*(start + np.array([2.0, 0.0]) * frame * 0.1), frame, n_bs=2)
                        for index, start in enumerate(positions) if index != victim]
        records = many.step(observations, frame * DT_NS)
    freed_slot_holder = next((record for record in records if record["slot"] == 0), None)
    reuse_ok = (not victim_alive and freed_slot_holder is not None
                and freed_slot_holder["track_key"] in slotless_keys and len(records) == 8)
    checks["eight_slot_policy"] = {"passed": over_eight_ok and reuse_ok and invariants(records),
                                   "detail": {"confirmed": len(confirmed), "initial_slotless_keys": slotless_keys,
                                              "victim_key": victim, "victim_deleted": not victim_alive,
                                              "freed_slot_holder": freed_slot_holder}}

    # 5. synthetic accuracy: CV and mild acceleration
    from calibrate_tracker import (make_observations, representative_covariance, trajectory,
                                     truth_velocities)

    covariance = representative_covariance(config)
    accuracy = {}
    for kind in ("cv", "ca"):
        local = copy.deepcopy(config)
        local["tracker"]["confirm_hits"] = 1
        local["tracker"]["confirm_requires_nbs2"] = False
        accuracy_tracker = CvKalmanTracker(local)
        truth_trajectory = trajectory(kind)
        truth_velocity = truth_velocities(truth_trajectory)
        observations = make_observations(truth_trajectory, covariance, 2026 if kind == "cv" else 2027)
        position_errors, velocity_errors, keys = [], [], set()
        for frame, obs in enumerate(observations):
            records = accuracy_tracker.step([obs], obs["time_ns"])
            if not records:
                continue
            record = records[0]
            keys.add(record["track_key"])
            if frame >= 10:
                truth = truth_trajectory[frame]
                position_errors.append(math.hypot(record["state_hat"][0] - truth[0],
                                                  record["state_hat"][1] - truth[1]))
                velocity_errors.append(np.array(record["state_hat"][2:]) - np.array(truth_velocity[frame]))
        accuracy[kind] = {
            "position_rmse_m": float(np.sqrt(np.mean(np.square(position_errors)))),
            "velocity_rmse_mps": float(np.sqrt(np.mean(np.square(np.asarray(velocity_errors))))),
            "track_keys": sorted(keys),
            "finite": all(math.isfinite(value) for value in position_errors),
        }
    accuracy_ok = (accuracy["cv"]["position_rmse_m"] < 0.5 and accuracy["cv"]["velocity_rmse_mps"] < 2.0
                   and accuracy["ca"]["position_rmse_m"] < 0.5 and accuracy["ca"]["velocity_rmse_mps"] < 2.0
                   and accuracy["ca"]["finite"]
                   and len(accuracy["cv"]["track_keys"]) == 1 and len(accuracy["ca"]["track_keys"]) == 1)
    checks["synthetic_accuracy"] = {"passed": bool(accuracy_ok), "detail": accuracy,
                                    "bounds": {"position_rmse_m": 0.5, "velocity_rmse_mps": 2.0,
                                               "note": "position bound below range resolution 1.61 m; velocity bound "
                                                       "one Doppler resolution cell 1.97 m/s; q_a comes from "
                                                       "calibrate_tracker.py and was not retuned for this test"}}

    passed = all(entry["passed"] for entry in checks.values())
    summary = {"test": "P4 tracker", "passed": passed, "q_a_m2_s3": config["tracker"]["q_a_m2_s3"],
               "checks": checks, "dropout_frames": details["dropout_frames"],
               "source_hashes": {"frontend/tracking/cv_kf.py":
                                 hashlib.sha256((ROOT / "frontend/tracking/cv_kf.py").read_bytes()).hexdigest()}}
    OUT.mkdir(parents=True, exist_ok=True)
    _common.write_json(OUT / "summary.json", summary)
    _common.write_json(OUT / "checks.json", {"test": "P4 tracker", "passed": passed, "checks": checks})
    print(json.dumps({"P4": "PASS" if passed else "FAIL",
                      "checks": {key: entry["passed"] for key, entry in checks.items()}},
                     ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()