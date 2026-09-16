"""SENS-SNR-REBUILD-06 Stage E: tracker q_a calibration on the real B64 chain.

Runs B64 production detection -> association/fusion -> CV-KF over multi-frame synthetic and A01
train streams at two SNR points, one-to-one matching, and selects q_a by the predeclared rule
(minimum mean position RMSE + mean velocity RMSE, ties broken by higher continuity). No downstream
metric is used. The chosen value is written into configs/shared_frontend.json.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import _common  # noqa: E402
import audit_snr_response as audit04  # noqa: E402

OUT = ROOT / "reports/f01e/snr_rebuild_06/tracker_calibration.json"
CANDIDATES = [0.25, 0.5, 1.0, 2.0, 4.0]
SNR_POINTS = [0.0, 10.0]
DT_NS = 100_000_000
GATE_M = 5.0


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("tracker calibration requires CUDA on the server")
    from frontend.echo_source import SourceEpisodes
    from frontend.fusion.association import fuse_frame
    from frontend.sensing import detector
    from frontend.tracking.cv_kf import CvKalmanTracker

    started = time.time()
    device = "cuda:0"
    config = _common.load_frontend_config()
    geometry = _common.load_geometry_config()
    waveform, array = _common.build_objects(config)
    stations, boresights = geometry["stations"], geometry["boresights"]
    height = float(geometry["height_difference_m"])
    multiplier = float(config["detector"]["cfar_threshold_multiplier"])
    lut = _common.load_production_lut(config)
    if lut is None:
        raise SystemExit("production covariance LUT missing; run Stage D first")
    resource = _common.load_resource(config)

    source = SourceEpisodes()
    train_snap, train_streams = audit04.train_scenes(source)
    streams = [audit04.synthetic_stream(3), audit04.synthetic_stream(5), audit04.synthetic_stream(8)]
    for entry in train_streams:
        positions, velocities, identities, keys = audit04.sample_episode_identity(source, entry["episode"],
                                                                                  4000, 12)
        entry.update({"positions": positions, "velocities": velocities, "identities": identities,
                      "keys": keys})
        streams.append(entry)

    def run_stream(stream, snr, q_a):
        local = copy.deepcopy(config)
        local["tracker"]["q_a_m2_s3"] = q_a
        tracker = CvKalmanTracker(local)
        position_errors, velocity_errors = [], []
        matched, coast, total = 0, 0, 0
        for frame in range(len(stream["positions"])):
            positions = [np.asarray(position, dtype=float) for position in stream["positions"][frame]]
            velocities = [np.asarray(velocity, dtype=float) for velocity in stream["velocities"][frame]]
            keys = stream.get("keys", [None] * 12)[frame] if stream["type"] == "train_stream" else \
                [20000 + index for index in range(len(positions))]
            echo = _common.synthesize(positions, velocities, keys, stations, boresights, waveform, array,
                                      local, snr, 9750 + len(stream.get("label", "")), frame, device,
                                      height_m=height)
            detections_by_bs = {}
            for bs in range(3):
                maps = detector.compute_maps(echo["Y"][bs], echo["X"][bs], waveform, array,
                                             local["detector"], resource=resource)
                detections, _, _ = detector.detect_from_maps(
                    maps, bs, frame, stations[bs], float(boresights[bs]), local, array, multiplier=multiplier,
                    covariance_lut=lut, height=height)
                detections_by_bs[bs] = detections
            observations, _ = fuse_frame(detections_by_bs, local)
            records = tracker.step(observations, frame * DT_NS)
            record_positions = [np.array([record["state_hat"][0], record["state_hat"][1]])
                                for record in records]
            frame_matched, _ = audit04.one_to_one_match(positions, record_positions, GATE_M)
            for gt_index, record_index, distance in frame_matched:
                matched += 1
                position_errors.append(distance)
                velocity_errors.append(np.asarray(records[record_index]["state_hat"][2:])
                                       - velocities[gt_index])
                if not records[record_index]["detected"]:
                    coast += 1
            total += len(positions)
        return {"position_rmse_m": float(np.sqrt(np.mean(np.square(position_errors))))
                if position_errors else None,
                "velocity_rmse_mps": float(np.sqrt(np.mean(np.square(np.asarray(velocity_errors)))))
                if velocity_errors else None,
                "continuity": matched / max(total, 1), "coast_ratio": coast / max(matched, 1),
                "targets": total}

    table = {}
    for candidate in CANDIDATES:
        entry = {}
        for snr in SNR_POINTS:
            rows = [run_stream(stream, snr, candidate) for stream in streams]
            entry[str(snr)] = {
                "mean_position_rmse_m": float(np.mean([row["position_rmse_m"] for row in rows
                                                       if row["position_rmse_m"] is not None])),
                "mean_velocity_rmse_mps": float(np.mean([row["velocity_rmse_mps"] for row in rows
                                                         if row["velocity_rmse_mps"] is not None])),
                "mean_continuity": float(np.mean([row["continuity"] for row in rows])),
                "mean_coast_ratio": float(np.mean([row["coast_ratio"] for row in rows])),
                "streams": len(rows)}
        table[str(candidate)] = entry
        print(f"  q_a={candidate}: " + json.dumps(entry, ensure_ascii=False), flush=True)

    def score(candidate: float) -> tuple:
        entry = table[str(candidate)]
        position = np.mean([entry[str(snr)]["mean_position_rmse_m"] for snr in SNR_POINTS])
        velocity = np.mean([entry[str(snr)]["mean_velocity_rmse_mps"] for snr in SNR_POINTS])
        continuity = np.mean([entry[str(snr)]["mean_continuity"] for snr in SNR_POINTS])
        return (position + velocity, -continuity)

    chosen = min(CANDIDATES, key=score)
    updated = json.loads((ROOT / "configs/shared_frontend.json").read_text())
    updated["tracker"]["q_a_m2_s3"] = float(chosen)
    (ROOT / "configs/shared_frontend.json").write_text(
        json.dumps(updated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {
        "stage": "SENS-SNR-REBUILD-06 Stage E",
        "method": "B64 production detection -> association/fusion -> CV-KF over 6 multi-frame streams "
                  "(3 synthetic + 3 A01 train, 12 frames each) at 0 and 10 dB; one-to-one GT<->track "
                  "matching with 5 m gate; predeclared selection: min(mean position RMSE + mean velocity "
                  "RMSE), ties by continuity",
        "candidates_m2_s3": CANDIDATES, "chosen_q_a_m2_s3": float(chosen),
        "selection_score_position_plus_velocity": {str(c): score(c)[0] for c in CANDIDATES},
        "snr_points": SNR_POINTS, "streams": [stream["label"] for stream in streams],
        "table": table, "cfar_multiplier": multiplier,
        "resource_mode": resource.mode, "active_symbols": resource.active_symbols,
        "config_hash": hashlib.sha256((ROOT / "configs/shared_frontend.json").read_bytes()).hexdigest(),
        "runtime_s": time.time() - started,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    _common.write_json(OUT, report)
    print(json.dumps({"chosen_q_a_m2_s3": chosen, "runtime_s": report["runtime_s"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()