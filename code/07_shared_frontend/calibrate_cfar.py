"""CFAR noise-only threshold multiplier calibration (P2, DECISIONS.md D08)."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import _common  # noqa: E402

OUT = ROOT / "reports/f01e/cfar_calibration.json"
MULTIPLIERS = [0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.17, 0.18, 0.185, 0.19, 0.195, 0.20,
               0.21, 0.22, 0.24, 0.28, 0.33, 0.40, 0.50, 0.70, 1.00, 1.50, 2.00]
COARSE_FRAMES = 512
VERIFY_FRAMES = 128
FINAL_VERIFY_FRAMES = 256
FINAL_VERIFY_EPISODE = 9102


def run_final_verify(target: float) -> dict:
    """Verify the config multiplier on fresh noise-only frames under the final code/config."""
    from frontend.sensing import detector as detector_module

    device = "cuda:0"
    config = _common.load_frontend_config()
    waveform, array = _common.build_objects(config)
    geometry = _common.load_geometry_config()
    stations, boresights = geometry["stations"], geometry["boresights"]
    multiplier = config["detector"].get("cfar_threshold_multiplier")
    if multiplier is None:
        raise SystemExit("config cfar_threshold_multiplier is null; freeze the calibrated value first")
    per_bs = [0, 0, 0]
    total = 0
    for frame in range(FINAL_VERIFY_FRAMES):
        echo = _common.synthesize([], [], [], stations, boresights, waveform, array, config, 20.0,
                                  FINAL_VERIFY_EPISODE, frame, device, noise=True)
        for bs in range(3):
            maps = detector_module.compute_maps(echo["Y"][bs], echo["X"][bs], waveform, array, config["detector"])
            detections, counters, _ = detector_module.detect_from_maps(
                maps, bs, 0, stations[bs], float(boresights[bs]), config, array, multiplier=float(multiplier))
            if counters["candidates_before_cap"] < len(detections):
                raise AssertionError("inconsistent detector counters")
            per_bs[bs] += len(detections)
            total += len(detections)
    rate = total / (FINAL_VERIFY_FRAMES * 3)
    return {
        "multiplier": float(multiplier),
        "frames_per_bs": FINAL_VERIFY_FRAMES,
        "episode": FINAL_VERIFY_EPISODE,
        "false_alarms_per_bs_frame": rate,
        "per_bs_counts": per_bs,
        "target_false_alarms_per_bs_frame": target,
        "within_tolerance": abs(rate - target) <= 0.25,
        "config_hash": hashlib.sha256((ROOT / "configs/shared_frontend.json").read_bytes()).hexdigest(),
        "detector_hash": hashlib.sha256((ROOT / "frontend/sensing/detector.py").read_bytes()).hexdigest(),
    }


def main(verify_only: bool = False) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CFAR calibration requires CUDA per frozen design")
    from frontend.sensing import detector as detector_module

    device = "cuda:0"
    config = _common.load_frontend_config()
    waveform, array = _common.build_objects(config)
    geometry = _common.load_geometry_config()
    stations, boresights = geometry["stations"], geometry["boresights"]
    detector_config = config["detector"]
    target = float(detector_config["target_false_alarms_per_bs_frame"])
    if verify_only:
        final_verify = run_final_verify(target)
        report = _common.read_json(OUT) if OUT.exists() else {}
        report["final_verify"] = final_verify
        report["config_multiplier"] = final_verify["multiplier"]
        report["note"] = "multiplier 0.19 approved and written to configs/shared_frontend.json; SNR levels remain null"
        _common.write_json(OUT, report)
        print(json.dumps({"final_verify": final_verify}, ensure_ascii=False))
        return
    radius = (int(detector_config["nms_radius"][0]), int(detector_config["nms_radius"][1]))
    nr, nv = 2 * waveform.K, 2 * waveform.N

    coarse = {multiplier: 0 for multiplier in MULTIPLIERS}
    verify = {multiplier: 0 for multiplier in MULTIPLIERS}
    per_bs_verify = {multiplier: [0, 0, 0] for multiplier in MULTIPLIERS}
    alpha_median = []

    def empty_echo(episode: int, frame: int):
        return _common.synthesize([], [], [], stations, boresights, waveform, array, config, 20.0,
                                  episode, frame, device, noise=True)

    for frame in range(COARSE_FRAMES):
        echo = empty_echo(9100, frame)
        for bs in range(3):
            maps = detector_module.compute_maps(echo["Y"][bs], echo["X"][bs], waveform, array, detector_config)
            if frame < 8:
                alpha_median.append(float(maps["alpha"].median()))
            for multiplier in MULTIPLIERS:
                mask = detector_module.cfar_mask(maps, multiplier)
                maxima = detector_module.local_maxima(mask, maps["P_RD"], radius)
                coarse[multiplier] += int(maxima.sum())

    for frame in range(VERIFY_FRAMES):
        echo = empty_echo(9101, frame)
        for bs in range(3):
            maps = detector_module.compute_maps(echo["Y"][bs], echo["X"][bs], waveform, array, detector_config)
            for multiplier in MULTIPLIERS:
                detections, _, _ = detector_module.detect_from_maps(
                    maps, bs, 0, stations[bs], float(boresights[bs]), config, array, multiplier=multiplier)
                verify[multiplier] += len(detections)
                per_bs_verify[multiplier][bs] += len(detections)

    coarse_rate = {str(m): coarse[m] / (COARSE_FRAMES * 3) for m in MULTIPLIERS}
    verify_rate = {str(m): verify[m] / (VERIFY_FRAMES * 3) for m in MULTIPLIERS}
    chosen = min(MULTIPLIERS, key=lambda m: (abs(verify_rate[str(m)] - target), m))

    report = {
        "method": "noise-only CPI; CFAR threshold multiplier sweep; no GT used",
        "target_false_alarms_per_bs_frame": target,
        "n_cells": nr * nv,
        "nr": nr, "nv": nv,
        "coarse_frames_per_bs": COARSE_FRAMES,
        "coarse_estimator": "local maxima count (no NMS/geometry) to bound cost",
        "coarse_false_alarms_per_bs_frame": coarse_rate,
        "verify_frames_per_bs": VERIFY_FRAMES,
        "verify_estimator": "full detector path (2D CFAR + NMS + geometry + cap)",
        "verify_false_alarms_per_bs_frame": verify_rate,
        "verify_per_bs_counts": {str(m): per_bs_verify[m] for m in MULTIPLIERS},
        "chosen_multiplier": chosen,
        "chosen_measured_false_alarms_per_bs_frame": verify_rate[str(chosen)],
        "chosen_coarse_false_alarms_per_bs_frame": coarse_rate[str(chosen)],
        "nominal_alpha_median": float(torch.tensor(alpha_median).median()) if alpha_median else None,
        "config_hash": hashlib.sha256((ROOT / "configs/shared_frontend.json").read_bytes()).hexdigest(),
        "detector_hash": hashlib.sha256((ROOT / "frontend/sensing/detector.py").read_bytes()).hexdigest(),
        "note": "multiplier 0.19 approved and written to configs/shared_frontend.json; SNR levels remain null",
        "within_tolerance": abs(verify_rate[str(chosen)] - target) <= 0.25,
    }
    _common.write_json(OUT, report)
    print(json.dumps({"chosen_multiplier": chosen,
                      "measured_false_alarms_per_bs_frame": verify_rate[str(chosen)],
                      "target": target, "within_tolerance": report["within_tolerance"]}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    arguments = parser.parse_args()
    main(verify_only=arguments.verify_only)