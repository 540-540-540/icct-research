"""SENS-SNR-REBUILD-06 Stage C: formal B64 CFAR multiplier calibration (noise-only, no GT).

Stage 1 sweeps candidate multipliers with the uncapped CFAR local-maxima count over 512
noise-only frames x 3 BS (calibration seeds). Stage 2 measures the full detector false-alarm
rate on 256 independent verification frames x 3 BS inside a local bracket around the proxy
crossing and freezes the multiplier closest to the 0.5/BS/frame target. The chosen value is
written into configs/shared_frontend.json.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import _common  # noqa: E402

OUT = ROOT / "reports/f01e/snr_rebuild_06/cfar_calibration.json"
MULTIPLIERS = [0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.19, 0.20, 0.21, 0.22, 0.24,
               0.28, 0.33, 0.40, 0.50, 0.70, 1.00, 1.50, 2.00]
CALIBRATION_FRAMES = 512
CALIBRATION_EPISODE = 7300
VERIFICATION_FRAMES = 256
VERIFICATION_EPISODE = 7400
FINAL_FRAMES = 256
FINAL_EPISODE = 7500
BRACKET_SCALE = [0.85, 0.93, 1.00, 1.08, 1.17]
TARGET = 0.5


def crossing(points: list[tuple[float, float]]) -> float | None:
    ordered = sorted(points)
    if ordered[0][1] < TARGET:
        return ordered[0][0]
    for (left_m, left_fa), (right_m, right_fa) in zip(ordered, ordered[1:]):
        if left_fa >= TARGET >= right_fa and left_fa > right_fa:
            fraction = (math.log(max(left_fa, 1e-3)) - math.log(TARGET)) / (
                math.log(max(left_fa, 1e-3)) - math.log(max(right_fa, 1e-3)))
            return float(left_m + fraction * (right_m - left_m))
    return None


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CFAR calibration requires CUDA on the server")
    from frontend.sensing import detector
    from frontend.sensing.waveform import SensingResource

    started = time.time()
    device = "cuda:0"
    config = _common.load_frontend_config()
    resource = SensingResource.from_config(config)
    if resource.mode != "contiguous_burst":
        raise SystemExit("production sensing_resource must be contiguous_burst for this calibration")
    geometry = _common.load_geometry_config()
    waveform, array = _common.build_objects(config)
    stations, boresights = geometry["stations"], geometry["boresights"]
    height = float(geometry["height_difference_m"])
    detector_config = config["detector"]
    radius = (int(detector_config["nms_radius"][0]), int(detector_config["nms_radius"][1]))
    nr, nv = 2 * waveform.K, 2 * resource.active_symbols

    def empty_echo(episode: int, frame: int):
        return _common.synthesize([], [], [], stations, boresights, waveform, array, config, 20.0,
                                  episode, frame, device, noise=True, height_m=height)

    coarse = {multiplier: 0 for multiplier in MULTIPLIERS}
    for frame in range(CALIBRATION_FRAMES):
        echo = empty_echo(CALIBRATION_EPISODE, frame)
        for bs in range(3):
            maps = detector.compute_maps(echo["Y"][bs], echo["X"][bs], waveform, array, detector_config,
                                         resource=resource)
            for multiplier in MULTIPLIERS:
                mask = detector.cfar_mask(maps, multiplier)
                coarse[multiplier] += int(detector.local_maxima(mask, maps["P_RD"], radius).sum())
        if (frame + 1) % 128 == 0:
            print(f"  coarse {frame + 1}/{CALIBRATION_FRAMES} ({time.time() - started:.0f}s)", flush=True)
    calibration_fa = {str(m): coarse[m] / (CALIBRATION_FRAMES * 3) for m in MULTIPLIERS}
    proxy = crossing([(m, calibration_fa[str(m)]) for m in MULTIPLIERS]) or 0.19
    brackets = sorted({round(proxy * scale, 4) for scale in BRACKET_SCALE})
    print(f"proxy multiplier {proxy:.4f}; bracket {brackets}", flush=True)

    verification = {multiplier: 0 for multiplier in brackets}
    for frame in range(VERIFICATION_FRAMES):
        echo = empty_echo(VERIFICATION_EPISODE, frame)
        for bs in range(3):
            maps = detector.compute_maps(echo["Y"][bs], echo["X"][bs], waveform, array, detector_config,
                                         resource=resource)
            for multiplier in brackets:
                detections, counters, _ = detector.detect_from_maps(
                    maps, bs, 0, stations[bs], float(boresights[bs]), config, array, multiplier=multiplier)
                if counters["candidates_before_cap"] < len(detections):
                    raise AssertionError("inconsistent detector counters")
                verification[multiplier] += len(detections)
        if (frame + 1) % 64 == 0:
            print(f"  verify {frame + 1}/{VERIFICATION_FRAMES} ({time.time() - started:.0f}s)", flush=True)
    verification_fa = {str(m): verification[m] / (VERIFICATION_FRAMES * 3) for m in brackets}
    refined = crossing([(m, verification_fa[str(m)]) for m in brackets])
    if refined is not None and not any(abs(refined - m) < 1e-9 for m in brackets):
        extra = 0
        for frame in range(VERIFICATION_FRAMES):
            echo = empty_echo(VERIFICATION_EPISODE, frame)
            for bs in range(3):
                maps = detector.compute_maps(echo["Y"][bs], echo["X"][bs], waveform, array, detector_config,
                                             resource=resource)
                detections, _, _ = detector.detect_from_maps(
                    maps, bs, 0, stations[bs], float(boresights[bs]), config, array, multiplier=refined)
                extra += len(detections)
        verification[refined] = extra
        brackets = sorted(brackets + [refined])
        verification_fa = {str(m): verification[m] / (VERIFICATION_FRAMES * 3) for m in brackets}
    chosen = min(brackets, key=lambda m: abs(verification_fa[str(m)] - TARGET))
    chosen_fa = verification_fa[str(chosen)]

    final_counts = 0
    for frame in range(FINAL_FRAMES):
        echo = empty_echo(FINAL_EPISODE, frame)
        for bs in range(3):
            maps = detector.compute_maps(echo["Y"][bs], echo["X"][bs], waveform, array, detector_config,
                                         resource=resource)
            detections, _, _ = detector.detect_from_maps(
                maps, bs, 0, stations[bs], float(boresights[bs]), config, array, multiplier=chosen)
            final_counts += len(detections)
    final_fa = final_counts / (FINAL_FRAMES * 3)

    updated = json.loads((ROOT / "configs/shared_frontend.json").read_text())
    updated["detector"]["cfar_threshold_multiplier"] = float(chosen)
    (ROOT / "configs/shared_frontend.json").write_text(
        json.dumps(updated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report = {
        "stage": "SENS-SNR-REBUILD-06 Stage C",
        "method": "noise-only frames through the production contiguous_burst RD path; 512-frame coarse "
                  "local-maxima sweep then 256-frame full-detector verification at a local bracket",
        "target_false_alarms_per_bs_frame": TARGET,
        "resource_mode": resource.mode, "active_symbols": resource.active_symbols,
        "start_symbol": resource.start_symbol, "total_symbols": resource.total_symbols,
        "rd_shape": [nr, nv],
        "calibration_frames_per_bs": CALIBRATION_FRAMES, "calibration_episode": CALIBRATION_EPISODE,
        "calibration_seed_range": [CALIBRATION_EPISODE, CALIBRATION_EPISODE, 0, CALIBRATION_FRAMES - 1],
        "candidate_multipliers": MULTIPLIERS, "calibration_false_alarms_per_bs_frame": calibration_fa,
        "proxy_crossing_multiplier": proxy,
        "verification_frames_per_bs": VERIFICATION_FRAMES, "verification_episode": VERIFICATION_EPISODE,
        "verification_seed_range": [VERIFICATION_EPISODE, VERIFICATION_EPISODE, 0, VERIFICATION_FRAMES - 1],
        "verification_bracket": brackets,
        "verification_false_alarms_per_bs_frame": verification_fa,
        "chosen_multiplier": float(chosen),
        "verification_false_alarms_per_bs_frame_at_chosen": chosen_fa,
        "final_verification_frames_per_bs": FINAL_FRAMES, "final_verification_episode": FINAL_EPISODE,
        "final_verification_false_alarms_per_bs_frame": final_fa,
        "within_acceptance_band_0p35_0p65": 0.35 <= final_fa <= 0.65,
        "noise_only": True, "gt_used": False,
        "config_hash_before": hashlib.sha256(
            (ROOT / "configs/shared_frontend.json").read_bytes()).hexdigest(),
        "detector_hash": hashlib.sha256((ROOT / "frontend/sensing/detector.py").read_bytes()).hexdigest(),
        "waveform_hash": hashlib.sha256((ROOT / "frontend/sensing/waveform.py").read_bytes()).hexdigest(),
        "runtime_s": time.time() - started,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    _common.write_json(OUT, report)
    print(json.dumps({"chosen_multiplier": chosen, "verification_fa": chosen_fa,
                      "verification_fa_all": verification_fa,
                      "acceptance": report["within_acceptance_band_0p35_0p65"],
                      "runtime_s": report["runtime_s"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()