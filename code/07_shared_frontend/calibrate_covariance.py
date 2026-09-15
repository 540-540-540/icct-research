"""Measurement covariance calibration: sigma_r(q), sigma_u(q) LUT (P2, DECISIONS.md D23)."""
from __future__ import annotations

import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import _common  # noqa: E402

OUT = ROOT / "reports/f01e/covariance_calibration.json"
CFAR_REPORT = ROOT / "reports/f01e/cfar_calibration.json"
RANGES_M = [30.0, 50.0, 80.0, 100.0, 150.0, 200.0, 250.0, 300.0]
BEARINGS_DEG = [-60.0, -30.0, 0.0, 30.0, 60.0]
SNR_LEVELS_DB = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
REALIZATIONS = 6
BIN_EDGES_DB = [6.0, 12.0, 20.0]
MIN_SAMPLES = 30
VELOCITY = [8.0, -4.0]


def robust_sigma(values: list[float]) -> float:
    if not values:
        return float("nan")
    median = statistics.median(values)
    return 1.4826 * statistics.median(abs(value - median) for value in values)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("covariance calibration requires CUDA per frozen design")
    from frontend.sensing import detector as detector_module

    device = "cuda:0"
    config = _common.load_frontend_config()
    waveform, array = _common.build_objects(config)
    geometry = _common.load_geometry_config()
    stations, boresights = geometry["stations"], geometry["boresights"]
    multiplier = 1.0
    if CFAR_REPORT.exists():
        multiplier = float(_common.read_json(CFAR_REPORT)["chosen_multiplier"])
    detector_config = config["detector"]

    rows = []
    combo_index = 0
    per_snr_detection = {level: [0, 0] for level in SNR_LEVELS_DB}
    for range_m in RANGES_M:
        for bearing_deg in BEARINGS_DEG:
            position, velocity = _common.place_target(stations[0], float(boresights[0]), range_m, bearing_deg,
                                                      VELOCITY, geometry["height_difference_m"])
            episode = 9200 + combo_index
            combo_index += 1
            for realization in range(REALIZATIONS):
                for snr in SNR_LEVELS_DB:
                    echo = _common.synthesize([position], [velocity], [1 + realization], stations, boresights,
                                              waveform, array, config, snr, episode, realization, device)
                    for bs in range(3):
                        _, r_gt, _, u_gt = _common.truth_polar(position, stations[bs], float(boresights[bs]),
                                                               geometry["height_difference_m"])
                        maps = detector_module.compute_maps(echo["Y"][bs], echo["X"][bs], waveform, array,
                                                            detector_config)
                        detections, _, _ = detector_module.detect_from_maps(
                            maps, bs, 0, stations[bs], float(boresights[bs]), config, array, multiplier=multiplier)
                        truth_r, truth_u = float(r_gt), float(u_gt)
                        match = _common.nearest_truth_detection(detections, truth_r, truth_u)
                        per_snr_detection[snr][1] += 1
                        if match is None:
                            continue
                        per_snr_detection[snr][0] += 1
                        rows.append({
                            "r_gt": truth_r, "u_gt": truth_u, "q_db": match["peak_to_noise_db"],
                            "e_r": match["r_m"] - truth_r, "e_u": match["u"] - truth_u,
                            "station_id": bs, "snr_ref_db": snr})

    binned = []
    for index in range(len(BIN_EDGES_DB) + 1):
        low = BIN_EDGES_DB[index - 1] if index > 0 else None
        high = BIN_EDGES_DB[index] if index < len(BIN_EDGES_DB) else None
        selected = [row for row in rows if (low is None or row["q_db"] > low)
                    and (high is None or row["q_db"] <= high)]
        binned.append({
            "q_min_db": low, "q_max_db": high, "samples": len(selected),
            "raw_sigma_r_m": robust_sigma([row["e_r"] for row in selected]),
            "raw_sigma_u": robust_sigma([row["e_u"] for row in selected]),
            "median_abs_e_r_m": statistics.median([abs(row["e_r"]) for row in selected]) if selected else None,
            "median_abs_e_u": statistics.median([abs(row["e_u"]) for row in selected]) if selected else None,
        })

    pooled_r = robust_sigma([row["e_r"] for row in rows]) if rows else float("nan")
    pooled_u = robust_sigma([row["e_u"] for row in rows]) if rows else float("nan")

    def conservative(own, higher, pooled, samples):
        value = own if math.isfinite(own) else (higher if math.isfinite(higher) else pooled)
        if math.isfinite(higher):
            value = max(value, higher)
        if samples < MIN_SAMPLES and math.isfinite(pooled):
            value = max(value, pooled)
        return value

    for index in range(len(binned) - 1, -1, -1):
        higher_r = binned[index + 1]["sigma_r_m"] if index + 1 < len(binned) else float("nan")
        higher_u = binned[index + 1]["sigma_u"] if index + 1 < len(binned) else float("nan")
        binned[index]["sigma_r_m"] = conservative(binned[index]["raw_sigma_r_m"], higher_r, pooled_r,
                                                  binned[index]["samples"])
        binned[index]["sigma_u"] = conservative(binned[index]["raw_sigma_u"], higher_u, pooled_u,
                                                binned[index]["samples"])

    report = {
        "method": "single-target synthetic sweep; C-domain residuals e_r=r_hat-r_gt, e_u=u_hat-u_gt; "
                  "binned by observed peak_to_noise_db; robust MAD sigma; no GT at inference",
        "sweep": {"ranges_m": RANGES_M, "bearings_deg": BEARINGS_DEG, "snr_ref_db": SNR_LEVELS_DB,
                  "realizations": REALIZATIONS, "stations_used": 3},
        "cfar_multiplier_used": multiplier,
        "rows": len(rows),
        "bin_edges_db": BIN_EDGES_DB,
        "min_samples": MIN_SAMPLES,
        "bins": [{"q_max_db": entry["q_max_db"], "sigma_r_m": entry["sigma_r_m"], "sigma_u": entry["sigma_u"],
                  "raw_sigma_r_m": entry["raw_sigma_r_m"], "raw_sigma_u": entry["raw_sigma_u"],
                  "samples": entry["samples"], "median_abs_e_r_m": entry["median_abs_e_r_m"],
                  "median_abs_e_u": entry["median_abs_e_u"]} for entry in binned],
        "pooled_sigma_r_m": pooled_r, "pooled_sigma_u": pooled_u,
        "detection_rate_by_snr": {str(level): per_snr_detection[level][0] / max(per_snr_detection[level][1], 1)
                                  for level in SNR_LEVELS_DB},
        "floor_m2": 0.01,
        "height_difference_m": geometry["height_difference_m"],
        "conservative_rule": "bins with fewer than min_samples inherit the next-higher-q sigma; "
                             "sigma enforced non-increasing with q",
        "config_hash": hashlib.sha256((ROOT / "configs/shared_frontend.json").read_bytes()).hexdigest(),
        "coords_hash": hashlib.sha256((ROOT / "frontend/sensing/coords.py").read_bytes()).hexdigest(),
    }
    _common.write_json(OUT, report)
    print(json.dumps({"rows": len(rows), "bins": report["bins"],
                      "detection_rate_by_snr": report["detection_rate_by_snr"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()