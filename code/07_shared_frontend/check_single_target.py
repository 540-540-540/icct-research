"""P2 single-target numerical sanity check: range, AoA, radial velocity and Cartesian position."""
from __future__ import annotations

import csv
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

OUT = ROOT / "reports/f01e/p2_detector"
RANGES_M = [30.0, 60.0, 100.0, 150.0, 200.0, 250.0, 300.0]
BEARINGS_DEG = [-60.0, -30.0, 0.0, 30.0, 60.0]
SNR_REF_DB = 20.0
REALIZATIONS = 3
VELOCITY = [8.0, -4.0]
FIELDS = ["r_gt_m", "u_gt", "vr_gt_mps", "r_hat_m", "u_hat", "vr_hat_mps", "e_r_m", "e_u", "e_vr_mps",
          "position_error_m", "x_m", "y_m", "pos_consistency_m", "detected", "peak_to_noise_db", "grid_index"]


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("P2 sanity check requires CUDA per frozen design")
    from frontend.sensing import detector as detector_module

    device = "cuda:0"
    config = _common.load_frontend_config()
    waveform, array = _common.build_objects(config)
    geometry = _common.load_geometry_config()
    _common.assert_height_alignment(config, geometry)
    stations, boresights = geometry["stations"], geometry["boresights"]
    station, boresight = stations[0], float(boresights[0])
    height = geometry["height_difference_m"]
    multiplier = config["detector"].get("cfar_threshold_multiplier")
    if multiplier is None:
        cfar_report = ROOT / "reports/f01e/cfar_calibration.json"
        multiplier = float(_common.read_json(cfar_report)["chosen_multiplier"]) if cfar_report.exists() else 1.0
    multiplier = float(multiplier)
    lut = _common.load_lut(ROOT)
    if lut is None:
        raise SystemExit("covariance LUT missing; run calibrate_covariance before the sanity check")

    rows = []
    case_index = 0
    for range_m in RANGES_M:
        for bearing_deg in BEARINGS_DEG:
            position, velocity = _common.place_target(station, boresight, range_m, bearing_deg, VELOCITY, height)
            _, r_gt, _, u_gt = _common.truth_polar(position, station, boresight, height)
            delta = torch.tensor(station, dtype=torch.float64) - torch.tensor(position, dtype=torch.float64)
            vr_gt = float((delta @ torch.tensor(velocity, dtype=torch.float64)) / r_gt)
            for realization in range(REALIZATIONS):
                echo = _common.synthesize([position], [velocity], [1], stations, boresights, waveform, array,
                                          config, SNR_REF_DB, 9400 + case_index, realization, device,
                                          height_m=float(height))
                maps = detector_module.compute_maps(echo["Y"][0], echo["X"][0], waveform, array, config["detector"])
                detections, _, _ = detector_module.detect_from_maps(
                    maps, 0, 0, station, boresight, config, array, multiplier=multiplier, covariance_lut=lut,
                    height=float(height))
                match = _common.nearest_truth_detection(detections, float(r_gt), float(u_gt))
                row = {"r_gt_m": float(r_gt), "u_gt": float(u_gt), "vr_gt_mps": vr_gt,
                       "detected": match is not None}
                if match is not None:
                    from frontend.sensing import coords

                    x_rebuilt, y_rebuilt = coords.polar_to_cartesian(match["r_m"], match["u"], station, boresight, height)
                    row.update({
                        "r_hat_m": match["r_m"], "u_hat": match["u"], "vr_hat_mps": match["vr_mps"],
                        "e_r_m": match["r_m"] - float(r_gt), "e_u": match["u"] - float(u_gt),
                        "e_vr_mps": match["vr_mps"] - vr_gt, "x_m": match["x_m"], "y_m": match["y_m"],
                        "pos_consistency_m": math.hypot(match["x_m"] - float(x_rebuilt), match["y_m"] - float(y_rebuilt)),
                        "position_error_m": math.hypot(match["x_m"] - position[0], match["y_m"] - position[1]),
                        "peak_to_noise_db": match["peak_to_noise_db"],
                        "grid_index": json.dumps(match["grid_index"])})
                rows.append(row)
            case_index += 1

    detected = [row for row in rows if row["detected"]]
    errors_r = [abs(row["e_r_m"]) for row in detected]
    errors_u = [abs(row["e_u"]) for row in detected]
    errors_vr = [abs(row["e_vr_mps"]) for row in detected]
    consistency = [row["pos_consistency_m"] for row in detected]

    checks = {
        "detection_rate": {"passed": len(detected) / len(rows) >= 0.95,
                           "detail": {"detected": len(detected), "total": len(rows)}},
        "range_accuracy": {"passed": statistics.median(errors_r) <= 0.8,
                           "detail": {"median_abs_e_r_m": statistics.median(errors_r),
                                      "max_abs_e_r_m": max(errors_r)}},
        "aoa_accuracy": {"passed": statistics.median(errors_u) <= 0.02,
                         "detail": {"median_abs_e_u": statistics.median(errors_u),
                                    "max_abs_e_u": max(errors_u)}},
        "radial_velocity_accuracy": {"passed": statistics.median(errors_vr) <= 0.5,
                                     "detail": {"median_abs_e_vr_mps": statistics.median(errors_vr),
                                                "max_abs_e_vr_mps": max(errors_vr)}},
        "cartesian_consistency": {"passed": max(consistency) <= 1e-9,
                                  "detail": {"max_consistency_m": max(consistency)}},
        "finite_outputs": {"passed": all(all(math.isfinite(value) for key, value in row.items()
                                             if isinstance(value, float))
                                         for row in detected)},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "single_target_accuracy.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    range_bias = {}
    for range_m in RANGES_M:
        selected = [row["e_r_m"] for row in detected if abs(row["r_gt_m"] - range_m) < 0.01]
        range_bias[str(range_m)] = {"samples": len(selected),
                                    "median_e_r_m": statistics.median(selected) if selected else None}
    worst_radial = sorted([row for row in detected if "e_vr_mps" in row],
                          key=lambda row: -abs(row["e_vr_mps"]))[:3]

    summary = {"test": "P2 single-target sanity", "snr_ref_db": SNR_REF_DB, "cfar_multiplier": multiplier,
               "height_difference_m": float(height),
               "cases": len(RANGES_M) * len(BEARINGS_DEG), "realizations": REALIZATIONS,
               "median_abs_e_r_m": statistics.median(errors_r), "median_abs_e_u": statistics.median(errors_u),
               "median_abs_e_vr_mps": statistics.median(errors_vr),
               "max_position_error_m": max(row["position_error_m"] for row in detected),
               "range_bias_by_range_m": range_bias,
               "worst_radial_velocity_rows": [{k: row.get(k) for k in
                                               ("r_gt_m", "u_gt", "vr_gt_mps", "vr_hat_mps", "e_vr_mps",
                                                "e_r_m", "peak_to_noise_db")} for row in worst_radial],
               "coords_hash": hashlib.sha256((ROOT / "frontend/sensing/coords.py").read_bytes()).hexdigest(),
               "detector_hash": hashlib.sha256((ROOT / "frontend/sensing/detector.py").read_bytes()).hexdigest()}
    passed = all(entry["passed"] for entry in checks.values())
    summary["passed"] = passed
    _common.write_json(OUT / "summary.json", summary)
    _common.write_json(OUT / "checks.json", {"test": "P2 sanity", "passed": passed, "checks": checks})
    print(json.dumps({"P2_sanity": "PASS" if passed else "FAIL", "summary": summary,
                      "checks": {key: entry["passed"] for key, entry in checks.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()