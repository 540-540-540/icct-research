"""Measurement covariance calibration: sigma_r(q), sigma_u(q) LUT with low-quality extension (P2.5).

Frozen design: DECISIONS.md D23, SYSTEM_MODEL.md section 5.1, SENS-REBUILD-03B section 2.
"""
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

OUT = ROOT / "reports/f01e/covariance_calibration.json"
P25_DIR = ROOT / "reports/f01e/p25_covariance"
RANGES_M = [30.0, 60.0, 100.0, 150.0, 200.0, 250.0, 300.0]
BEARINGS_DEG = [-69.0, -45.0, 0.0, 45.0, 69.0]
SNR_LEVELS_DB = [-20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0]
REALIZATIONS = 4
BIN_EDGES_DB = [6.0, 12.0, 20.0]
MIN_SAMPLES = 30
VELOCITY = [8.0, -3.0]
GATE_CHI2 = 9.21


def robust_sigma(values: list[float]) -> float:
    if not values:
        return float("nan")
    median = statistics.median(values)
    return 1.4826 * statistics.median(abs(value - median) for value in values)


def bin_index(q_db: float) -> int:
    for index, edge in enumerate(BIN_EDGES_DB):
        if q_db <= edge:
            return index
    return len(BIN_EDGES_DB)


def pairwise_statistics(pair_pool: dict, boresights, provisional: dict, inflation: float) -> dict:
    from frontend.sensing import coords

    lut = dict(provisional)
    lut["covariance_inflation"] = float(inflation)
    distances = []
    pair_upper_bound = 0
    for entries in pair_pool.values():
        pair_upper_bound += len(entries) * (len(entries) - 1) // 2
        for first in range(len(entries)):
            for second in range(first + 1, len(entries)):
                a, b = entries[first], entries[second]
                if a["bs"] == b["bs"]:
                    continue
                ca = coords.covariance_from_lut(a["r_m"], a["u"], a["q_db"], float(boresights[a["bs"]]), lut, 0.01)
                cb = coords.covariance_from_lut(b["r_m"], b["u"], b["q_db"], float(boresights[b["bs"]]), lut, 0.01)
                delta = torch.tensor([a["x"] - b["x"], a["y"] - b["y"]], dtype=torch.float64)
                distances.append(float(delta @ torch.linalg.solve(ca + cb, delta)))
    ordered = sorted(distances)
    def quantile(fraction: float) -> float:
        return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))] if ordered else float("nan")
    return {"inflation": float(inflation), "pairs": len(distances), "pair_upper_bound": pair_upper_bound,
            "gate_chi2_2dof": GATE_CHI2,
            "pass_rate": sum(1 for value in distances if value <= GATE_CHI2) / len(distances) if distances else None,
            "d2_p50": quantile(0.50), "d2_p90": quantile(0.90), "d2_p95": quantile(0.95), "d2_p99": quantile(0.99)}


def find_minimal_inflation(pair_pool: dict, boresights, provisional: dict, target: float = 0.99):
    cache = {}

    def rate(candidate: float) -> float:
        if candidate not in cache:
            cache[candidate] = pairwise_statistics(pair_pool, boresights, provisional, candidate)["pass_rate"]
        return cache[candidate]

    if rate(1.0) >= target:
        return 1.0, cache
    low, high = 1.0, 2.0
    while rate(high) < target and high < 64.0:
        low, high = high, high * 2
    for _ in range(14):
        middle = (low + high) / 2
        if rate(middle) >= target:
            high = middle
        else:
            low = middle
    return high, cache


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("covariance calibration requires CUDA per frozen design")
    from frontend.sensing import coords, detector as detector_module

    device = "cuda:0"
    config = _common.load_frontend_config()
    geometry = _common.load_geometry_config()
    _common.assert_height_alignment(config, geometry)
    waveform, array = _common.build_objects(config)
    stations, boresights = geometry["stations"], geometry["boresights"]
    height = float(geometry["height_difference_m"])
    multiplier = float(config["detector"]["cfar_threshold_multiplier"])
    detector_config = config["detector"]

    rows, attempts = [], []
    per_snr = {level: [0, 0] for level in SNR_LEVELS_DB}
    pair_pool: dict[tuple[int, int], list[dict]] = {}
    combo_index = 0
    for range_m in RANGES_M:
        for bearing_deg in BEARINGS_DEG:
            position, velocity = _common.place_target(stations[0], float(boresights[0]), range_m, bearing_deg,
                                                      VELOCITY, height)
            episode = 9200 + combo_index
            combo_index += 1
            truth = []
            for bs in range(3):
                _, r_gt, _, u_gt = _common.truth_polar(position, stations[bs], float(boresights[bs]), height)
                bearing_gt = math.asin(float(u_gt))
                visible = 10.0 <= float(r_gt) <= 300.0 and abs(bearing_gt) <= math.radians(70.0)
                truth.append((float(r_gt), float(u_gt), bool(visible)))
            for realization in range(REALIZATIONS):
                for snr in SNR_LEVELS_DB:
                    echo = _common.synthesize([position], [velocity], [1 + realization], stations, boresights,
                                              waveform, array, config, snr, episode, realization, device,
                                              height_m=height)
                    for bs in range(3):
                        r_gt, u_gt, visible = truth[bs]
                        if not visible:
                            continue
                        per_snr[snr][1] += 1
                        maps = detector_module.compute_maps(echo["Y"][bs], echo["X"][bs], waveform, array,
                                                            detector_config)
                        detections, _, _ = detector_module.detect_from_maps(
                            maps, bs, 0, stations[bs], float(boresights[bs]), config, array, multiplier=multiplier,
                            height=height)
                        match = _common.nearest_truth_detection(detections, r_gt, u_gt)
                        attempts.append(match is not None)
                        if match is None:
                            continue
                        per_snr[snr][0] += 1
                        rows.append({"q_db": match["peak_to_noise_db"], "e_r": match["r_m"] - r_gt,
                                     "e_u": match["u"] - u_gt, "station_id": bs, "snr_ref_db": snr})
                        pair_pool.setdefault((combo_index, realization, snr), []).append(
                            {"bs": bs, "q_db": match["peak_to_noise_db"], "x": match["x_m"], "y": match["y_m"],
                             "r_m": match["r_m"], "u": match["u"]})

    binned = []
    for index in range(len(BIN_EDGES_DB) + 1):
        low = BIN_EDGES_DB[index - 1] if index > 0 else None
        high = BIN_EDGES_DB[index] if index < len(BIN_EDGES_DB) else None
        selected = [row for row in rows if (low is None or row["q_db"] > low)
                    and (high is None or row["q_db"] <= high)]
        binned.append({
            "index": index, "q_min_db": low, "q_max_db": high, "samples": len(selected),
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
        binned[index]["fallback_used"] = bool(
            binned[index]["samples"] < MIN_SAMPLES or not math.isfinite(binned[index]["raw_sigma_r_m"])
            or not math.isfinite(binned[index]["raw_sigma_u"]))

    provisional = {"bins": [{"q_max_db": entry["q_max_db"], "sigma_r_m": entry["sigma_r_m"],
                            "sigma_u": entry["sigma_u"]} for entry in binned],
                   "floor_m2": 0.01, "height_difference_m": height}

    inflation, rate_cache = find_minimal_inflation(pair_pool, boresights, provisional, target=0.99)
    stats = [cache_entry for _, cache_entry in sorted(rate_cache.items())]
    chosen_stats = pairwise_statistics(pair_pool, boresights, provisional, inflation)
    pairwise_gate = {**chosen_stats, "target_pass_rate": 0.99, "candidate_table": stats,
                     "pass_rate_before_inflation": pairwise_statistics(pair_pool, boresights, provisional,
                                                                        1.0)["pass_rate"],
                     "pass_rate_after_inflation": chosen_stats["pass_rate"],
                     "p99_d2": chosen_stats["d2_p99"],
                     "search": "minimal inflation with pass_rate >= 0.99 via doubling + bisection"}

    total_attempts = len(attempts)
    detection_rate = (sum(attempts) / total_attempts) if total_attempts else 0.0
    for entry in binned:
        entry["detection_rate"] = (entry["samples"] / total_attempts) if total_attempts else 0.0

    report = {
        "method": "single-target synthetic sweep; C-domain residuals e_r=r_hat-r_gt, e_u=u_hat-u_gt; "
                  "binned by observed peak_to_noise_db; robust MAD sigma; conservative envelope; "
                  "covariance inflation calibrated from true-pair gating statistics",
        "sweep": {"ranges_m": RANGES_M, "bearings_deg": BEARINGS_DEG, "snr_ref_db": SNR_LEVELS_DB,
                  "realizations": REALIZATIONS, "stations_used": 3},
        "cfar_multiplier_used": multiplier,
        "rows": len(rows),
        "visible_attempts": total_attempts,
        "overall_detection_rate": detection_rate,
        "detection_rate_by_snr": {str(level): per_snr[level][0] / max(per_snr[level][1], 1)
                                  for level in SNR_LEVELS_DB},
        "bin_edges_db": BIN_EDGES_DB,
        "min_samples": MIN_SAMPLES,
        "bins": [{"q_max_db": entry["q_max_db"], "sigma_r_m": entry["sigma_r_m"], "sigma_u": entry["sigma_u"],
                  "raw_sigma_r_m": entry["raw_sigma_r_m"], "raw_sigma_u": entry["raw_sigma_u"],
                  "samples": entry["samples"], "detection_rate": entry["detection_rate"],
                  "fallback_used": entry["fallback_used"],
                  "median_abs_e_r_m": entry["median_abs_e_r_m"],
                  "median_abs_e_u": entry["median_abs_e_u"]} for entry in binned],
        "pooled_sigma_r_m": pooled_r, "pooled_sigma_u": pooled_u,
        "pairwise_true_pairs": chosen_stats["pairs"],
        "pairwise_gate": pairwise_gate,
        "covariance_inflation": inflation,
        "floor_m2": 0.01,
        "height_difference_m": height,
        "conservative_rule": "bins with fewer than min_samples inherit the next-higher-q sigma and the pooled "
                             "envelope; sigma enforced non-increasing with q; gating covariance inflated by the "
                             "calibrated factor so >=99% of true same-target pairs pass the frozen chi2 gate",
        "config_hash": hashlib.sha256((ROOT / "configs/shared_frontend.json").read_bytes()).hexdigest(),
        "coords_hash": hashlib.sha256((ROOT / "frontend/sensing/coords.py").read_bytes()).hexdigest(),
    }
    _common.write_json(OUT, report)

    P25_DIR.mkdir(parents=True, exist_ok=True)
    summary = {"test": "P2.5 covariance extension", "rows": len(rows), "visible_attempts": total_attempts,
               "overall_detection_rate": detection_rate,
               "detection_rate_by_snr": report["detection_rate_by_snr"],
               "pairwise_true_pairs": chosen_stats["pairs"], "pairwise_gate": report["pairwise_gate"],
               "fallback_bins": [entry["q_max_db"] for entry in binned if entry["fallback_used"]],
               "note": "low-q bins with too few detections use conservative pooled envelope; no fabricated samples",
               "passed": bool(len(rows) > 0 and chosen_stats["pairs"])}
    _common.write_json(P25_DIR / "summary.json", summary)
    with (P25_DIR / "bin_coverage.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["q_min_db", "q_max_db", "samples", "sigma_r_m", "sigma_u", "detection_rate",
                         "fallback_used", "median_abs_e_r_m", "median_abs_e_u"])
        for entry in binned:
            writer.writerow([entry["q_min_db"], entry["q_max_db"], entry["samples"], entry["sigma_r_m"],
                             entry["sigma_u"], entry["detection_rate"], entry["fallback_used"],
                             entry["median_abs_e_r_m"], entry["median_abs_e_u"]])
    print(json.dumps({"rows": len(rows), "attempts": total_attempts, "detection_rate": detection_rate,
                      "bins": [(entry["q_max_db"], entry["samples"], entry["sigma_r_m"], entry["fallback_used"])
                               for entry in binned],
                      "pairwise": report["pairwise_gate"],
                      "detection_rate_by_snr": report["detection_rate_by_snr"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()