"""SENS-SNR-DESIGN-05-R2: tiny synthetic probe of sensing resource patterns (diagnostic only).

Compares admissible sensing resource subsets of the frozen K x N grid that keep the same
physical map cells (range bin 0.805 m, velocity bin 0.986 m/s):

  E0            full grid (production path, via detector.compute_maps)
  TS2 / TS4 / TS8   time-sparse: every 2nd / 4th / 8th symbol senses the full band
  COMB4         staggered comb-4 across consecutive symbols + per-Doppler-bin destagger
  TF22          comb-2 staggered over every 2nd symbol (minimal time-frequency sparse)

No production file, config or cache is modified. Clean and noisy shared echoes are generated
once per scene/SNR and re-processed by every model, so comparisons are paired. GT is used only
in this C-domain diagnostic (matching, cell diagnostics and the alias/ghost checks).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
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

OUT = ROOT / "reports/f01e/snr_design_05"
SNR_POINTS = [-5.0, 0.0, 5.0, 10.0, 15.0, 20.0]
RANGE_CLASSES = {"near": 40.0, "medium": 140.0, "far": 280.0}
COUNTS = [1, 3, 5, 8]
BEARINGS = [0.0, 40.0]
RANGE_CLASS_EDGES = [(0.0, 90.0, "near"), (90.0, 210.0, "medium"), (210.0, 400.0, "far")]

MODELS = [
    {"name": "E0_full_grid", "kind": "full", "stride": 1, "comb": 1, "role": "baseline"},
    {"name": "TS2_time_sparse", "kind": "time", "stride": 2, "comb": 1, "role": "candidate"},
    {"name": "TS4_time_sparse", "kind": "time", "stride": 4, "comb": 1, "role": "candidate"},
    {"name": "COMB4_destagger", "kind": "comb", "stride": 1, "comb": 4, "role": "candidate"},
    {"name": "TF22_comb2_stride2", "kind": "comb", "stride": 2, "comb": 2, "role": "candidate"},
    {"name": "TS8_diagnostic", "kind": "time", "stride": 8, "comb": 1, "role": "diagnostic"},
]
MODEL_BY_NAME = {model["name"]: model for model in MODELS}


def range_class(r_m: float) -> str:
    for low, high, name in RANGE_CLASS_EDGES:
        if low <= r_m < high:
            return name
    return "far"


def resource_budget(model: dict, waveform) -> dict:
    c, m = float(model["comb"]), float(model["stride"])
    K, N, T = waveform.K, waveform.N, waveform.T
    coherent_samples = N / (c * m)
    span_s = (N - m) * T
    return {
        "model": model["name"], "kind": model["kind"], "role": model["role"],
        "comb": int(c), "stride": int(m),
        "frequency_occupancy": 1.0 / c, "time_occupancy": 1.0 / m, "sensing_re_fraction": 1.0 / (c * m),
        "occupied_subcarriers_per_symbol": int(K / c), "sensing_symbols": int(N / m),
        "coherent_slow_time_samples_per_parity": coherent_samples,
        "observation_span_s": span_s, "symbol_block_s": N * T,
        "range_resolution_m": waveform.range_resolution,
        "unambiguous_range_m": waveform.unambiguous_range,
        "velocity_resolution_mps": waveform.velocity_resolution,
        "unambiguous_velocity_mps": waveform.unambiguous_velocity / (c * m),
        "coherent_gain_db": 10 * math.log10(4.0 * K * coherent_samples / 9.0),
        "relative_energy_db": -10 * math.log10(c * m),
    }


def _hann(length: int, device, dtype) -> torch.Tensor:
    from frontend.sensing.waveform import periodic_hann

    return periodic_hann(length, device, dtype)


def _cfar_fields(power: torch.Tensor, detector_cfg: dict, nr: int, nv: int):
    from frontend.sensing import detector

    train_r, train_v = (int(v) for v in detector_cfg["cfar_train"])
    guard_r, guard_v = (int(v) for v in detector_cfg["cfar_guard"])
    ring_sum = detector.box_sum(power, train_r, train_v) - detector.box_sum(power, guard_r, guard_v)
    ones = torch.ones_like(power)
    ring_count = (detector.box_sum(ones, train_r, train_v) - detector.box_sum(ones, guard_r, guard_v))
    ring_count = ring_count.clamp_min(1.0)
    noise = ring_sum / ring_count
    pfa_cell = float(detector_cfg["target_false_alarms_per_bs_frame"]) / (nr * nv)
    alpha = ring_count * (pfa_cell ** (-1.0 / ring_count) - 1.0)
    return noise, alpha, ring_count


@torch.no_grad()
def resource_maps(Y_b: torch.Tensor, X_b: torch.Tensor, waveform, array, detector_cfg: dict,
                  model: dict, compensate: bool = True) -> dict:
    from frontend.sensing import detector

    if model["kind"] == "full":
        return detector.compute_maps(Y_b, X_b, waveform, array, detector_cfg)
    device, dtype = Y_b.device, Y_b.real.dtype
    K, N, T = waveform.K, waveform.N, waveform.T
    nr = 2 * K
    range_window = _hann(K, device, dtype)[None, :, None]
    z = Y_b / X_b[None]
    if model["kind"] == "time":
        stride = int(model["stride"])
        used = torch.arange(0, N, stride, device=device)
        samples = used.numel()
        nv = 2 * samples
        spectrum = torch.fft.ifft(z[:, :, used] * range_window, n=nr, dim=1)
        window = _hann(samples, device, dtype)[None, None, :]
        spectrum = torch.fft.fftshift(torch.fft.fft(spectrum * window, n=nv, dim=2), dim=2)
        spacing = stride * T
    else:
        comb, stride = int(model["comb"]), int(model["stride"])
        used = torch.arange(0, N, stride, device=device)
        parity = torch.arange(used.numel(), device=device) % comb
        samples = int(used.numel() // comb)
        nv = 2 * samples
        columns = torch.arange(0, K, comb, device=device)
        cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64
        merged = torch.zeros((Y_b.shape[0], K, nv), device=device, dtype=cdtype)
        for p in range(comb):
            selected = used[parity == p]
            if selected.numel() != samples:
                raise ValueError("resource pattern must give equal parity sample counts")
            block = z[:, columns + p][:, :, selected]
            window = _hann(samples, device, dtype)[None, None, :]
            transformed = torch.fft.fftshift(torch.fft.fft(block * window, n=nv, dim=2), dim=2)
            merged[:, columns + p, :] = transformed
        if compensate:
            frequencies = torch.fft.fftshift(torch.fft.fftfreq(nv, d=comb * stride * T, device=device, dtype=dtype))
            for p in range(1, comb):
                phase = (2 * math.pi * frequencies) * (p * stride * T)
                merged[:, columns + p, :] *= torch.polar(torch.ones_like(phase), -phase)[None, None, :]
        spectrum = torch.fft.ifft(merged * range_window, n=nr, dim=1)
        spacing = comb * stride * T
    power = spectrum.abs().square().sum(dim=0)
    noise, alpha, ring_count = _cfar_fields(power, detector_cfg, nr, nv)
    ranges = torch.arange(nr, device=device, dtype=dtype) * (waveform.unambiguous_range / nr)
    velocities = torch.fft.fftshift(torch.fft.fftfreq(nv, d=spacing, device=device, dtype=dtype))
    velocities = velocities * (waveform.c / (2 * waveform.fc))
    return {"P_RD": power, "noise": noise, "alpha": alpha, "ring_count": ring_count,
            "spectrum": spectrum, "ranges": ranges, "velocities": velocities, "nr": nr, "nv": nv}


def peak_in_range_window(maps: dict, r_gt: float, half_bins: int = 4) -> tuple[float, float, float]:
    ranges, velocities = maps["ranges"], maps["velocities"]
    i = int(torch.argmin(torch.abs(ranges - r_gt)))
    lo, hi = max(0, i - half_bins), min(maps["nr"], i + half_bins + 1)
    block = maps["P_RD"][lo:hi]
    flat = int(torch.argmax(block))
    row, column = divmod(flat, block.shape[1])
    return float(ranges[lo + row]), float(velocities[column]), float(block[row, column])


def gt_cell_snr(maps_clean: dict, maps_noise: dict, r_gt: float, v_gt: float) -> float:
    """Detection-independent diagnostic: clean 3x3 max over the noise-only map floor.

    The noise floor is the mean over a large central block of the noise-only map (stationary
    white-noise map). A tiny local window is unusable here: the map cells within +-2 bins are
    strongly correlated (window mainlobe ~8 bins), so a 3x3 noise estimate carries ~1.5 dB
    realization scatter that differs between resource models.
    """
    i = int(torch.argmin(torch.abs(maps_clean["ranges"] - r_gt)))
    j = int(torch.argmin(torch.abs(maps_clean["velocities"] - v_gt)))
    clean = maps_clean["P_RD"][max(0, i - 1):i + 2, max(0, j - 1):j + 2]
    nr = maps_noise["nr"]
    noise = maps_noise["P_RD"][nr // 5:4 * nr // 5]
    return float(10 * math.log10(float(clean.max()) / max(float(noise.mean()), 1e-300)))


def scene_list():
    scenes = []
    for count in COUNTS:
        for range_m in RANGE_CLASSES.values():
            for bearing in BEARINGS:
                scenes.append(audit04.synthetic_snapshot(count, range_m, bearing))
    scenes.append(audit04.synthetic_snapshot(2, 100.0, 20.0, close=True))
    scenes.append(audit04.synthetic_snapshot(3, 100.0, 0.0, close=True))
    return scenes


def truth_visibility(position, station, boresight: float, height: float) -> tuple[bool, float, float]:
    _, r_gt, _, u_gt = _common.truth_polar(position.tolist(), station, boresight, height)
    bearing = math.asin(float(u_gt))
    visible = 10.0 <= float(r_gt) <= 300.0 and abs(bearing) <= math.radians(70.0)
    return visible, float(r_gt), float(u_gt)


def alias_check(waveform, array, config, stations, boresights, height, lut, device) -> dict:
    from frontend.sensing import detector

    radial = 45.0
    station = stations[0]
    position, _ = _common.place_target(station, float(boresights[0]), 140.0, 0.0, (0.0, 0.0), height)
    direction = (np.asarray(station, dtype=float) - np.asarray(position, dtype=float))
    direction = direction / np.linalg.norm(direction)
    velocity = (radial * direction).tolist()
    echo = _common.synthesize([position], [velocity], [77], stations, boresights, waveform, array, config,
                              20.0, 9800, 0, device, noise=True, height_m=height)
    rows = []
    for name in ("E0_full_grid", "TS2_time_sparse", "TS4_time_sparse", "TF22_comb2_stride2", "TS8_diagnostic"):
        model = MODEL_BY_NAME[name]
        maps = resource_maps(echo["Y"][0], echo["X"][0], waveform, array, config["detector"], model)
        r_hat, v_hat, _ = peak_in_range_window(maps, 140.0)
        detections, _, _ = detector.detect_from_maps(
            maps, 0, 0, station, float(boresights[0]), config, array,
            multiplier=float(config["detector"]["cfar_threshold_multiplier"]), covariance_lut=lut, height=height)
        rows.append({"model": name, "radial_truth_mps": radial, "peak_r_m": r_hat, "peak_vr_mps": v_hat,
                     "unambiguous_velocity_mps": resource_budget(model, waveform)["unambiguous_velocity_mps"],
                     "detections": len(detections),
                     "aliased": abs(v_hat - radial) > 1.0})
    return {"rows": rows}


def cfar_recalibration_check(waveform, array, config, stations, boresights, height, device) -> dict:
    from frontend.sensing import detector

    multipliers = [0.19, 0.24, 0.30, 0.38, 0.48, 0.60, 0.76]
    radius = (int(config["detector"]["nms_radius"][0]), int(config["detector"]["nms_radius"][1]))
    frames = 6
    rows = []
    for model in MODELS:
        counts = {multiplier: 0 for multiplier in multipliers}
        for frame in range(frames):
            echo = _common.synthesize([], [], [], stations, boresights, waveform, array, config,
                                      0.0, 9950 + frame, frame, device, noise=True, height_m=height)
            for bs in range(3):
                maps = resource_maps(echo["Y"][bs], echo["X"][bs], waveform, array, config["detector"], model)
                for multiplier in multipliers:
                    peaks = detector.local_maxima(detector.cfar_mask(maps, multiplier), maps["P_RD"], radius)
                    counts[multiplier] += int(peaks.sum())
        fa = {str(multiplier): counts[multiplier] / (frames * 3) for multiplier in multipliers}
        floor = 0.25 / (frames * 3)
        fitted = None
        for left, right in zip(multipliers, multipliers[1:]):
            if fa[str(left)] >= 0.5 >= fa[str(right)] and fa[str(left)] > fa[str(right)]:
                log_left = math.log(max(fa[str(left)], floor))
                log_right = math.log(max(fa[str(right)], floor))
                fraction = (log_left - math.log(0.5)) / (log_left - log_right)
                fitted = float(left * (right / left) ** fraction)
                break
        rows.append({"model": model["name"], "false_alarms_per_bs_frame": fa,
                     "multiplier_for_0p5_fa": fitted})
    return {"multipliers": multipliers, "frames_per_model": frames, "rows": rows}


def comb_phase_check(waveform, array, config, stations, boresights, height, device) -> dict:
    station = stations[0]
    position, _ = _common.place_target(station, float(boresights[0]), 140.0, 0.0, (0.0, 0.0), height)
    direction = (np.asarray(station, dtype=float) - np.asarray(position, dtype=float))
    direction = direction / np.linalg.norm(direction)
    rows = []
    for label, radial in (("static", 0.0), ("moving_45mps", 45.0)):
        velocity = (radial * direction).tolist()
        echo = _common.synthesize([position], [velocity], [78], stations, boresights, waveform, array, config,
                                  20.0, 9801, 0, device, noise=False, height_m=height)
        model = MODEL_BY_NAME["COMB4_destagger"]
        for compensate in (True, False):
            maps = resource_maps(echo["Y"][0], echo["X"][0], waveform, array, config["detector"], model,
                                 compensate=compensate)
            i = int(torch.argmin(torch.abs(maps["ranges"] - 140.0)))
            row = maps["P_RD"][max(0, i - 6):i + 7]
            peak = float(row.max())
            flat = int(torch.argmax(row))
            row_index, column = divmod(flat, row.shape[1])
            r_peak = float(maps["ranges"][max(0, i - 6) + row_index])
            v_peak = float(maps["velocities"][column])
            ghost_offsets = []
            for offset_m in (103.0, -103.0):
                g = int(torch.argmin(torch.abs(maps["ranges"] - (140.0 + offset_m))))
                ghost_offsets.append(float(maps["P_RD"][max(0, g - 2):g + 3].max()) / peak)
            rows.append({"case": label, "compensate": compensate, "peak_r_m": r_peak, "peak_vr_mps": v_peak,
                         "peak_power": peak, "ghost_plus103m_ratio": ghost_offsets[0],
                         "ghost_minus103m_ratio": ghost_offsets[1],
                         "ghost_max_db": 10 * math.log10(max(ghost_offsets) + 1e-300)})
    return {"rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="two scenes, one SNR point (timing smoke)")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("resource probe requires CUDA")
    from frontend.sensing import detector

    started = time.time()
    device = "cuda:0"
    torch.manual_seed(0)
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

    scenes = scene_list()
    snr_points = SNR_POINTS[:1] if args.quick else SNR_POINTS
    if args.quick:
        scenes = scenes[:2]
    rows = []
    aggregate = {model["name"]: {snr: {"pairs": 0, "matched": 0, "misses": 0, "fa": 0, "detections": 0,
                                       "frames": 0, "targets": 0, "targets_matched": 0,
                                       "range_abs": [], "u_abs": [], "xy_abs": [], "peak": [],
                                       "by_range": {}, "gt_rd": {}} for snr in snr_points}
                 for model in MODELS}
    gt_rd_rows = []

    for scene_index, scene in enumerate(scenes):
        positions = [np.asarray(position, dtype=float) for position in scene["positions"]]
        velocities = [np.asarray(velocity, dtype=float) for velocity in scene["velocities"]]
        count = scene["count"]
        for snr in snr_points:
            keys = [20000 + scene_index * 100 + index for index in range(len(positions))]
            echo = _common.synthesize(positions, velocities, keys, stations, boresights, waveform, array, config,
                                      snr, 9600 + scene_index, 0, device, noise=True, height_m=height)
            single_target = count == 1
            if single_target:
                clean = _common.synthesize(positions, velocities, keys, stations, boresights, waveform, array,
                                           config, snr, 9600 + scene_index, 0, device, noise=False, height_m=height)
            for model in MODELS:
                name = model["name"]
                per = aggregate[name][snr]
                per["frames"] += 1
                per["targets"] += len(positions)
                gt_rd_clean = gt_rd_noise = None
                if single_target:
                    gt_rd_clean = resource_maps(clean["Y"][0], clean["X"][0], waveform, array,
                                                config["detector"], model)
                    gt_rd_noise = resource_maps(echo["W"][0], clean["X"][0], waveform, array,
                                                config["detector"], model)
                    _, r_gt, _, u_gt = _common.truth_polar(positions[0].tolist(), stations[0],
                                                           float(boresights[0]), height)
                    delta = stations[0] - torch.tensor(positions[0], dtype=torch.float64)
                    radius = float(torch.linalg.vector_norm(delta))
                    radial = float(torch.dot(delta, torch.tensor(velocities[0], dtype=torch.float64))) / radius
                    value = gt_cell_snr(gt_rd_clean, gt_rd_noise, float(r_gt), radial)
                    per["gt_rd"].setdefault(range_class(float(r_gt)), []).append(value)
                    gt_rd_rows.append({"model": name, "snr_ref_db": snr, "scene": scene["label"],
                                       "range_class": range_class(float(r_gt)), "r_gt_m": float(r_gt),
                                       "radial_mps": radial, "gt_rd_snr_db": value})
                detections_by_bs = {}
                for bs in range(3):
                    maps = resource_maps(echo["Y"][bs], echo["X"][bs], waveform, array,
                                         config["detector"], model)
                    detections, _, _ = detector.detect_from_maps(
                        maps, bs, 0, stations[bs], float(boresights[bs]), config, array, multiplier=multiplier,
                        covariance_lut=lut, height=height)
                    detections_by_bs[bs] = detections
                    per["detections"] += len(detections)
                    visible_positions, visible_index = [], []
                    for index, position in enumerate(positions):
                        visible, r_gt, u_gt = truth_visibility(position, stations[bs], float(boresights[bs]),
                                                               height)
                        if visible:
                            visible_positions.append(position)
                            visible_index.append(index)
                    matched, unmatched = audit04.one_to_one_match(
                        [np.asarray(p, dtype=float) for p in visible_positions],
                        [np.array([d["x_m"], d["y_m"]]) for d in detections])
                    per["fa"] += len(unmatched)
                    matched_set = {visible_index[i] for i, _, _ in matched}
                    for index, position in enumerate(positions):
                        visible, r_gt, u_gt = truth_visibility(position, stations[bs], float(boresights[bs]),
                                                               height)
                        if not visible:
                            continue
                        bin_name = range_class(float(r_gt))
                        entry = per["by_range"].setdefault(bin_name, {"pairs": 0, "matched": 0})
                        per["pairs"] += 1
                        entry["pairs"] += 1
                        row = {"model": name, "snr_ref_db": snr, "scene": scene["label"], "count": count,
                               "range_class": bin_name, "bs": bs, "target_index": index, "visible": True,
                               "matched": index in matched_set, "false_alarms_bs": len(unmatched),
                               "r_gt_m": float(r_gt), "u_gt": float(u_gt), "r_hat_m": None, "u_hat": None,
                               "vr_hat_mps": None, "range_abs_m": None, "u_abs": None, "xy_abs_m": None,
                               "peak_to_noise_db": None, "gt_rd_snr_db": None}
                        if index in matched_set:
                            detection = detections[next(j for i, j, _ in matched if visible_index[i] == index)]
                            per["matched"] += 1
                            entry["matched"] += 1
                            error_range = abs(detection["r_m"] - float(r_gt))
                            error_u = abs(detection["u"] - float(u_gt))
                            error_xy = math.hypot(detection["x_m"] - position[0],
                                                  detection["y_m"] - position[1])
                            per["range_abs"].append(error_range)
                            per["u_abs"].append(error_u)
                            per["xy_abs"].append(error_xy)
                            per["peak"].append(detection["peak_to_noise_db"])
                            row.update({"r_hat_m": detection["r_m"], "u_hat": detection["u"],
                                        "vr_hat_mps": detection["vr_mps"], "range_abs_m": error_range,
                                        "u_abs": error_u, "xy_abs_m": error_xy,
                                        "peak_to_noise_db": detection["peak_to_noise_db"]})
                        else:
                            per["misses"] += 1
                        if single_target and bin_name in per["gt_rd"]:
                            row["gt_rd_snr_db"] = per["gt_rd"][bin_name][-1]
                        rows.append(row)
        print(f"scene {scene_index + 1}/{len(scenes)} {scene['label']} done "
              f"({time.time() - started:.1f} s)", flush=True)

    def median(values):
        return statistics.median(values) if values else None

    summary = {"probe": "SENS-SNR-DESIGN-05-R2 tiny resource probe",
               "snr_points": snr_points, "scenes": len(scenes), "models": [m["name"] for m in MODELS],
               "map_cells": {"range_bin_m": waveform.unambiguous_range / (2 * waveform.K),
                             "velocity_bin_mps": waveform.velocity_resolution},
               "model_budget": [resource_budget(model, waveform) for model in MODELS],
               "by_model_snr": {}, "checks": {}, "runtime_s": None}
    for model in MODELS:
        name = model["name"]
        summary["by_model_snr"][name] = {}
        for snr in snr_points:
            per = aggregate[name][snr]
            by_range = {bin_name: {"pairs": entry["pairs"], "matched": entry["matched"],
                                   "recall": entry["matched"] / max(entry["pairs"], 1)}
                        for bin_name, entry in sorted(per["by_range"].items())}
            summary["by_model_snr"][name][str(snr)] = {
                "station_recall": per["matched"] / max(per["pairs"], 1),
                "station_misses": per["misses"], "station_pairs": per["pairs"],
                "false_alarms_per_bs_frame": per["fa"] / max(per["frames"] * 3, 1),
                "detections_per_bs_frame": per["detections"] / max(per["frames"] * 3, 1),
                "range_median_abs_m": median(per["range_abs"]),
                "range_p90_abs_m": per["range_abs"] and float(np.percentile(per["range_abs"], 90)),
                "u_median_abs": median(per["u_abs"]),
                "xy_median_m": median(per["xy_abs"]),
                "peak_to_noise_median_db": median(per["peak"]),
                "recall_by_range_class": by_range,
                "gt_rd_median_db_by_range_class": {bin_name: median(values)
                                                   for bin_name, values in sorted(per["gt_rd"].items())},
            }

    gt_offsets = {}
    for model in MODELS:
        name = model["name"]
        values = [row["gt_rd_snr_db"] - (row["snr_ref_db"] + 40.0 * math.log10(100.0 / row["r_gt_m"]))
                  for row in gt_rd_rows if row["model"] == name]
        gt_offsets[name] = median(values)
    baseline_offset = gt_offsets.get("E0_full_grid")
    for model in MODELS:
        name = model["name"]
        if name == "E0_full_grid" or baseline_offset is None or gt_offsets[name] is None:
            continue
        measured = gt_offsets[name] - baseline_offset
        theory = -10 * math.log10(float(model["comb"]) * float(model["stride"]))
        summary["checks"][f"measured_drop_{name}_within_2db"] = abs(measured - theory) < 2.0
        summary["by_model_snr"][name]["measured_gt_rd_drop_vs_E0_db"] = measured
        summary["by_model_snr"][name]["measured_processing_gain_db"] = gt_offsets[name]
    summary["by_model_snr"]["E0_full_grid"]["absolute_gt_rd_offset_db"] = baseline_offset
    summary["by_model_snr"]["E0_full_grid"]["measured_processing_gain_db"] = baseline_offset
    summary["checks"]["e0_gt_rd_offset_matches_full_grid_theory_within_2db"] = (
        baseline_offset is not None
        and abs(baseline_offset - (10 * math.log10(4 * waveform.K * waveform.N / 9.0))) < 2.0)

    for model in MODELS:
        name = model["name"]
        gain = summary["by_model_snr"][name].get("measured_processing_gain_db")
        if gain is None:
            continue
        summary["by_model_snr"][name]["far_class_transition_snr_db_at_rd_5db"] = 5.0 - gain + 40.0 * math.log10(280.0 / 100.0)
        summary["by_model_snr"][name]["far_class_transition_snr_db_at_rd_8db"] = 8.0 - gain + 40.0 * math.log10(280.0 / 100.0)

    summary["alias_check"] = alias_check(waveform, array, config, stations, boresights, height, lut, device)
    summary["checks"]["ts4_45mps_unaliased"] = all(
        not row["aliased"] for row in summary["alias_check"]["rows"]
        if row["model"] in ("E0_full_grid", "TS2_time_sparse", "TS4_time_sparse", "TF22_comb2_stride2"))
    summary["checks"]["ts8_45mps_aliased"] = any(
        row["aliased"] for row in summary["alias_check"]["rows"] if row["model"] == "TS8_diagnostic")
    summary["cfar_recalibration_check"] = cfar_recalibration_check(waveform, array, config, stations, boresights,
                                                                   height, device)
    summary["checks"]["cfar_recalibration_reaches_0p5_fa"] = all(
        row["multiplier_for_0p5_fa"] is not None for row in summary["cfar_recalibration_check"]["rows"])
    summary["comb_phase_check"] = comb_phase_check(waveform, array, config, stations, boresights, height, device)
    compensated = [row for row in summary["comb_phase_check"]["rows"] if row["compensate"]]
    uncompensated = [row for row in summary["comb_phase_check"]["rows"] if not row["compensate"]]
    summary["checks"]["comb_doppler_compensation_removes_ghosts"] = bool(
        compensated and uncompensated
        and max(row["ghost_max_db"] for row in compensated) < -20.0
        and max(row["ghost_max_db"] for row in uncompensated)
        > max(row["ghost_max_db"] for row in compensated) + 3.0)

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "resource_probe.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary["runtime_s"] = time.time() - started
    summary["environment"] = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__}
    with (OUT / "resource_probe_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")
    print(json.dumps({"checks": summary["checks"], "alias_check": summary["alias_check"],
                      "comb_phase_check": summary["comb_phase_check"], "runtime_s": summary["runtime_s"]},
                     ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()