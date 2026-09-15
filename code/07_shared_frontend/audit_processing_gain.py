"""SENS-SNR-AUDIT-04 internal probes: processing-size, array-size and geometry contribution.

Diagnostic only: production configs are never modified; probes rebuild local waveforms/arrays.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import _common  # noqa: E402

OUT = ROOT / "reports/f01e/snr_audit_04"
SNR_POINTS = [-10.0, 0.0, 10.0, 20.0]
PROBE_TARGETS = [("near", 60.0), ("medium", 140.0), ("far", 280.0)]
PROCESSING_SIZES = [(64, 64), (128, 128), (128, 256), (256, 256)]
ARRAY_SIZES = [4, 8, 16]


@torch.no_grad()
def maps_with_truncation(Y_b, X_b, waveform, array, detector, k_eff, n_eff):
    """Diagnostic copy of detector.compute_maps with truncated coherent processing."""
    from frontend.sensing.waveform import periodic_hann

    device, dtype = Y_b.device, Y_b.real.dtype
    nr, nv = 2 * k_eff, 2 * n_eff
    z = Y_b[:, :k_eff, :n_eff] / X_b[:k_eff, :n_eff][None]
    range_window = periodic_hann(k_eff, device, dtype)[None, :, None]
    doppler_window = periodic_hann(n_eff, device, dtype)[None, None, :]
    spectrum = torch.fft.ifft(z * range_window, n=nr, dim=1)
    spectrum = torch.fft.fftshift(torch.fft.fft(spectrum * doppler_window, n=nv, dim=2), dim=2)
    power = spectrum.abs().square().sum(dim=0)
    from frontend.sensing.detector import box_sum

    train_r, train_v = (int(v) for v in detector["cfar_train"])
    guard_r, guard_v = (int(v) for v in detector["cfar_guard"])
    ring_sum = box_sum(power, train_r, train_v) - box_sum(power, guard_r, guard_v)
    ring_count = (box_sum(torch.ones_like(power), train_r, train_v)
                  - box_sum(torch.ones_like(power), guard_r, guard_v)).clamp_min(1.0)
    noise = ring_sum / ring_count
    target_false_alarms = float(detector["target_false_alarms_per_bs_frame"])
    pfa_cell = target_false_alarms / (nr * nv)
    alpha = ring_count * (pfa_cell ** (-1.0 / ring_count) - 1.0)
    ranges = torch.arange(nr, device=device, dtype=dtype) * (waveform.unambiguous_range / nr)
    velocities = torch.fft.fftshift(torch.fft.fftfreq(nv, d=waveform.T, device=device, dtype=dtype))
    velocities = velocities * (waveform.c / (2 * waveform.fc))
    return {"P_RD": power, "noise": noise, "alpha": alpha, "spectrum": spectrum,
            "ranges": ranges, "velocities": velocities, "nr": nr, "nv": nv}


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("processing-gain audit requires CUDA per frozen design")
    from frontend.echo_source import SourceEpisodes
    from frontend.sensing import detector as detector_module
    from frontend.sensing.coords import cartesian_to_polar
    from frontend.sensing.waveform import ArrayConfig, PaperWaveform

    device = "cuda:0"
    config = _common.load_frontend_config()
    geometry = _common.load_geometry_config()
    waveform, array = _common.build_objects(config)
    stations, boresights = geometry["stations"], geometry["boresights"]
    height = float(geometry["height_difference_m"])
    multiplier = float(config["detector"]["cfar_threshold_multiplier"])
    lut = _common.load_lut(ROOT)
    station = stations[0]
    boresight = float(boresights[0])

    rows = []
    for label, range_m in PROBE_TARGETS:
        rho = math.sqrt(range_m ** 2 - height ** 2)
        position = [float(station[0]) + rho, float(station[1])]
        for snr in SNR_POINTS:
            echo = _common.synthesize([position], [[8.0, -3.0]], [7], stations, boresights, waveform, array,
                                      config, snr, 9800, 0, device, height_m=height)
            _, r_gt, _, u_gt = cartesian_to_polar(torch.tensor(position, dtype=torch.float64), station,
                                                  boresight, height)
            received = snr + 40.0 * math.log10(100.0 / float(r_gt))
            # processing-size probe (diagnostic waveforms only)
            for k_eff, n_eff in PROCESSING_SIZES:
                maps = maps_with_truncation(echo["Y"][0], echo["X"][0], waveform, array, config["detector"],
                                            k_eff, n_eff)
                detections, _, _ = detector_module.detect_from_maps(
                    maps, 0, 0, station, boresight, config, ArrayConfig(), multiplier=multiplier,
                    covariance_lut=lut, height=height)
                match = _common.nearest_truth_detection(detections, float(r_gt), float(u_gt))
                rows.append({"probe": "processing", "setting": f"K{k_eff}_N{n_eff}", "target": label,
                             "snr_ref_db": snr, "received_snr_db": received,
                             "peak_to_noise_db": match["peak_to_noise_db"] if match else None,
                             "detected": match is not None, "g_emp_db": (match["peak_to_noise_db"] - received)
                             if match else None})
            # array-size probe (diagnostic array config only)
            for elements in ARRAY_SIZES:
                probe_array = ArrayConfig(elements=elements)
                detections, _, _ = detector_module.detect_from_maps(
                    detector_module.compute_maps(echo["Y"][0][:elements], echo["X"][0], waveform, probe_array,
                                                 config["detector"]),
                    0, 0, station, boresight, config, probe_array, multiplier=multiplier,
                    covariance_lut=lut, height=height)
                match = _common.nearest_truth_detection(detections, float(r_gt), float(u_gt))
                rows.append({"probe": "array", "setting": f"A{elements}", "target": label,
                             "snr_ref_db": snr, "received_snr_db": received,
                             "peak_to_noise_db": match["peak_to_noise_db"] if match else None,
                             "detected": match is not None, "g_emp_db": (match["peak_to_noise_db"] - received)
                             if match else None})
    with (OUT / "processing_probe.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    source = SourceEpisodes()
    ranges, received_snrs = [], []
    for episode in source.episodes:
        if episode["split"] != "train":
            continue
        states, _slots, _used = source.at_time(episode, (int(episode["start_ms"]) + 5000) * 1_000_000)
        for state in states:
            for bs in range(3):
                _, r_m, _bearing, u_m = cartesian_to_polar(torch.tensor(state[:2], dtype=torch.float64),
                                                           stations[bs], float(boresights[bs]), height)
                bearing = math.asin(float(u_m))
                if 10.0 <= float(r_m) <= 300.0 and abs(bearing) <= math.radians(70.0):
                    ranges.append(float(r_m))
                    received_snrs.append(40.0 * math.log10(100.0 / float(r_m)))
    geometry_report = {
        "train_targets": len(ranges),
        "range_m": {"p10": float(np.quantile(ranges, 0.10)), "p50": float(np.quantile(ranges, 0.50)),
                    "p90": float(np.quantile(ranges, 0.90))} if ranges else None,
        "received_snr_relative_to_ref_db": {
            "p10": float(np.quantile(received_snrs, 0.10)), "p50": float(np.quantile(received_snrs, 0.50)),
            "p90": float(np.quantile(received_snrs, 0.90))} if received_snrs else None,
        "note": "received_snr relative to snr_ref (40 log10(100/r)); add snr_ref for absolute dB",
    }
    _common.write_json(OUT / "geometry_probe.json", geometry_report)
    print(json.dumps({"processing_probe_rows": len(rows), "geometry": geometry_report},
                     ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()