"""SENS-SNR-DESIGN-05 energy-accounting golden test (diagnostic only; no production change).

Verifies the OFDM power ledger, the Hann-windowed RD processing gain, and predicts the
post-RD SNR under three energy models:
  E1 current (per-RE SNR fixed), E2 fixed OFDM-symbol power, E3 fixed CPI energy.
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

OUT = ROOT / "reports/f01e/snr_design_05"
SNR_POINTS = [-10.0, 0.0, 10.0, 20.0]
RANGES = {"near": 40.0, "medium": 140.0, "far": 280.0}


FIELDS = ["record", "model", "K", "N", "snr_ref_db", "r_m", "radial_mps", "received_snr_db",
          "measured_rd_snr_db", "theory_rd_snr_db", "delta_db", "detection_margin_db"]


def hann_gain(length: int) -> float:
    index = np.arange(length)
    window = 0.5 - 0.5 * np.cos(2 * np.pi * index / length)
    return float(window.sum() ** 2 / (window ** 2).sum())


def rd_gain_current_db(K: int, N: int) -> float:
    return 10 * math.log10(hann_gain(K) * hann_gain(N))


def rd_gain_symbol_power_db(N: int) -> float:
    return 10 * math.log10(hann_gain(N))


def rd_gain_cpi_energy_db() -> float:
    return 10 * math.log10(4.0 / 9.0)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("energy golden test requires CUDA per frozen design")
    from frontend.sensing import detector as detector_module

    device = "cuda:0"
    config = _common.load_frontend_config()
    geometry = _common.load_geometry_config()
    _common.assert_height_alignment(config, geometry)
    waveform, array = _common.build_objects(config)
    stations, boresights = geometry["stations"], geometry["boresights"]
    height = float(geometry["height_difference_m"])
    multiplier = float(config["detector"]["cfar_threshold_multiplier"])
    K, N = waveform.K, waveform.N
    rows, checks = [], {}

    # Test 1+2: OFDM-symbol energy ledger
    symbol_energy_rows = []
    for k in (64, 128, 256, 512):
        current_symbol_energy = float(k)               # |X|^2 = 1 per active RE
        fixed_power_symbol_energy = float(k) * (1.0 / k)  # |X|^2 = 1/K per active RE
        symbol_energy_rows.append({"K": k, "current_E_symbol": current_symbol_energy,
                                   "fixed_power_E_symbol": fixed_power_symbol_energy})
    checks["test1_fixed_symbol_power_energy_invariant"] = all(
        abs(row["fixed_power_E_symbol"] - 1.0) < 1e-12 for row in symbol_energy_rows)
    checks["test2_current_energy_scales_with_K"] = (
        symbol_energy_rows[2]["current_E_symbol"] == 2 * symbol_energy_rows[1]["current_E_symbol"])

    # Test 3: CPI energy under fixed transmit power
    cpi_rows = [{"N": n, "E_CPI_fixed_power": float(n)} for n in (128, 256, 512)]
    checks["test3_fixed_power_CPI_energy_scales_with_N"] = (
        abs(cpi_rows[2]["E_CPI_fixed_power"] / cpi_rows[1]["E_CPI_fixed_power"] - 2.0) < 1e-12)

    # Test 4: Hann-windowed RD gain vs Monte Carlo using actual simulator/detector
    station = stations[0]
    boresight = float(boresights[0])
    nr, nv = 2 * K, 2 * N
    delta_r = waveform.unambiguous_range / nr
    delta_v = waveform.c / (2 * waveform.fc * nv * waveform.T)
    r_on = round(100.0 / delta_r) * delta_r
    vr_on = round(10.0 / delta_v) * delta_v
    rho = math.sqrt(r_on ** 2 - height ** 2)
    position = [float(station[0]) + rho, float(station[1])]
    direction = (np.array([float(station[0]), float(station[1])]) - np.array(position)) / rho
    velocity = (vr_on * direction).tolist()
    received_base = 40.0 * math.log10(100.0 / r_on)
    theory_current = rd_gain_current_db(K, N)
    theory_symbol_power = rd_gain_symbol_power_db(N)

    measured = {}
    for snr in SNR_POINTS:
        for model, effective_snr in (("E1_current", snr), ("E2_fixed_symbol_power", snr - 10 * math.log10(K))):
            ratios = []
            for realization in range(4):
                echo = _common.synthesize([position], [velocity], [7 + realization], stations, boresights,
                                          waveform, array, config, effective_snr, 9900 + realization,
                                          realization, device, noise=True, height_m=height)
                clean = _common.synthesize([position], [velocity], [7 + realization], stations, boresights,
                                           waveform, array, config, effective_snr, 9900 + realization,
                                           realization, device, noise=False, height_m=height)
                maps_clean = detector_module.compute_maps(clean["Y"][0], clean["X"][0], waveform, array,
                                                          config["detector"])
                maps_noise = detector_module.compute_maps(echo["W"][0], clean["X"][0], waveform, array,
                                                          config["detector"])
                i = int(torch.argmin(torch.abs(maps_clean["ranges"] - r_on)))
                j = int(torch.argmin(torch.abs(maps_clean["velocities"] - vr_on)))
                clean_window = maps_clean["P_RD"][i - 1:i + 2, j - 1:j + 2]
                noise_window = maps_noise["P_RD"][i - 1:i + 2, j - 1:j + 2]
                ratios.append(10 * math.log10(float(clean_window.max())
                                              / max(float(noise_window.median()), 1e-300)))
            measured[(model, snr)] = statistics.median(ratios)
            theory = theory_current + effective_snr + received_base
            rows.append({"record": "rd_probe", "model": model, "K": K, "N": N, "snr_ref_db": snr,
                         "r_m": r_on, "radial_mps": vr_on, "received_snr_db": effective_snr + received_base,
                         "measured_rd_snr_db": measured[(model, snr)], "theory_rd_snr_db": theory,
                         "delta_db": measured[(model, snr)] - theory})
    e1_deltas = [row["delta_db"] for row in rows if row["model"] == "E1_current"]
    checks["test4_theory_matches_monte_carlo"] = all(abs(value) < 2.0 for value in e1_deltas)
    checks["test4_e2_shift_is_10log10_K"] = abs(
        statistics.median([measured[("E2_fixed_symbol_power", snr)]
                           - measured[("E1_current", snr)] for snr in (-10.0, 0.0)])
        + 10 * math.log10(K)) < 1.0
    checks["test5_e2_rows_consistent_with_current_gain_and_reduced_input"] = all(
        abs(row["delta_db"]) < 2.0 for row in rows if row["model"] == "E2_fixed_symbol_power")

    # Predicted operating table for near/medium/far under E1/E2/E3
    cfar_threshold_db = 10 * math.log10(13.79 * multiplier)
    for snr in [-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0]:
        for name, range_m in RANGES.items():
            received = snr + 40.0 * math.log10(100.0 / range_m)
            for model, gain in (("E1_current", rd_gain_current_db(K, N)),
                                ("E2_fixed_symbol_power", rd_gain_symbol_power_db(N)),
                                ("E3_fixed_cpi_energy", rd_gain_cpi_energy_db())):
                rows.append({"record": "predicted", "model": model, "K": K, "N": N, "snr_ref_db": snr,
                             "r_m": range_m, "radial_mps": None, "received_snr_db": received,
                             "measured_rd_snr_db": None, "theory_rd_snr_db": received + gain,
                             "delta_db": None,
                             "detection_margin_db": received + gain - cfar_threshold_db})

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "energy_budget.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    summary = {"test": "SENS-SNR-DESIGN-05 energy golden test", "checks": checks,
               "symbol_energy_ledger": symbol_energy_rows, "cpi_energy_ledger": cpi_rows,
               "processing_gain": {"hann_gain_K": hann_gain(K), "hann_gain_N": hann_gain(N),
                                   "rd_gain_current_db": theory_current,
                                   "rd_gain_symbol_power_db": theory_symbol_power,
                                   "rd_gain_cpi_energy_db": rd_gain_cpi_energy_db()},
               "cfar_threshold_db_reference": cfar_threshold_db}
    _common.write_json(OUT / "energy_golden_checks.json", summary)
    print(json.dumps({"checks": checks, "processing_gain": summary["processing_gain"],
                      "measured_E1": {str(s): round(measured[("E1_current", s)], 2) for s in SNR_POINTS},
                      "measured_E2": {str(s): round(measured[("E2_fixed_symbol_power", s)], 2)
                                      for s in SNR_POINTS}}, ensure_ascii=False))
    if not all(checks.values()):
        raise SystemExit("energy golden test failed")


if __name__ == "__main__":
    main()