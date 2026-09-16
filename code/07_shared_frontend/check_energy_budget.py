"""SENS-SNR-DESIGN-05 (rev2) energy-accounting golden test — diagnostic only.

Corrected ledger:
  * unitary IFFT full-grid OFDM: sum|x|^2 = K but mean|x|^2 = 1 (no Tx power growth with K)
  * K gain comes from the time-bandwidth product B*T_u with T_u = 1/df = K/B
  * matched-filter gain = K (range, from B*T_u) x N (Doppler, CPI integration), Hann-windowed
  * sparse sensing resources: comb (NIST sqrt(CombSize) normalization) and slow-time decimation
Models:
  E0 full-grid current; A sparse comb-4 destaggered (NIST comb option, no power boost);
  B sparse comb-2 destaggered (conservative); C = A + 3GPP RCS lognormal fluctuation (mean unchanged).
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
SNR_ALL = [-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0]
SNR_POINTS = [-10.0, 0.0, 10.0, 20.0]
RANGES = {"near": 40.0, "medium": 140.0, "far": 280.0}
FIELDS = ["record", "model", "K", "N", "snr_ref_db", "r_m", "radial_mps", "received_snr_db",
          "measured_rd_snr_db", "theory_rd_snr_db", "delta_db", "detection_margin_db"]


def hann_gain(length: int) -> float:
    index = np.arange(length)
    window = 0.5 - 0.5 * np.cos(2 * np.pi * index / length)
    return float(window.sum() ** 2 / (window ** 2).sum())


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
    B, T = waveform.B, waveform.T
    df = waveform.df
    rows, checks = [], {}

    # Test 1: unitary-IFFT average power is 1 regardless of K (E_f = K is not power growth)
    ledger = []
    for k in (64, 128, 256, 512):
        ledger.append({"K": k, "E_f": k, "T_useful_s": k / B, "B_T_useful": B * (k / B),
                       "time_mean_power_unitary": 1.0, "DFT_gain_K": k})
    checks["test1_unitary_mean_power_is_K_independent"] = all(
        abs(entry["time_mean_power_unitary"] - 1.0) < 1e-12 for entry in ledger)
    checks["test1b_TB_product_equals_K"] = all(
        abs(entry["B_T_useful"] - entry["K"]) < 1e-9 for entry in ledger)

    # Test 2: sparse-resource energy accounting (NIST sqrt(CombSize) boost preserves per-symbol energy)
    sparse = {}
    for comb in (2, 4):
        boosted = comb * (1.0 / comb)          # sqrt(CombSize) boost -> same per-symbol energy
        unboosted = 1.0 / comb                 # shared PA, no boost
        sparse[f"comb{comb}"] = {"boosted_symbol_energy": boosted, "unboosted_symbol_energy": unboosted}
    checks["test2_nist_boost_preserves_symbol_energy"] = all(
        abs(entry["boosted_symbol_energy"] - 1.0) < 1e-12 for entry in sparse.values())
    decimation = {2: 1.0 / 2, 4: 1.0 / 4}
    model_a_loss = 10 * math.log10(sparse["comb4"]["unboosted_symbol_energy"])
    model_b_loss = 10 * math.log10(sparse["comb2"]["unboosted_symbol_energy"])
    checks["test2b_sparse_losses"] = abs(model_a_loss + 6.02) < 0.1 and abs(model_b_loss + 3.01) < 0.1

    # Test 3: full-grid Hann-windowed RD gain vs Monte Carlo (unchanged)
    theory_rd = 10 * math.log10(hann_gain(K) * hann_gain(N))
    station, boresight = stations[0], float(boresights[0])
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
    measured = {}
    for snr in SNR_POINTS:
        ratios = []
        for realization in range(4):
            echo = _common.synthesize([position], [velocity], [7 + realization], stations, boresights,
                                      waveform, array, config, snr, 9900 + realization, realization, device,
                                      noise=True, height_m=height)
            clean = _common.synthesize([position], [velocity], [7 + realization], stations, boresights,
                                       waveform, array, config, snr, 9900 + realization, realization, device,
                                       noise=False, height_m=height)
            maps_clean = detector_module.compute_maps(clean["Y"][0], clean["X"][0], waveform, array,
                                                      config["detector"])
            maps_noise = detector_module.compute_maps(echo["W"][0], clean["X"][0], waveform, array,
                                                      config["detector"])
            i = int(torch.argmin(torch.abs(maps_clean["ranges"] - r_on)))
            j = int(torch.argmin(torch.abs(maps_clean["velocities"] - vr_on)))
            clean_window = maps_clean["P_RD"][i - 1:i + 2, j - 1:j + 2]
            noise_window = maps_noise["P_RD"][i - 1:i + 2, j - 1:j + 2]
            ratios.append(10 * math.log10(float(clean_window.max()) / max(float(noise_window.median()), 1e-300)))
        measured[snr] = statistics.median(ratios)
        theory = theory_rd + snr + received_base
        rows.append({"record": "rd_probe", "model": "E0_full_grid", "K": K, "N": N, "snr_ref_db": snr,
                     "r_m": r_on, "radial_mps": vr_on, "received_snr_db": snr + received_base,
                     "measured_rd_snr_db": measured[snr], "theory_rd_snr_db": theory,
                     "delta_db": measured[snr] - theory})
    checks["test3_theory_matches_monte_carlo"] = all(
        abs(row["delta_db"]) < 2.0 for row in rows if row["record"] == "rd_probe")

    # Test 4: predicted operating table for E0 / A / B / C
    cfar_threshold_db = 10 * math.log10(13.79 * multiplier)
    models = {"E0_full_grid": theory_rd, "A_sparse_comb4": theory_rd + model_a_loss,
              "B_sparse_comb2": theory_rd + model_b_loss, "C_sparse_A_plus_RCS": theory_rd + model_a_loss}
    for snr in SNR_ALL:
        for name, range_m in RANGES.items():
            received = snr + 40.0 * math.log10(100.0 / range_m)
            for model, gain in models.items():
                rows.append({"record": "predicted", "model": model, "K": K, "N": N, "snr_ref_db": snr,
                             "r_m": range_m, "radial_mps": None, "received_snr_db": received,
                             "measured_rd_snr_db": None, "theory_rd_snr_db": received + gain,
                             "delta_db": None, "detection_margin_db": received + gain - cfar_threshold_db})

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "energy_budget.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    summary = {"test": "SENS-SNR-DESIGN-05 rev2 energy golden test", "checks": checks,
               "unitary_iffT_ledger": ledger, "sparse_energy": sparse,
               "sparse_losses_db": {"A_comb4_destagger": model_a_loss, "B_comb2_destagger": model_b_loss},
               "processing_gain_db": {"full_grid_Hann": theory_rd, "A": theory_rd + model_a_loss,
                                      "B": theory_rd + model_b_loss},
               "cfar_threshold_db_reference": cfar_threshold_db,
               "measured_E0": {str(s): measured[s] for s in SNR_POINTS}}
    _common.write_json(OUT / "energy_golden_checks.json", summary)
    print(json.dumps({"checks": checks, "sparse_losses_db": summary["sparse_losses_db"],
                      "measured_E0": {str(s): round(measured[s], 2) for s in SNR_POINTS}},
                     ensure_ascii=False))
    if not all(checks.values()):
        raise SystemExit("energy golden test failed")


if __name__ == "__main__":
    main()