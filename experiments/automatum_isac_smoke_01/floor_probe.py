"""SMOKE-01 supporting probe: deterministic single-target error floor vs noise level.

Explains why the SNR-to-error gradient of the smoke is compressed: the per-target RD
processing gain is ~+44.6 dB, so even -10 dB frame-level SNR leaves a large per-target
peak margin, and the residual error approaches the sub-bin interpolation floor around
+15 dB. C-domain evaluation only; the probe never feeds anything back to the detector.

Usage (repo root):
    python experiments/automatum_isac_smoke_01/floor_probe.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from experiments.automatum_isac_smoke_01.evaluate import pair_errors, truth_targets  # noqa: E402
from frontend.sensing.detector import detect_anonymous_measurements  # noqa: E402
from frontend.sensing.simulator import scale_shared_noise, synthesize_shared_frame_snr  # noqa: E402
from frontend.sensing.waveform import ArrayConfig, PaperWaveform, SensingResource  # noqa: E402

CONFIG = json.loads((ROOT / "configs/automatum_isac_smoke.json").read_text())
OUT = ROOT / "reports/isac_smoke/automatum_isac_smoke_01/floor_probe.json"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
WAVEFORM = PaperWaveform(**CONFIG["waveform"])
ARRAY = ArrayConfig(**CONFIG["array"])
RESOURCE = SensingResource.from_config(CONFIG)
VISIBILITY = CONFIG["visibility"]
PROBES = ((25.0, 0.0, 8.0, 180.0), (35.0, 12.0, 12.0, 200.0), (48.0, -20.0, 15.0, 160.0),
          (60.0, 30.0, 10.0, 210.0), (75.0, -40.0, 18.0, 140.0), (90.0, 55.0, 20.0, 235.0),
          (110.0, -60.0, 22.0, 120.0), (18.0, 45.0, 6.0, 225.0), (42.0, -8.0, 14.0, 170.0),
          (55.0, 5.0, -9.0, 0.0), (68.0, -52.0, 16.0, 130.0), (130.0, 65.0, 24.0, 245.0))
LEVELS_DB = (15.0, 30.0, 60.0)


def main() -> int:
    station = np.array([0.0, 0.0])
    stations = np.array([station, [10000.0, 10000.0], [-10000.0, 10000.0]])
    rows = []
    for index, (range_m, bearing_deg, speed, heading_deg) in enumerate(PROBES):
        rho = math.sqrt(range_m ** 2 - float(VISIBILITY["height_difference_m"]) ** 2)
        phi = math.radians(bearing_deg)
        states = np.array([[rho * math.cos(phi), rho * math.sin(phi),
                            speed * math.cos(math.radians(heading_deg)),
                            speed * math.sin(math.radians(heading_deg))]])
        base = synthesize_shared_frame_snr(states[:, :2], states[:, 2:], [index], stations,
                                           np.array([0.0, 0.0, 0.0]), WAVEFORM, ARRAY, [1.0],
                                           VISIBILITY, 0, index, snr_db=None, device=DEVICE)
        truth = truth_targets(states, station, 0.0, VISIBILITY)
        for snr_db in LEVELS_DB:
            noise = scale_shared_noise(base["S"][0][None], base["W0"][0][None], snr_db=snr_db)
            result = detect_anonymous_measurements(
                base["S"][0] + noise["W"][0], base["X"][0], 0, station, 0.0, WAVEFORM, ARRAY,
                CONFIG["detector"], VISIBILITY, resource=RESOURCE)
            measurement = min(result["measurements"],
                              key=lambda value: abs(value["range_hat"] - truth["r_m"][0]))
            errors = pair_errors(measurement, truth, 0)
            rows.append({"target_snr_db": snr_db, "range_gt_m": range_m, "bearing_gt_deg": bearing_deg,
                         "range_error_m": errors["range_error_m"],
                         "radial_velocity_error_mps": errors["radial_velocity_error_mps"],
                         "bearing_error_deg": errors["bearing_error_deg"],
                         "position_error_m": errors["position_error_m"]})
    summary = {}
    for snr_db in LEVELS_DB:
        subset = [row for row in rows if row["target_snr_db"] == snr_db]

        def rmse(key):
            return float(math.sqrt(np.mean([row[key] ** 2 for row in subset])))

        summary[str(int(snr_db))] = {"n": len(subset), "range_rmse_m": rmse("range_error_m"),
                                     "radial_velocity_rmse_mps": rmse("radial_velocity_error_mps"),
                                     "bearing_rmse_deg": rmse("bearing_error_deg"),
                                     "position_rmse_m": rmse("position_error_m")}
    payload = {"probe": "deterministic single-target floor", "levels_db": list(LEVELS_DB),
               "summary": summary, "rows": rows,
               "note": "clean-target detector bias/quantisation floor; C-domain diagnostic only"}
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())