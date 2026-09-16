"""SENS-SNR-REBUILD-06 Stage F: production regression orchestrator.

Runs the frozen P1-P4 / T2 / T6 checks and the module selfchecks against the B64 production
config, applies explicit PASS/FAIL criteria (including the B64 physics contract and the Stage
C/D/E calibration artifacts), and writes reports/f01e/snr_rebuild_06/regression_checks.json.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
OUT = ROOT / "reports/f01e/snr_rebuild_06/regression_checks.json"

SCRIPTS = [
    ("selfchecks", "run_selfchecks.py", None),
    ("shared_echo", "check_shared_echo.py", "reports/f01e/p1_shared_echo/summary.json"),
    ("no_gt", "check_no_gt.py", "reports/f01e/t2_no_gt/summary.json"),
    ("single_target", "check_single_target.py", "reports/f01e/p2_detector/summary.json"),
    ("association", "check_association.py", "reports/f01e/p3_association/summary.json"),
    ("tracker", "check_tracker.py", "reports/f01e/p4_tracker/summary.json"),
    ("bs_ablation", "check_bs_ablation.py", None),
]


def read(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def main() -> None:
    started = time.time()
    results = {}
    for name, script, summary_path in SCRIPTS:
        print(f"running {name} ...", flush=True)
        completed = subprocess.run([sys.executable, str(HERE / script)], cwd=str(ROOT),
                                   capture_output=True, text=True, timeout=7200)
        entry = {"returncode": completed.returncode, "stdout_tail": completed.stdout[-600:],
                 "stderr_tail": completed.stderr[-600:]}
        if summary_path:
            summary = read(ROOT / summary_path)
            checks_file = read((ROOT / summary_path).parent / "checks.json")
            passed = summary.get("passed") if summary else None
            if passed is None and isinstance(summary.get("checks"), dict):
                passed = all(value.get("passed") for value in summary["checks"].values()
                             if isinstance(value, dict))
            if passed is None and checks_file is not None:
                passed = checks_file.get("passed")
            entry["passed"] = bool(passed)
            failed_source = (summary or {}).get("checks") or (checks_file or {}).get("checks") or {}
            entry["failed_checks"] = [key for key, value in failed_source.items()
                                      if isinstance(value, dict) and not value.get("passed")]
        elif name == "selfchecks":
            try:
                payload = json.loads(completed.stdout[completed.stdout.index("{"):])
                entry["passed"] = all(item.get("passed") for item in payload.values())
                entry["failed_modules"] = [key for key, item in payload.items() if not item.get("passed")]
            except (ValueError, KeyError):
                entry["passed"] = False
        elif name == "bs_ablation":
            t6a = read(ROOT / "reports/f01e/t6a_bs_ablation/summary.json")
            t6b = read(ROOT / "reports/f01e/t6b_bs_tracker/summary.json")
            entry["t6a"] = t6a["results"] if t6a else None
            entry["t6b"] = {key: {k: value[k] for k in ("track_position_rmse_m", "continuity_after_warmup",
                                                        "id_switches")}
                            for key, value in (t6b["results"].items() if t6b else {})}
            if t6a and t6b:
                rmse_1 = t6a["results"]["1bs"]["mean_position_rmse_m"]
                rmse_3 = t6a["results"]["3bs"]["mean_position_rmse_m"]
                cont_1 = t6b["results"]["1bs"]["continuity_after_warmup"]
                cont_3 = t6b["results"]["3bs"]["continuity_after_warmup"]
                entry["criteria"] = {"t6a_3bs_rmse_not_worse_than_1bs": rmse_3 <= rmse_1 * 1.05,
                                     "t6b_3bs_continuity_not_worse_than_1bs": cont_3 >= cont_1 - 0.05,
                                     "t6a_rmse_1bs": rmse_1, "t6a_rmse_3bs": rmse_3,
                                     "t6b_continuity_1bs": cont_1, "t6b_continuity_3bs": cont_3}
                passed_criteria = {key: value for key, value in entry["criteria"].items()
                                   if isinstance(value, bool)}
                entry["passed"] = all(passed_criteria.values())
            else:
                entry["passed"] = False
        results[name] = entry

    config = read(ROOT / "configs/shared_frontend.json")
    resource = config["sensing_resource"]
    physics = {
        "mode_contiguous_burst": resource["mode"] == "contiguous_burst",
        "active_symbols_64": int(resource["active_symbols"]) == 64,
        "start_symbol_96": int(resource["start_symbol"]) == 96,
        "active_burst_inside_block": int(resource["start_symbol"]) + int(resource["active_symbols"])
        <= int(resource["total_symbols"]),
        "time_occupancy_25pct": abs(resource["active_symbols"] / resource["total_symbols"] - 0.25) < 1e-12,
    }
    from frontend.sensing.waveform import PaperWaveform, SensingResource

    waveform = PaperWaveform()
    sensing = SensingResource.from_config(config)
    theory = {
        "unambiguous_velocity_mps": waveform.unambiguous_velocity,
        "physical_doppler_resolution_mps": waveform.c / (2 * waveform.fc * sensing.active_symbols
                                                          * waveform.T),
        "velocity_grid_spacing_mps": waveform.c / (2 * waveform.fc * 2 * sensing.active_symbols
                                                    * waveform.T),
        "range_resolution_m": waveform.range_resolution,
        "range_grid_spacing_m": waveform.unambiguous_range / (2 * waveform.K),
        "unambiguous_range_m": waveform.unambiguous_range,
        "observation_span_s": (sensing.active_symbols - 1) * waveform.T,
        "coherent_gain_db": 10 * math.log10(4 * waveform.K * sensing.active_symbols / 9),
        "gain_drop_vs_full_db": 10 * math.log10(sensing.active_symbols / waveform.N),
    }
    physics["unambiguous_velocity_unchanged"] = abs(theory["unambiguous_velocity_mps"]
                                                    - 252.35055387205387) < 1e-6
    physics["gain_drop_minus_6p02db"] = abs(theory["gain_drop_vs_full_db"] + 6.020599913279624) < 1e-6
    physics["physical_doppler_resolution_7p886"] = abs(theory["physical_doppler_resolution_mps"]
                                                       - 7.885954808501683) < 1e-6

    cfar = read(ROOT / "reports/f01e/snr_rebuild_06/cfar_calibration.json")
    covariance = read(ROOT / "reports/f01e/snr_rebuild_06/covariance_calibration.json")
    tracker = read(ROOT / "reports/f01e/snr_rebuild_06/tracker_calibration.json")
    calibrations = {
        "cfar_multiplier_present": bool(cfar and cfar.get("chosen_multiplier") is not None
                                        and float(config["detector"]["cfar_threshold_multiplier"])
                                        == float(cfar["chosen_multiplier"])),
        "cfar_verification_fa_in_band": bool(cfar and 0.35 <=
                                             cfar["verification_false_alarms_per_bs_frame_at_chosen"] <= 0.65),
        "covariance_lut_wired_to_rebuild06": config["detector"]["covariance_lut"]
        == "reports/f01e/snr_rebuild_06/covariance_calibration.json",
        "covariance_true_pair_gate_pass": bool(covariance and covariance["pairwise_gate"]["pass_rate"] >= 0.99),
        "covariance_inflation": covariance["covariance_inflation"] if covariance else None,
        "tracker_q_a_present": bool(tracker and tracker.get("chosen_q_a_m2_s3") is not None
                                    and float(config["tracker"]["q_a_m2_s3"])
                                    == float(tracker["chosen_q_a_m2_s3"])),
    }

    checks = {"scripts": {name: entry.get("passed") for name, entry in results.items()},
              "physics": physics, "calibrations": calibrations}
    all_passed = all(checks["scripts"].values()) and all(physics.values()) and all(
        value for key, value in calibrations.items() if isinstance(value, bool))
    report = {
        "stage": "SENS-SNR-REBUILD-06 Stage F",
        "resource": resource, "physics_theory": theory,
        "checks": checks, "results": results, "passed": bool(all_passed),
        "runtime_s": time.time() - started,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")
    print(json.dumps({"regression": "PASS" if all_passed else "FAIL", "checks": checks["scripts"],
                      "physics": physics, "calibrations": calibrations,
                      "runtime_s": round(report["runtime_s"], 1)}, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()