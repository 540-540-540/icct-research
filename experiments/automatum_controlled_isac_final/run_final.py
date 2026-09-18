"""Route-B final execution orchestrator and freeze.

Phases: geometry -> calibration -> acceptance -> freeze. Test is never read; val is used for
verification only. The freeze phase writes the V2-FROZEN config and the freeze manifest.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.automatum_controlled_isac_final import common  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
REVISION = "AUTOMATUM-CONTROLLED-ISAC-V2-FROZEN"
BASELINE = "eaadbe6d826f9f6e83baf3caf3c01bf75ac093d1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(script: str) -> None:
    print(f"--- {script}")
    subprocess.run([sys.executable, str(SCRIPT_DIR / script)], check=True, cwd=ROOT)


def selected_geometry() -> dict:
    return json.loads((common.OUT_DIR / "geometry_search.json").read_text())[
        "selected_scene0_geometry"]


def selected_calibration() -> dict:
    return json.loads((common.OUT_DIR / "calibration_search.json").read_text())["selected"]


def freeze() -> int:
    config = common.load_config()
    geometry = selected_geometry()
    calibration = selected_calibration()
    final_metrics = json.loads((common.OUT_DIR / "final_metrics.json").read_text())
    if calibration["status"] != "FROZEN" or geometry["status"] != "FROZEN":
        raise SystemExit("cannot freeze: geometry/calibration not frozen")
    if not all(value["pass"] for value in final_metrics["bands"].values()):
        raise SystemExit("cannot freeze: target bands are not all in range")
    if not all(value["strictly_increasing_as_snr_drops"]
               for value in final_metrics["monotonicity"].values()):
        raise SystemExit("cannot freeze: five-level monotonicity fails")

    payload = common.config_with_geometry(config, geometry)
    payload.pop("scene0_geometry_status", None)
    payload["revision"] = REVISION
    payload["status"] = "FROZEN"
    payload["route"] = ("B: controlled multi-BS state sensing; detection and identity "
                        "association are assumed successful")
    payload["visibility"]["status"] = "FROZEN"
    payload["snr"]["status"] = "FROZEN"
    payload["snr"]["levels_db"] = [float(value) for value in common.LEVELS]
    payload["measurement"]["a"] = {key: float(value) for key, value in calibration["a"].items()}
    payload["measurement"]["sigma_rule"] = ("sigma_q(gamma) = sqrt(floor_q^2 + "
                                            "(a_q * 10^(-gamma/20))^2); k_q = 1 for all channels")
    payload["measurement"]["status"] = "FROZEN"
    payload["fusion"]["status"] = "FROZEN"
    payload["state_quality"]["status"] = ("DEPRECATED descriptive score; not used for "
                                          "calibration or evaluation")
    payload["freeze"] = {
        "baseline_head": BASELINE,
        "geometry_search": "reports/isac_final/geometry_search.json",
        "calibration_search": "reports/isac_final/calibration_search.json",
        "final_metrics": "reports/isac_final/final_metrics.json",
        "final_report": "reports/isac_final/final_report.md",
        "note": "frozen in the commit that contains this revision; see git log",
    }
    common.write_json(common.CONFIG_PATH, payload)

    evidence = {}
    for name in ("geometry_search.json", "geometry_final_audit.csv", "calibration_search.json",
                 "calibration_candidates.csv", "final_metrics.json", "final_metrics.csv",
                 "final_metrics_by_bs.csv", "final_speed_bins.csv", "final_stability.csv",
                 "final_report.md"):
        path = common.OUT_DIR / name
        if path.exists():
            evidence[f"reports/isac_final/{name}"] = sha256(path)
    manifest = {
        "revision": REVISION,
        "status": "FROZEN",
        "baseline_head": BASELINE,
        "config_sha256": sha256(common.CONFIG_PATH),
        "route_b": {
            "chain": ["GT [x,y,vx,vy]", "3 fixed BS", "per-BS range/bearing/radial-velocity truth",
                      "SNR-controlled measurement noise", "multi-BS position + velocity fusion",
                      "[x_hat,y_hat,vx_hat,vy_hat]"],
            "assumptions": ["target detection successful", "identity association successful",
                            "no missed detection / false alarm / CFAR in the formal chain",
                            "frame noise temporally independent", "no Kalman/temporal smoothing"],
            "model_state_dim": 4,
            "model_state_fields": ["x_hat", "y_hat", "vx_hat", "vy_hat"],
            "vehicle_id_role": "internal index only",
        },
        "geometry": {
            "scene_0": {"stations_xy_m": geometry["stations_xy_m"],
                        "boresights_deg": geometry["boresights_deg"],
                        "fov_half_angle_deg": geometry["fov_half_angle_deg"],
                        "change": "BS2 boresight 79.55183860023827 -> 64.55183860023827 deg; "
                                  "stations and FOV unchanged"},
            "scene_1": {"stations_xy_m": common.config_with_geometry(
                config)["scenes"][1]["stations_xy_m"],
                "boresights_deg": common.config_with_geometry(config)["scenes"][1]["boresights_deg"],
                "change": "unchanged"},
            "visibility": {"range_min_m": config["visibility"]["range_min_m"],
                           "range_max_m": config["visibility"]["range_max_m"],
                           "fov_half_angle_deg": geometry["fov_half_angle_deg"],
                           "height_difference_m": config["visibility"]["height_difference_m"]},
        },
        "snr": {"levels_db": list(common.LEVELS),
                "semantics": {"10.0": "Good", "0.0": "Medium", "-10.0": "Poor-but-usable"}},
        "measurement": {"sigma_rule": payload["measurement"]["sigma_rule"],
                        "a": payload["measurement"]["a"], "floors": payload["measurement"]["floors"],
                        "floor_source": payload["measurement"]["floor_source"],
                        "seed_rule": ("base standard normal keyed by "
                                      "(scene_id, frame, bs_id, vehicle_id, channel); SNR never "
                                      "enters the seed")},
        "fusion": {"velocity_prior_sigma_mps": config["fusion"]["velocity_prior_sigma_mps"],
                   "position_fusion": "covariance-weighted inverse-variance",
                   "velocity_fusion": "multi-BS radial velocities with 3D LOS direction"},
        "final_acceptance": {"bands": final_metrics["bands"],
                             "monotonicity": {key: value["strictly_increasing_as_snr_drops"]
                                              for key, value in final_metrics["monotonicity"].items()},
                             "trajectory_plots": final_metrics["trajectory_plots"]},
        "test_used": False,
        "evidence_sha256": evidence,
    }
    common.write_json(common.OUT_DIR / "freeze_manifest.json", manifest)
    print(f"FROZEN config written: {common.CONFIG_PATH}")
    print(json.dumps({"config_sha256": manifest["config_sha256"],
                      "evidence_files": len(evidence)}, indent=1))
    return 0


def main() -> int:
    started = time.time()
    phases = sys.argv[1:] or ["all"]
    if "all" in phases or "geometry" in phases:
        run("optimize_geometry.py")
    if "all" in phases or "calibrate" in phases:
        run("calibrate_final.py")
    if "all" in phases or "evaluate" in phases:
        run("evaluate_final.py")
    if "all" in phases or "freeze" in phases:
        freeze()
    print(f"total {time.time() - started:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())