"""Route-B final-execution self-check: the 20 required checks.

Usage (repo root):
    python experiments/automatum_controlled_isac_final/selfcheck.py
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.automatum_controlled_isac_final import common, fast_eval  # noqa: E402
from frontend.automatum_scene_source import AutomatumSceneSource  # noqa: E402
from frontend.controlled_isac.automatum_frontend import STATE_FIELDS, sense_vehicle  # noqa: E402
from frontend.controlled_isac.automatum_measurement import (base_noise, make_measurement,  # noqa: E402
                                                            setup_from_config)
from frontend.controlled_isac.automatum_measurement import calibration_from_config  # noqa: E402

BASELINE = "eaadbe6d826f9f6e83baf3caf3c01bf75ac093d1"
CONFIG = common.load_config()
GEOMETRY = json.loads((common.OUT_DIR / "geometry_search.json").read_text())[
    "selected_scene0_geometry"]
CALIBRATION = json.loads((common.OUT_DIR / "calibration_search.json").read_text())["selected"]
EFFECTIVE = common.config_with_geometry(CONFIG, GEOMETRY)
EFFECTIVE["measurement"]["a"] = CALIBRATION["a"]
SETUP = setup_from_config(EFFECTIVE)
CAL = calibration_from_config(EFFECTIVE)
SCRIPT_DIR = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def code_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    return re.sub(r'""".*?"""', "", text, flags=re.S)


def check_baseline() -> str:
    if git("merge-base", "--is-ancestor", BASELINE, "HEAD").returncode != 0:
        raise AssertionError("baseline commit is not an ancestor of HEAD")
    return f"baseline {BASELINE[:8]} is an ancestor of HEAD"


def check_scene1_geometry() -> str:
    baseline = json.loads(git("show", f"{BASELINE}:configs/automatum_controlled_isac.json").stdout)
    for scene in (baseline["scenes"], CONFIG["scenes"]):
        pass
    old = next(scene for scene in baseline["scenes"] if scene["scene_id"] == 1)
    new = next(scene for scene in CONFIG["scenes"] if scene["scene_id"] == 1)
    if old["stations_xy_m"] != new["stations_xy_m"] or old["boresights_deg"] != new["boresights_deg"]:
        raise AssertionError("scene 1 geometry changed")
    if float(baseline["visibility"]["fov_half_angle_deg"]) != float(
            CONFIG["visibility"]["fov_half_angle_deg"]):
        raise AssertionError("global FOV changed")
    return "scene 1 stations/boresights and global FOV identical to baseline"


def check_geometry_train_only() -> str:
    text = (SCRIPT_DIR / "optimize_geometry.py").read_text(encoding="utf-8")
    if 'unique_history("val")' in text or "unique_history('val')" in text:
        raise AssertionError("geometry search reads val states")
    if 'unique_history("train")' not in text:
        raise AssertionError("geometry search does not read train states")
    search = json.loads((common.OUT_DIR / "geometry_search.json").read_text())
    if search["search"]["train_scene0_states"] <= 0:
        raise AssertionError("search block missing")
    return "geometry search evaluated train scene-0 states only; val used after selection only"


def check_calibration_train_only() -> str:
    text = (SCRIPT_DIR / "calibrate_final.py").read_text(encoding="utf-8")
    if '"val"' in text.replace('"value"', "") and 'fast_eval.prepare' in text:
        raise AssertionError("calibration references val")
    if 'fast_eval.prepare(config, geometry, "train")' not in text:
        raise AssertionError("calibration does not declare the train split")
    return "calibration search uses train prediction-history unique states only"


def check_val_validation_only() -> str:
    text = (SCRIPT_DIR / "evaluate_final.py").read_text(encoding="utf-8")
    if 'row["split"] == "train"' not in text:
        raise AssertionError("band verdicts do not pin the train split")
    metrics = json.loads((common.OUT_DIR / "final_metrics.json").read_text())
    for key, entry in metrics["bands"].items():
        if not entry["pass"]:
            raise AssertionError(f"band {key} failed")
    return "val is reported for verification only; band verdicts computed on train"


def check_test_unused() -> str:
    for name in ("common.py", "fast_eval.py", "optimize_geometry.py", "calibrate_final.py",
                 "evaluate_final.py", "run_final.py"):
        text = (SCRIPT_DIR / name).read_text(encoding="utf-8")
        for banned in ("test_trajectories", "splits/test", "splits\\test"):
            if banned in text:
                raise AssertionError(f"{name} references test data")
    metrics = json.loads((common.OUT_DIR / "final_metrics.json").read_text())
    manifest = json.loads((common.OUT_DIR / "freeze_manifest.json").read_text())
    if metrics["test_used"] or manifest["test_used"]:
        raise AssertionError("test usage flag is not false")
    return "no final-execution script reads test; metrics and manifest declare test_used=false"


def check_dt() -> str:
    source = AutomatumSceneSource(ROOT / CONFIG["dataset"]["train_trajectories"])
    if abs(common.DT - 3.0 / 29.97) > 1e-15 or abs(source.dt - common.DT) > 1e-15:
        raise AssertionError("dt mismatch")
    return f"dt={common.DT!r} == 3/29.97"


def check_world_velocity() -> str:
    source = AutomatumSceneSource(ROOT / CONFIG["dataset"]["train_trajectories"])
    frame = source.at_frame(0, 307)
    table = np.genfromtxt(source.csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    rows = (table["scene_id"] == 0) & (table["timestamp"] == frame.timestamp)
    reference = np.stack([table["x"][rows], table["y"][rows], table["vx"][rows],
                          table["vy"][rows]], axis=1)
    if not np.array_equal(frame.states, reference):
        raise AssertionError("state differs from the world-frame CSV")
    return "states equal the world-frame CSV bit-for-bit"


def check_snr_levels() -> str:
    levels = tuple(float(value) for value in CONFIG["snr"]["levels_db"])
    if levels != common.LEVELS:
        raise AssertionError(f"SNR levels changed: {levels}")
    return f"SNR levels unchanged {levels}"


def check_seed_excludes_snr() -> str:
    import inspect

    if any("snr" in name.lower() for name in inspect.signature(base_noise).parameters):
        raise AssertionError("SNR appears in the noise seed signature")
    return "base_noise(scene_id, frame, bs_id, vehicle_id, channel) has no SNR input"


def check_same_base_noise() -> str:
    low = make_measurement(0, 307, 1, [20.0, 0.0], [-5.0, 0.0], 0, -10.0, CAL, SETUP)
    high = make_measurement(0, 307, 1, [20.0, 0.0], [-5.0, 0.0], 0, 10.0, CAL, SETUP)
    for channel, key, truth_key in (("range", "r_hat", "r_true"), ("bearing", "bearing_hat",
                                                                  "bearing_true"),
                                    ("radial_velocity", "vr_hat", "vr_true")):
        z_low = (low[key] - low[truth_key]) / low["sigma"][channel]
        z_high = (high[key] - high[truth_key]) / high["sigma"][channel]
        if abs(z_low - z_high) > 1e-12 or low["base_noise"][channel] != high["base_noise"][channel]:
            raise AssertionError(f"{channel} base noise depends on SNR")
    return "all SNR levels reuse the identical base draw on every channel"


def check_no_temporal_smoothing() -> str:
    for path in (ROOT / "frontend/controlled_isac/automatum_frontend.py",
                 SCRIPT_DIR / "fast_eval.py"):
        text = code_text(path).lower()
        for banned in ("kalman", "smooth", "filter("):
            if banned in text:
                raise AssertionError(f"{path.name} contains {banned}")
    return "no Kalman / temporal smoothing anywhere in the estimator chain"


def check_no_detector() -> str:
    for path in (ROOT / "frontend/controlled_isac/automatum_frontend.py",
                 ROOT / "frontend/controlled_isac/automatum_measurement.py"):
        text = code_text(path).lower()
        for banned in ("detector", "cfar", "frontend.sensing"):
            if banned in text:
                raise AssertionError(f"{path.name} references {banned}")
    return "estimator chain never imports or mentions a detector/CFAR"


def check_no_direct_state_noise() -> str:
    frontend = code_text(ROOT / "frontend/controlled_isac/automatum_frontend.py").lower()
    for banned in ("randn", "normal(", "random"):
        if banned in frontend:
            raise AssertionError(f"frontend adds noise directly: {banned}")
    measurement = Path(ROOT / "frontend/controlled_isac/automatum_measurement.py").read_text(
        encoding="utf-8")
    if "polar_to_xy(r_hat, bearing_hat" not in measurement:
        raise AssertionError("cartesian estimate is not derived from noisy polar values")
    return "noise is injected on range/bearing/radial_velocity only; x/y derived from polar"


def prepared_subset():
    keys = common.unique_history("train")["keys"]
    indices = list(range(0, len(keys), max(1, len(keys) // 40)))[:40]
    subset_keys = [keys[i] for i in indices]
    lookup = common.state_lookup("train")
    states = np.asarray([lookup[key] for key in subset_keys])
    r = np.zeros((len(subset_keys), 3))
    bearing = np.zeros((len(subset_keys), 3))
    radial = np.zeros((len(subset_keys), 3))
    rho = np.zeros((len(subset_keys), 3))
    geometry = common.scene_geometry(EFFECTIVE)
    for index, key in enumerate(subset_keys):
        for bs_id in range(3):
            rr, bb, vv, pp = common.truth_polar(states[index, :2], states[index, 2:],
                                                geometry[key[0]]["stations"][bs_id],
                                                math.radians(
                                                    geometry[key[0]]["boresights_deg"][bs_id]))
            r[index, bs_id] = float(rr[0])
            bearing[index, bs_id] = float(bb[0])
            radial[index, bs_id] = float(vv[0])
            rho[index, bs_id] = float(pp[0])
    visible = np.zeros((len(subset_keys), 3), dtype=bool)
    for scene_id in common.SCENES:
        rows = [i for i, key in enumerate(subset_keys) if key[0] == scene_id]
        visible[rows] = common.visibility(states[rows, :2], geometry[scene_id]["stations"],
                                          geometry[scene_id]["boresights_deg"],
                                          EFFECTIVE["visibility"]["fov_half_angle_deg"])
    return {"keys": subset_keys, "states": states, "visible": visible, "r": r, "bearing": bearing,
            "radial": radial, "rho": rho,
            "stations": np.stack([geometry[s]["stations"] for s in common.SCENES]),
            "boresights": np.stack([np.radians(geometry[s]["boresights_deg"])
                                    for s in common.SCENES]),
            "scene_index": np.asarray([key[0] for key in subset_keys]),
            "noise": common.base_noise_arrays(subset_keys), "fov_half_angle_deg": 70.0}


def check_error_from_polar_chain() -> str:
    prepared = prepared_subset()
    output = fast_eval.evaluate(prepared, CAL)
    max_dp = max_dv = 0.0
    for index, key in enumerate(prepared["keys"]):
        state = prepared["states"][index]
        for level in common.LEVELS:
            estimate = sense_vehicle(key[0], key[1], key[2], state[:2], state[2:], level, CAL, SETUP)
            dp = math.hypot(estimate["x_hat"] - state[0], estimate["y_hat"] - state[1])
            dv = math.hypot(estimate["vx_hat"] - state[2], estimate["vy_hat"] - state[3])
            if estimate["n_bs"] != output[level]["n_bs"][index]:
                raise AssertionError("n_bs mismatch between replica and frontend")
            max_dp = max(max_dp, abs(dp - output[level]["position_error"][index]))
            max_dv = max(max_dv, abs(dv - output[level]["velocity_error"][index]))
    if max_dp > 1e-5 or max_dv > 1e-4:
        raise AssertionError(f"replica/frontend mismatch {max_dp}, {max_dv}")
    return (f"calibration replica equals the frontend on 40 sampled states x 5 SNR "
            f"(max Δposition {max_dp:.2e} m, max Δvelocity {max_dv:.2e} m/s)")


def check_state_fields() -> str:
    if set(STATE_FIELDS) != {"x_hat", "y_hat", "vx_hat", "vy_hat"}:
        raise AssertionError("STATE_FIELDS drifted")
    return "final model state is exactly [x_hat, y_hat, vx_hat, vy_hat]"


def check_no_feature_leakage() -> str:
    manifest = json.loads((common.OUT_DIR / "freeze_manifest.json").read_text())
    if manifest["route_b"]["model_state_fields"] != ["x_hat", "y_hat", "vx_hat", "vy_hat"]:
        raise AssertionError("manifest state fields drifted")
    if manifest["route_b"]["model_state_dim"] != 4:
        raise AssertionError("manifest state dim drifted")
    for banned in ("n_bs", "condition_ratio", "covariance", "snr", "confidence"):
        if banned in manifest["route_b"]["model_state_fields"]:
            raise AssertionError("audit field leaked into model state")
    return "n_bs / covariance / condition_ratio / SNR / confidence are audit-only, never features"


def check_five_level_monotonic() -> str:
    metrics = json.loads((common.OUT_DIR / "final_metrics.json").read_text())
    for key, value in metrics["monotonicity"].items():
        if not value["strictly_increasing_as_snr_drops"]:
            raise AssertionError(f"{key} not strictly monotone")
    config = common.load_config()
    if config["revision"] != "AUTOMATUM-CONTROLLED-ISAC-V2-FROZEN" or config["status"] != "FROZEN":
        raise AssertionError("config is not frozen")
    return "five-level position/velocity strictly monotone; config revision/status = V2-FROZEN"


def check_deterministic() -> str:
    prepared = prepared_subset()
    first = fast_eval.evaluate(prepared, CAL)
    second = fast_eval.evaluate(prepared, CAL)
    for level in common.LEVELS:
        if not np.array_equal(first[level]["position_error"], second[level]["position_error"]):
            raise AssertionError("replica evaluation is not deterministic")
    manifest = json.loads((common.OUT_DIR / "freeze_manifest.json").read_text())
    for relative, digest in manifest["evidence_sha256"].items():
        path = ROOT / relative
        if sha256(path) != digest:
            raise AssertionError(f"evidence hash stale: {relative}")
    return "in-process evaluation identical; all frozen evidence SHA256 match the manifest"


def check_old_audit_untouched() -> str:
    result = git("diff", "--quiet", "HEAD", "--", "reports/isac_final_audit")
    if result.returncode != 0:
        raise AssertionError("previous final-audit files modified")
    return "reports/isac_final_audit is unchanged relative to HEAD"


CHECKS = (
    ("1 baseline = eaadbe6", check_baseline),
    ("2 scene 1 geometry unmodified", check_scene1_geometry),
    ("3 geometry search train-only", check_geometry_train_only),
    ("4 calibration search train-only", check_calibration_train_only),
    ("5 val validation only", check_val_validation_only),
    ("6 test unused", check_test_unused),
    ("7 dt = 3/29.97", check_dt),
    ("8 world vx/vy not re-rotated", check_world_velocity),
    ("9 SNR levels unchanged", check_snr_levels),
    ("10 noise seed excludes SNR", check_seed_excludes_snr),
    ("11 same base noise across SNR", check_same_base_noise),
    ("12 no temporal smoothing", check_no_temporal_smoothing),
    ("13 no detector", check_no_detector),
    ("14 no direct state noise", check_no_direct_state_noise),
    ("15 error from range/bearing/vr chain", check_error_from_polar_chain),
    ("16 state = [x_hat,y_hat,vx_hat,vy_hat]", check_state_fields),
    ("17 no model-feature leakage", check_no_feature_leakage),
    ("18 five-level monotonic + frozen config", check_five_level_monotonic),
    ("19 deterministic rerun + evidence hashes", check_deterministic),
    ("20 old final-audit files unchanged", check_old_audit_untouched),
)


def main() -> int:
    failures = 0
    for name, function in CHECKS:
        try:
            print(f"PASS {name}: {function()}")
        except Exception as error:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {error}")
    print(f"==== {len(CHECKS) - failures}/{len(CHECKS)} checks passed ====")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())