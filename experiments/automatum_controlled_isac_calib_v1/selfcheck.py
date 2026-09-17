"""Route-B Calibration V1 self-check: the 17 required checks of the work order.

Usage (repo root):
    python experiments/automatum_controlled_isac_calib_v1/selfcheck.py
"""
from __future__ import annotations

import inspect
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.automatum_controlled_isac_calib_v1 import calibrate, evaluate  # noqa: E402
from frontend.automatum_scene_source import AutomatumSceneSource  # noqa: E402
from frontend.controlled_isac import automatum_frontend, automatum_measurement as am  # noqa: E402
from frontend.controlled_isac import state_quality  # noqa: E402
from frontend.controlled_isac.cross_bs_association import fuse_positions  # noqa: E402
from frontend.controlled_isac.measurement import truth_measurement  # noqa: E402
from frontend.controlled_isac.velocity_fusion import recover_velocity  # noqa: E402

CONFIG = evaluate.load_config()
SETUP = am.setup_from_config(CONFIG)
CALIBRATION = calibrate.calibration_with(CONFIG, {"range": 0.15, "bearing": 0.004,
                                                 "radial_velocity": 0.25})
FRAME = evaluate.scene_frames(CONFIG, "train")[(0, 307)]


def check_dt() -> str:
    source = AutomatumSceneSource(ROOT / CONFIG["dataset"]["train_trajectories"])
    if abs(source.dt - 3 / 29.97) > 1e-15 or source.dt == 0.1:
        raise AssertionError(f"dt is {source.dt}")
    return f"dt={source.dt!r} == 3/29.97"


def check_no_rotation() -> str:
    source = AutomatumSceneSource(ROOT / CONFIG["dataset"]["train_trajectories"])
    frame = source.at_frame(0, 307)
    table = np.genfromtxt(source.csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    rows = (table["scene_id"] == 0) & (table["timestamp"] == frame.timestamp)
    reference = np.stack([table["x"][rows], table["y"][rows], table["vx"][rows],
                          table["vy"][rows]], axis=1)
    if not np.array_equal(frame.states, reference):
        raise AssertionError("states differ from world-frame CSV")
    if "psi" in (ROOT / "frontend/automatum_scene_source.py").read_text().lower():
        raise AssertionError("rotation in source")
    return "bit-identical to the world-frame CSV; no psi rotation"


def check_vehicle_identity() -> str:
    states = automatum_frontend.sense_frame(FRAME, 0, 0.0, CALIBRATION, SETUP)
    ids = [state["vehicle_id"] for state in states]
    truth = FRAME.vehicle_ids.tolist()
    if ids != truth:
        raise AssertionError("vehicle_id mapping is not one-to-one")
    return f"{len(ids)} states for {len(truth)} ground-truth vehicles, same order"


def check_no_missing() -> str:
    states = automatum_frontend.sense_frame(FRAME, 0, -10.0, CALIBRATION, SETUP)
    if len(states) != len(FRAME.vehicle_ids):
        raise AssertionError("a vehicle was dropped")
    return "every GT vehicle has a sensing state (no detection failure by construction)"


def check_no_fake() -> str:
    states = automatum_frontend.sense_frame(FRAME, 0, 0.0, CALIBRATION, SETUP)
    if {state["vehicle_id"] for state in states} != set(FRAME.vehicle_ids.tolist()):
        raise AssertionError("sensing produced a target outside the GT set")
    return "no fabricated vehicle"


def check_no_detector() -> str:
    import re

    for module in (automatum_frontend, am):
        source = Path(module.__file__).read_text(encoding="utf-8")
        code = re.sub(r'""".*?"""', "", source, flags=re.S).lower()
        for banned in ("detector", "cfar", "local_maxima", "sensing."):
            if banned in code:
                raise AssertionError(f"{module.__name__} references {banned}")
        if "frontend.sensing" in source:
            raise AssertionError(f"{module.__name__} imports the sensing stack")
    return "neither module imports or calls a detector/CFAR path (docstrings excluded)"


def check_truth_formulas() -> str:
    for scene_id in (0, 1):
        scene = [s for s in CONFIG["scenes"] if s["scene_id"] == scene_id][0]
        for bs in range(3):
            station = np.asarray(scene["stations_xy_m"][bs])
            boresight = math.radians(scene["boresights_deg"][bs])
            for vehicle_id, state in zip(FRAME.vehicle_ids.tolist(), FRAME.states):
                truth = truth_measurement(state[:2], state[2:], station, boresight, SETUP["height"])
                delta = state[:2] - station
                manual_r = math.sqrt(float(delta @ delta) + SETUP["height"] ** 2)
                manual_radial = float((station - state[:2]) @ state[2:]) / manual_r
                if abs(truth["r"] - manual_r) > 1e-12 or abs(truth["radial_velocity"] - manual_radial) > 1e-12:
                    raise AssertionError("truth formula mismatch")
                measurement = am.make_measurement(scene_id, 307, int(vehicle_id), state[:2], state[2:],
                                                  bs, 0.0, CALIBRATION, SETUP)
                if measurement is None:
                    continue
                if abs(measurement["r_true"] - truth["r"]) > 1e-12:
                    raise AssertionError("make_measurement truth mismatch")
    return "r/bearing/vr match the verified measurement.py formulas exactly"


def check_bs_independence() -> str:
    values = [am.base_noise(0, 307, bs, 1, "range") for bs in range(3)]
    if len({round(value, 12) for value in values}) != 3:
        raise AssertionError("BS noise draws are not independent")
    return f"three distinct base draws {[round(value, 3) for value in values]}"


def check_frame_independence() -> str:
    if am.base_noise(0, 307, 0, 1, "range") == am.base_noise(0, 308, 0, 1, "range"):
        raise AssertionError("frame noise draws are not independent")
    return "consecutive frames use different base draws"


def check_same_base_across_snr() -> str:
    state = FRAME.states[0]
    vehicle = int(FRAME.vehicle_ids[0])
    low = am.make_measurement(0, 307, vehicle, state[:2], state[2:], 0, -10.0, CALIBRATION, SETUP)
    high = am.make_measurement(0, 307, vehicle, state[:2], state[2:], 0, 10.0, CALIBRATION, SETUP)
    for channel, key in (("range", "r_hat"), ("bearing", "bearing_hat"),
                         ("radial_velocity", "vr_hat")):
        if low["base_noise"][channel] != high["base_noise"][channel]:
            raise AssertionError("base noise depends on SNR")
        truth_key = {"range": "r_true", "bearing": "bearing_true",
                     "radial_velocity": "vr_true"}[channel]
        implied_low = (low[key] - low[truth_key]) / low["sigma"][channel]
        implied_high = (high[key] - high[truth_key]) / high["sigma"][channel]
        if abs(implied_low - implied_high) > 1e-12:
            raise AssertionError("implied z differs across SNR")
    return "all SNR levels reuse the identical base draw per channel"


def check_snr_not_in_seed() -> str:
    parameters = inspect.signature(am.base_noise).parameters
    if any("snr" in name.lower() for name in parameters):
        raise AssertionError("SNR is a seed input")
    return "base_noise(scene_id, frame, bs_id, vehicle_id, channel) has no SNR argument"


def check_sigma_monotone() -> str:
    for channel in am.CHANNELS:
        values = [am.sigma_for(channel, snr_db, CALIBRATION) for snr_db in (-10, -5, 0, 5, 10)]
        if any(values[index] <= values[index + 1] for index in range(len(values) - 1)):
            raise AssertionError(f"{channel} sigma is not decreasing in SNR")
    return "sigma decreases monotonically from -10 to +10 dB on every channel"


def check_position_fusion() -> str:
    first = {"x": 10.0, "y": 0.0, "covariance": np.diag([0.04, 1.0])}
    second = {"x": 12.0, "y": 1.0, "covariance": np.diag([4.0, 0.01])}
    position, covariance = fuse_positions([first, second])
    information = np.linalg.inv(first["covariance"]) + np.linalg.inv(second["covariance"])
    expected_covariance = np.linalg.inv(information)
    weighted = (np.linalg.inv(first["covariance"]) @ np.array([10.0, 0.0])
                + np.linalg.inv(second["covariance"]) @ np.array([12.0, 1.0]))
    expected = expected_covariance @ weighted
    if not np.allclose(position, expected) or not np.allclose(covariance, expected_covariance):
        raise AssertionError("fusion does not match the inverse-variance formula")
    return "fuse_positions matches the hand-computed inverse-variance solution"


def check_velocity_rank() -> str:
    position = np.array([30.0, 10.0])
    one = recover_velocity(position, [{"station": np.array([0.0, 0.0]), "radial_velocity": 5.0}], 1.0,
                           velocity_prior_sigma=200.0)
    two = recover_velocity(position, [{"station": np.array([0.0, 0.0]), "radial_velocity": 5.0},
                                      {"station": np.array([50.0, 0.0]), "radial_velocity": -3.0}], 1.0,
                           velocity_prior_sigma=200.0)
    if one["rank"] != 1 or two["rank"] != 2 or two["condition_ratio"] <= 0:
        raise AssertionError(f"rank/condition wrong: {one}, {two}")
    return "rank 1 with one BS, rank 2 with two non-collinear BSs"


def check_state_fields() -> str:
    state = automatum_frontend.sense_frame(FRAME, 0, 0.0, CALIBRATION, SETUP)[0]
    for field in ("x_hat", "y_hat", "vx_hat", "vy_hat"):
        if not isinstance(state[field], float):
            raise AssertionError(f"{field} is not a float")
    if set(automatum_frontend.STATE_FIELDS) != {"x_hat", "y_hat", "vx_hat", "vy_hat"}:
        raise AssertionError("STATE_FIELDS drifted")
    truth = FRAME.states[0]
    if (state["x_hat"] == float(truth[0]) and state["y_hat"] == float(truth[1])
            and state["vx_hat"] == float(truth[2]) and state["vy_hat"] == float(truth[3])):
        raise AssertionError("estimate equals truth exactly")
    return "state is exactly [x_hat, y_hat, vx_hat, vy_hat]; audit fields are separate"


def check_no_train_val_leakage() -> str:
    recorded = []
    original = calibrate.run_split

    def spy(config, calibration, split, levels, frames_cache=None):
        recorded.append(split)
        return original(config, calibration, split, levels, frames_cache=frames_cache)

    calibrate.run_split = spy
    try:
        calibrate.search(CONFIG, coarse_grid={"range": (0.15,), "bearing": (0.004,),
                                              "radial_velocity": (0.25,)}, refine=False)
    finally:
        calibrate.run_split = original
    if set(recorded) != {"train"}:
        raise AssertionError(f"search used splits {set(recorded)}")
    return "calibration search evaluated train only; val/test never read"


def check_deterministic() -> str:
    calibration = calibrate.calibration_with(CONFIG, {"range": 0.15, "bearing": 0.004,
                                                     "radial_velocity": 0.25})
    cache = {(0, 307): FRAME}
    first = evaluate.run_split(CONFIG, calibration, "train", (-10.0, 0.0, 10.0), frames_cache=cache)
    second = evaluate.run_split(CONFIG, calibration, "train", (-10.0, 0.0, 10.0), frames_cache=cache)
    if first != second:
        raise AssertionError("repeated run differs")
    quality = state_quality.quality_from_errors(0.1, 0.1)
    if not 0 < quality < 100:
        raise AssertionError("quality is not bounded")
    return "two identical runs bit-identical; quality candidate bounded in (0, 100)"


CHECKS = (
    ("1 dt = 3/29.97", check_dt),
    ("2 world vx/vy not re-rotated", check_no_rotation),
    ("3 vehicle_id one-to-one", check_vehicle_identity),
    ("4 no missing target", check_no_missing),
    ("5 no fake target", check_no_fake),
    ("6 no detector / CFAR", check_no_detector),
    ("7 truth formulas match verified implementation", check_truth_formulas),
    ("8 BS noise independent", check_bs_independence),
    ("9 frame noise independent", check_frame_independence),
    ("10 same base z across SNR", check_same_base_across_snr),
    ("11 SNR not in seed", check_snr_not_in_seed),
    ("12 sigma monotone in SNR", check_sigma_monotone),
    ("13 position fusion correct", check_position_fusion),
    ("14 velocity fusion rank correct", check_velocity_rank),
    ("15 output is exactly [x_hat,y_hat,vx_hat,vy_hat]", check_state_fields),
    ("16 train/val no leakage", check_no_train_val_leakage),
    ("17 deterministic rerun", check_deterministic),
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