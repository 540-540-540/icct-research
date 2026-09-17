"""SMOKE-01 unit/smoke checks (the 18 required checks of the work order).

Usage (repo root):
    python experiments/automatum_isac_smoke_01/selfcheck.py
"""
from __future__ import annotations

import inspect
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

import frontend.sensing.detector as detector_module  # noqa: E402
from frontend.automatum_scene_source import AutomatumSceneSource  # noqa: E402
from frontend.sensing import coords  # noqa: E402
from frontend.sensing.simulator import scale_shared_noise, synthesize_shared_frame_snr  # noqa: E402
from frontend.sensing.waveform import ArrayConfig, PaperWaveform, SensingResource  # noqa: E402

CONFIG = json.loads((ROOT / "configs/automatum_isac_smoke.json").read_text())
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
WAVEFORM = PaperWaveform(**CONFIG["waveform"])
ARRAY = ArrayConfig(**CONFIG["array"])
RESOURCE = SensingResource.from_config(CONFIG)
VISIBILITY = CONFIG["visibility"]
FORBIDDEN = re.compile(r"\b(vehicle_ids?|truth|ground_truth|source_keys|target_count|gt_count)\b")


def code_text(function) -> str:
    return re.sub(r'""".*?"""', "", inspect.getsource(function), flags=re.S)


def training_source() -> AutomatumSceneSource:
    return AutomatumSceneSource(ROOT / CONFIG["dataset"]["train_trajectories"],
                                source_hz=float(CONFIG["dataset"]["source_hz"]),
                                stride=int(CONFIG["dataset"]["stride"]))


def synth(states, keys, scene_id=0, frame=0, snr_db=None, stations=None, boresights=None):
    states = np.asarray(states, dtype=np.float64).reshape(-1, 4)
    if stations is None:
        stations = np.array([[0.0, 0.0], [10000.0, 10000.0], [-10000.0, 10000.0]])
    if boresights is None:
        boresights = np.array([0.0, 0.0, 0.0])
    return synthesize_shared_frame_snr(states[:, :2], states[:, 2:], list(keys), stations, boresights,
                                       WAVEFORM, ARRAY, [1.0] * len(states), VISIBILITY, scene_id, frame,
                                       snr_db=snr_db, device=DEVICE)


# 1. world vx/vy never rotated again
def check_no_rotation() -> str:
    source = training_source()
    for scene_id in source.scenes:
        frame = source.at_frame(scene_id, source.frame_bounds(scene_id)[0])
        table = np.genfromtxt(source.csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
        rows = (table["scene_id"] == scene_id) & (table["timestamp"] == frame.timestamp)
        reference = np.stack([table["x"][rows], table["y"][rows], table["vx"][rows], table["vy"][rows]],
                             axis=1)
        if not np.array_equal(frame.states, reference):
            raise AssertionError("scene-level states differ from the world-frame CSV values")
    text = (ROOT / "frontend/automatum_scene_source.py").read_text(encoding="utf-8")
    if "psi" in text.lower() or "np.cos" in text or "np.sin" in text:
        raise AssertionError("scene source contains a rotation")
    return "states equal the world-frame CSV vx/vy bit-for-bit; no psi rotation in the source"


# 2. dt = 3/29.97
def check_true_dt() -> str:
    source = training_source()
    if abs(source.dt - 3 / 29.97) > 1e-15 or source.dt == 0.1:
        raise AssertionError(f"dt wrong: {source.dt}")
    first, last = source.frame_bounds(source.scenes[0])
    step = source.at_frame(source.scenes[0], first + 1).timestamp - source.at_frame(source.scenes[0], first).timestamp
    if abs(step - 3 / 29.97) > 1e-9:
        raise AssertionError("frame timestamps are not spaced by the true dt")
    return f"dt={source.dt!r} == 3/29.97 and frame timestamps follow it"


# 3. scene/frame lookup correct
def check_lookup() -> str:
    source = training_source()
    table = np.genfromtxt(source.csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    for index in (0, 1234, len(table) - 1):
        scene_id, timestamp = int(table["scene_id"][index]), float(table["timestamp"][index])
        frame_id = int(round(timestamp * 29.97)) // 3
        frame = source.at_frame(scene_id, frame_id)
        if abs(frame.timestamp - timestamp) > 1e-12:
            raise AssertionError("frame timestamp lookup mismatch")
        if int(table["vehicle_id"][index]) not in frame.vehicle_ids.tolist():
            raise AssertionError("vehicle missing from its own canonical frame")
    return "round(timestamp*29.97)//3 reproduces the frame and its rows"


# 4. source returns every vehicle of the frame
def check_all_vehicles() -> str:
    source = training_source()
    table = np.genfromtxt(source.csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    checked = 0
    for index in range(0, len(table), 4001):
        scene_id, timestamp = int(table["scene_id"][index]), float(table["timestamp"][index])
        expected = int(np.count_nonzero((table["scene_id"] == scene_id) & (table["timestamp"] == timestamp)))
        frame = source.at_frame(scene_id, int(round(timestamp * 29.97)) // 3)
        if len(frame) != expected:
            raise AssertionError(f"frame returns {len(frame)} of {expected} vehicles")
        checked += 1
    return f"all rows of {checked} sampled frames are returned by the source"


# 5. prediction cohort does not enter the echo
def check_cohort_isolation() -> str:
    source_text = (ROOT / "frontend/automatum_scene_source.py").read_text(encoding="utf-8")
    if "np.load" in source_text:
        raise AssertionError("scene source reads a npz/cohort artifact")
    samples = np.load(ROOT / CONFIG["dataset"]["splits_dir"] / "train/samples.npz")
    source = training_source()
    rng = np.random.default_rng(2026)
    for frame_id in (307, 634, 5027, int(rng.integers(0, 5000))):
        frame = source.at_frame(0, frame_id)
        overlapping = (samples["scene_id"] == 0) & (samples["start_frame"] <= frame_id) \
            & (frame_id <= samples["start_frame"] + 39)
        cohorts = set()
        for sample_index in np.flatnonzero(overlapping):
            mask = samples["vehicle_mask"][sample_index]
            cohorts.update(samples["vehicle_ids"][sample_index][mask].tolist())
        extras = [int(v) for v in frame.vehicle_ids if int(v) not in cohorts]
        if extras:
            return (f"frame 0/{frame_id}: {len(frame)} vehicles, {len(extras)} of them outside every "
                    "overlapping prediction cohort")
    raise AssertionError("no frame found with vehicles outside the prediction cohort")


# 6. all three BS observe one identical physical frame
def check_single_frame_per_bs() -> str:
    source = training_source()
    frame = source.at_frame(0, 307)
    first = synth(frame.states, frame.vehicle_ids, scene_id=0, frame=307)
    second = synth(frame.states, frame.vehicle_ids, scene_id=0, frame=307)
    if not torch.equal(first["S"], second["S"]) or not torch.equal(first["W0"], second["W0"]):
        raise AssertionError("repeated synthesis of the same frame differs")
    signature = inspect.signature(synthesize_shared_frame_snr).parameters
    for banned in ("measurement_time", "measurement_time_ns", "station_offset", "extrapolate"):
        if banned in signature:
            raise AssertionError(f"simulator exposes a per-BS timing hook: {banned}")
    runner = (ROOT / "experiments/automatum_isac_smoke_01/run.py").read_text(encoding="utf-8")
    if "echo_source" in runner or "measurement_time_ns" in runner:
        raise AssertionError("runner still uses the legacy staggered-time adapter")
    return "one state vector drives all three BS; repeated synthesis is identical"


# 7. fixed RCS = 1 m^2
def check_fixed_rcs() -> str:
    if float(CONFIG["power"]["fixed_rcs_m2"]) != 1.0:
        raise AssertionError("config RCS is not 1.0 m^2")
    runner = (ROOT / "experiments/automatum_isac_smoke_01/run.py").read_text(encoding="utf-8")
    if 'config["power"]["fixed_rcs_m2"]' not in runner:
        raise AssertionError("runner does not read the fixed RCS")
    return "RCS is a single fixed 1.0 m^2 value for every vehicle"


# 8. 1/R^2 amplitude relationship
def check_amplitude_law() -> str:
    station = np.array([[0.0, 0.0], [10000.0, 10000.0], [-10000.0, 10000.0]])
    near, far = 20.0, 80.0
    states = np.array([[math.sqrt(near ** 2 - 25), 0.0, -5.0, 0.0],
                       [math.sqrt(far ** 2 - 25), 0.0, -5.0, 0.0]])
    result = synthesize_shared_frame_snr(states[:, :2], states[:, 2:], [1, 2], station,
                                         [0.0, 0.0, 0.0], WAVEFORM, ARRAY, [1.0, 1.0], VISIBILITY,
                                         0, 0, snr_db=None, device=DEVICE)
    alpha = result["alpha"].cpu().numpy()
    ratio = float(alpha[0, 0] / alpha[0, 1])
    expected = (far / near) ** 2
    if abs(ratio - expected) > 1e-9 * expected:
        raise AssertionError(f"amplitude ratio {ratio} != (R2/R1)^2 {expected}")
    return f"alpha(20m)/alpha(80m) = {ratio:.4f} = (80/20)^2"


# 9. all 256 communication symbols reused for sensing
def check_full_resource() -> str:
    if RESOURCE.mode != "full" or RESOURCE.active_symbols != WAVEFORM.N or RESOURCE.start_symbol != 0:
        raise AssertionError("sensing resource is not the full 256-symbol block")
    generator = torch.Generator(device=DEVICE).manual_seed(7)
    Y = torch.complex(torch.randn((16, 256, 256), generator=generator, device=DEVICE, dtype=torch.float64),
                      torch.randn((16, 256, 256), generator=generator, device=DEVICE, dtype=torch.float64))
    X = torch.polar(torch.ones((256, 256), dtype=torch.float64, device=DEVICE),
                    torch.zeros((256, 256), dtype=torch.float64, device=DEVICE))
    maps = detector_module.rd_spectrum(Y, X, WAVEFORM, ARRAY, resource=RESOURCE)
    if maps["nv"] != 2 * 256 or maps["samples"] != 256:
        raise AssertionError("Doppler grid does not integrate all 256 symbols")
    return "resource=full, nv=512 from 256 slow-time symbols, no burst slicing"


# 10. shared clean echo is the multi-target sum
def check_shared_sum() -> str:
    stations = np.array([[0.0, 0.0], [10000.0, 10000.0], [-10000.0, 10000.0]])
    first = np.array([[30.0, 0.0, -4.0, 1.0]])
    second = np.array([[-40.0, 10.0, 3.0, -2.0]])
    both = np.concatenate([first, second])
    echo_first = synth(first, [11], stations=stations)
    echo_second = synth(second, [22], stations=stations)
    echo_both = synth(both, [11, 22], stations=stations)
    if not torch.allclose(echo_both["S"], echo_first["S"] + echo_second["S"], atol=1e-12, rtol=1e-10):
        raise AssertionError("S_total is not the per-target sum")
    return "S_total == S_target1 + S_target2 (all three BS, float64 tolerance)"


# 11. noise is added once per BS/frame
def check_single_noise() -> str:
    result = synth([[30.0, 0.0, -4.0, 1.0]], [11], snr_db=5.0)
    if not torch.allclose(result["Y"] - result["S"], result["W"], atol=1e-12, rtol=0):
        raise AssertionError("Y - S != W")
    if not torch.equal(result["W"], result["W0"] * result["scale"][:, None, None, None]):
        raise AssertionError("W is not one scaled base realisation")
    for bs in range(3):
        for other in range(bs + 1, 3):
            if torch.equal(result["W"][bs], result["W"][other]):
                raise AssertionError("different BS share the same noise realisation")
    return "Y = S + W exactly; one realisation per BS; different BS use different seeds"


# 12. achieved SNR matches the target
def check_achieved_snr() -> str:
    source = training_source()
    frame = source.at_frame(0, 307)
    result = synth(frame.states, frame.vehicle_ids, scene_id=0, frame=307)
    worst = 0.0
    for snr_db in CONFIG["snr"]["sweep_db"]:
        noise = scale_shared_noise(result["S"], result["W0"], snr_db=float(snr_db))
        achieved = noise["achieved_snr_db"].cpu().numpy()
        for bs in range(3):
            if result["visible"][bs].any():
                worst = max(worst, abs(float(achieved[bs]) - float(snr_db)))
    if worst > float(CONFIG["snr"]["achieved_tolerance_db"]):
        raise AssertionError(f"achieved SNR error {worst} dB exceeds tolerance")
    return f"max |achieved - target| = {worst:.3e} dB over the whole sweep"


# 13. base noise decoupled from SNR
def check_noise_seed_independence() -> str:
    source = training_source()
    frame = source.at_frame(0, 307)
    low = synth(frame.states, frame.vehicle_ids, scene_id=0, frame=307, snr_db=-10.0)
    high = synth(frame.states, frame.vehicle_ids, scene_id=0, frame=307, snr_db=15.0)
    if not torch.equal(low["W0"], high["W0"]) or not torch.equal(low["X"], high["X"]):
        raise AssertionError("base noise or waveform depends on the SNR level")
    if torch.equal(low["W"], high["W"]):
        raise AssertionError("noise amplitude did not change with SNR")
    return "W0 and X are bit-identical at -10 and +15 dB; only the scale changes"


# 14. detector API cannot see identity
def check_detector_blind() -> str:
    for function in (detector_module.detect_anonymous_measurements, detector_module.extract_rd_peaks,
                     detector_module.aoa_angle_peaks, detector_module.rd_spectrum):
        parameters = list(inspect.signature(function).parameters)
        for name in parameters:
            if FORBIDDEN.search(name):
                raise AssertionError(f"{function.__name__} accepts forbidden parameter {name}")
        if FORBIDDEN.search(code_text(function)):
            raise AssertionError(f"{function.__name__} code references identity/truth")
    result = synth([[30.0, 0.0, -4.0, 1.0], [34.0, 2.0, 2.0, -1.0]], [1, 2], snr_db=0.0)
    first = detector_module.detect_anonymous_measurements(
        result["Y"][0], result["X"][0], 0, [0.0, 0.0], 0.0, WAVEFORM, ARRAY, CONFIG["detector"],
        VISIBILITY, resource=RESOURCE)
    second = detector_module.detect_anonymous_measurements(
        result["Y"][0], result["X"][0], 0, [0.0, 0.0], 0.0, WAVEFORM, ARRAY, CONFIG["detector"],
        VISIBILITY, resource=RESOURCE)
    if first["measurements"] != second["measurements"]:
        raise AssertionError("detector output is not a pure function of the echo")
    return "no identity/count parameter or accessor in the detector API; output is a pure echo function"


# 15. the Automatum path never calls CFAR
def check_no_cfar() -> str:
    original = detector_module.cfar_mask

    def forbidden(*args, **kwargs):
        raise AssertionError("CFAR mask called on the Automatum path")

    detector_module.cfar_mask = forbidden
    try:
        result = synth([[30.0, 0.0, -4.0, 1.0]], [1], snr_db=0.0)
        detector_module.detect_anonymous_measurements(
            result["Y"][0], result["X"][0], 0, [0.0, 0.0], 0.0, WAVEFORM, ARRAY, CONFIG["detector"],
            VISIBILITY, resource=RESOURCE)
    finally:
        detector_module.cfar_mask = original
    return "detect_anonymous_measurements works while cfar_mask is poisoned"


# 16. one RD peak can yield multiple angle peaks
def multi_target_probe():
    rho = math.sqrt(60.0 ** 2 - 25.0)
    states = []
    for bearing_deg in (-10.0, 10.0):
        phi = math.radians(bearing_deg)
        states.append([rho * math.cos(phi), rho * math.sin(phi),
                       -10.0 * math.cos(phi), -10.0 * math.sin(phi)])
    return synth(np.array(states), [1, 2], snr_db=0.0)


def check_multi_angle() -> str:
    result = multi_target_probe()
    detection = detector_module.detect_anonymous_measurements(
        result["Y"][0], result["X"][0], 0, [0.0, 0.0], 0.0, WAVEFORM, ARRAY, CONFIG["detector"],
        VISIBILITY, resource=RESOURCE)
    if not detection["rd_peaks"]:
        raise AssertionError("probe produced no RD peak")
    peak = max(detection["rd_peaks"], key=lambda value: value["peak_power"])
    cell = peak["grid_index"]
    angles = [measurement for measurement in detection["measurements"]
              if measurement["grid_index"][:2] == cell]
    if len(angles) < 2:
        raise AssertionError(f"multi-angle path returned {len(angles)} angle peaks")
    u_values = sorted(measurement["u_hat"] for measurement in angles)
    if not (u_values[0] < -0.1 and u_values[-1] > 0.1):
        raise AssertionError(f"angle peaks not split around boresight: {u_values}")
    i, j = cell
    snapshot = detection["spectrum"][:, i, j]
    legacy_u, _ = detector_module.aoa_estimate(snapshot, ARRAY)
    if not (abs(legacy_u) < 0.1 or len({round(value, 3) for value in u_values}) > 1):
        raise AssertionError("legacy path behaviour unclear")
    return (f"same RD peak -> {len(angles)} angle peaks at u={[round(v, 3) for v in u_values]}; "
            f"legacy single-angle path returns u={legacy_u:.3f}")


# 17. detector never reads a ground-truth target count
def check_no_gt_count() -> str:
    signature = inspect.signature(detector_module.detect_anonymous_measurements)
    for name in signature.parameters:
        if "count" in name.lower() or FORBIDDEN.search(name):
            raise AssertionError(f"detector accepts {name}")
    if FORBIDDEN.search(code_text(detector_module.detect_anonymous_measurements)):
        raise AssertionError("detector source references ground truth")
    return "no count/GT parameter and no forbidden identifier in the detector source"


# 18. the smoke is deterministic
def check_determinism() -> str:
    from experiments.automatum_isac_smoke_01.run import select_frames

    source = training_source()
    scene = CONFIG["scenes"][0]
    first, _, _ = select_frames(source, 0, scene["junction_center_xy_m"], 6, 36)
    second, _, _ = select_frames(source, 0, scene["junction_center_xy_m"], 6, 36)
    if first != second:
        raise AssertionError("frame selection is not deterministic")
    frame = source.at_frame(0, first[0])
    stations = np.asarray(scene["stations_xy_m"], dtype=np.float64)
    boresights = np.radians(np.asarray(scene["boresights_deg"], dtype=np.float64))
    for run in range(2):
        base = synthesize_shared_frame_snr(frame.states[:, :2], frame.states[:, 2:], frame.vehicle_ids,
                                           stations, boresights, WAVEFORM, ARRAY,
                                           [1.0] * len(frame), VISIBILITY, 0, int(frame.frame),
                                           snr_db=None, device=DEVICE)
        noise = scale_shared_noise(base["S"][0][None], base["W0"][0][None], snr_db=0.0)
        result = detector_module.detect_anonymous_measurements(
            base["S"][0] + noise["W"][0], base["X"][0], 0, stations[0], boresights[0], WAVEFORM,
            ARRAY, CONFIG["detector"], VISIBILITY, resource=RESOURCE)
        if run == 0:
            reference = result["measurements"]
        elif result["measurements"] != reference:
            raise AssertionError("repeated smoke run differs")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return f"selection and detector output reproduce exactly (HEAD {head.stdout.strip()[:8]})"


CHECKS = (
    ("1 world vx/vy not re-rotated", check_no_rotation),
    ("2 dt = 3/29.97", check_true_dt),
    ("3 scene/frame lookup", check_lookup),
    ("4 source returns all frame vehicles", check_all_vehicles),
    ("5 prediction cohort not in echo", check_cohort_isolation),
    ("6 one physical frame for three BS", check_single_frame_per_bs),
    ("7 fixed RCS = 1 m^2", check_fixed_rcs),
    ("8 1/R^2 amplitude law", check_amplitude_law),
    ("9 all 256 symbols used for sensing", check_full_resource),
    ("10 shared echo is multi-target sum", check_shared_sum),
    ("11 one noise realisation per BS/frame", check_single_noise),
    ("12 achieved SNR matches target", check_achieved_snr),
    ("13 base noise independent of SNR", check_noise_seed_independence),
    ("14 detector API blind to identity", check_detector_blind),
    ("15 new path never calls CFAR", check_no_cfar),
    ("16 one RD peak -> multiple angles", check_multi_angle),
    ("17 detector reads no GT count", check_no_gt_count),
    ("18 smoke deterministic", check_determinism),
)


def main() -> int:
    failures = 0
    for name, function in CHECKS:
        try:
            detail = function()
            print(f"PASS {name}: {detail}")
        except Exception as error:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {error}")
    print(f"==== {len(CHECKS) - failures}/{len(CHECKS)} checks passed ====")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())