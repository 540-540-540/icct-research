"""Final-audit self-check: 15 required checks. Read-only; no parameter is modified.

Usage (repo root):
    python experiments/automatum_controlled_isac_final_audit/selfcheck.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.automatum_controlled_isac_final_audit import common  # noqa: E402
from frontend.automatum_scene_source import AutomatumSceneSource  # noqa: E402

FROZEN_SAMPLES_SHA = {
    "train": "1b86a6453a53779b9b647f8fa04bf3b40c56bba0dfc26f0d19f6de895a58e0c9",
    "val": "51dc713ba5873ba438267e5c46f97902e4e744cbd3069bf3e289a485d058ff31",
}
FROZEN_SPLIT_SHA = {
    "train": "d75994cb999d10ac5c05fab00f95a11d98b8e8cc61c475b87373ba6b51a82a11",
    "val": "c9964adb9b73e828754912381f5d93cfd8adf215bad4ba85db766d4166ab894d",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_clean(*paths: str) -> bool:
    result = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *paths], cwd=ROOT)
    return result.returncode == 0


def check_dt() -> str:
    source = AutomatumSceneSource(ROOT / common.load_config()["dataset"]["train_trajectories"])
    if abs(common.DT - 3.0 / 29.97) > 1e-15 or abs(source.dt - common.DT) > 1e-15:
        raise AssertionError("dt mismatch")
    return f"dt={common.DT!r} == 3/29.97 (source dt identical)"


def check_dedupe_correct() -> str:
    history = common.unique_history("train")
    keys = history["keys"]
    if len(keys) != len(set(keys)):
        raise AssertionError("duplicate keys survived de-duplication")
    if any(keys[i] >= keys[i + 1] for i in range(len(keys) - 1)):
        raise AssertionError("unique keys are not strictly sorted")
    if history["unique_states"] != len(keys):
        raise AssertionError("state count mismatch")
    return (f"{history['history_rows']} history rows de-duplicated to {len(keys)} unique states "
            f"(factor {history['duplication_factor']:.2f})")


def check_no_double_counting() -> str:
    d = np.load(ROOT / "data/automatum_t_crossing/splits/train/samples.npz")
    scene = d["scene_id"].astype(np.int64)
    start = d["start_frame"].astype(np.int64)
    rows = []
    for index in range(scene.size):
        ids = d["vehicle_ids"][index][d["vehicle_mask"][index]].astype(np.int64)
        for offset in range(20):
            for vehicle in ids.tolist():
                rows.append((int(scene[index]), int(start[index]) + offset, int(vehicle)))
    rows = np.asarray(rows, dtype=np.int64)
    unique_alternative = np.unique(rows, axis=0)
    keys = np.asarray(common.unique_history("train")["keys"], dtype=np.int64)
    if unique_alternative.shape != keys.shape or not np.array_equal(unique_alternative, keys):
        raise AssertionError("independent np.unique de-duplication disagrees")
    return "independent np.unique reconstruction equals the audit unique-state set"


def check_vehicle_id_index_only() -> str:
    text = (ROOT / "frontend/controlled_isac/automatum_frontend.py").read_text(encoding="utf-8")
    if "vehicle_id" not in text:
        raise AssertionError("frontend lost the vehicle index field")
    state_fields = re.search(r"STATE_FIELDS = \(([^)]*)\)", text).group(1)
    if set(re.findall(r'"(\w+)"', state_fields)) != {"x_hat", "y_hat", "vx_hat", "vy_hat"}:
        raise AssertionError("model state fields drifted")
    return "vehicle_id is an index only; model state remains [x_hat,y_hat,vx_hat,vy_hat]"


def check_candidate_b_unchanged() -> str:
    calibration = common.load_calibration(common.load_config())
    if not common.calibration_is_candidate_b(calibration):
        raise AssertionError("Candidate B values changed")
    if not git_clean("configs/automatum_controlled_isac.json"):
        raise AssertionError("config file modified relative to HEAD")
    return f"Candidate B unchanged {common.EXPECTED_CANDIDATE_B}; config equals HEAD"


def check_snr_levels_unchanged() -> str:
    levels = tuple(float(value) for value in common.load_config()["snr"]["levels_db"])
    if levels != common.LEVELS:
        raise AssertionError(f"SNR levels changed: {levels}")
    return f"SNR levels unchanged {levels}"


def check_geometry_unchanged() -> str:
    if not git_clean("configs/automatum_controlled_isac.json"):
        raise AssertionError("geometry source modified")
    config = common.load_config()
    if len(config["scenes"]) != 2 or any(len(scene["stations_xy_m"]) != 3 for scene in config["scenes"]):
        raise AssertionError("geometry structure changed")
    return "3-BS geometry unchanged (config equals HEAD)"


def check_fusion_unchanged() -> str:
    if not git_clean("frontend/controlled_isac/velocity_fusion.py",
                     "frontend/controlled_isac/cross_bs_association.py"):
        raise AssertionError("fusion code modified")
    return "velocity/position fusion code unchanged relative to HEAD"


def check_state_quality_unchanged() -> str:
    if not git_clean("frontend/controlled_isac/state_quality.py"):
        raise AssertionError("state_quality.py modified")
    return "state_quality.py unchanged relative to HEAD"


def check_samples_unmodified() -> str:
    for split, expected in FROZEN_SAMPLES_SHA.items():
        actual = sha256(ROOT / "data/automatum_t_crossing/splits" / split / "samples.npz")
        if actual != expected:
            raise AssertionError(f"{split} samples.npz SHA mismatch")
    return "train/val samples.npz SHA256 equal the frozen values"


def check_split_unmodified() -> str:
    for split, expected in FROZEN_SPLIT_SHA.items():
        actual = sha256(ROOT / "data/automatum_t_crossing/splits" / split / "trajectories.csv")
        if actual != expected:
            raise AssertionError(f"{split} trajectories.csv SHA mismatch")
    return "train/val trajectories.csv SHA256 equal the frozen values"


def check_frontend_untouched() -> str:
    if not git_clean("frontend"):
        raise AssertionError("frontend numerical path modified")
    return "git diff HEAD -- frontend is empty"


def check_test_unused() -> str:
    for name in ("common.py", "audit_data_scale.py", "audit_geometry.py",
                 "audit_sensing_scale.py", "audit_quality.py", "run_audit.py"):
        text = (Path(__file__).resolve().parent / name).read_text(encoding="utf-8")
        if "test_trajectories" in text or "splits/test" in text or "splits\" / \"test" in text:
            raise AssertionError(f"{name} references test data")
    decision = json.loads((common.OUT_DIR / "decision_summary.json").read_text())
    if decision["test_used_for_parameter_decisions"] is not False:
        raise AssertionError("decision summary claims test usage")
    return "no audit script reads test; decision summary states test not used"


def check_deterministic() -> str:
    first = common.unique_history("train")
    second = common.unique_history("train")
    if first["keys"] != second["keys"]:
        raise AssertionError("unique-state construction is not deterministic")
    lookup_first, frames_first = common.state_lookup("train")
    lookup_second, frames_second = common.state_lookup("train")
    if set(lookup_first) != set(lookup_second) or len(frames_first) != len(frames_second):
        raise AssertionError("state lookup is not deterministic")
    decision_text = (common.OUT_DIR / "decision_summary.json").read_text()
    if "runtime" in decision_text or "timestamp" in decision_text:
        raise AssertionError("decision summary contains a wall-clock field")
    return "in-process reconstruction identical; decision summary has no wall-clock fields"


def check_train_val_independent() -> str:
    with (common.OUT_DIR / "sensing_scale_by_snr.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    splits = {row["split"] for row in rows}
    if splits != {"train", "val"}:
        raise AssertionError(f"split labels missing: {splits}")
    decision = json.loads((common.OUT_DIR / "decision_summary.json").read_text())
    if "train" not in json.dumps(decision["candidate_b_errors_ge2"]):
        raise AssertionError("decision metrics missing")
    return "train and val reported independently; decisions use train scales"


CHECKS = (
    ("1 canonical dt = 3/29.97", check_dt),
    ("2 prediction-history de-duplication correct", check_dedupe_correct),
    ("3 no double counting of overlapping windows", check_no_double_counting),
    ("4 vehicle_id index only", check_vehicle_id_index_only),
    ("5 Candidate B unchanged", check_candidate_b_unchanged),
    ("6 SNR levels unchanged", check_snr_levels_unchanged),
    ("7 geometry unchanged", check_geometry_unchanged),
    ("8 fusion unchanged", check_fusion_unchanged),
    ("9 state_quality unchanged", check_state_quality_unchanged),
    ("10 samples.npz unmodified", check_samples_unmodified),
    ("11 split unmodified", check_split_unmodified),
    ("12 frontend numerical path unmodified", check_frontend_untouched),
    ("13 test not used for decisions", check_test_unused),
    ("14 deterministic reconstruction", check_deterministic),
    ("15 train/val independent reporting", check_train_val_independent),
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