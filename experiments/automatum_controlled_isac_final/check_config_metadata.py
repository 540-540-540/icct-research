"""Consistency check for the metadata-only cleanup of the frozen V2 config.

Verifies against the freeze commit that no numerical/algorithmic field changed: only the
dataset calibration/validation frame provenance metadata may differ, and the freeze manifest
may differ only in config_sha256 plus the explanatory note.

Usage (repo root):
    python experiments/automatum_controlled_isac_final/check_config_metadata.py
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CONFIG = ROOT / "configs/automatum_controlled_isac.json"
MANIFEST = ROOT / "reports/isac_final/freeze_manifest.json"
BASELINE = "bb9dbe93ac3347ce770601e0c4bc413230ae7698"
REMOVED_KEYS = ("calibration_frames", "validation_frames")
ADDED_KEYS = ("calibration_scope", "validation_scope", "test_scope")
NUMERIC_ROOT_KEYS = ("waveform", "scenes", "visibility", "snr", "measurement", "fusion",
                     "sensing_resource", "array", "power", "state_quality")


def git_show(path: str) -> str:
    result = subprocess.run(["git", "show", f"{BASELINE}:{path}"], cwd=ROOT,
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"cannot read {path} at {BASELINE}: {result.stderr}")
    return result.stdout


def assert_unchanged(label: str, old, new) -> None:
    if old != new:
        raise AssertionError(f"{label} changed")


def main() -> int:
    baseline_config = json.loads(git_show("configs/automatum_controlled_isac.json"))
    current_config = json.loads(CONFIG.read_text(encoding="utf-8"))
    baseline_manifest = json.loads(git_show("reports/isac_final/freeze_manifest.json"))
    current_manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert_unchanged("revision", baseline_config["revision"], current_config["revision"])
    assert_unchanged("status", baseline_config["status"], current_config["status"])
    if current_config["revision"] != "AUTOMATUM-CONTROLLED-ISAC-V2-FROZEN" \
            or current_config["status"] != "FROZEN":
        raise AssertionError("config is not V2-FROZEN")

    baseline_dataset = dict(baseline_config["dataset"])
    current_dataset = dict(current_config["dataset"])
    for key in REMOVED_KEYS:
        baseline_dataset.pop(key, None)
    for key in ADDED_KEYS:
        if key not in current_dataset:
            raise AssertionError(f"missing scope field {key}")
        current_dataset.pop(key)
    assert_unchanged("dataset (numbers/paths)", baseline_dataset, current_dataset)

    for key in NUMERIC_ROOT_KEYS:
        assert_unchanged(f"config.{key}", baseline_config.get(key), current_config.get(key))
    assert_unchanged("config top-level keys",
                     sorted(set(baseline_config) - {"dataset"}),
                     sorted(set(current_config) - {"dataset"}))

    manifest_old = dict(baseline_manifest)
    manifest_new = dict(current_manifest)
    manifest_old.pop("config_sha256", None)
    manifest_new.pop("config_sha256", None)
    manifest_old.pop("config_sha256_note", None)
    manifest_new.pop("config_sha256_note", None)
    assert_unchanged("freeze manifest (frozen values)", manifest_old, manifest_new)
    if current_manifest["config_sha256"] != hashlib.sha256(CONFIG.read_bytes()).hexdigest():
        raise AssertionError("manifest config_sha256 does not match the cleaned config")
    if "metadata-only" not in current_manifest.get("config_sha256_note", ""):
        raise AssertionError("manifest is missing the metadata-only SHA note")

    for relative, digest in current_manifest["evidence_sha256"].items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        if actual != digest:
            raise AssertionError(f"evidence hash stale: {relative}")

    print("PASS revision/status = AUTOMATUM-CONTROLLED-ISAC-V2-FROZEN / FROZEN")
    print("PASS all numeric config fields identical to the freeze commit")
    print("PASS dataset diff is exactly the calibration/validation provenance metadata")
    print("PASS freeze manifest frozen values identical; config SHA updated with note")
    print("PASS all frozen evidence SHA256 still match")
    print("==== metadata cleanup consistency: PASS ====")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())