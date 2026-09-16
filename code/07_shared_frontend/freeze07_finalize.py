"""SENS-FREEZE-07 Stage H: integrity comparison, cache manifest and freeze summary."""
from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import _common  # noqa: E402

DATA = ROOT / "data/f01e"
FREEZE = ROOT / "reports/f01e/freeze_07"
SNR_LIST = [-5, 0, 5, 10, 15, 20]
SPLITS = ["train", "V_select", "V_confirm", "test"]
PRE_MANIFEST = Path("/tmp/f01d_pre_freeze_manifest.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_manifest(base: Path) -> dict:
    manifest = {}
    for path in sorted(base.rglob("*")):
        if path.is_file():
            manifest[str(path.relative_to(base))] = {"size_bytes": path.stat().st_size,
                                                     "sha256": sha256_file(path)}
    return manifest


def npz_info(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as payload:
        return {key: {"shape": list(payload[key].shape), "dtype": str(payload[key].dtype)}
                for key in payload.files}


def snr_name(snr: float) -> str:
    return "snr_" + str(int(snr)).replace("-", "m")


def main() -> None:
    config = _common.load_frontend_config()
    config_hash = sha256_file(ROOT / "configs/shared_frontend.json")
    detector_hash = sha256_file(ROOT / "frontend/sensing/detector.py")
    waveform_hash = sha256_file(ROOT / "frontend/sensing/waveform.py")
    lut_hash = sha256_file(ROOT / config["detector"]["covariance_lut"])

    integrity = {"stage": "SENS-FREEZE-07 F01-D integrity"}
    if PRE_MANIFEST.exists():
        pre = json.loads(PRE_MANIFEST.read_text())
        post = file_manifest(Path(pre["base"]))
        changed = sorted(set(pre["files"]) ^ set(post)) + \
            [key for key in pre["files"] if key in post
             and (pre["files"][key]["sha256"] != post[key]["sha256"]
                  or pre["files"][key]["size_bytes"] != post[key]["size_bytes"])]
        integrity.update({"f01d_present": True, "files": len(post),
                          "pre_manifest_files": len(pre["files"]), "changed": changed[:20],
                          "unchanged": not changed})
    else:
        integrity.update({"f01d_present": Path("/home/dell/YrM/ICCT/data/f01d").exists(),
                          "unchanged": not Path("/home/dell/YrM/ICCT/data/f01d").exists()})

    equivalence = {}
    for snr in (-5.0, 20.0):
        name = f"sequences/{snr_name(snr)}/episode_000.npz"
        full = DATA / name
        dry = ROOT / "data/f01e_dryrun" / name
        equivalence[str(snr)] = {
            "full_sha256": sha256_file(full) if full.exists() else None,
            "dry_run_sha256": sha256_file(dry) if dry.exists() else None}
        equivalence[str(snr)]["identical"] = (equivalence[str(snr)]["full_sha256"] is not None
                                              and equivalence[str(snr)]["full_sha256"]
                                              == equivalence[str(snr)]["dry_run_sha256"])

    dry_run = json.loads((FREEZE / "dry_run_raw.json").read_text())
    totals_frames = 0
    for snr in SNR_LIST:
        totals_frames += 280 * 199
    log_seconds = 0.0
    log_episodes = 0
    for snr in SNR_LIST:
        log = FREEZE / "logs" / f"generate_{int(snr)}.log"
        if log.exists():
            for line in log.read_text(errors="ignore").splitlines():
                try:
                    entry = json.loads(line)
                except ValueError:
                    continue
                if "seconds" in entry and "episode" in entry:
                    log_seconds += float(entry["seconds"])
                    log_episodes += 1
    projected = {
        "dry_run_seconds_per_frame": dry_run["seconds_per_frame"],
        "projected_total_frames": totals_frames,
        "projected_runtime_hours_single_process": dry_run["seconds_per_frame"] * totals_frames / 3600.0,
        "measured_generation_seconds_aggregate": log_seconds,
        "measured_episodes_logged": log_episodes,
        "measured_seconds_per_frame": log_seconds / max(log_episodes * 199, 1),
        "parallelism": "6 processes (one per SNR) round-robin on 2 GPUs",
        "projected_storage_bytes": sum(path.stat().st_size for path in DATA.rglob("*") if path.is_file()),
    }

    cache_files = []
    split_entries = {}
    for path in sorted(DATA.rglob("*")):
        if not path.is_file():
            continue
        relative = str(path.relative_to(DATA))
        entry = {"relative_path": relative, "size_bytes": path.stat().st_size,
                 "sha256": sha256_file(path)}
        if path.suffix == ".npz" and "/sequences/" not in relative.replace("\\", "/"):
            entry["contents"] = npz_info(path)
        cache_files.append(entry)
    for split in SPLITS:
        metadata_payload = json.loads((DATA / "metadata" / f"{split}.json").read_text())
        episode_indices = sorted({int(sample["episode_index"]) for sample in metadata_payload["samples"]})
        split_entries[split] = {"inputs": {}, "labels": {}, "metadata": f"metadata/{split}.json",
                                "episode_indices": episode_indices}
        for snr in SNR_LIST:
            name = f"{split}_{snr_name(snr)}.npz"
            input_path = DATA / "inputs" / name
            with np.load(input_path, allow_pickle=False) as payload:
                split_entries[split]["inputs"][str(snr)] = {
                    "relative_path": f"inputs/{name}", "sample_count": int(payload["state_hat"].shape[0]),
                    "shape": list(payload["state_hat"].shape), "dtype": str(payload["state_hat"].dtype),
                    "sha256": sha256_file(input_path), "size_bytes": input_path.stat().st_size}
            label_path = DATA / "labels" / name
            if label_path.exists():
                with np.load(label_path, allow_pickle=False) as payload:
                    split_entries[split]["labels"][str(snr)] = {
                        "relative_path": f"labels/{name}", "sample_count": int(payload["future_position"].shape[0]),
                        "shape": list(payload["future_position"].shape),
                        "sha256": sha256_file(label_path), "size_bytes": label_path.stat().st_size}
        reference_label = DATA / "labels" / f"{split}.npz"
        split_entries[split]["labels"]["reference"] = {
            "relative_path": f"labels/{split}.npz", "sha256": sha256_file(reference_label),
            "note": f"aligned to {max(SNR_LIST)} dB (F01-D-compatible reference)"}
        split_entries[split]["metadata_samples"] = len(metadata_payload["samples"])
        split_entries[split]["sequences"] = {
            str(snr): sum(1 for index in episode_indices
                          if (DATA / "sequences" / snr_name(snr) / f"episode_{index:03d}.npz").exists())
            for snr in SNR_LIST}

    contract = {
        "authoritative_loader": "frontend/symbol_dataset.py (SharedPredictionInputs, read_inputs)",
        "authoritative_packer": "frontend/pack_symbol_dataset.py",
        "f01d_layout": ["inputs/{split}_snr_{s}.npz", "labels/{split}.npz", "metadata/{split}.json",
                        "sequences/snr_{s}/episode_{NNN}.npz", "diagnostics/snr_{s}/episode_{NNN}.json"],
        "whitelist_input_fields": ["state_hat", "track_exists", "detected", "timestamp"],
        "state_shape": [20, 8, 4], "label_shape": [20, 8, 2], "state_dimension": 4,
        "history_frames": 20, "future_frames": 20, "max_slots": 8, "dt_s": 0.1,
        "slot_semantics": "anonymous sticky tracker slots 0..7 (frozen shared frontend); tentative and "
                          "empty are zero padded with track_exists=0",
        "label_rule": "offline C-domain slot<->source-vehicle alignment over the 20 history frames "
                      "(mean position distance, >=3 common frames, Hungarian, 5 m gate); raw future xy "
                      "with an independent validity mask; GT never enters sensing",
        "sequence_frame_grid": "episode start + (1..199)*100 ms; timestamps in seconds",
        "split_source": "frozen A01 episodes.jsonl (train 179, V_select 25, V_confirm 22, test 54)",
        "contract_conflict_resolution": {
            "issue": "F01-D slots were source-slot ordered (identity-fixed), so one labels/{split}.npz "
                     "served all SNRs; B64 uses anonymous sticky slots whose numbering drifts between SNR "
                     "realizations. Measured on the dry run: 198/310 windows pure slot permutations and "
                     "99/310 set differences at -5 dB vs the 20 dB reference.",
            "resolution": "per-SNR label files labels/{split}_snr_{s}.npz with identical keys/shapes/"
                          "semantics, plus the reference labels/{split}.npz aligned at 20 dB for existing "
                          "call sites; metadata records label_files, label_alignment_snr and per-SNR "
                          "slot_alignment maps. Loader contract (inputs only) is unchanged.",
            "status": "documented deviation, flagged for independent ratification"},
    }

    cache_manifest = {
        "freeze_stage": "SENS-FREEZE-07",
        "generation_commit": "f319faf357ee152a1ee1d4984d8eb33e19965ab1 (production code; freeze commit adds evidence)",
        "production_config_sha256": config_hash,
        "production_detector_sha256": detector_hash,
        "production_waveform_sha256": waveform_hash,
        "covariance_lut_sha256": lut_hash,
        "snr_levels": SNR_LIST,
        "cache_schema_version": "F01-E v1 (F01-D container schema; anonymous sticky slots; per-SNR labels)",
        "history_frames": 20, "future_frames": 20, "max_slots": 8, "state_dimension": 4, "dt_s": 0.1,
        "contract_discovery": contract,
        "dry_run": projected,
        "equivalence_test": equivalence,
        "splits": split_entries,
        "files": cache_files,
        "total_files": len(cache_files),
        "total_bytes": sum(entry["size_bytes"] for entry in cache_files),
    }

    regression = json.loads((ROOT / "reports/f01e/snr_rebuild_06/regression_checks.json").read_text())
    interface_path = FREEZE / "interface_check.json"
    interface = json.loads(interface_path.read_text()) if interface_path.exists() else {}
    quality_path = FREEZE / "train_quality_audit.json"
    quality = json.loads(quality_path.read_text()) if quality_path.exists() else {}
    decomposition_path = FREEZE / "train_error_decomposition.json"
    decomposition_raw = json.loads(decomposition_path.read_text()) if decomposition_path.exists() else {}
    decomposition = {key: decomposition_raw.get(key) for key in
                     ("alignment_cost", "window_p90_gt1_fraction",
                      "window_p90_detected_only_gt1_fraction", "note")} if decomposition_raw else {}
    covariance_report = json.loads((ROOT / "reports/f01e/snr_rebuild_06/covariance_calibration.json")
                                   .read_text())
    freeze_summary = {
        "stage": "SENS-FREEZE-07",
        "production_base_commit": "f319faf357ee152a1ee1d4984d8eb33e19965ab1",
        "production": {"resource": "B64_contiguous_burst", "active_symbols": 64, "start_symbol": 96,
                       "cfar_multiplier": config["detector"]["cfar_threshold_multiplier"],
                       "covariance_inflation": covariance_report["covariance_inflation"],
                       "tracker_q_a": config["tracker"]["q_a_m2_s3"]},
        "snr_levels_db": SNR_LIST,
        "checks": {
            "rebuild06_regression": bool(regression.get("passed")),
            "no_gt": bool(regression["checks"]["scripts"]["no_gt"]),
            "f01d_unchanged": bool(integrity.get("unchanged")),
            "all_caches_generated": all(split_entries[split]["sequences"][str(snr)] == len(split_entries[split]["episode_indices"])
                                        for split in SPLITS for snr in SNR_LIST),
            "cache_hashes_complete": len(cache_files) > 0,
            "sticky_slot_continuity": quality.get("sticky_slot_continuity_violations") == 0,
            "interface_compatible": bool(interface.get("passed")),
            "no_future_leakage": bool(interface.get("checks", {}).get("no_forbidden_input_keys")),
            "split_integrity": bool(interface.get("checks", {}).get("split_integrity")),
            "train_quality_noncatastrophic": bool(quality) and quality["overall"][
                "problematic_window_fraction_overall"] <= 0.50,
        },
"train_quality": {
            "problematic_window_fraction_overall": quality.get("overall", {}).get(
                "problematic_window_fraction_overall"),
            "problematic_window_fraction_by_snr": {str(snr): quality["by_snr"][str(snr)][
                "problematic_window_fraction"] for snr in SNR_LIST} if quality else {},
            "coast_ratio_by_snr": {str(snr): quality["by_snr"][str(snr)]["coast_ratio"]
                                   for snr in SNR_LIST} if quality else {},
            "density_strata_at_minus5": quality.get("by_snr", {}).get("-5", {}).get("by_density", {}),
            "position_error_overall": quality.get("overall", {}).get("position_error_m"),
            "velocity_error_overall": quality.get("overall", {}).get("velocity_mps"),
            "error_decomposition": decomposition,
            "blocking_issue": "predeclared train-quality gate (problematic window fraction > 0.50) "
                              "triggered; driven by dense-scene coast frames (26-29% of alive frames, "
                              "SNR-independent structural association merging); freeze withheld for "
                              "independent judgement",
        },
        "cache": {"schema_version": cache_manifest["cache_schema_version"],
                  "total_files": cache_manifest["total_files"], "total_bytes": cache_manifest["total_bytes"],
                  "total_windows": sum(split_entries[split]["inputs"][str(snr)]["sample_count"]
                                       for split in SPLITS for snr in SNR_LIST)},
        "upstream_isac_frozen": None,
        "overall_status": None,
    }
    hard = all(freeze_summary["checks"].values())
    freeze_summary["upstream_isac_frozen"] = bool(hard)
    freeze_summary["overall_status"] = "PASS" if hard else "FAIL"

    FREEZE.mkdir(parents=True, exist_ok=True)
    for name, payload in (("cache_manifest.json", cache_manifest), ("integrity_checks.json", integrity),
                          ("freeze_summary.json", freeze_summary)):
        with (FREEZE / name).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
            handle.write("\n")
    print(json.dumps({"integrity": integrity.get("unchanged"), "equivalence": equivalence,
                      "checks": freeze_summary["checks"],
                      "overall": freeze_summary["overall_status"],
                      "total_files": cache_manifest["total_files"],
                      "total_bytes": cache_manifest["total_bytes"]}, ensure_ascii=False)[:1500])


if __name__ == "__main__":
    main()