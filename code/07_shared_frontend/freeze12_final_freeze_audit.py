"""F01E-FINALIZE-11R2 closure: reproducibility audit of the final F01E freeze (CPU-only).

Re-derives the expected final labels from the frozen sequences and the recorded metadata assignments
under the origin-safe matcher and the same-segment IDENTITY_SWITCH_PM1_MASK, verifies them against
the packed labels, runs the formal loader positive/negative guards, checks that the frozen ISAC
upstream is byte-identical, and writes the long-term final manifest / freeze summary / integrity
evidence under ``reports/f01e/final_freeze_11``. ``data/f01e`` is read-only here.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import freeze07_generate_f01e as generator  # noqa: E402
import freeze10_identity_forensics as audit10  # noqa: E402

DATA = ROOT / "data/f01e"
OUT = ROOT / "reports/f01e/final_freeze_11"
SNR_LIST = [-5.0, 0.0, 5.0, 10.0, 15.0, 20.0]
SPLITS = ("train", "V_select", "V_confirm", "test")
MASK_RADIUS = 1
AUDIT10_BRANCH = "f01e-identity-audit-10"
AUDIT10_COMMIT = "30f899a8b05be9ec0b5b06364ce53bb935e11fc8"
FINALIZE11_COMMIT = "689213a9e36d864489229f991cb93107030fdd17"
AUDIT10_INTEGRITY = ROOT / "reports/f01e/identity_audit_10/integrity.json"
FINALIZE11_SUMMARY = ROOT / "reports/f01e/finalize_11/summary.json"
PINNED_SHA256 = {
    "configs/shared_frontend.json":
        "5c86e3b19e5ad860dc9965fcffc9c715f7320c35626499fffce3d83725b0b4fd",
    "frontend/sensing/detector.py":
        "f6b8d70dcd0b6941b846678107e83bfcf541d6de902becb6dcf7028b89cf383d",
    "frontend/sensing/waveform.py":
        "5f9b18564a7c1030171f1a64a38433e5a68d5ada9a6b58fb7ace87b297b8f783",
    "frontend/tracking/cv_kf.py":
        "9d3d3338350ff2c438505bab67aeffba36fc66bb5b966dc915b1f12b48e54dd5",
    "reports/f01e/snr_rebuild_06/covariance_calibration.json":
        "57d555f9fc280c99f00d2004621455a21caefaaa1023c03d92aa6ed5558675f4",
}
FORBIDDEN_INPUT_TOKENS = ("future", "gt", "truth", "label", "source_key")


def snr_key(snr: float) -> str:
    return str(int(snr))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reconstruct_raw_labels(source, episode: dict, origins: list, maps: list) -> tuple:
    """Raw aligned future labels from the recorded metadata maps (no matcher re-run needed)."""
    window_count = len(origins)
    future = np.zeros((window_count, 20, 8, 2), np.float32)
    valid = np.zeros((window_count, 20, 8), bool)
    for window, mapping in enumerate(maps):
        for slot, key in mapping.items():
            times = origins[window] + np.arange(1, 21, dtype=np.int64) * 100
            xy, exact = generator.gt_track_xy(source.tracks[int(key)], times)
            usable = exact & np.isfinite(xy[:, 0])
            future[window, usable, slot] = xy[usable].astype(np.float32)
            valid[window, :, slot] = usable
    return future, valid


def scan_split(source, split: str, document: dict, actual: dict) -> dict:
    """Independent expected-label reconstruction + R1/R2 mask accounting for one split."""
    samples = document["samples"]
    by_episode = defaultdict(list)
    for sample in samples:
        by_episode[sample["episode_index"]].append(sample)
    per_snr = {snr: Counter() for snr in SNR_LIST}
    old_by_split = Counter()
    representative = []
    for episode_index, episode_samples in sorted(by_episode.items()):
        episode = source.episodes[episode_index]
        origins = [int(sample["origin_ms"]) for sample in episode_samples]
        base = int(episode_samples[0]["sample_index"])
        window_count = len(origins)
        sequences = {snr: generator.load_sequence(generator.sequence_path(DATA, snr, episode_index))
                     for snr in SNR_LIST}
        for snr in SNR_LIST:
            maps = [{int(slot): int(key) for slot, key in
                     sample["slot_alignment_by_snr"][snr_key(snr)].items()}
                    for sample in episode_samples]
            plan = generator.identity_switch_pm1_plan(sequences[snr], origins, maps,
                                                      radius=MASK_RADIUS)
            raw_future, raw_valid = reconstruct_raw_labels(source, episode, origins, maps)
            expected_future = raw_future.copy()
            expected_valid = raw_valid.copy()
            generator.apply_identity_switch_pm1_mask(
                {"future_position": expected_future, "label_valid": expected_valid}, plan["cells"])
            actual_future = actual[snr]["future_position"][base:base + window_count]
            actual_valid = actual[snr]["label_valid"][base:base + window_count]

            stats = per_snr[snr]
            stats["switches"] += len(plan["switches"])
            stats["samples"] += window_count
            pm1_cells = plan["cells"]
            old_cells = set(pm1_cells)
            for switch in plan["switches"]:
                for position in switch["cross_segment_windows"]:
                    old_cells.add((position, switch["slot"]))
            stats["pm1_cells_expected"] += sum(1 for (w, s) in pm1_cells
                                               if raw_valid[w, :, s].any())
            stats["pm1_cells_actual"] += sum(1 for (w, s) in pm1_cells
                                             if raw_valid[w, :, s].any()
                                             and not actual_valid[w, :, s].any())
            stats["missing_pm1_masks"] += sum(1 for (w, s) in pm1_cells
                                              if raw_valid[w, :, s].any()
                                              and actual_valid[w, :, s].any())
            stats["old_cells_labelled"] += sum(1 for (w, s) in old_cells
                                               if raw_valid[w, :, s].any())
            stats["corrected_cells_labelled"] += sum(1 for (w, s) in pm1_cells
                                                     if raw_valid[w, :, s].any())
            stats["cross_segment_positions"] += len(old_cells - set(pm1_cells))
            for (window, slot) in old_cells - set(pm1_cells):
                if raw_valid[window, :, slot].any() and not actual_valid[window, :, slot].any():
                    stats["cross_segment_masks_in_final"] += 1
            for window in range(window_count):
                for slot in range(8):
                    if actual_valid[window, :, slot].any():
                        if not expected_valid[window, :, slot].any():
                            stats["valid_surplus_in_actual"] += 1
                        continue
                    if expected_valid[window, :, slot].any():
                        stats["valid_missing_in_actual"] += 1
                        continue
                    if raw_valid[window, :, slot].any():
                        if (window, slot) not in pm1_cells:
                            stats["unexpected_invalid_beyond_policy"] += 1
                    else:
                        stats["invalid_from_raw_availability"] += 1
            if not np.array_equal(actual_valid, expected_valid):
                stats["valid_array_mismatch"] += 1
            if np.array_equal(actual_future, expected_future):
                stats["future_array_mismatch"] += 0
            else:
                stats["future_array_mismatch"] += 1
                stats["future_abs_diff_max"] = max(
                    stats["future_abs_diff_max"],
                    float(np.max(np.abs(actual_future.astype(np.float64)
                                        - expected_future.astype(np.float64)))))
            for switch in plan["switches"]:
                if not switch["cross_segment_windows"] or len(representative) >= 8:
                    continue
                neighbour = switch["cross_segment_windows"][0]
                _, _, neighbour_segments = generator.origin_eligibility(sequences[snr],
                                                                        origins[neighbour])
                representative.append({
                    "split": split, "episode": episode_index, "snr": snr, "slot": switch["slot"],
                    "switch_origin_ms": origins[switch["window"]],
                    "gt_before": switch["gt_before"], "gt_after": switch["gt_after"],
                    "cross_segment_neighbour_ms": origins[neighbour],
                    "neighbour_origin_eligible": switch["slot"] in neighbour_segments,
                    "neighbour_contiguous_alive_length":
                        neighbour_segments.get(switch["slot"], {}).get("contiguous_alive_length", 0),
                    "neighbour_raw_label_present": bool(raw_valid[neighbour, :, switch["slot"]].any()),
                    "reason": "slot death/rebirth gap between switch origin and +1 neighbour"})
    old_by_split["old_cells_labelled"] = sum(entry["old_cells_labelled"]
                                             for entry in per_snr.values())
    old_by_split["corrected_cells_labelled"] = sum(entry["corrected_cells_labelled"]
                                                   for entry in per_snr.values())
    old_by_split["cross_segment_positions"] = sum(entry["cross_segment_positions"]
                                                  for entry in per_snr.values())
    old_by_split["cross_segment_masks_in_final"] = sum(entry["cross_segment_masks_in_final"]
                                                       for entry in per_snr.values())
    expected_keys = ("switches", "samples", "pm1_cells_expected", "pm1_cells_actual",
                     "missing_pm1_masks", "old_cells_labelled", "corrected_cells_labelled",
                     "cross_segment_positions", "cross_segment_masks_in_final",
                     "valid_surplus_in_actual", "valid_missing_in_actual",
                     "unexpected_invalid_beyond_policy", "invalid_from_raw_availability",
                     "valid_array_mismatch", "future_array_mismatch")
    for stats in per_snr.values():
        for key in expected_keys:
            stats.setdefault(key, 0)
        stats.setdefault("future_abs_diff_max", 0.0)
    return {"per_snr": {snr_key(snr): dict(per_snr[snr]) for snr in SNR_LIST},
            "totals": dict(old_by_split), "representative_cases": representative}


def loader_guards(source_document: dict) -> dict:
    from frontend import f01e_dataset

    results = {"positive_loads": {}, "negative": {}}
    for split in SPLITS:
        for snr in SNR_LIST:
            dataset = f01e_dataset.F01EDataset.from_root(DATA, split, snr)
            sample = dataset[len(dataset) // 2]
            results["positive_loads"][f"{split}_{snr_key(snr)}"] = {
                "samples": len(dataset),
                "shapes_ok": sample["model_input"]["state_hat"].shape == (20, 8, 4)
                and sample["labels"]["future_position"].shape == (20, 8, 2),
                "finite": bool(np.isfinite(sample["model_input"]["state_hat"]).all()
                               and np.isfinite(sample["labels"]["future_position"]).all()),
                "binding_ok": f01e_dataset.snr_name(snr) in dataset.label_path.name}
    results["forbidden_input_keys"] = [key for key in
                                       f01e_dataset.F01EDataset.from_root(DATA, "train", -5)[0]["model_input"]
                                       if any(token in key.lower()
                                              for token in FORBIDDEN_INPUT_TOKENS)]

    tamper_root = Path("/tmp/f01e_final_guard_r2")
    if tamper_root.exists():
        shutil.rmtree(tamper_root)
    (tamper_root / "inputs").mkdir(parents=True)
    (tamper_root / "labels").mkdir()
    (tamper_root / "metadata").mkdir()
    name = f01e_dataset.snr_name(-5)
    shutil.copy(DATA / "inputs" / f"train_{name}.npz", tamper_root / "inputs" / f"train_{name}.npz")
    shutil.copy(DATA / "labels" / f"train_{name}.npz", tamper_root / "labels" / f"train_{name}.npz")

    def attempt(label, mutate, expected):
        document = json.loads(json.dumps(source_document))
        mutate(document)
        (tamper_root / "metadata" / "train.json").write_text(json.dumps(document))
        try:
            f01e_dataset.F01EDataset.from_root(tamper_root, "train", -5)
            results["negative"][label] = {"rejected": False, "expected": expected.__name__}
        except expected:
            results["negative"][label] = {"rejected": True, "expected": expected.__name__}
        except Exception as error:  # noqa: BLE001
            results["negative"][label] = {"rejected": False, "expected": expected.__name__,
                                          "actual": repr(error)}

    attempt("missing_finalized", lambda doc: doc.pop("finalized", None), RuntimeError)
    attempt("sanitization_none", lambda doc: doc.update(label_sanitization="NONE"), RuntimeError)
    attempt("radius_two", lambda doc: doc.update(mask_radius_origins=2), RuntimeError)
    attempt("scope_changed",
            lambda doc: doc.update(mask_scope="cross_segment_allowed"), RuntimeError)
    try:
        f01e_dataset.F01EDataset.from_paths(DATA, "train", -5, label_snr_db=None)
        results["negative"]["generic_label"] = {"rejected": False, "expected": "RuntimeError"}
    except RuntimeError:
        results["negative"]["generic_label"] = {"rejected": True, "expected": "RuntimeError"}
    try:
        f01e_dataset.F01EDataset.from_root(ROOT / "data/f01d", "train", -5)
        results["negative"]["f01d_root"] = {"rejected": False, "expected": "RuntimeError"}
    except RuntimeError:
        results["negative"]["f01d_root"] = {"rejected": True, "expected": "RuntimeError"}
    try:
        f01e_dataset.F01EDataset.from_paths(DATA, "train", -5, label_snr_db=20)
        results["negative"]["wrong_snr_pairing"] = {"rejected": False, "expected": "ValueError"}
    except ValueError:
        results["negative"]["wrong_snr_pairing"] = {"rejected": True, "expected": "ValueError"}
    results["negative_pass"] = all(entry["rejected"] for entry in results["negative"].values())
    results["positive_pass"] = all(entry["shapes_ok"] and entry["finite"] and entry["binding_ok"]
                                   for entry in results["positive_loads"].values())
    return results


def main() -> None:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    from frontend import f01e_dataset
    from frontend.echo_source import SourceEpisodes
    source = SourceEpisodes()

    manifests = {name: audit10.subtree_manifest(DATA / name)
                 for name in ("inputs", "sequences", "labels", "metadata")}
    audit10_integrity = json.loads(AUDIT10_INTEGRITY.read_text())
    audit10_baseline = audit10_integrity["before"]
    finalize11_summary = json.loads(FINALIZE11_SUMMARY.read_text())

    inputs_identical = all(manifests["inputs"] == audit10_integrity[side]["inputs"]
                           for side in ("before", "after"))
    sequences_identical = all(manifests["sequences"] == audit10_integrity[side]["sequences"]
                              for side in ("before", "after"))

    split_results = {}
    representative_cases = []
    final_labelled_total = 0
    for split in SPLITS:
        metadata_path = DATA / "metadata" / f"{split}.json"
        document = json.loads(metadata_path.read_text())
        f01e_dataset.require_final_metadata(document, metadata_path)
        actual = {}
        for snr in SNR_LIST:
            with np.load(DATA / "labels" / f"{split}_{generator.snr_name(snr)}.npz",
                         allow_pickle=False) as payload:
                actual[snr] = {"future_position": payload["future_position"],
                               "label_valid": payload["label_valid"]}
        result = scan_split(source, split, document, actual)
        split_results[split] = result
        representative_cases.extend(result["representative_cases"])
        final_labelled_total += sum(int(actual[snr]["label_valid"].any(axis=1).sum())
                                    for snr in SNR_LIST)
        print(json.dumps({"split": split, "totals": result["totals"],
                          "elapsed_s": round(time.time() - started, 1)}), flush=True)

    # final-label structural invariants against the frozen inputs
    dead_slot_labels = 0
    origin_ineligible_labels = 0
    for split in SPLITS:
        for snr in SNR_LIST:
            with np.load(DATA / "inputs" / f"{split}_{generator.snr_name(snr)}.npz",
                         allow_pickle=False) as payload:
                exists = payload["track_exists"]
            eligible = f01e_dataset.origin_eligibility(exists)
            with np.load(DATA / "labels" / f"{split}_{generator.snr_name(snr)}.npz",
                         allow_pickle=False) as payload:
                valid_any = payload["label_valid"].any(axis=1)
            dead_slot_labels += int((valid_any & ~exists[:, -1, :]).sum())
            origin_ineligible_labels += int((valid_any & ~eligible).sum())

    config = json.loads((ROOT / "configs/shared_frontend.json").read_text())
    config_checks = {
        "b64_resource_unchanged": config["sensing_resource"] == {
            "mode": "contiguous_burst", "total_symbols": 256, "active_symbols": 64,
            "start_symbol": 96},
        "waveform_unchanged": config["waveform"] == {
            "fc": 24000000000.0, "B": 93100000.0, "K": 256, "N": 256, "T": 1.2375e-05,
            "c": 299792458.0},
        "cfar_multiplier_unchanged": (config["detector"]["cfar_threshold_multiplier"]
                                      == 0.19929921825469216),
        "covariance_lut_reference_unchanged": (config["detector"]["covariance_lut"]
                                               == "reports/f01e/snr_rebuild_06/"
                                                  "covariance_calibration.json"),
        "association_gate_unchanged": config["association"]["gate_chi2_2dof"] == 9.21,
        "tracker_q_a_unchanged": config["tracker"]["q_a_m2_s3"] == 2.0,
        "snr_levels_unchanged": config["snr_levels_db"] == [-5, 0, 5, 10, 15, 20],
    }
    hash_checks = {rel: {"expected": expected, "actual": sha256_file(ROOT / rel),
                         "unchanged": sha256_file(ROOT / rel) == expected}
                   for rel, expected in PINNED_SHA256.items()}
    diagnostics_detector = set()
    diagnostics_waveform = set()
    for path in sorted((DATA / "diagnostics").rglob("*.json")):
        report = json.loads(path.read_text())
        diagnostics_detector.add(report.get("detector_sha256"))
        diagnostics_waveform.add(report.get("waveform_sha256"))
    provenance_checks = {
        "detector_matches_sequence_provenance": diagnostics_detector == {
            PINNED_SHA256["frontend/sensing/detector.py"]},
        "waveform_matches_sequence_provenance": diagnostics_waveform == {
            PINNED_SHA256["frontend/sensing/waveform.py"]},
    }

    guards = loader_guards(json.loads((DATA / "metadata" / "train.json").read_text()))
    leakage_free = len(guards["forbidden_input_keys"]) == 0

    totals = Counter()
    for split in SPLITS:
        totals.update(split_results[split]["totals"])
    unexpected_total = sum(sum(entry["unexpected_invalid_beyond_policy"]
                               for entry in split_results[split]["per_snr"].values())
                           for split in SPLITS)
    missing_total = sum(sum(entry["missing_pm1_masks"]
                            for entry in split_results[split]["per_snr"].values())
                        for split in SPLITS)
    cross_segment_masks_in_final = sum(split_results[split]["totals"]["cross_segment_masks_in_final"]
                                       for split in SPLITS)

    r1_expected = {split: finalize11_summary["by_split"][split]["masked_label_cells"]
                   for split in SPLITS}
    r1_observed = {split: split_results[split]["totals"]["old_cells_labelled"] for split in SPLITS}
    audit10_r1 = json.loads((ROOT / "reports/f01e/identity_audit_10/policy_impact.json").read_text())
    audit10_r1_expectation = audit10_r1["policies"]["P2_pm1"]["masked_labelled_slots"]

    hard_checks = {
        "production_sensing_unchanged": (hash_checks["frontend/sensing/detector.py"]["unchanged"]
                                         and hash_checks["frontend/sensing/waveform.py"]["unchanged"]
                                         and provenance_checks["detector_matches_sequence_provenance"]
                                         and provenance_checks["waveform_matches_sequence_provenance"]),
        "sensing_config_unchanged": hash_checks["configs/shared_frontend.json"]["unchanged"],
        "b64_unchanged": config_checks["b64_resource_unchanged"],
        "cfar_unchanged": config_checks["cfar_multiplier_unchanged"],
        "covariance_unchanged": (config_checks["covariance_lut_reference_unchanged"]
                                 and hash_checks["reports/f01e/snr_rebuild_06/"
                                                 "covariance_calibration.json"]["unchanged"]),
        "q_a_unchanged": (config_checks["tracker_q_a_unchanged"]
                          and hash_checks["frontend/tracking/cv_kf.py"]["unchanged"]),
        "association_gate_unchanged": config_checks["association_gate_unchanged"],
        "snr_unchanged": config_checks["snr_levels_unchanged"],
        "inputs_byte_identical": inputs_identical,
        "sequences_byte_identical": sequences_identical,
        "dead_slot_labels_zero": dead_slot_labels == 0,
        "origin_ineligible_labels_zero": origin_ineligible_labels == 0,
        "cross_segment_pm1_masking_zero": cross_segment_masks_in_final == 0,
        "missing_pm1_masks_zero": missing_total == 0,
        "unexpected_invalid_beyond_policy_zero": unexpected_total == 0,
        "same_snr_guard_pass": guards["positive_pass"],
        "final_metadata_guard_pass": guards["negative_pass"],
        "generic_label_rejection_pass": guards["negative"]["generic_label"]["rejected"],
        "f01d_rejection_pass": guards["negative"]["f01d_root"]["rejected"],
        "future_leakage_false": leakage_free,
        "train_reconciliation_pass": (r1_observed["train"] == audit10_r1_expectation
                                      and all(r1_observed[split] == r1_expected[split]
                                              for split in SPLITS)),
        "non_train_structural_smoke_pass": all(
            all(entry["valid_array_mismatch"] == 0 and entry["future_array_mismatch"] == 0
                for entry in split_results[split]["per_snr"].values())
            for split in ("V_select", "V_confirm", "test")),
    }
    upstream_isac_frozen = all(hard_checks.values())

    final_manifest = {
        "stage": "F01E-FINALIZE-11R2", "formal_dataset": "F01E",
        "final_branch": "f01e-finalize-11",
        "finalize11_baseline_commit": FINALIZE11_COMMIT,
        "commit_status": "precommit tree (R2 closure commit created after this audit)",
        "precommit_provenance": {
            "generator_sha256": sha256_file(ROOT / "code/07_shared_frontend/"
                                                   "freeze07_generate_f01e.py"),
            "loader_sha256": sha256_file(ROOT / "frontend/f01e_dataset.py"),
            "freeze11_sha256": sha256_file(ROOT / "code/07_shared_frontend/"
                                                  "freeze11_finalize_pm1_mask.py"),
            "freeze12_sha256": sha256_file(Path(__file__))},
        "audit10_evidence": {"branch": AUDIT10_BRANCH, "commit": AUDIT10_COMMIT,
                             "decision": "IDENTITY_SWITCH_PM1_MASK"},
        "contract": {
            "snr_db": [-5, 0, 5, 10, 15, 20], "dt_s": 0.1, "history_frames": 20,
            "future_frames": 20, "max_slots": 8, "state_dim": 4,
            "model_input_keys": ["state_hat", "track_exists", "detected", "timestamp",
                                 "origin_eligible"],
            "label_keys": ["future_position", "label_valid"],
            "formal_loader": "frontend.f01e_dataset.F01EDataset",
            "label_alignment": "origin-safe contiguous segment",
            "sanitization_policy": "IDENTITY_SWITCH_PM1_MASK",
            "mask_radius_origins": 1,
            "mask_scope": "same_continuous_segment_only",
            "policy_source": "F01E-IDENTITY-AUDIT-10 train-only frozen decision"},
        "file_sha256": {
            "sensing_config": sha256_file(ROOT / "configs/shared_frontend.json"),
            "detector": sha256_file(ROOT / "frontend/sensing/detector.py"),
            "waveform": sha256_file(ROOT / "frontend/sensing/waveform.py"),
            "tracker": sha256_file(ROOT / "frontend/tracking/cv_kf.py"),
            "covariance_lut": sha256_file(ROOT / "reports/f01e/snr_rebuild_06/"
                                                "covariance_calibration.json"),
            "generator": sha256_file(ROOT / "code/07_shared_frontend/"
                                           "freeze07_generate_f01e.py"),
            "loader": sha256_file(ROOT / "frontend/f01e_dataset.py")},
        "manifests": manifests,
        "audit10_baseline_manifests": {name: audit10_baseline[name]
                                       for name in ("inputs", "sequences")},
        "inputs_byte_identical": inputs_identical,
        "sequences_byte_identical": sequences_identical,
        "label_rule": ("origin-safe contiguous-segment alignment then same-segment "
                       "IDENTITY_SWITCH_PM1_MASK sanitization"),
        "generated_by": "code/07_shared_frontend/freeze12_final_freeze_audit.py"}

    freeze_summary = {
        "stage": "F01E-FINALIZE-11R2", "formal_dataset": "F01E",
        "upstream_isac_frozen": upstream_isac_frozen,
        "hard_checks": hard_checks,
        "config_value_checks": config_checks,
        "hash_checks": hash_checks,
        "provenance_checks": provenance_checks,
        "pm1_same_segment_correction": {
            "old_semantics": "window offset +/-1 without same-segment check (R1)",
            "corrected_semantics": "same continuous tracker segment only (R2)",
            "by_split": {split: split_results[split]["totals"] for split in SPLITS},
            "audit10_expected_old_train": audit10_r1_expectation,
            "finalize11_r1_expected": r1_expected,
            "finalize11_r1_observed": r1_observed,
            "cross_segment_masks_prevented": totals["cross_segment_positions"],
            "cross_segment_labelled_cells_prevented": (totals["old_cells_labelled"]
                                                       - totals["corrected_cells_labelled"]),
            "representative_cases": representative_cases},
        "final_label_verification": {split: split_results[split]["per_snr"] for split in SPLITS},
        "label_totals": {
            "final_labelled_slots": final_labelled_total,
            "train_final_labelled_slots":
                finalize11_summary["by_split"]["train"]["labels_after"]["labelled_slots"],
            "train_final_coverage":
                finalize11_summary["by_split"]["train"]["coverage_after"]},
        "invariants": {"dead_slot_labels": dead_slot_labels,
                       "origin_ineligible_labels": origin_ineligible_labels,
                       "missing_pm1_masks": missing_total,
                       "unexpected_invalid_beyond_policy": unexpected_total,
                       "cross_segment_pm1_masking": cross_segment_masks_in_final},
        "loader_guards": guards,
        "next_action": "REVIEW_REQUIRED"}

    integrity = {
        "stage": "F01E-FINALIZE-11R2", "data_f01e_read_only": True,
        "inputs": {"manifest": manifests["inputs"],
                   "byte_identical_vs_audit10": inputs_identical},
        "sequences": {"manifest": manifests["sequences"],
                      "byte_identical_vs_audit10": sequences_identical},
        "labels": {"manifest": manifests["labels"]},
        "metadata": {"manifest": manifests["metadata"]},
        "audit10_baseline": {name: audit10_baseline[name] for name in ("inputs", "sequences")},
        "final_label_verification": {split: split_results[split]["per_snr"] for split in SPLITS},
        "invariants": freeze_summary["invariants"],
        "loader_guards_pass": guards["positive_pass"] and guards["negative_pass"],
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__},
        "runtime_seconds": round(time.time() - started, 1)}

    for name, payload in (("final_manifest.json", final_manifest),
                          ("freeze_summary.json", freeze_summary),
                          ("integrity.json", integrity)):
        with (OUT / name).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
            handle.write("\n")

    print(json.dumps({"upstream_isac_frozen": upstream_isac_frozen,
                      "hard_checks": hard_checks,
                      "pm1_correction": freeze_summary["pm1_same_segment_correction"]["by_split"],
                      "cross_segment_masks_prevented": totals["cross_segment_positions"],
                      "missing_pm1_masks": missing_total,
                      "dead_slot_labels": dead_slot_labels,
                      "origin_ineligible_labels": origin_ineligible_labels,
                      "final_labelled_slots": final_labelled_total,
                      "declared_next_action": "REVIEW_REQUIRED"}, ensure_ascii=False))


if __name__ == "__main__":
    main()