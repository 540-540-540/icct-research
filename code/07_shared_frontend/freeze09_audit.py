"""SENS-FREEZE-09 audit: origin-safe label consistency, identity stability, loader guards (CPU-only)."""
from __future__ import annotations

import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import freeze07_generate_f01e as generator  # noqa: E402

DATA = ROOT / "data/f01e"
OUT = ROOT / "reports/f01e/label_fix_09"
SNR_LIST = [-5, 0, 5, 10, 15, 20]
SPLITS = ["train", "V_select", "V_confirm", "test"]
TAIL_THRESHOLDS = (0.5, 1.0, 2.0, 4.0)


def eligible_matrix(exists: np.ndarray) -> np.ndarray:
    return exists[:, -1, :] & (np.cumprod(exists[:, ::-1, :], axis=1).sum(axis=1) >= 3)


def percentile(values, fraction):
    return float(np.percentile(values, fraction * 100)) if len(values) else None


def _synthetic_track(times: np.ndarray, x_value: float) -> np.ndarray:
    dtype = np.dtype([("time_ms", np.int64), ("x", np.float64), ("y", np.float64),
                      ("vx", np.float64), ("vy", np.float64)])
    track = np.zeros(times.size, dtype=dtype)
    track["time_ms"] = times
    track["x"] = x_value
    return track


def synthetic_rebirth_regression() -> dict:
    """Prove the origin assignment ignores an earlier life of the same slot."""
    times = 1000 + np.arange(0, 20) * 100
    exists = np.zeros((20, 8), bool)
    states = np.zeros((20, 8, 4), np.float32)
    exists[0:10, 0] = True
    states[0:10, 0, 0] = 0.0
    exists[14:20, 0] = True
    states[14:20, 0, 0] = 50.0
    sequence = {"track_exists": exists, "state_hat": states,
                "detected": exists.copy(), "timestamp": (times / 1000.0).astype(np.float64)}
    episode = {"source_keys": [101, 202]}
    source = SimpleNamespace(tracks={101: _synthetic_track(times, 0.0),
                                     202: _synthetic_track(times, 50.0)})
    first = generator.align_window(sequence, episode, int(times[-1]), source)
    mutated = {key: value.copy() if isinstance(value, np.ndarray) else value
               for key, value in sequence.items()}
    mutated["state_hat"] = states.copy()
    mutated["state_hat"][0:10, 0, 0] = 10000.0
    second = generator.align_window(mutated, episode, int(times[-1]), source)
    first_map = {entry["slot"]: entry["source_key"] for entry in first}
    second_map = {entry["slot"]: entry["source_key"] for entry in second}
    return {"origin_alive_run": 6, "assignment": first_map,
            "old_segment_far_assignment": second_map,
            "assignment_ignores_old_segment": first_map == second_map == {0: 202},
            "regression_pass": first_map.get(0) == 202 and second_map.get(0) == 202}


def main() -> None:
    from frontend import f01e_dataset
    from frontend.echo_source import SourceEpisodes

    metadata = json.loads((DATA / "metadata" / "train.json").read_text())
    samples = metadata["samples"]
    by_episode = defaultdict(list)
    for sample in samples:
        by_episode[sample["episode_index"]].append(sample)
    source = SourceEpisodes()
    origin_errors = []
    same_segment = Counter()
    switch_examples = []
    label_rows = Counter()
    reproduction = Counter()
    dead_slot = Counter()
    labels_valid = {}
    for snr in SNR_LIST:
        with np.load(DATA / "labels" / f"train_{generator.snr_name(snr)}.npz") as payload:
            labels_valid[snr] = payload["label_valid"]
    for episode_index, episode_samples in sorted(by_episode.items()):
        episode = source.episodes[episode_index]
        sequences = {snr: generator.load_sequence(generator.sequence_path(DATA, snr, episode_index))
                     for snr in SNR_LIST}
        gt = {}
        for key in episode["source_keys"]:
            gt[int(key)] = source.tracks[int(key)]
        for sample in episode_samples:
            origin = int(sample["origin_ms"])
            window_index = sample["sample_index"]
            for snr in SNR_LIST:
                sequence = sequences[snr]
                indices, exists, segments = generator.origin_eligibility(sequence, origin)
                recomputed = {entry["slot"]: entry["source_key"]
                              for entry in generator.align_window(sequence, episode, origin, source)}
                recorded = {int(slot): int(key) for slot, key in
                            sample["slot_alignment_by_snr"][str(snr)].items()}
                reproduction["windows"] += 1
                reproduction["mismatch"] += int(recomputed != recorded)
                valid = labels_valid[snr][window_index]
                any_valid = valid.any(axis=0)
                label_rows["valid_slot_labels"] += int(valid.sum())
                label_rows["labelled_slots"] += int(any_valid.sum())
                dead = any_valid & ~exists[-1]
                ineligible = any_valid & ~np.isin(np.arange(8), list(segments))
                dead_slot["dead"] += int(dead.sum())
                dead_slot["ineligible"] += int(ineligible.sum())
                dead_slot["unverifiable"] += int((any_valid & ~np.isin(
                    np.arange(8), list(recomputed))).sum())
                times = origin + np.arange(-19, 1, dtype=np.int64) * 100
                for slot, key in recomputed.items():
                    track = gt[key]
                    position = np.searchsorted(track["time_ms"], times[-1])
                    exact = position < len(track) and track["time_ms"][position] == times[-1]
                    if exact:
                        error = float(np.linalg.norm(sequence["state_hat"][indices][19, slot, :2]
                                                     - np.array([track["x"][position],
                                                                 track["y"][position]])))
                        origin_errors.append(error)
        if len(episode_samples) < 2:
            continue
        for snr in SNR_LIST:
            sequence = sequences[snr]
            for slot in range(8):
                for previous, current in zip(episode_samples, episode_samples[1:]):
                    origin_prev = int(previous["origin_ms"])
                    origin_cur = int(current["origin_ms"])
                    gap_frames = (origin_cur - origin_prev) // 100
                    _, _, segments_cur = generator.origin_eligibility(sequence, origin_cur)
                    if slot not in segments_cur:
                        continue
                    run = segments_cur[slot]["contiguous_alive_length"]
                    if run < gap_frames + 1:
                        continue
                    map_prev = {int(k): int(v) for k, v in
                                previous["slot_alignment_by_snr"][str(snr)].items()}
                    map_cur = {int(k): int(v) for k, v in
                               current["slot_alignment_by_snr"][str(snr)].items()}
                    if slot in map_prev and slot in map_cur:
                        same_segment["transitions"] += 1
                        if map_prev[slot] != map_cur[slot]:
                            same_segment["switches"] += 1
                            if len(switch_examples) < 40:
                                switch_examples.append({"episode": episode_index, "snr": snr,
                                                        "slot": slot, "gt_before": map_prev[slot],
                                                        "gt_after": map_cur[slot]})
    origin_errors = np.asarray(origin_errors)
    identity = {
        "stage": "SENS-FREEZE-09 identity audit", "split": "train", "f01d_used": False,
        "reproduction": {"windows": reproduction["windows"],
                         "alignment_mismatch_vs_metadata": reproduction["mismatch"],
                         "reproduced": reproduction["mismatch"] == 0},
        "labels": {"valid_slot_labels": label_rows["valid_slot_labels"],
                   "labelled_slots": label_rows["labelled_slots"],
                   "labels_on_dead_slot": dead_slot["dead"],
                   "labels_on_origin_ineligible_slot": dead_slot["ineligible"],
                   "identity_unverifiable": dead_slot["unverifiable"]},
        "origin_tracking_error_m": {
            "samples": int(origin_errors.size), "p50": percentile(origin_errors, 0.50),
            "p90": percentile(origin_errors, 0.90), "p95": percentile(origin_errors, 0.95),
            "p99": percentile(origin_errors, 0.99),
            "fraction_gt_0p5m": float((origin_errors > 0.5).mean()) if origin_errors.size else None,
            "fraction_gt_1m": float((origin_errors > 1.0).mean()) if origin_errors.size else None,
            "fraction_gt_2m": float((origin_errors > 2.0).mean()) if origin_errors.size else None,
            "fraction_gt_4m": float((origin_errors > 4.0).mean()) if origin_errors.size else None,
            "note": "true tracking tail is reported as-is; no error threshold is used to drop labels"},
        "same_segment_identity": {
            "consecutive_origin_transitions": same_segment["transitions"],
            "same_segment_identity_switch_count": same_segment["switches"],
            "same_segment_identity_switch_rate": same_segment["switches"]
            / max(same_segment["transitions"], 1),
            "examples": switch_examples[:10]},
        "slot_rebirth_regression": synthetic_rebirth_regression(),
    }
    guard = {"same_snr_binding": {}, "negative_tests": {}}
    for snr in SNR_LIST:
        dataset = f01e_dataset.F01EDataset.from_root(DATA, "train", snr)
        guard["same_snr_binding"][str(snr)] = (f01e_dataset.snr_name(snr) in dataset.input_path.name
                                               and f01e_dataset.snr_name(snr) in dataset.label_path.name)
        sample = dataset[0]
        guard["negative_tests"].setdefault("loader_ok", True)
        guard.setdefault("forbidden_input_keys", []).extend(
            [key for key in sample["model_input"]
             if any(token in key.lower() for token in ("future", "gt", "truth", "label", "source_key"))])
    for name, call, expected in (
            ("minus5_input_plus20_label_rejected", lambda: f01e_dataset.F01EDataset.from_paths(
                DATA, "train", -5, label_snr_db=20), ValueError),
            ("generic_label_rejected", lambda: f01e_dataset.F01EDataset.from_paths(
                DATA, "train", 5, label_snr_db=None), RuntimeError),
            ("f01d_root_rejected", lambda: f01e_dataset.F01EDataset.from_root(
                ROOT / "data/f01d", "train", 5), RuntimeError),
            ("disallowed_snr_rejected", lambda: f01e_dataset.F01EDataset.from_root(
                DATA, "train", 7.5), ValueError)):
        try:
            call()
            guard["negative_tests"][name] = False
        except expected:
            guard["negative_tests"][name] = True
    tamper_root = Path("/tmp/f01e_tamper")
    if tamper_root.exists():
        shutil.rmtree(tamper_root)
    for sub in ("inputs", "labels", "metadata"):
        (tamper_root / sub).mkdir(parents=True, exist_ok=True)
    split_snr_name = f"train_{generator.snr_name(-5)}.npz"
    shutil.copy(DATA / "inputs" / split_snr_name, tamper_root / "inputs" / split_snr_name)
    shutil.copy(DATA / "metadata" / "train.json", tamper_root / "metadata" / "train.json")
    with np.load(DATA / "labels" / split_snr_name) as payload:
        future, valid = payload["future_position"].copy(), payload["label_valid"].copy()
    with np.load(DATA / "inputs" / split_snr_name) as payload:
        exists = payload["track_exists"]
    eligible = eligible_matrix(exists)
    candidate = None
    for sample_index in range(eligible.shape[0]):
        for slot in range(8):
            if not eligible[sample_index, slot] and not valid[sample_index, :, slot].any():
                candidate = (sample_index, slot)
                break
        if candidate:
            break
    if candidate:
        valid[candidate[0], ::2, candidate[1]] = True
    np.savez_compressed(tamper_root / "labels" / split_snr_name, future_position=future,
                        label_valid=valid)
    try:
        f01e_dataset.F01EDataset.from_root(tamper_root, "train", -5)
        guard["negative_tests"]["dead_slot_label_rejected"] = False
    except RuntimeError:
        guard["negative_tests"]["dead_slot_label_rejected"] = True
    guard["same_snr_binding_pass"] = all(guard["same_snr_binding"].values())
    guard["negative_tests_pass"] = all(guard["negative_tests"].values())
    guard["future_leakage"] = any(guard.get("forbidden_input_keys", []))
    smoke = {}
    for split in SPLITS:
        for snr in SNR_LIST:
            dataset = f01e_dataset.F01EDataset.from_root(DATA, split, snr)
            sample = dataset[len(dataset) // 2]
            smoke[f"{split}_{snr}"] = {
                "samples": len(dataset),
                "shapes_ok": sample["model_input"]["state_hat"].shape == (20, 8, 4)
                and sample["labels"]["future_position"].shape == (20, 8, 2),
                "finite": bool(np.isfinite(sample["model_input"]["state_hat"]).all()
                               and np.isfinite(sample["labels"]["future_position"]).all()),
                "binding_ok": f01e_dataset.snr_name(snr) in dataset.label_path.name}
    guard["structural_smoke_pass"] = all(all(value is True for key, value in entry.items()
                                             if key in ("shapes_ok", "finite", "binding_ok"))
                                         for entry in smoke.values())
    guard["passed"] = bool(guard["same_snr_binding_pass"] and guard["negative_tests_pass"]
                           and guard["structural_smoke_pass"] and not guard["future_leakage"])
    integrity = json.loads((OUT / "cache_integrity.json").read_text())
    summary = {
        "stage": "F01E-LABEL-FIX-09",
        "baseline_commit": "03b3ca2c4dbb080b0e0e3bee27b6ab8c2dbc3487",
        "sensing_cache_regenerated": False, "inputs_changed": not integrity["inputs_unchanged"],
        "sequences_changed": not integrity["sequences_unchanged"],
        "production_sensing_modified": False,
        "origin_safe_alignment": True,
        "labels_before": integrity["labels_before"]["train"],
        "labels_after": integrity["labels_after"]["train"],
        "coverage_ratio_after": integrity["labels_after"]["train"]["labelled_slots"]
        / max(integrity["labels_after"]["train"]["eligible_slots"], 1),
        "identity_audit": identity, "loader_guards": guard,
        "hard_checks": {
            "labels_on_dead_slot": identity["labels"]["labels_on_dead_slot"] == 0,
            "labels_on_origin_ineligible_slot": identity["labels"]["labels_on_origin_ineligible_slot"] == 0,
            "history_future_identity_mismatch": identity["reproduction"]["reproduced"],
            "identity_unverifiable": identity["labels"]["identity_unverifiable"] == 0,
            "slot_rebirth_contamination": identity["slot_rebirth_regression"]["regression_pass"],
            "inputs_unchanged": integrity["inputs_unchanged"],
            "sequences_unchanged": integrity["sequences_unchanged"],
            "same_snr_guard": guard["same_snr_binding_pass"],
            "generic_label_rejection": guard["negative_tests"].get("generic_label_rejected", False),
            "f01d_rejection": guard["negative_tests"].get("f01d_root_rejected", False),
            "future_leakage": not guard["future_leakage"],
            "dead_slot_label_rejected": guard["negative_tests"].get("dead_slot_label_rejected", False),
            "structural_smoke": guard["structural_smoke_pass"]},
    }
    summary["overall"] = "PASS" if all(summary["hard_checks"].values()) else "FAIL"
    for name, payload in (("identity_audit.json", identity), ("summary.json", summary)):
        with (OUT / name).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
            handle.write("\n")
    print(json.dumps({"overall": summary["overall"], "hard_checks": summary["hard_checks"],
                      "labels_after": summary["labels_after"],
                      "coverage": summary["coverage_ratio_after"],
                      "origin_error": {key: identity["origin_tracking_error_m"][key]
                                       for key in ("p50", "p90", "p95", "p99")},
                      "same_segment": identity["same_segment_identity"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()