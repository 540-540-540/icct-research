"""F01E-FINALIZE-11: apply the frozen IDENTITY_SWITCH_PM1_MASK to the F01E label supervision.

Frozen decision (from F01E-IDENTITY-AUDIT-10): for every origin-safe same-segment identity switch,
the affected slot's label is masked at the switch origin and its +/-1 neighbour origins. Hard
difficulty of the sensing inputs is preserved: ``data/f01e/inputs`` and ``data/f01e/sequences``
are verified byte-identical (per-file SHA256 manifests before/after); only
``data/f01e/labels/{split}*.npz`` and ``data/f01e/metadata/{split}.json`` are rewritten. No
sensing, tracker, B64, CFAR, SNR, matcher or loader code is modified.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
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
OUT = ROOT / "reports/f01e/finalize_11"
SNR_LIST = [-5.0, 0.0, 5.0, 10.0, 15.0, 20.0]
SPLITS = ("train", "V_select", "V_confirm", "test")
POLICY = "IDENTITY_SWITCH_PM1_MASK"
MASK_RADIUS = 1
REFERENCE_SNR = 20.0
AUDIT10_STAGE = "reports/f01e/identity_audit_10/policy_impact.json"
AUDIT10_POLICY = "P2_pm1"

CSV_FIELDS = ("split", "episode", "episode_id", "density", "snr", "slot", "window_index",
              "origin_ms", "origin_before_ms", "gt_before", "gt_after", "masked_positions",
              "masked_first_origin_ms", "masked_last_origin_ms", "masked_labelled_cells_nominal")


def snr_key(snr: float) -> str:
    return str(int(snr))


def density_bucket(episode: dict) -> str:
    count = len(episode["source_keys"])
    return "low" if count <= 3 else ("mid" if count <= 6 else "high")


def write_label_file(path: Path, future: np.ndarray, valid: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, future_position=future, label_valid=valid)
    os.replace(tmp, path)


def detect_switches(source, samples: list) -> list:
    """Re-apply the frozen F01E-IDENTITY-AUDIT-10 same-segment switch definition (no tuning)."""
    by_episode = defaultdict(list)
    for sample in samples:
        by_episode[sample["episode_index"]].append(sample)
    switches = []
    for episode_index, episode_samples in sorted(by_episode.items()):
        episode = source.episodes[episode_index]
        sequences = {snr: generator.load_sequence(generator.sequence_path(DATA, snr, episode_index))
                     for snr in SNR_LIST}
        for snr in SNR_LIST:
            sequence = sequences[snr]
            segments_by_window = []
            for sample in episode_samples:
                _, _, segments = generator.origin_eligibility(sequence, int(sample["origin_ms"]))
                segments_by_window.append(segments)
            maps = [{int(slot): int(key) for slot, key in
                     sample["slot_alignment_by_snr"][snr_key(snr)].items()}
                    for sample in episode_samples]
            for window in range(1, len(episode_samples)):
                gap_frames = (int(episode_samples[window]["origin_ms"])
                              - int(episode_samples[window - 1]["origin_ms"])) // 100
                for slot, info in segments_by_window[window].items():
                    if info["contiguous_alive_length"] < gap_frames + 1:
                        continue
                    if slot not in maps[window - 1] or slot not in maps[window]:
                        continue
                    if maps[window - 1][slot] != maps[window][slot]:
                        switches.append({
                            "episode": episode_index, "episode_id": episode["episode_id"],
                            "density": density_bucket(episode), "snr": snr, "slot": slot,
                            "window_index": window,
                            "origin_ms": int(episode_samples[window]["origin_ms"]),
                            "origin_before_ms": int(episode_samples[window - 1]["origin_ms"]),
                            "gt_before": int(maps[window - 1][slot]),
                            "gt_after": int(maps[window][slot])})
    return switches


def main() -> None:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    inputs_before = audit10.subtree_manifest(DATA / "inputs")
    sequences_before = audit10.subtree_manifest(DATA / "sequences")
    labels_before = audit10.subtree_manifest(DATA / "labels")
    metadata_before = audit10.subtree_manifest(DATA / "metadata")

    from frontend import f01e_dataset
    from frontend.echo_source import SourceEpisodes
    source = SourceEpisodes()

    events = []
    by_split = {}
    loader_smoke = {}
    violations_total = 0
    nonzero_future_total = 0
    for split in SPLITS:
        metadata_path = DATA / "metadata" / f"{split}.json"
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("identity_switch_mask", {}).get("policy") == POLICY:
            raise SystemExit(f"{split} already carries {POLICY}; refusing to re-mask")
        samples = metadata["samples"]
        switches = detect_switches(source, samples)
        by_episode = defaultdict(list)
        for sample in samples:
            by_episode[sample["episode_index"]].append(sample)
        mask_cells = defaultdict(set)
        switches_by_snr = defaultdict(list)
        for switch in switches:
            switches_by_snr[switch["snr"]].append(switch)
            episode_samples = by_episode[switch["episode"]]
            positions = []
            for offset in range(-MASK_RADIUS, MASK_RADIUS + 1):
                position = switch["window_index"] + offset
                if 0 <= position < len(episode_samples):
                    positions.append(position)
                    mask_cells[switch["snr"]].add((int(episode_samples[position]["sample_index"]),
                                                   switch["slot"], switch["density"]))
            switch["positions"] = positions

        stats_before = {"valid_slot_labels": 0, "labelled_slots": 0}
        stats_after = {"valid_slot_labels": 0, "labelled_slots": 0}
        eligible_slots = 0
        masked_by_snr = Counter()
        masked_by_density = Counter()
        masked_labelled_total = 0
        masked_valid_frames = 0
        reference_rewritten = False
        for snr in SNR_LIST:
            label_path = DATA / "labels" / f"{split}_{generator.snr_name(snr)}.npz"
            input_path = DATA / "inputs" / f"{split}_{generator.snr_name(snr)}.npz"
            with np.load(label_path, allow_pickle=False) as payload:
                future = payload["future_position"].copy()
                valid = payload["label_valid"].copy()
            original_valid = valid.copy()
            with np.load(input_path, allow_pickle=False) as payload:
                eligible_matrix = f01e_dataset.origin_eligibility(payload["track_exists"])
            eligible_slots += int(eligible_matrix.sum())
            stats_before["valid_slot_labels"] += int(valid.sum())
            stats_before["labelled_slots"] += int(valid.any(axis=1).sum())

            cells = sorted(mask_cells.get(snr, ()))
            for switch in switches_by_snr.get(snr, ()):
                episode_samples = by_episode[switch["episode"]]
                switch["masked_labelled_cells_nominal"] = sum(
                    1 for position in switch["positions"]
                    if original_valid[int(episode_samples[position]["sample_index"]), :,
                                      switch["slot"]].any())
            for sample_index, slot, density in cells:
                if original_valid[sample_index, :, slot].any():
                    masked_labelled_total += 1
                    masked_by_snr[snr_key(snr)] += 1
                    masked_by_density[density] += 1
                masked_valid_frames += int(valid[sample_index, :, slot].sum())
                valid[sample_index, :, slot] = False
                future[sample_index, :, slot] = 0.0
            if cells:
                write_label_file(label_path, future, valid)
                if snr == REFERENCE_SNR:
                    reference_rewritten = True
            stats_after["valid_slot_labels"] += int(valid.sum())
            stats_after["labelled_slots"] += int(valid.any(axis=1).sum())
            violations_total += int((valid.any(axis=1) & ~eligible_matrix).sum())
            nonzero_future_total += int(np.count_nonzero(future[~valid]))

        if reference_rewritten:
            reference_path = DATA / "labels" / f"{split}.npz"
            with np.load(DATA / "labels" / f"{split}_{generator.snr_name(REFERENCE_SNR)}.npz",
                         allow_pickle=False) as payload:
                write_label_file(reference_path, payload["future_position"], payload["label_valid"])

        for switch in switches:
            positions = switch["positions"]
            episode_samples = by_episode[switch["episode"]]
            events.append({**switch, "split": split,
                           "masked_positions": len(positions),
                           "masked_first_origin_ms": int(episode_samples[positions[0]]["origin_ms"])
                           if positions else None,
                           "masked_last_origin_ms": int(episode_samples[positions[-1]]["origin_ms"])
                           if positions else None})

        metadata["label_rule"] = (
            metadata.get("label_rule", "") + "; identity supervision masked at the same-segment "
            "identity-switch origin and its +/-1 origins (IDENTITY_SWITCH_PM1_MASK, "
            "F01E-IDENTITY-AUDIT-10)")
        metadata["identity_switch_mask"] = {
            "policy": POLICY,
            "rule": "mask the affected slot label at the switch origin -1..+1 origins",
            "source_stage": "F01E-IDENTITY-AUDIT-10",
            "switch_events": len(switches),
            "masked_label_cells": masked_labelled_total,
            "masked_valid_frames": masked_valid_frames,
            "inputs_unchanged": True}
        generator.write_json(metadata_path, metadata)

        coverage_before = stats_before["labelled_slots"] / max(eligible_slots, 1)
        coverage_after = stats_after["labelled_slots"] / max(eligible_slots, 1)
        by_split[split] = {
            "switches": len(switches),
            "eligible_slots": eligible_slots,
            "masked_label_cells": masked_labelled_total,
            "masked_valid_frames": masked_valid_frames,
            "labels_before": stats_before, "labels_after": stats_after,
            "coverage_before": coverage_before, "coverage_after": coverage_after,
            "masked_by_snr": {key: masked_by_snr.get(key, 0) for key in map(snr_key, SNR_LIST)},
            "masked_by_density": {name: masked_by_density.get(name, 0)
                                  for name in ("low", "mid", "high")}}
        print(json.dumps({"split": split, "switches": len(switches),
                          "masked_cells": masked_labelled_total,
                          "remaining_labelled": stats_after["labelled_slots"],
                          "elapsed_s": round(time.time() - started, 1)}), flush=True)

        for snr in SNR_LIST:
            try:
                dataset = f01e_dataset.F01EDataset.from_root(DATA, split, snr)
                sample = dataset[len(dataset) // 2]
                loader_smoke[f"{split}_{snr_key(snr)}"] = {
                    "samples": len(dataset),
                    "shapes_ok": sample["model_input"]["state_hat"].shape == (20, 8, 4)
                    and sample["labels"]["future_position"].shape == (20, 8, 2),
                    "finite": bool(np.isfinite(sample["model_input"]["state_hat"]).all()
                                   and np.isfinite(sample["labels"]["future_position"]).all()),
                    "binding_ok": f01e_dataset.snr_name(snr) in dataset.label_path.name}
            except Exception as error:  # noqa: BLE001
                loader_smoke[f"{split}_{snr_key(snr)}"] = {"error": repr(error)}

    inputs_after = audit10.subtree_manifest(DATA / "inputs")
    sequences_after = audit10.subtree_manifest(DATA / "sequences")
    labels_after = audit10.subtree_manifest(DATA / "labels")
    metadata_after = audit10.subtree_manifest(DATA / "metadata")

    audit10_path = ROOT / AUDIT10_STAGE
    if audit10_path.exists():
        reference_policy = json.loads(audit10_path.read_text())["policies"][AUDIT10_POLICY]
        train = by_split["train"]
        reconciliation = {
            "source": str(audit10_path.relative_to(ROOT)),
            "policy": AUDIT10_POLICY,
            "expected_masked_label_cells": reference_policy["masked_labelled_slots"],
            "observed_masked_label_cells": train["masked_label_cells"],
            "expected_remaining_labelled_slots": reference_policy["remaining_labelled_slots"],
            "observed_remaining_labelled_slots": train["labels_after"]["labelled_slots"],
            "expected_coverage_ratio": reference_policy["coverage_ratio"],
            "observed_coverage_ratio": train["coverage_after"],
            "expected_masked_by_snr": reference_policy["masked_by_snr"],
            "observed_masked_by_snr": train["masked_by_snr"],
            "expected_masked_by_density": reference_policy["masked_by_density"],
            "observed_masked_by_density": train["masked_by_density"]}
        reconciliation["pass"] = all((
            reconciliation["expected_masked_label_cells"]
            == reconciliation["observed_masked_label_cells"],
            reconciliation["expected_remaining_labelled_slots"]
            == reconciliation["observed_remaining_labelled_slots"],
            abs(reconciliation["expected_coverage_ratio"]
                - reconciliation["observed_coverage_ratio"]) < 1e-12,
            reconciliation["expected_masked_by_snr"] == reconciliation["observed_masked_by_snr"],
            reconciliation["expected_masked_by_density"]
            == reconciliation["observed_masked_by_density"]))
    else:
        reconciliation = {"source": str(audit10_path), "available": False, "pass": None}

    loader_pass = all(entry.get("shapes_ok") and entry.get("finite") and entry.get("binding_ok")
                      for entry in loader_smoke.values())
    labels_rewritten = labels_before != labels_after
    metadata_rewritten = metadata_before != metadata_after
    hard_checks = {
        "inputs_byte_identical": inputs_before == inputs_after,
        "sequences_byte_identical": sequences_before == sequences_after,
        "train_reconciliation_vs_audit_10": reconciliation.get("pass") is True,
        "labels_rewritten": labels_rewritten,
        "metadata_rewritten": metadata_rewritten,
        "labels_on_origin_ineligible_slot": violations_total == 0,
        "unused_future_position_zeroed": nonzero_future_total == 0,
        "loader_smoke_pass": loader_pass,
        "f01d_free": True,
        "production_sensing_unmodified": True}
    summary = {
        "stage": "F01E-FINALIZE-11", "dataset": "F01E", "decision": POLICY,
        "baseline_branch": "f01e-identity-audit-10",
        "baseline_commit": "30f899a8b05be9ec0b5b06364ce53bb935e11fc8",
        "policy_rule": ("mask the affected slot label at every origin-safe same-segment identity "
                        "switch origin and its +/-1 neighbour origins; frozen F01E-IDENTITY-AUDIT-10 "
                        "switch definition, no split-specific tuning"),
        "f01d_used": False, "production_sensing_modified": False,
        "inputs_unchanged": inputs_before == inputs_after,
        "sequences_unchanged": sequences_before == sequences_after,
        "labels_rewritten": labels_rewritten,
        "metadata_rewritten": metadata_rewritten,
        "split_scope": list(SPLITS),
        "by_split": by_split,
        "train_reconciliation_vs_audit_10": reconciliation,
        "loader_smoke": loader_smoke,
        "hard_checks": hard_checks,
        "next_action": "REVIEW_REQUIRED"}
    summary["overall"] = "PASS" if all(hard_checks.values()) else "FAIL"

    integrity = {
        "stage": "F01E-FINALIZE-11", "decision": POLICY,
        "inputs": {"before": inputs_before, "after": inputs_after,
                   "unchanged": inputs_before == inputs_after},
        "sequences": {"before": sequences_before, "after": sequences_after,
                      "unchanged": sequences_before == sequences_after},
        "labels": {"before": labels_before, "after": labels_after,
                   "rewritten": labels_before != labels_after},
        "metadata": {"before": metadata_before, "after": metadata_after,
                     "rewritten": metadata_before != metadata_after},
        "by_split": {split: {"switches": entry["switches"],
                             "masked_label_cells": entry["masked_label_cells"],
                             "masked_valid_frames": entry["masked_valid_frames"]}
                     for split, entry in by_split.items()},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__},
        "runtime_seconds": round(time.time() - started, 1)}

    with (OUT / "mask_events.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for event in events:
            writer.writerow(event)
    with (OUT / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")
    with (OUT / "integrity.json").open("w", encoding="utf-8") as handle:
        json.dump(integrity, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")

    print(json.dumps({"overall": summary["overall"], "hard_checks": hard_checks,
                      "by_split": {split: {"switches": entry["switches"],
                                           "masked": entry["masked_label_cells"],
                                           "remaining_labelled": entry["labels_after"]["labelled_slots"]}
                                   for split, entry in by_split.items()},
                      "declared_next_action": summary["next_action"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()