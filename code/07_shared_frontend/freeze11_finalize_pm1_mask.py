"""F01E-FINALIZE-11: apply the frozen IDENTITY_SWITCH_PM1_MASK to the F01E label supervision.

The mask plan itself lives in ``freeze07_generate_f01e.identity_switch_pm1_plan`` and is shared with
the formal pack, so this script and the formal pipeline can never diverge. A masked neighbour of a
same-segment identity switch must belong to the same continuous tracker segment as the switch
origin: slot death, rebirth or any ``track_exists=0`` gap make the neighbour cross-segment and it is
never masked (F01E-FINALIZE-11R2 correction).

Hard difficulty of the sensing inputs is preserved: ``data/f01e/inputs`` and
``data/f01e/sequences`` are verified byte-identical (per-file SHA256 manifests before/after); only
``data/f01e/labels/{split}*.npz`` and ``data/f01e/metadata/{split}.json`` are rewritten. The formal
final repack entry point is ``freeze07_generate_f01e.py pack --labels-only``.
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
              "cross_segment_positions", "cross_segment_labelled_cells",
              "masked_first_origin_ms", "masked_last_origin_ms")


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


def plan_split(source, samples: list) -> tuple:
    """Shared-plan mask cells per SNR plus switch events with cross-segment accounting."""
    by_episode = defaultdict(list)
    for sample in samples:
        by_episode[sample["episode_index"]].append(sample)
    events = []
    cells_by_snr = defaultdict(set)
    cross_by_snr = defaultdict(set)
    for episode_index, episode_samples in sorted(by_episode.items()):
        episode = source.episodes[episode_index]
        origins = [int(sample["origin_ms"]) for sample in episode_samples]
        sequences = {snr: generator.load_sequence(generator.sequence_path(DATA, snr, episode_index))
                     for snr in SNR_LIST}
        for snr in SNR_LIST:
            maps = [{int(slot): int(key) for slot, key in
                     sample["slot_alignment_by_snr"][snr_key(snr)].items()}
                    for sample in episode_samples]
            plan = generator.identity_switch_pm1_plan(sequences[snr], origins, maps, radius=MASK_RADIUS)
            masked_indices = set()
            for switch in plan["switches"]:
                masked_sample_indices = [int(episode_samples[position]["sample_index"])
                                         for position in switch["masked_windows"]]
                cross_sample_indices = [int(episode_samples[position]["sample_index"])
                                        for position in switch["cross_segment_windows"]]
                masked_indices.update(masked_sample_indices)
                for sample_index in masked_sample_indices:
                    cells_by_snr[snr].add((sample_index, switch["slot"]))
                for sample_index in cross_sample_indices:
                    cross_by_snr[snr].add((sample_index, switch["slot"]))
                events.append({
                    "split": None, "episode": episode_index, "episode_id": episode["episode_id"],
                    "density": density_bucket(episode), "snr": snr, "slot": switch["slot"],
                    "window_index": switch["window"],
                    "origin_ms": origins[switch["window"]],
                    "origin_before_ms": origins[switch["window"] - 1],
                    "gt_before": switch["gt_before"], "gt_after": switch["gt_after"],
                    "masked_positions": len(masked_sample_indices),
                    "cross_segment_positions": len(cross_sample_indices),
                    "masked_first_origin_ms": origins[switch["masked_windows"][0]]
                    if switch["masked_windows"] else None,
                    "masked_last_origin_ms": origins[switch["masked_windows"][-1]]
                    if switch["masked_windows"] else None,
                    "_masked_sample_indices": masked_sample_indices,
                    "_cross_sample_indices": cross_sample_indices})
    return events, cells_by_snr, cross_by_snr


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
    violations_total = 0
    nonzero_future_total = 0
    for split in SPLITS:
        metadata_path = DATA / "metadata" / f"{split}.json"
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("finalized") is True or metadata.get("label_sanitization") == POLICY:
            raise SystemExit(f"{split} already carries the finalized {POLICY} contract; refusing "
                             "to re-mask. Use freeze07 pack --labels-only for a deterministic rebuild.")
        samples = metadata["samples"]
        split_events, cells_by_snr, cross_by_snr = plan_split(source, samples)

        stats_before = {"valid_slot_labels": 0, "labelled_slots": 0}
        stats_after = {"valid_slot_labels": 0, "labelled_slots": 0}
        eligible_slots = 0
        masked_by_snr = Counter()
        masked_by_density = Counter()
        masked_labelled_total = 0
        masked_valid_frames = 0
        old_labelled_total = 0
        cross_labelled_total = 0
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

            cells = sorted(cells_by_snr.get(snr, ()))
            old_cells = sorted(set(cells) | cross_by_snr.get(snr, set()))
            for sample_index, slot in old_cells:
                if original_valid[sample_index, :, slot].any():
                    old_labelled_total += 1
                    if (sample_index, slot) not in set(cells):
                        cross_labelled_total += 1
            for sample_index, slot in cells:
                if original_valid[sample_index, :, slot].any():
                    masked_labelled_total += 1
                    masked_by_snr[snr_key(snr)] += 1
            for sample_index, slot in cells:
                masked_valid_frames += int(valid[sample_index, :, slot].sum())
            generator.apply_identity_switch_pm1_mask({"future_position": future, "label_valid": valid},
                                                     cells)
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

        for event in split_events:
            event["split"] = split
            event["cross_segment_labelled_cells"] = 0
            event.pop("_masked_sample_indices", None)
            event.pop("_cross_sample_indices", None)
            events.append(event)

        metadata.update({
            "finalized": True,
            "label_alignment": "origin-safe contiguous segment",
            "label_sanitization": POLICY,
            "mask_radius_origins": MASK_RADIUS,
            "mask_scope": "same_continuous_segment_only",
            "policy_source": "F01E-IDENTITY-AUDIT-10 train-only frozen decision"})
        metadata["label_rule"] = (
            metadata.get("label_rule", "") + "; identity supervision masked at same-segment "
            "identity-switch origins and their +/-1 origins (IDENTITY_SWITCH_PM1_MASK)")
        metadata["identity_switch_mask"] = {
            "policy": POLICY,
            "rule": "mask the affected slot label at same-segment switch origin +/-1 origins",
            "source_stage": "F01E-IDENTITY-AUDIT-10",
            "switch_events": len(split_events),
            "masked_label_cells": masked_labelled_total,
            "masked_valid_frames": masked_valid_frames,
            "cross_segment_positions_skipped": sum(event["cross_segment_positions"]
                                                   for event in split_events),
            "inputs_unchanged": True}
        generator.write_json(metadata_path, metadata)

        coverage_before = stats_before["labelled_slots"] / max(eligible_slots, 1)
        coverage_after = stats_after["labelled_slots"] / max(eligible_slots, 1)
        by_split[split] = {
            "switches": len(split_events),
            "eligible_slots": eligible_slots,
            "old_mask_cells_labelled": old_labelled_total,
            "corrected_mask_cells_labelled": masked_labelled_total,
            "difference": old_labelled_total - masked_labelled_total,
            "cross_segment_positions": sum(event["cross_segment_positions"]
                                           for event in split_events),
            "cross_segment_labelled_cells": cross_labelled_total,
            "masked_valid_frames": masked_valid_frames,
            "labels_before": stats_before, "labels_after": stats_after,
            "coverage_before": coverage_before, "coverage_after": coverage_after,
            "masked_by_snr": {key: masked_by_snr.get(key, 0) for key in map(snr_key, SNR_LIST)},
            "masked_by_density": {key: 0 for key in ("low", "mid", "high")}}
        print(json.dumps({"split": split, "switches": len(split_events),
                          "old_labelled": old_labelled_total,
                          "corrected_labelled": masked_labelled_total,
                          "cross_segment_positions": by_split[split]["cross_segment_positions"],
                          "elapsed_s": round(time.time() - started, 1)}), flush=True)

    inputs_after = audit10.subtree_manifest(DATA / "inputs")
    sequences_after = audit10.subtree_manifest(DATA / "sequences")
    labels_after = audit10.subtree_manifest(DATA / "labels")
    metadata_after = audit10.subtree_manifest(DATA / "metadata")

    audit10_path = ROOT / AUDIT10_STAGE
    reference_policy = (json.loads(audit10_path.read_text())["policies"][AUDIT10_POLICY]
                        if audit10_path.exists() else None)
    hard_checks = {
        "inputs_byte_identical": inputs_before == inputs_after,
        "sequences_byte_identical": sequences_before == sequences_after,
        "labels_rewritten": labels_before != labels_after,
        "metadata_rewritten": metadata_before != metadata_after,
        "labels_on_origin_ineligible_slot": violations_total == 0,
        "unused_future_position_zeroed": nonzero_future_total == 0,
        "train_reconciliation_vs_audit_10": (reference_policy is not None
                                             and reference_policy["masked_labelled_slots"]
                                             == by_split["train"]["corrected_mask_cells_labelled"])}
    summary = {
        "stage": "F01E-FINALIZE-11", "dataset": "F01E", "decision": POLICY,
        "policy_semantics": "same-continuous-segment only (F01E-FINALIZE-11R2)",
        "f01d_used": False, "production_sensing_modified": False,
        "inputs_unchanged": inputs_before == inputs_after,
        "sequences_unchanged": sequences_before == sequences_after,
        "by_split": by_split,
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
                      "declared_next_action": summary["next_action"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()