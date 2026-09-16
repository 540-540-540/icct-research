"""F01E-AUDIT-08 interface guard: same-SNR binding, negative tests, structural smoke (CPU-only)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data/f01e"
OUT = ROOT / "reports/f01e/audit_08"
SNR_LIST = [-5, 0, 5, 10, 15, 20]
SPLITS = ["train", "V_select", "V_confirm", "test"]
FORBIDDEN_INPUT_TOKENS = ("future", "gt", "truth", "label", "source_key", "target_id", "vehicle_id",
                          "nearest", "assigned", "alignment")


def main() -> None:
    from frontend import f01e_dataset

    guard = {"formal_loader": "frontend.f01e_dataset.F01EDataset",
             "formal_dataset_root": "data/f01e",
             "allowed_snr_db": list(f01e_dataset.ALLOWED_SNR_DB),
             "same_snr_binding": {}, "negative_tests": {}, "positive_tests": {},
             "legacy_generic_labels_used_by_formal_loader": False,
             "f01d_used_by_formal_loader": False, "future_leakage": False,
             "model_input_keys": list(f01e_dataset.MODEL_INPUT_KEYS),
             "label_keys": list(f01e_dataset.LABEL_FIELDS),
             "evaluation_only_keys": sorted({"sample_index", "episode_index", "origin_ms", "source_keys",
                                             "episode_id", "source_block", "source_groups",
                                             "label_alignment_snr", "slot_alignment",
                                             "slot_alignment_by_snr"})}
    for snr in SNR_LIST:
        dataset = f01e_dataset.F01EDataset.from_root(DATA, "train", snr)
        binding = (f01e_dataset.snr_name(snr) in dataset.input_path.name
                   and f01e_dataset.snr_name(snr) in dataset.label_path.name)
        guard["same_snr_binding"][str(snr)] = bool(binding)
        sample = dataset[0]
        inputs = sample["model_input"]
        labels = sample["labels"]
        guard["positive_tests"][str(snr)] = {
            "samples": len(dataset), "input": dataset.input_path.name, "labels": dataset.label_path.name,
            "state_shape": list(inputs["state_hat"].shape), "label_shape": list(labels["future_position"].shape),
            "finite": bool(np.isfinite(inputs["state_hat"]).all()
                           and np.isfinite(labels["future_position"]).all()),
            "forbidden_input_keys": [key for key in inputs if any(token in key.lower()
                                                                  for token in FORBIDDEN_INPUT_TOKENS)]}
    try:
        f01e_dataset.F01EDataset.from_paths(DATA, "train", -5, label_snr_db=20)
        guard["negative_tests"]["minus5_input_plus20_label_rejected"] = False
    except ValueError:
        guard["negative_tests"]["minus5_input_plus20_label_rejected"] = True
    try:
        f01e_dataset.F01EDataset.from_paths(DATA, "train", 5, label_snr_db=None)
        guard["negative_tests"]["plus5_input_generic_label_rejected"] = False
    except RuntimeError:
        guard["negative_tests"]["plus5_input_generic_label_rejected"] = True
    try:
        f01e_dataset.F01EDataset.from_root(ROOT / "data/f01d", "train", 5)
        guard["negative_tests"]["f01d_root_rejected"] = False
    except RuntimeError:
        guard["negative_tests"]["f01d_root_rejected"] = True
    try:
        f01e_dataset.F01EDataset.from_root(DATA, "train", 7.5)
        guard["negative_tests"]["disallowed_snr_rejected"] = False
    except ValueError:
        guard["negative_tests"]["disallowed_snr_rejected"] = True

    smoke = {}
    for split in SPLITS:
        smoke[split] = {}
        for snr in SNR_LIST:
            dataset = f01e_dataset.F01EDataset.from_root(DATA, split, snr)
            total = len(dataset)
            indices = sorted({0, total // 2, total - 1})
            checks = {"samples": total, "loaded": len(indices), "shapes_ok": True, "finite": True,
                      "mask_relation": True, "zero_padding": True, "labels_shapes_ok": True,
                      "binding_ok": (f01e_dataset.snr_name(snr) in dataset.label_path.name)}
            for index in indices:
                sample = dataset[index]
                inputs, labels = sample["model_input"], sample["labels"]
                checks["shapes_ok"] &= inputs["state_hat"].shape == (20, 8, 4) \
                    and inputs["track_exists"].shape == (20, 8)
                checks["finite"] &= bool(np.isfinite(inputs["state_hat"]).all()
                                         and np.isfinite(labels["future_position"]).all())
                checks["mask_relation"] &= bool(not np.any(inputs["detected"] & ~inputs["track_exists"]))
                checks["zero_padding"] &= bool(np.all(inputs["state_hat"][~inputs["track_exists"]] == 0))
                checks["labels_shapes_ok"] &= labels["future_position"].shape == (20, 8, 2)
            smoke[split][str(snr)] = {key: bool(value) if isinstance(value, (bool, np.bool_)) else value
                                      for key, value in checks.items()}
    guard["structural_smoke"] = smoke
    guard["structural_smoke_pass"] = all(
        all(value is True for key, value in checks.items()
            if key in ("shapes_ok", "finite", "mask_relation", "zero_padding", "labels_shapes_ok",
                       "binding_ok"))
        for split in smoke.values() for checks in split.values())
    guard["same_snr_binding_pass"] = all(guard["same_snr_binding"].values())
    guard["negative_tests_pass"] = all(guard["negative_tests"].values())
    guard["future_leakage"] = any(checks["forbidden_input_keys"] for checks in guard["positive_tests"].values())
    guard["passed"] = bool(guard["same_snr_binding_pass"] and guard["negative_tests_pass"]
                           and guard["structural_smoke_pass"] and not guard["future_leakage"])
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "interface_snr_guard.json").open("w", encoding="utf-8") as handle:
        json.dump(guard, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")

    contract = json.loads((OUT / "data_contract.json").read_text())
    tail = json.loads((OUT / "tail_attribution_summary.json").read_text())
    label = json.loads((OUT / "label_identity_audit.json").read_text())
    summary = {
        "stage": "F01E-AUDIT-08", "formal_dataset": "F01E", "f01d_used": False,
        "f01d_used_for_audit": False, "sensing_cache_regenerated": False,
        "production_sensing_modified": False,
        "baseline_note": "work order baseline 7f216f2feaa9de88e2618538e7a37e6dd3453987 is the git TREE of "
                         "commit 9473d177078277891c776d80896b7353103077b3 (tip of sens-freeze-07); audit "
                         "based on 9473d17 whose content is identical",
        "tail_attribution_complete": True,
        "current_assignment_reproduced_from_metadata": True,
        "label_identity_verifiable": label["window_labels"]["identity_unverifiable_count"] == 0,
        "label_mapping_internal_consistency": label["window_labels"]["history_future_identity_mismatch_count"] == 0,
        "history_future_identity_safe": label["training_pair_identity_check"]["inconsistent_gt_1m_total"] == 0,
        "training_pair_identity_check": {
            "valid_slot_labels": label["training_pair_identity_check"]["valid_slot_labels_total"],
            "origin_state_far_from_label_vehicle_gt_1m":
                label["training_pair_identity_check"]["inconsistent_gt_1m_total"],
            "fraction": label["training_pair_identity_check"]["inconsistent_gt_1m_fraction"],
            "labels_on_dead_slot": label["training_pair_identity_check"]["labels_on_dead_slot_total"]},
        "snr_loader_guard": guard["passed"],
        "future_leakage": bool(guard["future_leakage"]),
        "diagnosis": {
            "dominant_tail_category": max(tail["tail_gt_1m"]["cause_fraction_of_tail"],
                                          key=tail["tail_gt_1m"]["cause_fraction_of_tail"].get),
            "unassigned_alive_fraction_of_alive": tail["unassigned_alive_states"]["fraction_of_alive"],
            "alignment_suspect_with_slot_rebirth_in_window":
                tail["alignment_suspect_subtypes"]["slot_rebirth_inside_window"],
            "snr_dependence_of_tail": {snr: tail["by_snr"][snr]["tail_gt_1_fraction"]
                                       for snr in [str(value) for value in tail["snr_levels"]]},
            "alignment_suspect_fraction_of_tail": tail["tail_gt_1m"]["cause_fraction_of_tail"][
                "alignment_suspect"],
            "true_tracking_fraction_of_tail": tail["tail_gt_1m"]["cause_fraction_of_tail"][
                "true_tracking_tail"],
            "ambiguous_fraction_of_tail": tail["tail_gt_1m"]["cause_fraction_of_tail"][
                "ambiguous_close_vehicle"],
            "unresolved_fraction_of_tail": tail["tail_gt_1m"]["cause_fraction_of_tail"]["unresolved"],
            "detected_only_fraction_of_tail": tail["by_detection_state"]["detected"]["cause_fraction_of_tail"],
            "valid_states": tail["valid_states"],
            "tail_gt_1m_fraction": tail["tail_gt_1m"]["fraction_all_valid_states"],
        },
        "next_action_recommended_by_evidence": "REVIEW_REQUIRED",
    }
    with (OUT / "audit_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")
    print(json.dumps({"guard": guard["passed"], "binding": guard["same_snr_binding_pass"],
                      "negative": guard["negative_tests"], "smoke": guard["structural_smoke_pass"],
                      "leakage": guard["future_leakage"], "summary": summary["diagnosis"]}))


if __name__ == "__main__":
    main()