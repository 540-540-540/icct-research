#!/usr/bin/env python3
"""Separate validation-population changes from checkpoint changes; no training/test."""
from pathlib import Path
import json
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from frontend.sind_prediction_dataset import SinDPredictionDataset
from frontend.sind_target_dataset import SinDTargetPredictionDataset
from prediction.qgnn_raj_pennylane import build_model, build_self_baseline


def main():
    torch.set_num_threads(4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    old = SinDPredictionDataset("val", 0.0, ROOT, False)
    new = SinDTargetPredictionDataset("val", 0.0, ROOT, True)
    lookup = {(int(sc), int(fr), int(vid)): i for i, (sc, fr, vid) in enumerate(
              zip(new.scene_id, new.start_frame, new.vehicle_ids[:, 0]))}
    assert len(lookup) == len(new)
    old_rows, old_slots = np.nonzero(old.vehicle_mask)
    matched = np.array([lookup[(int(old.scene_id[i]), int(old.start_frame[i]), int(old.vehicle_ids[i, j]))]
                        for i, j in zip(old_rows, old_slots)])
    assert len(np.unique(matched)) == len(matched)
    old_history = old.state_hat[old.snr_idx, old._history_indices[old_rows, :, old_slots]]
    new_history = new.state_hat[new.history_indices[new.history_rows[matched, 0]]]
    old_future = old.future[old_rows, :, old_slots].astype(np.float32)
    new_future = new.state_gt[new.future_indices[matched]]
    assert old_history.shape == new_history.shape
    assert old_future.shape == new_future.shape
    hd = float(np.max(np.abs(old_history-new_history)))
    fd = float(np.max(np.abs(old_future-new_future)))
    td = float(np.max(np.abs(old.history_timestamp[old_rows]-new.history_timestamp[new.origin_index[matched]])))
    assert hd == fd == td == 0.0, (hd, fd, td)
    old_selected = np.zeros(len(new), bool)
    old_selected[matched] = True
    all_mask = np.ones(len(new), bool)

    def aggregate(errors, selector):
        origins = new.origin_index[selector]
        values = errors[selector]
        counts = np.bincount(origins, minlength=new.num_origins)
        active = counts > 0
        means = np.stack([np.bincount(origins, weights=values[:, col], minlength=new.num_origins)[active]
                          / counts[active] for col in range(2)], 1)
        return {"ADE": float(means[:, 0].mean()), "FDE": float(means[:, 1].mean()),
                "targets": int(selector.sum()), "origins": int(active.sum()),
                "target_ADE": float(values[:, 0].mean()), "target_FDE": float(values[:, 1].mean())}

    report = {"test_set_used": False, "optimizer_updates": 0,
              "common_target_input_identity": {"targets": len(matched), "history_max_difference": hd,
                  "future_max_difference": fd, "timestamp_max_difference": td},
              "old_targets": int(old_selected.sum()), "new_targets": len(new),
              "added_targets": int((~old_selected).sum()), "models": {}, "decomposition": {}}
    output = ROOT / "reports/qgnn/sind_target_self_repair"
    output.mkdir(parents=True, exist_ok=True)
    cases = [("old_"+k, k, ROOT / f"results/qgnn/raj_pennylane_p2/self_{k}_0db_seed2026")
             for k in ("lstm", "transformer", "tcn")]
    cases += [("new_"+k, k, ROOT / f"results/qgnn/self_repair_v1/target_{k}_ego_v1_0db_seed2026")
              for k in ("lstm", "transformer", "tcn", "llm")]
    cases += [("repaired_old_llm", "llm", ROOT / "results/qgnn/self_repair_v1/legacy_llm_ego_v1_0db_seed2026")]
    cases = [("cv", None, None)] + cases
    loader = DataLoader(new, batch_size=256, shuffle=False, num_workers=0, pin_memory=device.type=="cuda")
    errors_saved = {}
    for label, kind, folder in cases:
        started = time.perf_counter()
        model = None
        if kind is not None:
            model = (build_model(seed=2026, phase="self", self_frame="ego_v1") if kind=="llm"
                     else build_self_baseline(kind, seed=2026)).to(device)
            saved = torch.load(folder/"best.pt", map_location="cpu", weights_only=False)
            missing, unexpected = model.load_state_dict(saved["model_state"], strict=False)
            allowed = lambda k: k.startswith("core.") or (k.startswith("llm.base.gpt2.") and "lora_" not in k)
            assert not unexpected and all(allowed(k) for k in missing)
            model.eval()
            del saved
        parts = []
        with torch.no_grad():
            for batch in loader:
                history = batch["history_state"].to(device)
                times = batch["history_timestamp"].to(device)
                truth = batch["future_state"][:, :, 0, :2].to(device)
                if model is None:
                    dt = ((times[:, -1]-times[:, 0])/19).to(history.dtype)
                    horizon = torch.arange(1, 21, dtype=history.dtype, device=device)
                    prediction = history[:, -1:, 0, :2] + history[:, -1:, 0, 2:]*dt[:, None, None]*horizon[None, :, None]
                else:
                    prediction = model(history, batch["target_mask"].to(device), times)["prediction"][:, :, 0]
                distance = torch.linalg.vector_norm(prediction-truth, dim=-1)
                parts.append(torch.stack((distance.mean(1), distance[:, -1]), 1).cpu().numpy())
        errors = np.concatenate(parts).astype(np.float64)
        assert errors.shape == (len(new), 2) and np.isfinite(errors).all()
        record = {"old_selected_population": aggregate(errors, old_selected),
                  "new_full_population": aggregate(errors, all_mask),
                  "added_targets_only": aggregate(errors, ~old_selected),
                  "evaluation_seconds": time.perf_counter()-started}
        if folder is not None:
            reference = json.loads((folder/"summary.json").read_text())["selected_checkpoint"]
            scope = "old_selected_population" if label.startswith("old_") or label=="repaired_old_llm" else "new_full_population"
            deltas = {m: abs(record[scope][m]-reference[m]) for m in ("ADE", "FDE")}
            assert max(deltas.values()) < 2e-5, (label, deltas)
            record["saved_metric_replay_max_difference"] = max(deltas.values())
            record["selected_epoch"] = reference["epoch"]
        report["models"][label] = record
        errors_saved[label] = errors.astype(np.float32)
        print(label, json.dumps(record), flush=True)
        del model
        if device.type=="cuda":
            torch.cuda.empty_cache()
    for kind in ("lstm", "transformer", "tcn", "llm"):
        a = report["models"]["repaired_old_llm" if kind=="llm" else "old_"+kind]
        b = report["models"]["new_"+kind]
        decomposition = {}
        for metric in ("ADE", "FDE"):
            old_old = a["old_selected_population"][metric]
            old_new = a["new_full_population"][metric]
            new_old = b["old_selected_population"][metric]
            new_new = b["new_full_population"][metric]
            decomposition[metric] = {
                "old_checkpoint_old_population": old_old,
                "old_checkpoint_new_population": old_new,
                "new_checkpoint_old_population": new_old,
                "new_checkpoint_new_population": new_new,
                "apparent_improvement": old_old-new_new,
                "population_change_at_fixed_old_checkpoint": old_old-old_new,
                "checkpoint_change_on_fixed_new_population": old_new-new_new,
                "checkpoint_change_on_fixed_old_population": old_old-new_old}
        report["decomposition"][kind] = decomposition
    (output/"self_population_crossover.json").write_text(json.dumps(report, indent=2)+"\n")
    # Per-target errors support later paired diagnostics; predictions and raw data are not copied.
    arrays = ROOT/"results/qgnn/self_causality_audit"
    arrays.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays/"val_target_errors.npz", origin_index=new.origin_index,
                        old_selected=old_selected, **errors_saved)
    print("PASS", json.dumps(report["decomposition"]), flush=True)


if __name__ == "__main__":
    main()
