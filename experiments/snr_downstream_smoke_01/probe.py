"""SNR-DOWNSTREAM-SMOKE-01: inference-only SNR sensitivity probe on the frozen F01-E.

Same frozen checkpoints (one classical GNN, one QGNN) and the same model weights are evaluated at
all six F01-E channels [-5, 0, 5, 10, 15, 20] dB on V_select only. No training, no SNR tuning, no
V_confirm/test access, no modification of F01-E. Inputs come from the formal
``frontend.f01e_dataset.F01EDataset``; the only adapter is the documented F01-E train normalization
(``data/f01e/normalization.json``) that reconstructs the ``standardized_state`` field the legacy
model interface expects. Outputs are a descriptive smoke/sensitivity probe, not formal paper
results.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SPLIT = "V_select"
SNRS = (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0)
DT = 0.1
OUT = ROOT / "reports/f01e/snr_downstream_smoke_01"
CHECKPOINTS = {
    "QGNN": {"kind": "original", "path": "results/qgnn_directed_context/seed2026/original/best.pt",
             "sha256": "6853008762325fc18f8e0503feb9bc1d9e8f181f063829800c115436850288ab"},
    "GNN": {"kind": "gnn", "path": "results/qgnn_motionframe/seed2026/gnn/best.pt",
            "sha256": "30154a753f756b462c37e1893fdf7934b3eef0c152f26bd47fbae60cc546d191"},
}
MODELS = ("CV", "GNN", "QGNN")


def snr_key(snr: float) -> str:
    return str(int(snr))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_f01e():
    from frontend.f01e_dataset import F01EDataset

    data = ROOT / "data/f01e"
    normalization = json.loads((data / "normalization.json").read_text())
    if normalization["fit_split"] != "train":
        raise ValueError("Normalization must be fit on train only")
    loaders = {snr: F01EDataset.from_root(data, SPLIT, snr) for snr in SNRS}
    reference = loaders[20.0]
    n = len(reference)
    for snr, dataset in loaders.items():
        if len(dataset) != n:
            raise ValueError("V_select origin count differs across SNR")
        if not np.array_equal(reference.arrays["timestamp"], dataset.arrays["timestamp"]):
            raise ValueError("V_select timestamps differ across SNR")
        if [sample["sample_index"] for sample in reference.metadata] != \
                [sample["sample_index"] for sample in dataset.metadata]:
            raise ValueError("Metadata sample order differs across SNR loaders")
    return loaders, normalization


def standardize(state_hat, exists, normalization):
    mean = np.asarray(normalization["mean"], np.float32)
    std = np.asarray(normalization["std"], np.float32)
    z = ((state_hat - mean) / std).astype(np.float32)
    z[~exists] = 0.0
    return z


def cv_predictions(state_hat, exists):
    origin = state_hat[:, -1, :, :].astype(np.float64)
    horizon = np.arange(1, 21, dtype=np.float64)[None, :, None, None] * DT
    prediction = origin[:, None, :, :2] + horizon * origin[:, None, :, 2:]
    return prediction.astype(np.float32)


def load_models(device: str):
    import torch

    from experiments.qgnn_directed_context import run as directed
    from experiments.qgnn_motionframe import run as motion
    from prediction import training

    models, meta = {}, {}
    models["QGNN"] = directed.Predictor("original", directed.configuration())
    models["GNN"] = motion.Predictor("gnn", motion.configuration())
    for name, spec in CHECKPOINTS.items():
        path = ROOT / spec["path"]
        digest = sha256_file(path)
        if digest != spec["sha256"]:
            raise ValueError(f"{name} checkpoint SHA256 changed: {digest}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        training.load_weights(models[name], payload["model"])
        models[name].requires_grad_(False)
        models[name] = models[name].to(device).eval()
        meta[name] = {"path": spec["path"], "sha256": digest,
                      "checkpoint_epoch": int(payload["progress"]["epoch"]),
                      "graph_class": type(models[name].graph).__name__}
    return models, meta


def infer(model, arrays, normalization, device: str, batch: int = 16):
    import torch

    state = torch.as_tensor(arrays["state_hat"])
    exists = torch.as_tensor(arrays["track_exists"])
    detected = torch.as_tensor(arrays["detected"])
    standardized = torch.as_tensor(standardize(arrays["state_hat"], arrays["track_exists"],
                                               normalization))
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(state), batch):
            stop = min(start + batch, len(state))
            output = model(state_hat=state[start:stop].to(device),
                           standardized_state=standardized[start:stop].to(device),
                           track_exists=exists[start:stop].to(device),
                           detected=detected[start:stop].to(device))
            predictions.append(output["prediction"].detach().cpu().numpy().astype(np.float32))
    return np.concatenate(predictions)


def metric_row(model: str, cohort: str, snr: float, prediction, future, valid, eligible) -> dict:
    from experiments.qgnn_bottleneck import metrics

    scene_macro = next(row for row in metrics.evaluate_metrics(prediction, future, valid, eligible)
                       if row["horizon_steps"] == 20)
    distance = np.linalg.norm(prediction.astype(np.float64) - future.astype(np.float64), axis=-1)
    mask = valid & eligible[:, None, :]
    point_ade = float(distance[mask].mean()) if mask.any() else None
    end = mask[:, 19]
    point_fde = float(distance[:, 19][end].mean()) if end.any() else None
    return {"model": model, "cohort": cohort, "snr_db": int(snr),
            "ADE": scene_macro["ADE"], "FDE": scene_macro["FDE"],
            "ADE_point": point_ade, "FDE_point": point_fde,
            "ade_scenes": scene_macro["ade_scenes"], "valid_points": scene_macro["valid_points"],
            "ade_targets": scene_macro["ade_targets"], "fde_targets": scene_macro["fde_targets"]}


def collect_common_targets(loaders):
    """(window, source_key, {snr: slot}, joint valid steps) for the paired cohort."""
    n = len(loaders[20.0])
    maps = {snr: [{int(slot): int(key) for slot, key in
                   sample["slot_alignment_by_snr"][snr_key(snr)].items()}
                  for sample in loaders[snr].metadata] for snr in SNRS}
    label_valid = {snr: loaders[snr].label_arrays["label_valid"] for snr in SNRS}
    future = {snr: loaders[snr].label_arrays["future_position"] for snr in SNRS}
    targets = []
    for window in range(n):
        inverse = {snr: {key: slot for slot, key in maps[snr][window].items()} for snr in SNRS}
        shared_keys = [key for key in inverse[20.0]
                       if all(key in inverse[snr] for snr in SNRS)]
        for key in shared_keys:
            slots = {snr: inverse[snr][key] for snr in SNRS}
            joint = np.ones(20, bool)
            for snr in SNRS:
                joint &= label_valid[snr][window, :, slots[snr]]
            if not joint.any():
                continue
            reference = future[20.0][window, :, slots[20.0]]
            for snr in SNRS:
                if not np.array_equal(future[snr][window, :, slots[snr]][joint], reference[joint]):
                    raise ValueError("Common cohort GT future differs across SNR")
            targets.append((window, int(key), slots, joint))
    return targets


def build_common_arrays(targets, loaders, prediction):
    """Per-SNR evaluation tensors containing only the common paired targets."""
    common = {snr: {"prediction": np.zeros_like(prediction[snr]),
                    "future": np.zeros_like(loaders[snr].label_arrays["future_position"]),
                    "valid": np.zeros(loaders[snr].label_arrays["label_valid"].shape, bool),
                    "eligible": np.zeros((loaders[snr].label_arrays["label_valid"].shape[0], 8),
                                         bool)}
              for snr in SNRS}
    for window, _key, slots, joint in targets:
        for snr in SNRS:
            slot = slots[snr]
            common[snr]["prediction"][window, :, slot] = prediction[snr][window, :, slot]
            common[snr]["future"][window, :, slot] = \
                loaders[snr].label_arrays["future_position"][window, :, slot]
            common[snr]["valid"][window, :, slot] = joint
            common[snr]["eligible"][window, slot] = True
    return common


def edge_features_np(position, velocity):
    relative_position = position[:, None, :] - position[None, :, :]
    relative_velocity = velocity[:, None, :] - velocity[None, :, :]
    distance = np.linalg.norm(relative_position, axis=-1, keepdims=True)
    direction = relative_position / np.clip(distance, 1.0, None)
    closing = -(relative_velocity * direction).sum(axis=-1, keepdims=True)
    ttc = distance / np.clip(closing, 0.25, None)
    ttc = np.where(closing > 0.0, ttc, 20.0)
    return np.concatenate([relative_position / 30.0, relative_velocity / 15.0, distance / 45.0,
                           closing / 15.0, np.clip(ttc, None, 20.0) / 20.0], axis=-1)


def input_differences(loaders, targets):
    maps = {snr: [{int(slot): int(key) for slot, key in
                   sample["slot_alignment_by_snr"][snr_key(snr)].items()}
                  for sample in loaders[snr].metadata] for snr in SNRS}
    accumulator = {snr_key(snr): {"position_origin": [], "velocity_origin": [],
                                  "position_history": [], "velocity_history": [],
                                  "edge_feature_7d": []} for snr in SNRS if snr != 20.0}
    for window, key, slots, _joint in targets:
        reference_state = loaders[20.0].arrays["state_hat"][window]
        reference_exists = loaders[20.0].arrays["track_exists"][window]
        reference_slot = slots[20.0]
        reference_map = {key_: slot for slot, key_ in maps[20.0][window].items()}
        position = reference_state[:, reference_slot, :2].astype(np.float64)
        velocity = reference_state[:, reference_slot, 2:].astype(np.float64)
        reference_edges = edge_features_np(reference_state[-1, :, :2].astype(np.float64),
                                           reference_state[-1, :, 2:].astype(np.float64))
        for snr in SNRS:
            if snr == 20.0:
                continue
            entry = accumulator[snr_key(snr)]
            other_state = loaders[snr].arrays["state_hat"][window]
            other_exists = loaders[snr].arrays["track_exists"][window]
            other_map = {key_: slot for slot, key_ in maps[snr][window].items()}
            other_slot = slots[snr]
            other_position = other_state[:, other_slot, :2].astype(np.float64)
            other_velocity = other_state[:, other_slot, 2:].astype(np.float64)
            entry["position_origin"].append(float(np.linalg.norm(
                other_position[-1] - position[-1])))
            entry["velocity_origin"].append(float(np.linalg.norm(
                other_velocity[-1] - velocity[-1])))
            both = reference_exists[:, reference_slot] & other_exists[:, other_slot]
            if both.any():
                entry["position_history"].append(float(np.linalg.norm(
                    other_position[both] - position[both], axis=-1).mean()))
                entry["velocity_history"].append(float(np.linalg.norm(
                    other_velocity[both] - velocity[both], axis=-1).mean()))
            other_edges = edge_features_np(other_state[-1, :, :2].astype(np.float64),
                                          other_state[-1, :, 2:].astype(np.float64))
            pair_keys = [other_key for other_key in reference_map
                         if other_key in other_map and other_key != key
                         and reference_exists[-1, reference_map[other_key]]
                         and other_exists[-1, other_map[other_key]]]
            for other_key in pair_keys:
                difference = (other_edges[other_slot, other_map[other_key]]
                              - reference_edges[reference_slot, reference_map[other_key]])
                entry["edge_feature_7d"].append(float(np.linalg.norm(difference)))
    return {snr: {field: (float(np.mean(values)) if values else None)
                  for field, values in entry.items()}
            for snr, entry in accumulator.items()}


def summarise(values_by_snr, base=20.0):
    base_ade = values_by_snr[base]["ADE"]
    base_fde = values_by_snr[base]["FDE"]
    return {snr_key(snr): {"ADE_degradation_percent":
                           100.0 * (values_by_snr[snr]["ADE"] - base_ade) / base_ade
                           if base_ade else None,
                           "FDE_degradation_percent":
                           100.0 * (values_by_snr[snr]["FDE"] - base_fde) / base_fde
                           if base_fde else None}
            for snr in SNRS}


def separability(values_by_snr):
    change_ade = 100.0 * (values_by_snr[-5.0]["ADE"] - values_by_snr[20.0]["ADE"]) \
        / values_by_snr[20.0]["ADE"]
    change_fde = 100.0 * (values_by_snr[-5.0]["FDE"] - values_by_snr[20.0]["FDE"]) \
        / values_by_snr[20.0]["FDE"]
    worst = max(change_ade, change_fde)
    if worst >= 10.0:
        return "CLEAR", change_ade, change_fde
    if worst >= 3.0:
        return "WEAK", change_ade, change_fde
    return "NEAR-FLAT", change_ade, change_fde


def main() -> None:
    started = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    import torch
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    OUT.mkdir(parents=True, exist_ok=True)
    loaders, normalization = load_f01e()
    n = len(loaders[20.0])
    predictions = {model: {} for model in MODELS}
    predictions["CV"] = {snr: cv_predictions(loaders[snr].arrays["state_hat"],
                                             loaders[snr].arrays["track_exists"]) for snr in SNRS}
    models, checkpoint_meta = load_models(device)
    for name, model in models.items():
        for snr in SNRS:
            predictions[name][snr] = infer(model, loaders[snr].arrays, normalization, device,
                                           batch=args.batch)
        print(json.dumps({"model": name, "device": device,
                          "elapsed_s": round(time.time() - started, 1)}), flush=True)

    rows = []
    tables = {model: {} for model in MODELS}
    for model in MODELS:
        for snr in SNRS:
            arrays = loaders[snr].arrays
            future = loaders[snr].label_arrays["future_position"]
            valid = loaders[snr].label_arrays["label_valid"]
            eligible = valid.any(axis=1)
            row = metric_row(model, "native", snr, predictions[model][snr], future, valid, eligible)
            rows.append(row)
            tables[model][snr] = {"ADE": row["ADE"], "FDE": row["FDE"],
                                  "ADE_point": row["ADE_point"], "FDE_point": row["FDE_point"],
                                  "ade_targets": row["ade_targets"], "valid_points": row["valid_points"]}

    targets = collect_common_targets(loaders)
    common_rows = []
    common_tables = {model: {} for model in MODELS}
    common_data = {}
    for model in MODELS:
        common = build_common_arrays(targets, loaders, predictions[model])
        common_data[model] = common
        for snr in SNRS:
            row = metric_row(model, "common", snr, common[snr]["prediction"], common[snr]["future"],
                             common[snr]["valid"], common[snr]["eligible"])
            common_rows.append(row)
            common_tables[model][snr] = {"ADE": row["ADE"], "FDE": row["FDE"],
                                         "ADE_point": row["ADE_point"],
                                         "FDE_point": row["FDE_point"],
                                         "ade_targets": row["ade_targets"],
                                         "valid_points": row["valid_points"]}
    rows.extend(common_rows)

    degradation = {cohort: {model: summarise((tables if cohort == "native" else common_tables)[model])
                            for model in MODELS} for cohort in ("native", "common")}
    relative = {}
    separability_verdict = {}
    for cohort, table_set in (("native", tables), ("common", common_tables)):
        relative[cohort] = {}
        for model in MODELS:
            values = table_set[model]
            verdict, change_ade, change_fde = separability(values)
            separability_verdict[f"{cohort}_{model}"] = {
                "verdict": verdict, "minus5_vs_20db_ADE_percent": change_ade,
                "minus5_vs_20db_FDE_percent": change_fde,
                "absolute_ADE_difference": values[-5.0]["ADE"] - values[20.0]["ADE"],
                "absolute_FDE_difference": values[-5.0]["FDE"] - values[20.0]["FDE"]}
            relative[cohort][model] = separability_verdict[f"{cohort}_{model}"]

    input_diff = input_differences(loaders, targets)
    common_keys = {}
    for window, key, slots, joint in targets:
        common_keys.setdefault(key, 0)
        common_keys[key] += 1

    summary = {
        "stage": "SNR-DOWNSTREAM-SMOKE-01",
        "purpose": "inference-only SNR sensitivity probe of the frozen F01-E downstream models",
        "formal_dataset": "F01E", "final_loader": "frontend.f01e_dataset.F01EDataset",
        "split_used": SPLIT, "splits_not_opened": ["V_confirm", "test", "train (data)"],
        "training_performed": False, "snr_channels": [int(snr) for snr in SNRS],
        "normalization": {"source": "data/f01e/normalization.json", "fit_split": "train",
                          "note": "legacy model interface expects standardized_state; reconstructed "
                                  "from the F01-E train normalization; no other adapter"},
        "checkpoints": checkpoint_meta,
        "cohorts": {
            "native": {"definition": "per-SNR formal label_valid",
                       "cells": {snr_key(snr): int(loaders[snr].label_arrays["label_valid"]
                                                   .any(axis=1).sum()) for snr in SNRS}},
            "common": {"definition": "same source vehicle aligned in all six SNR with jointly valid "
                                     "future frames; identical GT future and identical valid steps",
                       "targets": len(targets), "unique_vehicles": len(common_keys),
                       "windows": len({window for window, _, _, _ in targets}),
                       "cells_by_snr": {snr_key(snr): int(common_data["CV"][snr]["eligible"].sum())
                                        for snr in SNRS}}},
        "metrics": {"primary": "scene-macro ADE/FDE over 20 future steps "
                               "(experiments.qgnn_bottleneck.metrics.evaluate_metrics)",
                    "auxiliary": "pointwise ADE/FDE"},
        "tables": {"native": tables, "common": common_tables},
        "degradation_vs_20db_percent": degradation,
        "minus5_vs_20db": relative,
        "input_diff_common": input_diff,
        "separability_rule": "worst of -5dB-vs-20dB relative changes: >=10% CLEAR, >=3% WEAK, "
                             "else NEAR-FLAT",
        "separability": separability_verdict,
        "descriptive_only": True,
        "runtime_seconds": round(time.time() - started, 1),
    }
    with (OUT / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")
    with (OUT / "metrics_by_snr.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": "complete", "common_targets": len(targets),
                      "separability": {key: value["verdict"]
                                       for key, value in separability_verdict.items()},
                      "runtime_seconds": summary["runtime_seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()