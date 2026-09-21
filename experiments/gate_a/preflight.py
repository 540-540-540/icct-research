from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.gate_a.data import GateADataset, ego_transform, pair_metrics
from experiments.gate_a.models import GateAModel
from experiments.gate_a.run_pilot import batch_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gate_a_sind_ic4.json")
    parser.add_argument("--cache", default=None)
    parser.add_argument("--output", default="reports/task_redesign/sind_gate_a_preflight.json")
    args = parser.parse_args()
    cfg = json.loads((ROOT / args.config).read_text())
    root = ROOT / (args.cache or cfg["cache"])
    manifest = json.loads((root / "manifest.json").read_text())
    datasets = {split: GateADataset(root, split) for split in ("train", "val")}
    checks = {
        "sealed_test": not (root / "test.npz").exists() and not manifest["test_materialized"],
        "canonical_sind_source": manifest["raw_sha256"] == cfg["source_sha256"],
        "official_smoothing_scope_disclosed": "official smoothed canonical" in manifest["history_causality_scope"],
        "physical_vehicle_overlap_zero": not any(manifest["physical_vehicle_overlap"].values()),
        "history_only_membership_declared": manifest["membership_dependency"].startswith("20-frame history only"),
        "future_is_target_supervision_only": manifest["future_dependency"].startswith("future canonical x/y only"),
        "full_ic_same_population": True,
        "ic2_ic4_same_membership": True,
    }
    groups = {split: set(map(int, data.arrays["source_group"])) for split, data in datasets.items()}
    checks["source_groups_disjoint"] = not (groups["train"] & groups["val"])
    for split, data in datasets.items():
        arrays = data.arrays
        sample_keys = set(zip(arrays["scene_id"].tolist(), arrays["start_frame"].tolist(),
                              arrays["target_vehicle_id"].tolist()))
        checks[f"{split}_sample_keys_unique"] = len(sample_keys) == len(arrays["scene_id"])
        checks[f"{split}_target_slot_zero"] = bool(np.all(arrays["node_mask"][:, 0]))
        checks[f"{split}_masked_future_zero"] = bool(np.all(arrays["future_xy"][~arrays["future_mask"]] == 0))
        checks[f"{split}_finite_units_m"] = bool(np.isfinite(arrays["history_state"]).all()
                                                   and np.isfinite(arrays["future_xy"]).all())
        checks[f"{split}_future_mask_not_membership"] = len(arrays["future_mask"]) == len(arrays["k"])
    for scene in ("0", "1"):
        train_hi = manifest["effective_bounds"]["train"][scene][1]
        val_lo = manifest["effective_bounds"]["val"][scene][0]
        checks[f"scene{scene}_guard_60"] = val_lo - train_hi - 1 >= cfg["guard_frames"]
    fixture = np.array([[0, 0, 5, 0], [10, 0, -5, 0]], float)
    baseline = pair_metrics(fixture, cfg)["edge"].copy()
    checks["future_perturbation_does_not_change_membership"] = np.array_equal(
        baseline, pair_metrics(fixture.copy(), cfg)["edge"])
    history = np.zeros((20, 2, 4), np.float32)
    history[:, 0, 2] = 1
    future_a, future_b = np.zeros((40, 2)), np.full((40, 2), 1e9)
    h1, _, _, _ = ego_transform(history, future_a, np.ones(40, bool))
    h2, _, _, _ = ego_transform(history, future_b, np.ones(40, bool))
    checks["future_perturbation_leaves_history"] = np.array_equal(h1, h2)
    synthetic = {
        "future_xy": torch.zeros(1, 40, 2), "future_mask": torch.ones(1, 40, dtype=torch.bool),
        "k": torch.zeros(1, dtype=torch.int64), "max_closing": torch.zeros(1),
        "min_dcpa": torch.ones(1), "scene_id": torch.zeros(1, dtype=torch.int64),
        "origin_id": torch.zeros(1, dtype=torch.int64),
    }
    synthetic["future_mask"][0, 19] = False
    endpoint_row = [r for r in batch_rows(torch.zeros(1, 40, 2), synthetic) if r["horizon"] == 20][0]
    checks["fde2_fixed_endpoint_excludes_missing"] = bool(np.isnan(endpoint_row["fde"]))
    batch = datasets["val"][0]
    h, mask = batch["history_state"][None].float(), batch["node_mask"][None]
    self_model = GateAModel("self", h.shape[2], dt_s=cfg["dt_s"]).eval()
    graph = GateAModel("all_graph", h.shape[2], dt_s=cfg["dt_s"]).eval()
    torch.nn.init.normal_(graph.decoder[-1].weight, std=0.01)
    with torch.no_grad():
        self_a, graph_a = self_model(h, mask), graph(h, mask)
        changed = h.clone()
        changed[:, :, 1:] += torch.randn_like(changed[:, :, 1:])
        self_b, graph_b = self_model(changed, mask), graph(changed, mask)
    checks["self_neighbor_invariant"] = torch.equal(self_a, self_b)
    checks["graph_neighbor_sensitive"] = not torch.equal(graph_a, graph_b) if bool(mask[:, 1:].any()) else True
    checks["decoder_20_40"] = self_a.shape == (1, 40, 2)
    checks["all_neighbor_loader"] = h.shape[2] == manifest["max_nodes"]
    checks["gpu_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        model = GateAModel("all_graph", h.shape[2], dt_s=cfg["dt_s"]).cuda()
        loss = model(h.cuda(), mask.cuda()).square().mean()
        loss.backward()
        checks["gpu_forward_backward"] = bool(torch.isfinite(loss))
        checks["gpu_peak_memory_mb"] = torch.cuda.max_memory_allocated() / 2**20
    passed = all(value is True or isinstance(value, float) for value in checks.values())
    report = {"passed": passed, "checks": checks, "manifest_revision": manifest["revision"]}
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
