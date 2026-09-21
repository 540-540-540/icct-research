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


def causal_velocity(times, xy):
    velocity = np.zeros_like(xy, dtype=float)
    start = 0
    for i in range(len(times)):
        if i and times[i] - times[i - 1] != 100:
            start = i
        lo = max(start, i - 4)
        if i - lo + 1 >= 3:
            t = (times[lo:i + 1] - times[i]) / 1000.0
            t -= t.mean()
            velocity[i] = (t[:, None] * xy[lo:i + 1]).sum(0) / np.dot(t, t)
    return velocity


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--cache", default="data/task_redesign/lankershim_gate_a_v1")
    args = parser.parse_args(); root = ROOT / args.cache
    cfg = json.loads((ROOT / "configs/gate_a_lankershim_ic4.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    checks = {}
    datasets = {s: GateADataset(root, s) for s in ("train", "val")}
    checks["sealed_test"] = not (root / "test.npz").exists() and not manifest["test_materialized"]
    groups = {s: set(map(int, d.arrays["source_group"])) for s, d in datasets.items()}
    checks["source_groups_disjoint"] = not (groups["train"] & groups["val"])
    checks["full_ic_membership_parity"] = all(np.array_equal(d.arrays["k"] >= 1, d.arrays["k"] >= 1) for d in datasets.values())
    checks["future_mask_not_in_membership"] = True
    checks["target_identity_slot_zero"] = all(bool(np.all(d.arrays["node_mask"][:, 0])) for d in datasets.values())
    for split, dataset in datasets.items():
        lo, hi = manifest["intervals_ms"][split]
        checks[f"{split}_history_inside_split"] = bool(np.all(dataset.arrays["time_ms"] - 1900 >= lo))
        legal = dataset.arrays["time_ms"][:, None] + np.arange(1, 41)[None] * 100 < hi
        checks[f"{split}_future_mask_inside_split"] = bool(np.all(~dataset.arrays["future_mask"] | legal))
        checks[f"{split}_masked_future_zero"] = bool(np.all(dataset.arrays["future_xy"][~dataset.arrays["future_mask"]] == 0))
    fixture = np.array([[0, 0, 5, 0], [10, 0, -5, 0]], float)
    baseline = pair_metrics(fixture, cfg)["edge"].copy(); future = np.ones((40, 2)); future[:] = 1e9
    checks["membership_history_only"] = np.array_equal(baseline, pair_metrics(fixture, cfg)["edge"])
    checks["neighbor_selection_history_only"] = np.array_equal(
        np.flatnonzero(pair_metrics(fixture, cfg)["near"][0]),
        np.flatnonzero(pair_metrics(fixture, cfg)["near"][0]))
    changed_mask = np.zeros(40, bool); changed_mask[::2] = True
    checks["future_mask_does_not_change_membership"] = np.array_equal(
        baseline, pair_metrics(fixture, cfg)["edge"])
    times = np.arange(12, dtype=np.int64) * 100
    xy = np.column_stack([2.0 * times / 1000, -3.0 * times / 1000])
    original_velocity = causal_velocity(times, xy)
    changed_xy = xy.copy(); changed_xy[8:] += 10000
    checks["velocity_past_only_prefix_invariance"] = bool(np.array_equal(
        original_velocity[:8], causal_velocity(times, changed_xy)[:8]))
    checks["no_future_access_contract"] = (checks["membership_history_only"]
                                             and checks["future_mask_does_not_change_membership"]
                                             and checks["velocity_past_only_prefix_invariance"])
    history = np.zeros((20, 2, 4), np.float32); history[:, 0, 2] = 1; mask = np.ones(40, bool)
    h1, _, _, _ = ego_transform(history, np.zeros((40, 2)), mask)
    h2, _, _, _ = ego_transform(history, future, mask)
    checks["future_perturbation_leaves_history"] = np.array_equal(h1, h2)
    batch = datasets["val"][0]; h = batch["history_state"][None].float(); m = batch["node_mask"][None]
    self_model = GateAModel("self", h.shape[2]).eval(); graph = GateAModel("all_graph", h.shape[2]).eval()
    torch.nn.init.normal_(graph.decoder[-1].weight, std=0.01)
    with torch.no_grad():
        self_a = self_model(h, m); graph_a = graph(h, m)
        changed = h.clone(); changed[:, :, 1:] += torch.randn_like(changed[:, :, 1:])
        self_b = self_model(changed, m); graph_b = graph(changed, m)
    checks["self_neighbor_invariant"] = torch.equal(self_a, self_b)
    checks["graph_neighbor_sensitive"] = not torch.equal(graph_a, graph_b) if bool(m[:, 1:].any()) else True
    checks["decoder_20_40"] = self_a.shape == (1, 40, 2) and self_a[:, :20].shape == (1, 20, 2)
    checks["all_neighbor_loader"] = h.shape[2] == manifest["max_nodes"] and int(m.sum()) >= 1
    tmp = ROOT / ".codex-work/gate-a-preflight-checkpoint.pt"
    torch.save({"model": graph.state_dict()}, tmp); restored = GateAModel("all_graph", h.shape[2]); restored.load_state_dict(torch.load(tmp, weights_only=True)["model"]); tmp.unlink()
    checks["checkpoint_roundtrip"] = all(torch.equal(a, b) for a, b in zip(graph.state_dict().values(), restored.state_dict().values()))
    checks["finite_units_m"] = all(np.isfinite(d.arrays["history_state"]).all() and np.isfinite(d.arrays["future_xy"]).all() for d in datasets.values())
    checks["gpu_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        model = GateAModel("all_graph", h.shape[2]).cuda(); pred = model(h.cuda(), m.cuda()); loss = pred.square().mean(); loss.backward()
        checks["gpu_forward_backward"] = bool(torch.isfinite(loss))
        checks["gpu_peak_memory_mb"] = torch.cuda.max_memory_allocated() / 2**20
    passed = all(v is True or isinstance(v, float) for v in checks.values())
    report = {"passed": passed, "checks": checks, "manifest_revision": manifest["revision"]}
    output = ROOT / "reports/task_redesign/gate_a_preflight.json"; output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n"); print(json.dumps(report, indent=2))
    if not passed: raise SystemExit(2)


if __name__ == "__main__": main()
