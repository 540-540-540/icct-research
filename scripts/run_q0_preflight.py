"""G0 correctness/preflight checks for the Automatum Q0 harness."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import time

import torch
from torch.utils.data import DataLoader

from frontend.automatum_prediction_dataset import AutomatumPredictionDataset, CANONICAL_DT as DATA_DT
from prediction.q0.contracts import CANONICAL_DT
from prediction.q0.features import normalize_state_and_edges, physical_edge_features
from prediction.q0.graph import EdgeGATv2, EdgeResidualMPNN, NoGraphCore
from prediction.q0.metrics import trajectory_metrics
from prediction.q0.model import build_q0_model
from prediction.q0.normalization import (
    compute_train_normalization,
    load_normalization,
    normalization_path,
    save_normalization,
)
from prediction.q0.temporal import constant_velocity_baseline



def _permute_edges(edge: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    return edge[:, :, perm][:, :, :, perm]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snr", type=float, default=0.0)
    parser.add_argument("--recompute-normalization", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    report: dict[str, object] = {"snr_db": args.snr, "checks": {}}
    if abs(CANONICAL_DT - DATA_DT) > 1e-15:
        raise AssertionError("Dataset and Q0 dt disagree")
    report["checks"]["canonical_dt"] = CANONICAL_DT

    norm_path = normalization_path(ROOT, args.snr)
    if args.recompute_normalization or not norm_path.exists():
        start = time.perf_counter()
        stats = compute_train_normalization(args.snr, root_dir=ROOT, batch_size=64)
        save_normalization(norm_path, stats, args.snr)
        report["normalization_seconds"] = time.perf_counter() - start
    else:
        stats = load_normalization(norm_path)

    train = AutomatumPredictionDataset("train", args.snr, ROOT, return_tensors=True)
    val = AutomatumPredictionDataset("val", args.snr, ROOT, return_tensors=True)
    batch = next(iter(DataLoader(train, batch_size=4, shuffle=False)))
    history = batch["history_state"].float()
    mask = batch["vehicle_mask"].bool()

    timestamps = batch["history_timestamp"].double()
    delta = timestamps[:, 1:] - timestamps[:, :-1]
    if not torch.allclose(delta, torch.full_like(delta, CANONICAL_DT), atol=1e-12, rtol=0):
        raise AssertionError("History timestamps violate canonical dt")
    report["checks"]["timestamp_grid"] = True

    edge, pair_mask = physical_edge_features(history, mask)
    if edge.shape != (history.shape[0], 20, 8, 8, 8):
        raise AssertionError("Edge feature shape mismatch")
    if edge[~pair_mask[:, None].expand(edge.shape[:4])].abs().max().item() != 0:
        raise AssertionError("Invalid/padding edges are nonzero")
    report["checks"]["physical_edges"] = True

    standardized_state, standardized_edge, pair_mask = normalize_state_and_edges(
        history, mask, stats
    )
    perm = torch.tensor([3, 0, 7, 2, 5, 1, 6, 4])
    inv = torch.argsort(perm)
    p_history = history[:, :, perm]
    p_mask = mask[:, perm]
    p_state, p_edge, p_pair = normalize_state_and_edges(p_history, p_mask, stats)

    for core in (
        NoGraphCore(hidden_dim=32, graph_dim=32),
        EdgeResidualMPNN(hidden_dim=32, graph_dim=32, layers=2),
        EdgeGATv2(hidden_dim=32, graph_dim=32, layers=2, heads=4),
    ):
        core.eval()
        with torch.no_grad():
            original = core(standardized_state, standardized_edge, pair_mask, mask)
            shuffled = core(p_state, p_edge, p_pair, p_mask)[:, :, inv]
        error = (original - shuffled).abs().max().item()
        if error > 2e-5:
            raise AssertionError(f"{core.display_name} permutation error {error}")
        report["checks"][f"{core.display_name}_permutation_max_abs"] = error

    cv = constant_velocity_baseline(history)
    expected_first = history[:, -1, :, :2] + CANONICAL_DT * history[:, -1, :, 2:4]
    valid = mask[..., None].expand_as(expected_first)
    if not torch.allclose(cv[:, 0][valid], expected_first[valid], atol=1e-6, rtol=1e-6):
        raise AssertionError("Constant-velocity baseline is not using canonical dt")
    report["checks"]["cv_uses_canonical_dt"] = True

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tiny = next(iter(DataLoader(val, batch_size=1, shuffle=False)))
    model_reports = {}
    for name in ("nograph", "mpnn", "gatv2"):
        model = build_q0_model(name, stats, graph_dim=64, hidden_dim=64, graph_layers=2).to(device)
        model.train()
        h = tiny["history_state"].to(device).float()
        m = tiny["vehicle_mask"].to(device).bool()
        future = tiny["future_state"].to(device).float()
        output = model(h, m)
        metric = trajectory_metrics(output["prediction"], future, m)
        metric["loss"].backward()
        finite_grad = all(
            torch.isfinite(p.grad).all().item()
            for p in model.parameters()
            if p.grad is not None
        )
        if not finite_grad:
            raise FloatingPointError(f"{name} has non-finite gradients")
        model_reports[name] = {
            "prediction_shape": list(output["prediction"].shape),
            "graph_shape": list(output["graph_features"].shape),
            "loss": float(metric["loss"].detach().cpu()),
            "parameters": model.parameter_summary(),
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report["models"] = model_reports
    report["train_samples"] = len(train)
    report["val_samples"] = len(val)
    report_path = ROOT / "reports/q0/g0_preflight.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

