"""Validation-only per-scene evaluation and interaction diagnostics for Q0."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from frontend.automatum_prediction_dataset import AutomatumPredictionDataset
from prediction.q0.features import physical_edge_features
from prediction.q0.model import build_q0_model
from prediction.q0.normalization import load_normalization, normalization_path


def scene_errors(prediction, future_state, mask):
    target = future_state[..., :2]
    distance = torch.linalg.vector_norm(prediction - target, dim=-1)
    valid = mask[:, None, :]
    distance = torch.where(valid, distance, torch.zeros_like(distance))
    n = mask.sum(dim=1).clamp_min(1)
    ade = distance.sum(dim=(1, 2)) / (n * distance.shape[1])
    fde = distance[:, -1].sum(dim=1) / n
    return ade, fde


def interaction_stats(history, mask):
    edge, pair = physical_edge_features(history, mask)
    last = edge[:, -1]
    n = mask.sum(dim=1)
    dist = last[..., 4]
    closing = last[..., 5]
    d_cpa = last[..., 7]
    pair_f = pair.to(history.dtype)
    inf = torch.full_like(dist, float("inf"))

    close20 = (((dist < 20.0) & pair).sum(dim=(1, 2)) // 2)
    close30 = (((dist < 30.0) & pair).sum(dim=(1, 2)) // 2)
    closing30 = ((((dist < 30.0) & (closing > 0.5) & pair).sum(dim=(1, 2))) // 2)
    min_distance = torch.where(pair, dist, inf).amin(dim=(1, 2))
    min_cpa = torch.where(pair, d_cpa, inf).amin(dim=(1, 2))
    max_closing = torch.where(pair, closing, torch.full_like(closing, -float("inf"))).amax(dim=(1, 2))
    return {
        "vehicle_count": n,
        "close_pairs_20m": close20,
        "close_pairs_30m": close30,
        "closing_pairs_30m_gt_0p5": closing30,
        "min_pair_distance_m": min_distance,
        "min_cpa_distance_m": min_cpa,
        "max_closing_rate_mps": max_closing,
    }


def load_checkpoint(model, path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["model_state"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise ValueError(f"Unexpected checkpoint keys: {unexpected}")
    # Frozen GPT keys may be absent in newer mutable checkpoints.
    if any(not k.startswith("temporal.llm.") for k in missing):
        raise ValueError(f"Unexpected missing checkpoint keys: {missing}")
    return payload


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["nograph", "mpnn", "gatv2", "gated_mpnn", "pair_triplet"], required=True)
    p.add_argument("--snr", type=float, default=0.0)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    stats = load_normalization(normalization_path(ROOT, args.snr))
    model = build_q0_model(args.model, stats, init_seed=args.seed).to(device)
    checkpoint = load_checkpoint(model, args.checkpoint)
    model.eval()

    dataset = AutomatumPredictionDataset("val", args.snr, ROOT, return_tensors=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    rows = []
    cursor = 0
    with torch.no_grad():
        for batch in loader:
            history = batch["history_state"].to(device).float()
            mask = batch["vehicle_mask"].to(device).bool()
            future = batch["future_state"].to(device).float()
            output = model(history, mask)
            ade, fde = scene_errors(output["prediction"], future, mask)
            stats_i = interaction_stats(history, mask)
            b = history.shape[0]
            for j in range(b):
                row = {
                    "dataset_index": cursor + j,
                    "scene_id": int(batch["scene_id"][j]),
                    "start_frame": int(batch["start_frame"][j]),
                    "ADE": float(ade[j].cpu()),
                    "FDE": float(fde[j].cpu()),
                }
                for key, values in stats_i.items():
                    value = values[j].cpu()
                    row[key] = int(value) if not value.dtype.is_floating_point else float(value)
                rows.append(row)
            cursor += b

    grouped = {}
    for n in range(2, 9):
        part = [r for r in rows if r["vehicle_count"] == n]
        if part:
            grouped[str(n)] = {
                "scenes": len(part),
                "ADE": sum(r["ADE"] for r in part) / len(part),
                "FDE": sum(r["FDE"] for r in part) / len(part),
            }
    summary = {
        "model": args.model,
        "snr_db": args.snr,
        "checkpoint_validation": checkpoint.get("validation"),
        "scenes": len(rows),
        "ADE": sum(r["ADE"] for r in rows) / len(rows),
        "FDE": sum(r["FDE"] for r in rows) / len(rows),
        "by_vehicle_count": grouped,
        "test_set_used": False,
    }
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

