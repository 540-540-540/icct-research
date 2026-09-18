"""Validation-only evaluator for frozen-base interaction residual models."""
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
from prediction.q0.interaction import build_interaction_model
from prediction.q0.model import build_q0_model
from prediction.q0.normalization import load_normalization, normalization_path


def scene_errors(prediction, future_state, mask):
    target = future_state[..., :2]
    d = torch.linalg.vector_norm(prediction - target, dim=-1)
    d = torch.where(mask[:, None, :], d, torch.zeros_like(d))
    n = mask.sum(1).clamp_min(1)
    return d.sum((1, 2)) / (n * d.shape[1]), d[:, -1].sum(1) / n


def interaction_stats(history, mask):
    edge, pair = physical_edge_features(history, mask)
    last = edge[:, -1]
    dist, closing, dcpa = last[..., 4], last[..., 5], last[..., 7]
    inf = torch.full_like(dist, float("inf"))
    return {
        "vehicle_count": mask.sum(1),
        "close_pairs_20m": (((dist < 20) & pair).sum((1, 2)) // 2),
        "close_pairs_30m": (((dist < 30) & pair).sum((1, 2)) // 2),
        "closing_pairs_30m_gt_0p5": ((((dist < 30) & (closing > .5) & pair).sum((1, 2))) // 2),
        "min_pair_distance_m": torch.where(pair, dist, inf).amin((1, 2)),
        "min_cpa_distance_m": torch.where(pair, dcpa, inf).amin((1, 2)),
        "max_closing_rate_mps": torch.where(
            pair, closing, torch.full_like(closing, -float("inf"))
        ).amax((1, 2)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=["local", "pairwise", "pair_triplet"], required=True)
    p.add_argument("--base-checkpoint", required=True)
    p.add_argument("--interaction-checkpoint", required=True)
    p.add_argument("--snr", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    stats = load_normalization(normalization_path(ROOT, args.snr))
    base = build_q0_model("nograph", stats, init_seed=args.seed)
    base_payload = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    missing, unexpected = base.load_state_dict(base_payload["model_state"], strict=False)
    if missing or unexpected:
        raise ValueError(f"Base mismatch missing={missing}, unexpected={unexpected}")
    model = build_interaction_model(base, args.kind).to(device)
    interaction_payload = torch.load(
        args.interaction_checkpoint, map_location="cpu", weights_only=False
    )
    missing, unexpected = model.load_state_dict(
        interaction_payload["interaction_state"], strict=False
    )
    if unexpected or any(not k.startswith("base.") for k in missing):
        raise ValueError(f"Interaction mismatch missing={missing}, unexpected={unexpected}")
    model.eval()
    model.base.eval()

    dataset = AutomatumPredictionDataset("val", args.snr, ROOT, return_tensors=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    rows, cursor = [], 0
    with torch.no_grad():
        for batch in loader:
            h = batch["history_state"].to(device).float()
            m = batch["vehicle_mask"].to(device).bool()
            f = batch["future_state"].to(device).float()
            out = model(h, m)
            ade, fde = scene_errors(out["prediction"], f, m)
            phys = interaction_stats(h, m)
            for j in range(h.shape[0]):
                row = {
                    "dataset_index": cursor + j,
                    "scene_id": int(batch["scene_id"][j]),
                    "start_frame": int(batch["start_frame"][j]),
                    "ADE": float(ade[j].cpu()),
                    "FDE": float(fde[j].cpu()),
                }
                for key, v in phys.items():
                    x = v[j].cpu()
                    row[key] = float(x) if x.dtype.is_floating_point else int(x)
                rows.append(row)
            cursor += h.shape[0]

    summary = {
        "model": f"frozen_base_{args.kind}",
        "snr_db": args.snr,
        "checkpoint_validation": interaction_payload.get("validation"),
        "scenes": len(rows),
        "ADE": sum(r["ADE"] for r in rows) / len(rows),
        "FDE": sum(r["FDE"] for r in rows) / len(rows),
        "test_set_used": False,
    }
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

