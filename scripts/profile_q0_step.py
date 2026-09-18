"""End-to-end Q0 training-step profiler; train split only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from frontend.automatum_prediction_dataset import AutomatumPredictionDataset
from prediction.q0.metrics import trajectory_metrics
from prediction.q0.model import build_q0_model
from prediction.q0.normalization import load_normalization, normalization_path
from scripts.train_q0 import seed_all, trainable_groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["nograph", "mpnn", "gatv2", "gated_mpnn", "pair_triplet"], required=True)
    parser.add_argument("--snr", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    seed_all(2026)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    stats = load_normalization(normalization_path(ROOT, args.snr))
    dataset = AutomatumPredictionDataset("train", args.snr, ROOT, return_tensors=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = build_q0_model(args.model, stats, graph_dim=64, hidden_dim=64, graph_layers=3, init_seed=2026).to(device)
    optimizer = torch.optim.AdamW(trainable_groups(model, 3e-4), weight_decay=1e-4)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    timings = []
    iterator = iter(loader)
    total = args.warmup + args.steps
    for step in range(total):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        history = batch["history_state"].to(device, non_blocking=True).float()
        mask = batch["vehicle_mask"].to(device, non_blocking=True).bool()
        future = batch["future_state"].to(device, non_blocking=True).float()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        tick = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        output = model(history, mask)
        metric = trajectory_metrics(output["prediction"], future, mask)
        metric["loss"].backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - tick
        if step >= args.warmup:
            timings.append(elapsed)

    ordered = sorted(timings)
    p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
    result = {
        "model": args.model,
        "snr_db": args.snr,
        "batch_size": args.batch_size,
        "measured_steps": len(timings),
        "mean_step_seconds": statistics.mean(timings),
        "median_step_seconds": statistics.median(timings),
        "p95_step_seconds": p95,
        "min_step_seconds": min(timings),
        "max_step_seconds": max(timings),
        "peak_vram_gb": (
            torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
        ),
        "parameters": model.parameter_summary(),
        "includes": "encoder+graph+GPT2/LoRA+head+loss+backward+optimizer",
        "test_set_used": False,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

