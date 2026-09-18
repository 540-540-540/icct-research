"""Validation-only Q0 trainer. This script deliberately has no test-set mode."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from frontend.automatum_prediction_dataset import AutomatumPredictionDataset
from prediction.q0.metrics import trajectory_metrics
from prediction.q0.model import build_q0_model
from prediction.q0.normalization import load_normalization, normalization_path



def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def subset(dataset, limit: int | None, seed: int) -> torch.utils.data.Dataset:
    if limit is None or limit >= len(dataset):
        return dataset
    ids = np.random.default_rng(seed).permutation(len(dataset))[:limit].tolist()
    return Subset(dataset, ids)


def mutable_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        k: v.detach().cpu()
        for k, v in model.state_dict().items()
        if not k.startswith("temporal.llm.") or "lora_" in k
    }


def trainable_groups(model: torch.nn.Module, lr: float) -> list[dict]:
    graph, lora, rest = [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("graph."):
            graph.append(parameter)
        elif "lora_" in name:
            lora.append(parameter)
        else:
            rest.append(parameter)
    groups = []
    if graph:
        groups.append({"params": graph, "lr": lr})
    if rest:
        groups.append({"params": rest, "lr": lr})
    if lora:
        groups.append({"params": lora, "lr": lr * 0.25})
    return groups


@torch.no_grad()
def evaluate(model, loader, device) -> dict[str, float]:
    model.eval()
    ade_sum = fde_sum = 0.0
    scenes = 0
    for batch in loader:
        history = batch["history_state"].to(device, non_blocking=True).float()
        mask = batch["vehicle_mask"].to(device, non_blocking=True).bool()
        future = batch["future_state"].to(device, non_blocking=True).float()
        output = model(history, mask)
        metric = trajectory_metrics(output["prediction"], future, mask)
        b = history.shape[0]
        ade_sum += float(metric["scene_ade"].sum().cpu())
        fde_sum += float(metric["scene_fde"].sum().cpu())
        scenes += b
    ade, fde = ade_sum / scenes, fde_sum / scenes
    return {"ADE": ade, "FDE": fde, "J": ade + 0.5 * fde, "scenes": scenes}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["nograph", "mpnn", "gatv2", "gated_mpnn", "pair_triplet"], required=True)
    parser.add_argument("--snr", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    stats = load_normalization(normalization_path(ROOT, args.snr))
    train_full = AutomatumPredictionDataset("train", args.snr, ROOT, return_tensors=True)
    val_full = AutomatumPredictionDataset("val", args.snr, ROOT, return_tensors=True)
    train_data = subset(train_full, args.train_limit, args.seed)
    val_data = subset(val_full, args.val_limit, args.seed + 1)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    model = build_q0_model(args.model, stats, graph_dim=64, hidden_dim=64, graph_layers=3, init_seed=args.seed).to(device)
    optimizer = torch.optim.AdamW(trainable_groups(model, args.lr), weight_decay=1e-4)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {"normalization": str(normalization_path(ROOT, args.snr))}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (run_dir / "parameters.json").write_text(
        json.dumps(model.parameter_summary(), indent=2) + "\n"
    )

    total_steps = args.epochs * len(train_loader)
    global_step = 0
    best = None
    best_state = None
    start_run = time.perf_counter()
    history_log = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        for step, batch in enumerate(train_loader, 1):
            tick = time.perf_counter()
            history = batch["history_state"].to(device, non_blocking=True).float()
            mask = batch["vehicle_mask"].to(device, non_blocking=True).bool()
            future = batch["future_state"].to(device, non_blocking=True).float()
            optimizer.zero_grad(set_to_none=True)
            output = model(history, mask)
            metric = trajectory_metrics(output["prediction"], future, mask)
            metric["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 5.0
            )
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_seconds = time.perf_counter() - tick
            global_step += 1
            elapsed = time.perf_counter() - start_run
            rate = elapsed / global_step
            remaining = rate * (total_steps - global_step)
            if step == 1 or step == len(train_loader) or step % 10 == 0:
                memory = (
                    torch.cuda.max_memory_allocated(device) / 2**30
                    if device.type == "cuda" else 0.0
                )
                print(
                    f"epoch={epoch}/{args.epochs} step={step}/{len(train_loader)} "
                    f"global={global_step}/{total_steps} loss={metric['loss'].item():.6f} "
                    f"ADE={metric['ADE'].item():.6f} FDE={metric['FDE'].item():.6f} "
                    f"step_s={step_seconds:.3f} elapsed_s={elapsed:.1f} "
                    f"eta_s={remaining:.1f} peak_vram_gb={memory:.2f}",
                    flush=True,
                )

        validation = evaluate(model, val_loader, device)
        record = {"epoch": epoch, "validation": validation, "elapsed_seconds": time.perf_counter() - start_run}
        history_log.append(record)
        print("validation=" + json.dumps(record), flush=True)
        if best is None or validation["J"] < best["J"]:
            best = validation
            best_state = mutable_state(model)
            torch.save(
                {"model_state": best_state, "validation": best, "config": config},
                run_dir / "best.pt",
            )

    (run_dir / "training.json").write_text(json.dumps(history_log, indent=2) + "\n")
    summary = {
        "best_validation": best,
        "elapsed_seconds": time.perf_counter() - start_run,
        "parameters": model.parameter_summary(),
        "test_set_used": False,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

