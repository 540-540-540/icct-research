"""Train a cross-vehicle interaction residual on a frozen NoGraph+GPT-2 base."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from frontend.automatum_prediction_dataset import AutomatumPredictionDataset
from prediction.q0.interaction import build_interaction_model
from prediction.q0.metrics import trajectory_metrics
from prediction.q0.model import build_q0_model
from prediction.q0.normalization import load_normalization, normalization_path


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    model.base.eval()
    ade = fde = 0.0
    n = 0
    for batch in loader:
        history = batch["history_state"].to(device).float()
        mask = batch["vehicle_mask"].to(device).bool()
        future = batch["future_state"].to(device).float()
        out = model(history, mask)
        metric = trajectory_metrics(out["prediction"], future, mask)
        b = history.shape[0]
        ade += float(metric["scene_ade"].sum().cpu())
        fde += float(metric["scene_fde"].sum().cpu())
        n += b
    ade, fde = ade / n, fde / n
    return {"ADE": ade, "FDE": fde, "J": ade + 0.5 * fde, "scenes": n}


def interaction_state(model):
    return {
        k: v.detach().cpu()
        for k, v in model.state_dict().items()
        if not k.startswith("base.")
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=["local", "pairwise", "pair_triplet"], required=True)
    p.add_argument("--base-checkpoint", required=True)
    p.add_argument("--snr", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--run-dir", required=True)
    args = p.parse_args()

    seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    stats = load_normalization(normalization_path(ROOT, args.snr))

    base = build_q0_model("nograph", stats, init_seed=args.seed)
    payload = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    missing, unexpected = base.load_state_dict(payload["model_state"], strict=False)
    if unexpected or missing:
        raise ValueError(f"Base checkpoint mismatch missing={missing} unexpected={unexpected}")
    model = build_interaction_model(base, args.kind, hidden_dim=64, interaction_dim=64, layers=3).to(device)

    train = AutomatumPredictionDataset("train", args.snr, ROOT, return_tensors=True)
    val = AutomatumPredictionDataset("val", args.snr, ROOT, return_tensors=True)
    train_loader = DataLoader(
        train, batch_size=args.batch_size, shuffle=True, num_workers=0,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )

    run_dir = ROOT / args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (run_dir / "parameters.json").write_text(
        json.dumps(model.parameter_summary(), indent=2) + "\n"
    )

    initial = evaluate(model, val_loader, device)
    base_val = payload.get("validation")
    if base_val is not None:
        if abs(initial["ADE"] - base_val["ADE"]) > 2e-6 or abs(initial["FDE"] - base_val["FDE"]) > 2e-6:
            raise AssertionError(f"Epoch-0 model is not the frozen base: {initial} vs {base_val}")
    print("epoch0=" + json.dumps(initial), flush=True)
    best = initial
    best_epoch = 0
    best_state = interaction_state(model)
    torch.save(
        {"interaction_state": best_state, "validation": best, "epoch": 0, "config": config},
        run_dir / "best_interaction.pt",
    )

    logs = [{"epoch": 0, "validation": initial}]
    total_steps = args.epochs * len(train_loader)
    global_step = 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        model.base.eval()
        for step, batch in enumerate(train_loader, 1):
            tick = time.perf_counter()
            history = batch["history_state"].to(device, non_blocking=True).float()
            mask = batch["vehicle_mask"].to(device, non_blocking=True).bool()
            future = batch["future_state"].to(device, non_blocking=True).float()
            optimizer.zero_grad(set_to_none=True)
            out = model(history, mask)
            metric = trajectory_metrics(out["prediction"], future, mask)
            metric["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 5.0
            )
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            global_step += 1
            if step == 1 or step == len(train_loader) or step % 20 == 0:
                elapsed = time.perf_counter() - started
                eta = elapsed / global_step * (total_steps - global_step)
                print(
                    f"epoch={epoch}/{args.epochs} step={step}/{len(train_loader)} "
                    f"loss={metric['loss'].item():.6f} ADE={metric['ADE'].item():.6f} "
                    f"FDE={metric['FDE'].item():.6f} step_s={time.perf_counter()-tick:.3f} "
                    f"elapsed_s={elapsed:.1f} eta_s={eta:.1f}",
                    flush=True,
                )

        validation = evaluate(model, val_loader, device)
        record = {"epoch": epoch, "validation": validation, "elapsed_seconds": time.perf_counter() - started}
        logs.append(record)
        print("validation=" + json.dumps(record), flush=True)
        if validation["J"] < best["J"]:
            best = validation
            best_epoch = epoch
            best_state = interaction_state(model)
            torch.save(
                {"interaction_state": best_state, "validation": best, "epoch": epoch, "config": config},
                run_dir / "best_interaction.pt",
            )

    summary = {
        "initial_validation": initial,
        "best_validation": best,
        "best_epoch": best_epoch,
        "gain_vs_frozen_base_ADE_percent": 100.0 * (1.0 - best["ADE"] / initial["ADE"]),
        "gain_vs_frozen_base_FDE_percent": 100.0 * (1.0 - best["FDE"] / initial["FDE"]),
        "elapsed_seconds": time.perf_counter() - started,
        "parameters": model.parameter_summary(),
        "test_set_used": False,
    }
    (run_dir / "training.json").write_text(json.dumps(logs, indent=2) + "\n")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

