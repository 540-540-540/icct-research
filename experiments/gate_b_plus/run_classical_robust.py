from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.gate_a.data import GateADataset
from experiments.gate_a.models import GateAModel
from experiments.gate_a.run_pilot import evaluate
from experiments.gate_b_plus.run_screen import screen_loss


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def atomic_json(path: Path, payload: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n"); tmp.replace(path)


def optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("graph", "all_graph"), required=True)
    parser.add_argument("--seed", type=int, choices=tuple(range(2026, 2031)), required=True)
    parser.add_argument("--order-seed", type=int, default=None,
                        help="Fixed batch-order seed. Omit to retain the legacy seed-coupled order.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=0,
                        help="Stop after this many validation epochs without a new best; 0 disables.")
    parser.add_argument("--fde-weight", type=float, choices=(0.5, 0.75, 1.0), default=0.5)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    cfg = json.loads((ROOT / "configs/gate_b_plus_screen.json").read_text())
    seed_all(args.seed); order_seed = args.seed if args.order_seed is None else args.order_seed
    out = ROOT / args.output
    if out.exists() and not args.resume: raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    train = GateADataset(ROOT / cfg["benchmark"], "train", limit=36643, seed=args.seed)
    val = GateADataset(ROOT / cfg["benchmark"], "val")
    val_loader = DataLoader(val, cfg["training"]["batch_size"] * 2, shuffle=False, num_workers=2)
    device = torch.device("cuda:0")
    model = GateAModel(args.kind, train.arrays["history_state"].shape[2], dt_s=cfg["dt_s"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"],
                                  weight_decay=cfg["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=2, min_lr=1e-5)
    best_path, last_path = out / "best.pt", out / "last.pt"
    records, best, best_epoch, ema, updates, elapsed = [], float("inf"), 0, None, 0, 0.0
    if last_path.exists():
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        if (state["kind"] != args.kind or state["seed"] != args.seed
                or state.get("order_seed", state["seed"]) != order_seed
                or state.get("patience", 0) != args.patience
                or state.get("fde_weight", 0.5) != args.fde_weight):
            raise RuntimeError("resume mismatch")
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        optimizer_to(optimizer, device)
        scheduler.load_state_dict(state["scheduler"])
        records, best, best_epoch, ema = state["records"], state["best"], state["best_epoch"], state["ema"]
        ema = {key: value.to(device) for key, value in ema.items()}
        updates, elapsed = state["updates"], state["elapsed_s"]
    started = time.monotonic(); stopped_early = False
    for epoch in range(len(records) + 1, args.epochs + 1):
        order = torch.randperm(
            len(train), generator=torch.Generator().manual_seed(order_seed + 1009 * epoch)
        ).tolist()
        loader = DataLoader(Subset(train, order), cfg["training"]["batch_size"], shuffle=False, num_workers=2)
        model.train(); losses = []
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = screen_loss(model(batch["history_state"], batch["node_mask"]), batch, args.fde_weight)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            current = model.state_dict()
            if ema is None: ema = {key: value.detach().clone() for key, value in current.items()}
            else:
                for key, value in current.items():
                    if value.is_floating_point() or value.is_complex(): ema[key].mul_(.999).add_(value.detach(), alpha=.001)
                    else: ema[key].copy_(value)
            losses.append(float(loss.detach())); updates += 1
        raw = {key: value.detach().clone() for key, value in model.state_dict().items()}; model.load_state_dict(ema)
        metrics = evaluate(model, val_loader, device, dt_s=cfg["dt_s"]); score = metrics["ic4"]["J_m"]
        scheduler.step(score)
        records.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation": metrics})
        if score < best:
            best, best_epoch = score, epoch
            torch.save({"model": model.state_dict(), "kind": args.kind, "seed": args.seed,
                        "epoch": epoch, "score": score}, best_path)
        model.load_state_dict(raw); elapsed_now = elapsed + time.monotonic() - started
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "kind": args.kind, "seed": args.seed, "records": records, "best": best, "best_epoch": best_epoch,
                 "order_seed": order_seed, "patience": args.patience,
                 "fde_weight": args.fde_weight,
                 "ema": ema, "updates": updates, "elapsed_s": elapsed_now}
        tmp = last_path.with_suffix(".tmp"); torch.save(state, tmp); tmp.replace(last_path)
        print(json.dumps({"kind": args.kind, "epoch": epoch, "ic4": metrics["ic4"],
                          "best_epoch": best_epoch}), flush=True)
        if args.patience and epoch - best_epoch >= args.patience:
            stopped_early = True
            break
    saved = torch.load(best_path, map_location=device, weights_only=True); model.load_state_dict(saved["model"])
    final = evaluate(model, val_loader, device, dt_s=cfg["dt_s"])
    summary = {"status": "COMPLETED", "not_formal_evidence": True, "test_accessed": False,
               "kind": args.kind, "seed": args.seed, "train_samples": len(train), "ema_decay": .999,
               "fde_weight": args.fde_weight,
               "model_seed": args.seed, "order_seed": order_seed,
               "target_epochs": args.epochs, "early_stopping_patience": args.patience,
               "stopped_early": stopped_early,
               "parameters": sum(parameter.numel() for parameter in model.parameters()),
               "epochs_completed": len(records), "checkpoint_epoch": saved["epoch"],
               "optimizer_updates": updates, "runtime_s": elapsed + time.monotonic() - started,
               "validation": final, "training": records}
    atomic_json(out / "summary.json", summary)
    print(json.dumps({"kind": args.kind, "seed": args.seed, "ic4": final["ic4"]}, indent=2))


if __name__ == "__main__": main()
