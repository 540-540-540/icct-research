from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.gate_a.data import GateADataset
from experiments.gate_a.run_pilot import batch_loss, evaluate
from experiments.gate_b.models import GateBRajModel


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Formal Gate B launcher; validation only, test remains sealed")
    parser.add_argument("--config", default="configs/gate_b_raj_migration.json")
    parser.add_argument("--seed", type=int, choices=(2026, 2027), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    cfg = json.loads((ROOT / args.config).read_text())
    cache, out = ROOT / cfg["benchmark"], ROOT / args.output
    out.mkdir(parents=True, exist_ok=False)
    train = GateADataset(cache, "train", cfg["training"]["train_limit"], args.seed)
    val = GateADataset(cache, "val")
    val_loader = DataLoader(val, cfg["training"]["batch_size"] * 2, num_workers=2)
    summary = {"revision": cfg["revision"], "seed": args.seed, "test_accessed": False, "models": {}}
    for kind in cfg["models"]:
        seed_all(args.seed)
        model = GateBRajModel(kind, dt_s=cfg["dt_s"], rounds=cfg["rounds"]).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"],
                                      weight_decay=cfg["training"]["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, min_lr=1e-5)
        best, patience, records = float("inf"), 0, []
        checkpoint = out / f"{kind}.pt"
        for epoch in range(1, cfg["training"]["epochs"] + 1):
            order = torch.Generator().manual_seed(args.seed + 1009 * epoch)
            loader = DataLoader(train, cfg["training"]["batch_size"], shuffle=True, generator=order, num_workers=2)
            model.train(); losses = []
            for batch in loader:
                batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                pred = model(batch["history_state"][:, :, :8], batch["node_mask"][:, :8])
                loss = batch_loss(pred, batch)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
                losses.append(float(loss.detach()))
            metrics = evaluate(model, val_loader, torch.device("cuda"), dt_s=cfg["dt_s"])
            score = metrics["ic4"]["J_m"]; scheduler.step(score)
            records.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation": metrics})
            if score < best - 1e-5:
                best, patience = score, 0
                torch.save({"model": model.state_dict(), "epoch": epoch, "score": score}, checkpoint)
            else:
                patience += 1
                if patience >= cfg["training"]["patience"]: break
        saved = torch.load(checkpoint, map_location="cuda", weights_only=True); model.load_state_dict(saved["model"])
        summary["models"][kind] = {
            "parameters": model.parameter_audit(), "train_samples": len(train), "validation_samples": len(val),
            "epochs_completed": len(records), "checkpoint_epoch": saved["epoch"],
            "validation": evaluate(model, val_loader, torch.device("cuda"), dt_s=cfg["dt_s"]), "training": records,
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
