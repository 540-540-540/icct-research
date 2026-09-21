from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.gate_a.data import GateADataset
from experiments.gate_a.models import GateAModel


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def batch_loss(pred, batch):
    mask = batch["future_mask"]
    error = torch.linalg.vector_norm(pred - batch["future_xy"], dim=-1)
    weight = batch["target_weight"]
    ade_valid = mask.any(1)
    ade = (error * mask).sum(1) / mask.sum(1).clamp_min(1)
    loss = (ade[ade_valid] * weight[ade_valid]).sum() / weight[ade_valid].sum().clamp_min(1e-8)
    endpoint = mask[:, -1]
    if endpoint.any():
        fde = error[:, -1]
        loss = loss + 0.5 * (fde[endpoint] * weight[endpoint]).sum() / weight[endpoint].sum().clamp_min(1e-8)
    return loss


def batch_rows(pred, batch):
    error = torch.linalg.vector_norm(pred.cpu() - batch["future_xy"], dim=-1)
    mask = batch["future_mask"]
    origin = batch.get("origin_id", batch.get("time_ms"))
    rows = []
    for horizon in (20, 40):
        hmask = mask[:, :horizon]
        ade = (error[:, :horizon] * hmask).sum(1) / hmask.sum(1).clamp_min(1)
        for i in range(len(error)):
            rows.append({
                "horizon": horizon,
                "ade": float(ade[i]) if bool(hmask[i].any()) else float("nan"),
                "fde": float(error[i, horizon - 1]) if bool(mask[i, horizon - 1]) else float("nan"),
                "complete": bool(hmask[i].all()), "k": int(batch["k"][i]),
                "closing": float(batch["max_closing"][i]), "dcpa": float(batch["min_dcpa"][i]),
                "scene": int(batch.get("scene_id", torch.zeros(len(error), dtype=torch.int32))[i]),
                "origin": int(origin[i]),
            })
    return rows


def aggregate(rows):
    result = {}
    for metric in ("ade", "fde"):
        groups = defaultdict(list)
        for row in rows:
            if np.isfinite(row[metric]):
                groups[row["origin"]].append(row[metric])
        result[f"{metric}_m"] = float(np.mean([np.mean(v) for v in groups.values()])) if groups else float("nan")
        result[f"{metric}_n"] = int(sum(map(len, groups.values())))
        result[f"{metric}_origins"] = len(groups)
    result["J_m"] = result["ade_m"] + 0.5 * result["fde_m"]
    return result


def summarize_rows(rows):
    result = {}
    predicates = {
        "full": lambda r: True, "ic": lambda r: r["k"] >= 1,
        "k0": lambda r: r["k"] == 0, "k1": lambda r: r["k"] == 1,
        "k2plus": lambda r: r["k"] >= 2, "closing1": lambda r: r["closing"] >= 1.0,
        "cpa5": lambda r: r["dcpa"] <= 5.0,
        "complete_full": lambda r: r["complete"],
        "complete_ic": lambda r: r["complete"] and r["k"] >= 1,
    }
    for scene in sorted({r["scene"] for r in rows}):
        predicates[f"city{scene}_full"] = lambda r, scene=scene: r["scene"] == scene
        predicates[f"city{scene}_ic"] = lambda r, scene=scene: r["scene"] == scene and r["k"] >= 1
    for horizon in (20, 40):
        suffix = horizon // 10
        horizon_rows = [r for r in rows if r["horizon"] == horizon]
        for name, predicate in predicates.items():
            selected = [r for r in horizon_rows if predicate(r)]
            if selected:
                result[f"{name}{suffix}"] = aggregate(selected)
    return result


@torch.no_grad()
def evaluate(model, loader, device, cv=False, dt_s=0.1):
    if model is not None:
        model.eval()
    rows = []
    for batch in loader:
        if cv:
            velocity = batch["history_state"][:, -1, 0, 2:4]
            times = torch.arange(1, 41, dtype=velocity.dtype) * dt_s
            pred = velocity[:, None] * times[None, :, None]
        else:
            pred = model(batch["history_state"].to(device), batch["node_mask"].to(device)).cpu()
        rows.extend(batch_rows(pred, batch))
    return summarize_rows(rows)


def train(kind, seed, cfg, cache, out, device):
    train_set = GateADataset(cache, "train", cfg["pilot"]["train_limit"], seed)
    validation = GateADataset(cache, "val")
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_set, cfg["pilot"]["batch_size"], shuffle=True, generator=generator,
                              num_workers=2, persistent_workers=True)
    validation_loader = DataLoader(validation, cfg["pilot"]["batch_size"] * 2, num_workers=2)
    model = GateAModel(kind, train_set.arrays["history_state"].shape[2], dt_s=cfg["dt_s"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["pilot"]["learning_rate"],
                                  weight_decay=cfg["pilot"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, min_lr=1e-5)
    scaler = torch.amp.GradScaler("cuda")
    best, patience, updates = float("inf"), 0, 0
    checkpoint = out / f"{kind}.pt"
    log, started = [], time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(cfg["pilot"]["epochs"]):
        model.train()
        losses = []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                loss = batch_loss(model(batch["history_state"], batch["node_mask"]), batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
            updates += 1
        validation_metrics = evaluate(model, validation_loader, device, dt_s=cfg["dt_s"])
        score = validation_metrics["ic4"]["J_m"]
        scheduler.step(score)
        log.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
                    "learning_rate": optimizer.param_groups[0]["lr"], "validation": validation_metrics})
        print(kind, {"epoch": epoch + 1, "loss": log[-1]["train_loss"], "ic4": validation_metrics["ic4"]}, flush=True)
        if score < best - 1e-5:
            best, patience = score, 0
            torch.save({"model": model.state_dict(), "epoch": epoch + 1, "score": best,
                        "kind": kind, "seed": seed}, checkpoint)
        else:
            patience += 1
            if patience >= cfg["pilot"]["patience"]:
                break
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["model"])
    descending = len(log) >= 3 and all(log[i]["validation"]["ic4"]["J_m"] > log[i + 1]["validation"]["ic4"]["J_m"]
                                       for i in range(len(log) - 3, len(log) - 1))
    return {
        "kind": kind, "seed": seed, "parameters": sum(p.numel() for p in model.parameters()),
        "total_train_samples": len(train_set.arrays["history_state"]), "actual_train_samples": len(train_set),
        "train_fraction": len(train_set) / len(train_set.arrays["history_state"]),
        "batch_size": cfg["pilot"]["batch_size"], "optimizer_updates": updates,
        "epochs_completed": len(log), "checkpoint_epoch": saved["epoch"],
        "undertraining_warning": bool(saved["epoch"] >= len(log) - 1 and descending),
        "runtime_s": time.monotonic() - started,
        "gpu_peak_memory_mb": torch.cuda.max_memory_allocated() / 2**20,
        "validation": evaluate(model, validation_loader, device, dt_s=cfg["dt_s"]), "training": log,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--config", default="configs/gate_a_sind_ic4.json")
    parser.add_argument("--cache", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--models", default="self,own,pool,graph,all_graph")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--train-limit", type=int, default=None)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    cfg = json.loads((ROOT / args.config).read_text())
    if args.epochs is not None:
        cfg["pilot"]["epochs"] = args.epochs
    if args.train_limit is not None:
        cfg["pilot"]["train_limit"] = args.train_limit
    cache = ROOT / (args.cache or cfg["cache"])
    out = ROOT / (args.output or f"reports/task_redesign/sind_gate_a_seed{args.seed}")
    out.mkdir(parents=True, exist_ok=True)
    validation = GateADataset(cache, "val")
    validation_loader = DataLoader(validation, cfg["pilot"]["batch_size"] * 2, num_workers=2)
    results = {
        "revision": cfg["revision"], "seed": args.seed, "test_accessed": False,
        "checkpoint_rule": "minimum validation IC4 J; fixed 4s endpoint; one checkpoint reports all views",
        "cv_baseline": evaluate(None, validation_loader, torch.device("cpu"), cv=True, dt_s=cfg["dt_s"]), "models": {},
    }
    for kind in args.models.split(","):
        seed_all(args.seed)
        results["models"][kind] = train(kind, args.seed, cfg, cache, out, torch.device("cuda"))
        (out / "summary.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"seed": args.seed, "models": {k: v["validation"]["ic4"] for k, v in results["models"].items()}}, indent=2))


if __name__ == "__main__":
    main()
