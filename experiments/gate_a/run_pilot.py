from __future__ import annotations

import argparse
import json
import os
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
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def batch_loss(pred, batch):
    mask = batch["future_mask"]
    error = torch.linalg.vector_norm(pred - batch["future_xy"], dim=-1)
    ade = (error * mask).sum(1) / mask.sum(1).clamp_min(1)
    last = (mask.to(torch.int64).sum(1) - 1).clamp_min(0)
    fde = error.gather(1, last[:, None]).squeeze(1)
    valid = mask.any(1)
    value = ade[valid] + 0.5 * fde[valid]
    weight = batch["target_weight"][valid]
    return (value * weight).sum() / weight.sum().clamp_min(1e-8)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); rows = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        pred = model(batch["history_state"], batch["node_mask"])
        error = torch.linalg.vector_norm(pred - batch["future_xy"], dim=-1)
        for horizon in (20, 40):
            mask = batch["future_mask"][:, :horizon]
            valid = mask.any(1)
            ade = (error[:, :horizon] * mask).sum(1) / mask.sum(1).clamp_min(1)
            last = (mask.to(torch.int64).sum(1) - 1).clamp_min(0)
            fde = error[:, :horizon].gather(1, last[:, None]).squeeze(1)
            for i in torch.nonzero(valid, as_tuple=False).flatten().tolist():
                rows.append({"horizon": horizon, "ade": float(ade[i]), "fde": float(fde[i]),
                             "k": int(batch["k"][i]), "closing": float(batch["max_closing"][i]),
                             "dcpa": float(batch["min_dcpa"][i]), "time_ms": int(batch["time_ms"][i])})
    result = {}
    strata = {"full": lambda r: True, "ic": lambda r: r["k"] >= 1, "k0": lambda r: r["k"] == 0,
              "k1": lambda r: r["k"] == 1, "k2plus": lambda r: r["k"] >= 2,
              "closing1": lambda r: r["closing"] >= 1.0,
              "cpa5": lambda r: r["dcpa"] <= 5.0}
    for horizon in (20, 40):
        for name, predicate in strata.items():
            selected = [r for r in rows if r["horizon"] == horizon and predicate(r)]
            if selected:
                origins = defaultdict(list)
                for row in selected: origins[row["time_ms"]].append(row)
                ade = float(np.mean([np.mean([r["ade"] for r in group]) for group in origins.values()]))
                fde = float(np.mean([np.mean([r["fde"] for r in group]) for group in origins.values()]))
                result[f"{name}{horizon // 10}"] = {"n": len(selected), "ade_m": ade, "fde_m": fde,
                                                     "J_m": ade + 0.5 * fde, "origins": len(origins)}
    return result


def train(kind, seed, cfg, cache, out, device):
    train = GateADataset(cache, "train", cfg["pilot"]["train_limit"], seed)
    validation = GateADataset(cache, "val")
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train, cfg["pilot"]["batch_size"], shuffle=True, generator=generator,
                              num_workers=2, persistent_workers=True)
    validation_loader = DataLoader(validation, cfg["pilot"]["batch_size"] * 2, num_workers=2)
    model = GateAModel(kind, train.arrays["history_state"].shape[2]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["pilot"]["learning_rate"],
                                  weight_decay=cfg["pilot"]["weight_decay"])
    scaler = torch.amp.GradScaler("cuda")
    best, patience = float("inf"), 0
    checkpoint = out / f"{kind}.pt"
    log = []
    for epoch in range(cfg["pilot"]["epochs"]):
        model.train(); losses = []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                loss = batch_loss(model(batch["history_state"], batch["node_mask"]), batch)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer); scaler.update(); losses.append(float(loss.detach()))
        validation_metrics = evaluate(model, validation_loader, device)
        score = validation_metrics["ic4"]["J_m"]
        log.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "validation": validation_metrics})
        print(kind, log[-1], flush=True)
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
    return {"kind": kind, "seed": seed, "parameters": sum(p.numel() for p in model.parameters()),
            "checkpoint_epoch": saved["epoch"], "validation": evaluate(model, validation_loader, device),
            "training": log}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cache", default="data/task_redesign/lankershim_gate_a_v1")
    parser.add_argument("--output", default=None)
    parser.add_argument("--models", default="self,own,pool,graph,all_graph")
    args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    cfg = json.loads((ROOT / "configs/gate_a_lankershim_ic4.json").read_text())
    out = ROOT / (args.output or f"reports/task_redesign/gate_a_seed{args.seed}")
    out.mkdir(parents=True, exist_ok=True)
    results = {"revision": cfg["revision"], "seed": args.seed, "test_accessed": False,
               "checkpoint_rule": "minimum validation IC4 J; one checkpoint reports all horizons/strata", "models": {}}
    for kind in args.models.split(","):
        seed_all(args.seed)
        results["models"][kind] = train(kind, args.seed, cfg, ROOT / args.cache, out,
                                          torch.device("cuda"))
        (out / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__": main()
