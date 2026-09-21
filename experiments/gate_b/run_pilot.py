from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.gate_a.data import GateADataset
from experiments.gate_a.run_pilot import batch_loss, batch_rows, summarize_rows
from experiments.gate_b.models import GateBRajModel


def atomic_json(path: Path, payload: object, allow_nan: bool = False) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=allow_nan) + "\n")
    tmp.replace(path)


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def compatible_sources(old: dict, new: dict) -> bool:
    runner = "experiments/gate_b/run_pilot.py"
    return all(old.get(key) == value for key, value in new.items() if key != runner)


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_rows(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict, list[dict]]:
    model.eval(); rows, offset = [], 0
    for batch in loader:
        pred = model(batch["history_state"].to(device), batch["node_mask"].to(device)).cpu()
        current = batch_rows(pred, batch); size = len(pred)
        for position, row in enumerate(current):
            row["sample_index"] = offset + position % size
        rows.extend(current); offset += size
    return summarize_rows(rows), rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable formal Gate B validation; test remains sealed")
    parser.add_argument("--config", default="configs/gate_b_raj_migration.json")
    parser.add_argument("--seed", type=int, choices=(2026, 2027), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    cfg_path = ROOT / args.config; cfg = json.loads(cfg_path.read_text())
    cache, out = ROOT / cfg["benchmark"], ROOT / args.output
    if out.exists() and not args.resume: raise FileExistsError("output exists; pass --resume")
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    train = GateADataset(cache, "train", cfg["training"]["train_limit"], args.seed)
    val = GateADataset(cache, "val")
    val_loader = DataLoader(val, cfg["training"]["batch_size"] * 2, shuffle=False, num_workers=2)
    subset_hash = sha(np.asarray(train.indices).tobytes())
    config_hash = sha(cfg_path.read_bytes())
    source_files = [ROOT / "experiments/gate_b/models.py", Path(__file__), ROOT / "experiments/gate_a/run_pilot.py",
                    ROOT / "prediction/qgnn_paper_native/raj_paper.py", ROOT / "prediction/qgnn_paper_native/raj_subset.py"]
    source_hashes = {str(path.relative_to(ROOT)): sha(path.read_bytes()) for path in source_files}
    summary = {
        "revision": cfg["revision"], "status": "RUNNING", "seed": args.seed, "test_accessed": False,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "config_sha256": config_hash, "source_sha256": source_hashes,
        "train_indices_sha256": subset_hash, "train_samples": len(train), "validation_samples": len(val),
        "checkpoint_rule": cfg["checkpoint_rule"], "models": {},
    }
    summary_path = out / "summary.json"
    if summary_path.exists():
        previous = json.loads(summary_path.read_text())
        for key in ("revision", "seed", "config_sha256", "train_indices_sha256"):
            if previous[key] != summary[key]: raise RuntimeError(f"resume mismatch: {key}")
        if not compatible_sources(previous["source_sha256"], source_hashes): raise RuntimeError("resume mismatch: source_sha256")
        if previous["source_sha256"] != source_hashes:
            previous.setdefault("runner_repairs", []).append({"reason": "runner-only serialization/resume repair",
                "old_sha256": previous["source_sha256"]["experiments/gate_b/run_pilot.py"],
                "new_sha256": source_hashes["experiments/gate_b/run_pilot.py"]})
        summary = previous; summary["source_sha256"] = source_hashes; summary["status"] = "RUNNING"
    atomic_json(summary_path, summary)

    def train_to(kind: str, target_epochs: int, reset_patience: bool = False) -> dict:
        seed_all(args.seed)
        model = GateBRajModel(kind, dt_s=cfg["dt_s"], rounds=cfg["rounds"]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"],
                                      weight_decay=cfg["training"]["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, min_lr=1e-5)
        last_path, best_path = out / f"{kind}_last.pt", out / f"{kind}_best.pt"
        best, best_epoch, patience, records, updates, elapsed = float("inf"), 0, 0, [], 0, 0.0
        if last_path.exists():
            state = torch.load(last_path, map_location="cpu", weights_only=False)
            for key, value in (("kind", kind), ("seed", args.seed), ("config_sha256", config_hash),
                               ("train_indices_sha256", subset_hash)):
                if state[key] != value: raise RuntimeError(f"checkpoint mismatch: {key}")
            if not compatible_sources(state["source_sha256"], source_hashes): raise RuntimeError("checkpoint mismatch: source_sha256")
            model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"]); best = state["best"]; best_epoch = state["best_epoch"]
            patience, records, updates, elapsed = state["patience"], state["records"], state["updates"], state["elapsed_s"]
            random.setstate(state["python_rng"]); np.random.set_state(state["numpy_rng"])
            torch.set_rng_state(state["torch_rng"]); torch.cuda.set_rng_state_all(state["cuda_rng"])
        if reset_patience: patience = 0
        started = time.monotonic(); torch.cuda.reset_peak_memory_stats(device)
        for epoch in range(len(records) + 1, target_epochs + 1):
            order = torch.randperm(len(train), generator=torch.Generator().manual_seed(args.seed + 1009 * epoch)).tolist()
            loader = DataLoader(Subset(train, order), cfg["training"]["batch_size"], shuffle=False, num_workers=2)
            model.train(); losses, gradnorms = [], []
            for batch in loader:
                batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                prediction = model(batch["history_state"], batch["node_mask"]); loss = batch_loss(prediction, batch)
                if not torch.isfinite(loss): raise FloatingPointError("nonfinite loss")
                loss.backward(); gradnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                if not torch.isfinite(gradnorm): raise FloatingPointError("nonfinite gradient")
                optimizer.step(); losses.append(float(loss.detach())); gradnorms.append(float(gradnorm)); updates += 1
            metrics, _ = evaluate_rows(model, val_loader, device); score = metrics["ic4"]["J_m"]; scheduler.step(score)
            record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "gradient_norm": float(np.mean(gradnorms)),
                      "learning_rate": optimizer.param_groups[0]["lr"], "validation": metrics}
            records.append(record)
            if score < best - 1e-5:
                best, best_epoch, patience = score, epoch, 0
                torch.save({"model": model.state_dict(), "epoch": epoch, "score": score, "kind": kind,
                            "seed": args.seed, "config_sha256": config_hash, "source_sha256": source_hashes,
                            "train_indices_sha256": subset_hash}, best_path)
            else: patience += 1
            elapsed_now = elapsed + time.monotonic() - started
            state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                     "kind": kind, "seed": args.seed, "config_sha256": config_hash, "source_sha256": source_hashes,
                     "train_indices_sha256": subset_hash, "best": best, "best_epoch": best_epoch, "patience": patience,
                     "records": records, "updates": updates, "elapsed_s": elapsed_now, "python_rng": random.getstate(),
                     "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
                     "cuda_rng": torch.cuda.get_rng_state_all()}
            tmp = last_path.with_suffix(".tmp"); torch.save(state, tmp); tmp.replace(last_path)
            atomic_json(out / f"{kind}_training.json", records)
            atomic_json(out / "heartbeat.json", {"seed": args.seed, "kind": kind, "epoch": epoch,
                                                   "target_epochs": target_epochs, "best_epoch": best_epoch,
                                                   "best_ic4_J": best, "elapsed_s": elapsed_now, "pid": os.getpid()})
            print(kind, json.dumps({"epoch": epoch, "loss": record["train_loss"], "ic4": metrics["ic4"],
                                    "best_epoch": best_epoch}), flush=True)
            if epoch >= cfg["training"]["base_epochs"] and patience >= cfg["training"]["patience"]: break
        elapsed += time.monotonic() - started
        saved = torch.load(best_path, map_location=device, weights_only=True); model.load_state_dict(saved["model"])
        validation, rows = evaluate_rows(model, val_loader, device)
        atomic_json(out / f"{kind}_validation_rows.json", {"kind": kind, "seed": args.seed,
                                                            "checkpoint_epoch": saved["epoch"], "rows": rows,
                                                            "test_accessed": False}, allow_nan=True)
        decreasing = len(records) >= 3 and all(records[i]["validation"]["ic4"]["J_m"] > records[i + 1]["validation"]["ic4"]["J_m"]
                                                   for i in range(len(records) - 3, len(records) - 1))
        return {"kind": kind, "parameters": model.parameter_audit(), "epochs_completed": len(records),
                "checkpoint_epoch": saved["epoch"], "optimizer_updates": updates, "runtime_s": elapsed,
                "gpu_peak_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20,
                "undertraining_warning": bool(saved["epoch"] >= len(records) - 1 and decreasing),
                "validation": validation, "training": records}

    try:
        base = cfg["training"]["base_epochs"]
        for kind in cfg["models"]:
            summary["models"][kind] = train_to(kind, base)
            atomic_json(summary_path, summary)
        trigger = any(result["checkpoint_epoch"] >= base - 1 and result["undertraining_warning"]
                      for result in summary["models"].values())
        summary["convergence_extension_triggered"] = trigger
        if trigger:
            for kind in cfg["models"]:
                summary["models"][kind] = train_to(kind, cfg["training"]["extension_epochs"], reset_patience=True)
                atomic_json(summary_path, summary)
        summary["status"] = "COMPLETED"; atomic_json(summary_path, summary)
        print(json.dumps({"seed": args.seed, "extension": trigger,
                          "ic4": {kind: value["validation"]["ic4"] for kind, value in summary["models"].items()}}, indent=2))
    except Exception as exc:
        summary["status"] = "FAILED"; summary["error"] = repr(exc); summary["traceback"] = traceback.format_exc()
        atomic_json(summary_path, summary); raise


if __name__ == "__main__": main()
