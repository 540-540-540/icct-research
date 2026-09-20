#!/usr/bin/env python3
"""Resumable Self-stage training for the frozen Raj residual architecture."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from frontend.sind_prediction_dataset import SinDPredictionDataset
from prediction.q0.metrics import trajectory_metrics
from prediction.qgnn_raj_pennylane import build_model, build_self_baseline
from scripts.train_q0_motion_llm import token_loss


def atomic_json(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def self_state(model):
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if not name.startswith("core.")
        and not (
            name.startswith("llm.base.gpt2.") and "lora_" not in name
        )
    }


def subset_indices(length, limit, seed):
    indices = np.random.default_rng(seed).permutation(length)
    return indices if limit is None else indices[:limit]


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ade = fde = 0.0
    scenes = 0
    for batch in loader:
        history = batch["history_state"].to(device, non_blocking=True)
        mask = batch["vehicle_mask"].to(device, non_blocking=True)
        future = batch["future_state"].to(device, non_blocking=True)
        timestamps = batch["history_timestamp"].to(device, non_blocking=True)
        metric = trajectory_metrics(
            model(history, mask, timestamps)["prediction"], future, mask
        )
        ade += float(metric["scene_ade"].sum())
        fde += float(metric["scene_fde"].sum())
        scenes += len(history)
    ade, fde = ade / scenes, fde / scenes
    return {
        "ADE": ade,
        "FDE": fde,
        "selection_score": ade + 0.5 * fde,
        "scenes": scenes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=("llm", "lstm", "transformer", "tcn"), default="llm"
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--snr", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--token-weight", type=float, default=0.035)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=7200.0)
    parser.add_argument("--save-steps", type=int, default=100)
    args = parser.parse_args()

    torch.set_num_threads(4)
    seed_all(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    run_dir = ROOT / args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "last.pt").exists() and not args.resume:
        raise FileExistsError("Use --resume or a new run directory")

    source_files = list((ROOT / "prediction/qgnn_raj_pennylane").glob("*.py"))
    source_files += [Path(__file__), ROOT / "configs/qgnn_final_tokens.json"]
    source_hashes = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_files
    }
    config = vars(args) | {
        "phase": "self",
        "model": args.model,
        "device": str(device),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "test_set_used": False,
        "checkpoint_selection": "minimum ADE + 0.5 * FDE",
        "source_sha256": source_hashes,
    }
    if not args.resume:
        atomic_json(run_dir / "config.json", config)

    model = (
        build_model(seed=args.seed, phase="self")
        if args.model == "llm"
        else build_self_baseline(args.model, seed=args.seed)
    ).to(device)
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    regular = [p for name, p in trainable if "lora_" not in name]
    lora = [p for name, p in trainable if "lora_" in name]
    groups = [{"params": regular, "lr": args.lr}]
    if lora:
        groups.append({"params": lora, "lr": args.lr * 0.25})
    optimizer = torch.optim.AdamW(groups, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.08
    )

    train_data = SinDPredictionDataset("train", args.snr, ROOT, True)
    val_data = SinDPredictionDataset("val", args.snr, ROOT, True)
    train_ids = subset_indices(len(train_data), args.train_limit, args.seed)
    val_ids = subset_indices(len(val_data), args.val_limit, args.seed + 1)
    train_data = Subset(train_data, train_ids.tolist())
    val_data = Subset(val_data, val_ids.tolist())
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    best = None
    records = []
    start_epoch = 1
    skip_batches = global_step = 0
    elapsed_before = resumed_loss = 0.0
    resumed_seen = 0
    if args.resume:
        saved = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
        if saved["source_sha256"] != source_hashes:
            raise ValueError("Resume source hash mismatch")
        for key in (
            "model", "seed", "snr", "epochs", "batch_size", "lr", "token_weight",
            "train_limit", "val_limit",
        ):
            if saved["config"][key] != vars(args)[key]:
                raise ValueError(f"Resume config mismatch: {key}")
        missing, unexpected = model.load_state_dict(saved["model_state"], strict=False)
        disallowed = [
            name for name in missing
            if not name.startswith("core.")
            and not (name.startswith("llm.base.gpt2.") and "lora_" not in name)
        ]
        if unexpected or disallowed:
            raise ValueError(
                f"Checkpoint mismatch missing={disallowed} unexpected={unexpected}"
            )
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        start_epoch = saved["next_epoch"]
        skip_batches = saved["next_batch"]
        global_step = saved["global_step"]
        best = saved["best"]
        records = saved["records"]
        elapsed_before = saved["elapsed_seconds"]
        resumed_loss = saved["epoch_loss_sum"]
        resumed_seen = saved["epoch_seen"]

    started = time.perf_counter()

    def elapsed():
        return elapsed_before + time.perf_counter() - started

    def checkpoint(next_epoch, next_batch, epoch_loss_sum=0.0, epoch_seen=0):
        payload = {
            "model_state": self_state(model),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": vars(args),
            "source_sha256": source_hashes,
            "next_epoch": next_epoch,
            "next_batch": next_batch,
            "global_step": global_step,
            "best": best,
            "records": records,
            "elapsed_seconds": elapsed(),
            "epoch_loss_sum": epoch_loss_sum,
            "epoch_seen": epoch_seen,
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }
        temporary = run_dir / "last.tmp"
        torch.save(payload, temporary)
        temporary.replace(run_dir / "last.pt")

    def summary(status):
        atomic_json(
            run_dir / "summary.json",
            {
                "status": status,
                "model": args.model,
                "selected_checkpoint": best,
                "selection_rule": "minimum ADE + 0.5 * FDE",
                "initial_validation": records[0]["validation"] if records else None,
                "epochs_completed": sum(row["epoch"] > 0 for row in records),
                "global_step": global_step,
                "elapsed_seconds": elapsed(),
                "train_samples": len(train_data),
                "validation_samples": len(val_data),
                "trainable_parameters": sum(p.numel() for _, p in trainable),
                "test_set_used": False,
            },
        )

    if not args.resume:
        initial = evaluate(model, val_loader, device)
        best = {"epoch": 0, **initial}
        records.append({"epoch": 0, "train_loss": None, "validation": initial})
        torch.save(
            {"model_state": self_state(model), "validation": initial, "config": config},
            run_dir / "best.pt",
        )
        atomic_json(run_dir / "training.json", records)
        print("EPOCH0 " + json.dumps(records[0]), flush=True)
    summary("RUNNING")
    for epoch in range(start_epoch, args.epochs + 1):
        order = torch.randperm(
            len(train_data), generator=torch.Generator().manual_seed(args.seed + 1009 * epoch)
        ).tolist()
        skip = skip_batches if epoch == start_epoch else 0
        epoch_data = Subset(train_data, order[skip * args.batch_size:])
        loader = DataLoader(
            epoch_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        loss_sum = resumed_loss if epoch == start_epoch else 0.0
        seen = resumed_seen if epoch == start_epoch else 0
        model.train()
        for step, batch in enumerate(loader, start=skip):
            history = batch["history_state"].to(device, non_blocking=True)
            mask = batch["vehicle_mask"].to(device, non_blocking=True)
            future = batch["future_state"].to(device, non_blocking=True)
            timestamps = batch["history_timestamp"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = model(history, mask, timestamps)
            coordinate = trajectory_metrics(output["prediction"], future, mask)
            loss = coordinate["loss"]
            if args.model == "llm":
                tokens = token_loss(
                    output["token_logits"],
                    model.llm.future_token_ids(history, future),
                    mask,
                )
                loss = loss + args.token_weight * tokens
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for _, p in trainable], 3.0
            )
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("Nonfinite training gradient")
            optimizer.step()
            global_step += 1
            loss_sum += float(loss.detach()) * len(history)
            seen += len(history)
            if global_step % args.save_steps == 0:
                checkpoint(epoch, step + 1, loss_sum, seen)
            if elapsed() > args.max_seconds:
                checkpoint(epoch, step + 1, loss_sum, seen)
                summary("PAUSED_BUDGET")
                print("PAUSED_BUDGET", flush=True)
                return

        scheduler.step()
        validation = evaluate(model, val_loader, device)
        record = {
            "epoch": epoch,
            "train_loss": loss_sum / max(seen, 1),
            "validation": validation,
        }
        records.append(record)
        candidate = {"epoch": epoch, **validation}
        if best is None or candidate["selection_score"] < best["selection_score"]:
            best = candidate
            torch.save(
                {"model_state": self_state(model), "validation": validation, "config": config},
                run_dir / "best.pt",
            )
        checkpoint(epoch + 1, 0)
        atomic_json(run_dir / "training.json", records)
        summary("RUNNING")
        print("EPOCH " + json.dumps(record), flush=True)

    summary("COMPLETED")
    print("COMPLETED", flush=True)


if __name__ == "__main__":
    main()
