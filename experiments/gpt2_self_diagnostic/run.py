#!/usr/bin/env python3
"""Resumable GPT-2 mechanism diagnosis on the unchanged SinD target dataset.

Validation is reused for mechanism diagnosis, not an independent confirmation.
Every checkpoint includes the complete model, including random/fine-tuned GPT.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from frontend.sind_target_dataset import SinDTargetPredictionDataset
from prediction.q0.metrics import trajectory_metrics
from scripts.train_raj_residual_self import (
    atomic_json, batch_inputs, evaluate, seed_all, subset_indices,
)


def atomic_checkpoint(path, payload):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def full_model_state(model):
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def source_files():
    """Snapshot runtime dependencies, excluding separately edited checks/launchers."""
    paths = [
        "experiments/gpt2_self_diagnostic/run.py",
        "experiments/gpt2_self_diagnostic/model.py",
        "frontend/sind_target_dataset.py",
        "scripts/train_raj_residual_self.py",
        "scripts/train_q0_motion_llm.py",
        "configs/qgnn_final_tokens.json",
        "configs/sind_target_prediction.json",
        "prediction/q0/contracts.py",
        "prediction/q0/metrics.py",
        "prediction/q0/temporal.py",
        "prediction/q0/motion_token_llm.py",
        "prediction/qgnn_final/model.py",
        "prediction/qgnn_raj_pennylane/__init__.py",
        "prediction/qgnn_raj_pennylane/common.py",
        "prediction/qgnn_raj_pennylane/model.py",
        "prediction/qgnn_raj_pennylane/residual.py",
        "prediction/qgnn_raj_pennylane/self_baselines.py",
    ]
    return sorted(paths)


def snapshot_sources(run_dir, paths, resume):
    snapshot = run_dir / "source_snapshot"
    for relative in paths:
        current = (ROOT / relative).read_bytes()
        saved = snapshot / relative
        if resume:
            if not saved.is_file() or saved.read_bytes() != current:
                raise ValueError(f"Resume source snapshot mismatch: {relative}")
        else:
            saved.parent.mkdir(parents=True, exist_ok=True)
            saved.write_bytes(current)


def rng_state():
    return {
        "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(saved):
    random.setstate(saved["python_rng"])
    np.random.set_state(saved["numpy_rng"])
    torch.set_rng_state(saved["torch_rng"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(saved["cuda_rng"])


def empty_epoch_sums():
    return {"loss": 0.0, "coordinate": 0.0, "token_ce": 0.0,
            "seen": 0, "step_seconds": 0.0}


def weighted_objective(model, output, history, mask, future, weights, token_weight):
    coordinate = trajectory_metrics(output["prediction"], future, mask)
    coordinate_loss = ((coordinate["scene_ade"] + 0.5 * coordinate["scene_fde"]) * weights).mean()
    token_ce = coordinate_loss.new_zeros(())
    if token_weight:
        targets = model.llm.future_token_ids(history, future)
        logits = output["token_logits"]
        ce = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none",
        ).reshape_as(targets)
        valid = mask[:, None, :].expand_as(targets)
        per_target = (ce * valid).sum((1, 2)) / valid.sum((1, 2))
        token_ce = (per_target * weights).mean()
    return coordinate_loss + token_weight * token_ce, coordinate_loss, token_ce


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("tcn", "llm"), default="llm")
    parser.add_argument("--frame", choices=("global", "ego", "ego_heading"), default="ego_heading")
    parser.add_argument("--initialization", choices=("pretrained", "random"), default="pretrained")
    parser.add_argument("--adaptation", choices=("lora", "full"), default="lora")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Actual and effective batch size; no gradient accumulation")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peripheral learning rate")
    parser.add_argument("--backbone-lr", type=float, default=7.5e-5,
                        help="GPT backbone or LoRA rate, using the same cosine schedule")
    parser.add_argument("--train-limit", type=int, help="Short preflight only; omit for formal runs")
    parser.add_argument("--val-limit", type=int, help="Short preflight only; omit for formal runs")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=7200., help="Budget per invocation after initialization")
    parser.add_argument("--max-steps", type=int,
                        help="Absolute optimizer-step ceiling; omit on resume to finish scheduled epochs")
    parser.add_argument("--save-steps", type=int, default=100)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.save_steps) < 1:
        parser.error("epochs, batch-size and save-steps must be positive")
    if args.lr <= 0 or args.backbone_lr <= 0 or args.max_seconds <= 0:
        parser.error("learning rates and max-seconds must be positive")
    if args.max_steps is not None and args.max_steps < 0:
        parser.error("max-steps must be nonnegative")
    if any(limit is not None and limit < 1 for limit in (args.train_limit, args.val_limit)):
        parser.error("subset limits must be positive")
    if args.model == "tcn" and (args.frame == "ego_heading" or args.initialization != "pretrained" or args.adaptation != "lora"):
        parser.error("TCN supports global/ego; initialization/adaptation must retain their default GPT-only placeholders")
    if args.model == "llm" and args.frame == "global":
        parser.error("LLM supports ego/ego_heading; legacy is not a global-state control")
    return args


def main():
    args = parse_args()
    from experiments.gpt2_self_diagnostic.model import build_diagnostic_model, parameter_groups

    torch.set_num_threads(4)
    seed_all(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    run_dir = (ROOT / args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = (run_dir / ".run.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(f"Another process owns {run_dir}") from error
    lock.seek(0)
    lock.truncate()
    lock.write(str(os.getpid()) + "\n")
    lock.flush()
    if args.resume and not (run_dir / "last.pt").is_file():
        raise FileNotFoundError("Resume requires last.pt; preserve this directory and choose a new run directory")
    if not args.resume and any((run_dir / name).exists() for name in ("config.json", "last.pt", "best.pt", "source_snapshot")):
        raise FileExistsError("Existing experiment; use --resume or a new run directory")

    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    token_weight = .035 if args.model == "llm" else 0.0
    training_config = {key: getattr(args, key) for key in (
        "model", "frame", "initialization", "adaptation", "seed", "epochs",
        "batch_size", "lr", "backbone_lr", "train_limit", "val_limit",
    )}
    training_config.update(token_weight=token_weight, weight_decay=2e-4, gradient_clip=3.0,
                           snr_db=0.0, history_length=20, prediction_length=20,
                           metric_unit="origin_macro", scheduler="CosineAnnealingLR",
                           scheduler_eta_min=args.lr * .08)
    data_manifest = {
        split: json.loads((ROOT / "data/sind/target_views_v1" / split / "manifest.json").read_text())
        for split in ("train", "val")
    }
    paths = source_files()
    previous_config = None
    if args.resume:
        previous_config = json.loads((run_dir / "config.json").read_text())
        for key, value in (("training", training_config), ("data_manifest", data_manifest), ("source_files", paths)):
            if previous_config.get(key) != value:
                raise ValueError(f"Resume {key} mismatch")
        if previous_config.get("device_type") != device.type:
            raise ValueError("Resume device type mismatch")
    snapshot_sources(run_dir, paths, args.resume)

    train_data = SinDTargetPredictionDataset("train", 0.0, ROOT, True)
    val_data = SinDTargetPredictionDataset("val", 0.0, ROOT, True)
    if (len(train_data), len(val_data)) != (319855, 31211):
        raise ValueError("The diagnostic protocol freezes 319855 train / 31211 val targets")
    if args.train_limit is not None and args.train_limit > len(train_data):
        raise ValueError("train-limit exceeds the dataset")
    if args.val_limit is not None and args.val_limit > len(val_data):
        raise ValueError("val-limit exceeds the dataset")
    train_ids = subset_indices(len(train_data), args.train_limit, args.seed)
    val_ids = subset_indices(len(val_data), args.val_limit, args.seed + 1)
    train_data = Subset(train_data, train_ids.tolist())
    val_data = Subset(val_data, val_ids.tolist())
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=device.type == "cuda")
    model = build_diagnostic_model(kind=args.model, seed=args.seed, frame=args.frame,
                                   initialization=args.initialization, adaptation=args.adaptation).to(device)
    # Production builders call torch.manual_seed inside a CPU-only fork, which
    # also leaves CUDA seeded at seed+300003. Match that dropout stream explicitly.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + 300003)
    groups = parameter_groups(model, args.backbone_lr, args.lr)
    optimizer = torch.optim.AdamW(groups, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * .08,
    )
    trainable = [p for p in model.parameters() if p.requires_grad]
    parameter_counts = model.parameter_summary()
    config = {
        **vars(args), "training": training_config, "data_manifest": data_manifest,
        "source_files": paths, "source_provenance": "source_snapshot byte-for-byte comparison",
        "device": str(device), "device_type": device.type,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "dropout_cuda_seed": args.seed + 300003,
        "test_set_used": False, "validation_usage": "reused_validation_for_mechanism_diagnosis",
        "random_scope": "GPT h,wpe,ln_f only; peripheral motion embedding shared",
        "coordinate_comparison_scope": "Interface factor; GPT uses 19 motion tokens plus own states and TCN uses 20 displacement/velocity steps",
        "checkpoint_selection": "minimum origin-macro ADE + 0.5 * FDE",
        "parameters": parameter_counts, "train_samples": len(train_data), "validation_samples": len(val_data),
        "optimizer_groups": [{"name": g.get("name", str(i)), "lr": g["lr"],
                              "parameters": sum(p.numel() for p in g["params"])}
                             for i, g in enumerate(optimizer.param_groups)],
        "effective_batch_size": args.batch_size, "time_budget_scope": "per_invocation",
        "max_steps_scope": "absolute_optimizer_steps", "checkpoint_state": "complete_model_state_dict",
    }
    if not args.resume:
        atomic_json(run_dir / "config.json", config)

    best, records = None, []
    start_epoch, skip_batches, global_step = 1, 0, 0
    elapsed_before = 0.0
    resumed_sums = empty_epoch_sums()
    if args.resume:
        saved = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
        if saved.get("training_config") != training_config or saved.get("data_manifest") != data_manifest:
            raise ValueError("Resume checkpoint protocol mismatch")
        model.load_state_dict(saved["model_state"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start_epoch, skip_batches, global_step = saved["next_epoch"], saved["next_batch"], saved["global_step"]
        best, records = saved["best"], saved["records"]
        elapsed_before, resumed_sums = saved["elapsed_seconds"], saved["epoch_sums"]
        restore_rng(saved)
        del saved
    started = time.perf_counter()

    def elapsed():
        return elapsed_before + time.perf_counter() - started

    def checkpoint(next_epoch, next_batch, sums=None):
        atomic_checkpoint(run_dir / "last.pt", {
            "model_state": full_model_state(model), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "training_config": training_config,
            "data_manifest": data_manifest, "next_epoch": next_epoch, "next_batch": next_batch,
            "global_step": global_step, "best": best, "records": records,
            "elapsed_seconds": elapsed(), "epoch_sums": sums if sums is not None else empty_epoch_sums(),
            **rng_state(),
        })

    def save_best(validation):
        atomic_checkpoint(run_dir / "best.pt", {
            "model_state": full_model_state(model), "validation": validation,
            "config": previous_config if args.resume else config,
            "global_step": global_step, "checkpoint_state": "complete_model_state_dict",
        })

    def summary(status):
        atomic_json(run_dir / "summary.json", {
            "status": status, "model": args.model, "frame": args.frame,
            "initialization": args.initialization, "adaptation": args.adaptation,
            "selected_checkpoint": best, "selection_rule": "minimum origin-macro ADE + 0.5 * FDE",
            "initial_validation": records[0]["validation"] if records else None,
            "last_validation": records[-1]["validation"] if records else None,
            "epochs_completed": sum(row["epoch"] > 0 for row in records),
            "global_step": global_step, "elapsed_seconds": elapsed(),
            "train_samples": len(train_data), "validation_samples": len(val_data),
            "parameters": parameter_counts, "test_set_used": False,
            "validation_usage": "reused_validation_for_mechanism_diagnosis",
            "metric_unit": "origin_macro", "supervision": "center_only",
            "checkpoint_state": "complete_model_state_dict",
        })

    if start_epoch > args.epochs:
        summary("COMPLETED")
        print("COMPLETED (no further updates)", flush=True)
        return
    if not records:
        checkpoint(1, 0)
        validation_started = time.perf_counter()
        initial = evaluate(model, val_loader, device)
        best = {"epoch": 0, **initial}
        records.append({"epoch": 0, "train_loss": None, "train_coordinate": None,
                        "train_token_ce": None, "validation": initial,
                        "validation_seconds": time.perf_counter() - validation_started,
                        "elapsed_seconds": elapsed(), "lr": [g["lr"] for g in optimizer.param_groups]})
        save_best(best)
        atomic_json(run_dir / "training.json", records)
        checkpoint(1, 0)
        print("EPOCH0 " + json.dumps(records[0]), flush=True)

    def pause_reason():
        if stop_requested:
            return "PAUSED_SIGNAL"
        if args.max_steps is not None and global_step >= args.max_steps:
            return "PAUSED_STEPS"
        if time.perf_counter() - started >= args.max_seconds:
            return "PAUSED_BUDGET"
        return None

    reason = pause_reason()
    if reason:
        summary(reason)
        print(reason, flush=True)
        return
    summary("RUNNING")
    for epoch in range(start_epoch, args.epochs + 1):
        order = torch.randperm(len(train_data), generator=torch.Generator().manual_seed(args.seed + 1009 * epoch)).tolist()
        skip = skip_batches if epoch == start_epoch else 0
        epoch_data = Subset(train_data, order[skip * args.batch_size:])
        loader = DataLoader(epoch_data, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=device.type == "cuda")
        sums = dict(resumed_sums) if epoch == start_epoch else empty_epoch_sums()
        rates = [group["lr"] for group in optimizer.param_groups]
        model.train()
        # A replacement mid-epoch iterator draws an extra CPU base seed even at
        # num_workers=0. Cancel that draw to keep dropout/resume exact.
        iterator_rng = torch.get_rng_state() if skip else None
        iterator = iter(loader)
        if iterator_rng is not None:
            torch.set_rng_state(iterator_rng)
        for batch_index, batch in enumerate(iterator, start=skip):
            step_started = time.perf_counter()
            history, mask, future, timestamps = batch_inputs(batch, device)
            weights = batch["target_weight"].to(device, dtype=history.dtype)
            if not torch.isfinite(weights).all() or torch.any(weights <= 0):
                raise ValueError("Invalid target-to-origin loss weights")
            optimizer.zero_grad(set_to_none=True)
            output = model(history, mask, timestamps)
            loss, coordinate, ce = weighted_objective(model, output, history, mask, future, weights, token_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(trainable, 3.0)
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite training gradient")
            optimizer.step()
            global_step += 1
            count = len(history)
            sums["loss"] += float(loss.detach()) * count
            sums["coordinate"] += float(coordinate.detach()) * count
            sums["token_ce"] += float(ce.detach()) * count
            sums["seen"] += count
            sums["step_seconds"] += time.perf_counter() - step_started
            reason = pause_reason()
            if reason or global_step % args.save_steps == 0:
                checkpoint(epoch, batch_index + 1, sums)
            if reason:
                summary(reason)
                print(reason, flush=True)
                return
        scheduler.step()
        validation_started = time.perf_counter()
        validation = evaluate(model, val_loader, device)
        record = {
            "epoch": epoch, "train_loss": sums["loss"] / sums["seen"],
            "train_coordinate": sums["coordinate"] / sums["seen"],
            "train_token_ce": sums["token_ce"] / sums["seen"],
            "token_weight": token_weight, "training_step_seconds": sums["step_seconds"],
            "validation": validation, "validation_seconds": time.perf_counter() - validation_started,
            "lr": rates, "next_lr": [g["lr"] for g in optimizer.param_groups],
            "global_step": global_step, "elapsed_seconds": elapsed(),
        }
        records.append(record)
        candidate = {"epoch": epoch, **validation}
        if candidate["selection_score"] < best["selection_score"]:
            best = candidate
            save_best(best)
        checkpoint(epoch + 1, 0)
        atomic_json(run_dir / "training.json", records)
        summary("RUNNING")
        print("EPOCH " + json.dumps(record), flush=True)
        reason = pause_reason()
        if reason and epoch < args.epochs:
            summary(reason)
            print(reason, flush=True)
            return
    summary("COMPLETED")
    print("COMPLETED", flush=True)


if __name__ == "__main__":
    main()
