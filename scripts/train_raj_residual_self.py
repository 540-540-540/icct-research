#!/usr/bin/env python3
"""Resumable Self-stage training for the frozen Raj residual architecture."""
from __future__ import annotations

import argparse
import hashlib
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


def batch_inputs(batch, device):
    """The Self phase evaluates targets only; context is for later interaction."""
    history = batch["history_state"].to(device, non_blocking=True)
    context = batch["vehicle_mask"].to(device, non_blocking=True)
    mask = batch.get("target_mask", batch["vehicle_mask"]).to(device, non_blocking=True)
    if mask.dtype != torch.bool or mask.shape != context.shape:
        raise ValueError("Expected boolean target mask matching the context")
    if torch.any(mask & ~context) or not torch.all(mask.any(dim=1)):
        raise ValueError("Every sample needs a valid supervised target")
    future = batch["future_state"].to(device, non_blocking=True)
    timestamps = batch["history_timestamp"].to(device, non_blocking=True)
    return history, mask, future, timestamps


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ade = fde = 0.0
    samples = 0
    origins = {}
    target_views = False
    for batch in loader:
        history, mask, future, timestamps = batch_inputs(batch, device)
        metric = trajectory_metrics(
            model(history, mask, timestamps)["prediction"], future, mask
        )
        a, f = metric["scene_ade"].cpu(), metric["scene_fde"].cpu()
        ade += float(a.sum())
        fde += float(f.sum())
        samples += len(history)
        if "target_mask" in batch:
            target_views = True
            for sc, frame, av, fv in zip(
                batch["scene_id"].tolist(), batch["start_frame"].tolist(),
                a.tolist(), f.tolist(),
            ):
                value = origins.setdefault((int(sc), int(frame)), [0.0, 0.0, 0])
                value[0] += av
                value[1] += fv
                value[2] += 1
    if not samples:
        raise ValueError("Validation subset is empty")
    target_ade, target_fde = ade / samples, fde / samples
    if target_views:
        ade = sum(a / n for a, _, n in origins.values()) / len(origins)
        fde = sum(f / n for _, f, n in origins.values()) / len(origins)
    else:
        ade, fde = target_ade, target_fde
    result = {
        "ADE": ade, "FDE": fde, "selection_score": ade + 0.5 * fde,
        "scenes": len(origins) if target_views else samples,
        "metric_unit": "origin_macro" if target_views else "legacy_window_macro",
        "samples": samples,
    }
    if target_views:
        result.update(target_ADE=target_ade, target_FDE=target_fde, targets=samples)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=("llm", "lstm", "transformer", "tcn"), default="llm"
    )
    parser.add_argument("--dataset", choices=("legacy", "target"), default="target")
    parser.add_argument("--self-frame", choices=("legacy", "ego_v1"), default="ego_v1")
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

    if args.epochs < 1 or args.batch_size < 1 or args.save_steps < 1:
        parser.error("epochs, batch-size, and save-steps must be positive")
    if args.max_seconds <= 0 or args.token_weight < 0:
        parser.error("max-seconds must be positive and token-weight nonnegative")
    for limit in (args.train_limit, args.val_limit):
        if limit is not None and limit < 1:
            parser.error("subset limits must be positive")
    if args.dataset == "target" and args.snr != 0.0:
        parser.error("target_views_v1 currently freezes 0 dB only")
    torch.set_num_threads(4)
    seed_all(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    run_dir = ROOT / args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    # The supported server is Linux. Keep one writer per experiment directory.
    import fcntl
    run_lock = (run_dir / ".run.lock").open("a+")
    try:
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(f"Another process owns this run: {run_dir}") from error
    run_lock.seek(0)
    run_lock.truncate()
    run_lock.write(str(os.getpid()) + "\n")
    run_lock.flush()
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    if args.resume and not (run_dir / "last.pt").is_file():
        raise FileNotFoundError(
            "Initialization did not reach a checkpoint. Preserve this directory "
            "and select a new --run-dir; no resumable optimizer state exists."
        )
    if any((run_dir / name).exists() for name in ("config.json", "best.pt", "last.pt")) and not args.resume:
        raise FileExistsError("Use --resume or a new run directory")

    source_files = list((ROOT / "prediction/qgnn_raj_pennylane").glob("*.py"))
    source_files += [Path(__file__), ROOT / "configs/qgnn_final_tokens.json",
                     ROOT / "prediction/q0/metrics.py",
                     ROOT / "prediction/q0/motion_token_llm.py",
                     ROOT / "prediction/qgnn_final/model.py"]
    if args.dataset == "target":
        source_files += [ROOT / "frontend/sind_target_dataset.py",
                         ROOT / "configs/sind_target_prediction.json"]
    else:
        source_files += [ROOT / "frontend/sind_prediction_dataset.py"]
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
        "time_budget_scope": "per_invocation",
        "metric_unit": "origin_macro" if args.dataset == "target" else "legacy_window_macro",
        "supervision": "center_only" if args.dataset == "target" else "all_selected_vehicles",
        "self_frame_applied": args.self_frame if args.model == "llm" else "not_applicable",
        "source_sha256": source_hashes,
    }
    model = (
        build_model(seed=args.seed, phase="self", self_frame=args.self_frame)
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

    if args.dataset == "target":
        from frontend.sind_target_dataset import SinDTargetPredictionDataset
        dataset_class = SinDTargetPredictionDataset
    else:
        dataset_class = SinDPredictionDataset
    train_data = dataset_class("train", args.snr, ROOT, True)
    val_data = dataset_class("val", args.snr, ROOT, True)
    if args.dataset == "target":
        config["data_manifest"] = {
            split: json.loads((ROOT / "data/sind/target_views_v1" / split / "manifest.json").read_text())
            for split in ("train", "val")
        }
    if not args.resume:
        atomic_json(run_dir / "config.json", config)
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
            "train_limit", "val_limit", "dataset", "self_frame",
        ):
            if saved["config"][key] != vars(args)[key]:
                raise ValueError(f"Resume config mismatch: {key}")
        saved_run_config = json.loads((run_dir / "config.json").read_text())
        if saved_run_config.get("data_manifest") != config.get("data_manifest"):
            raise ValueError("Resume data manifest mismatch")
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
                "dataset": args.dataset,
                "self_frame": config["self_frame_applied"],
                "metric_unit": config["metric_unit"],
                "supervision": config["supervision"],
                "completed_data_passes": sum(row["epoch"] > 0 for row in records),
            },
        )

    if not records:
        # Save before the first validation/optimizer update, including epoch-0
        # RNG state. A stopped initialization can safely resume the same model.
        checkpoint(1, 0)
        initial = evaluate(model, val_loader, device)
        best = {"epoch": 0, **initial}
        records.append({"epoch": 0, "train_loss": None, "validation": initial})
        torch.save(
            {"model_state": self_state(model), "validation": initial, "config": config},
            run_dir / "best.pt",
        )
        atomic_json(run_dir / "training.json", records)
        checkpoint(1, 0)
        print("EPOCH0 " + json.dumps(records[0]), flush=True)
    summary("RUNNING")
    if stop_requested:
        summary("PAUSED_SIGNAL")
        return
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
        # A replacement mid-epoch iterator draws a CPU base seed even with
        # num_workers=0. Cancel that extra draw so dropout resumes exactly.
        iterator_rng = torch.get_rng_state() if skip else None
        iterator = iter(loader)
        if iterator_rng is not None:
            torch.set_rng_state(iterator_rng)
        for step, batch in enumerate(iterator, start=skip):
            history, mask, future, timestamps = batch_inputs(batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(history, mask, timestamps)
            coordinate = trajectory_metrics(output["prediction"], future, mask)
            weights = batch.get("target_weight")
            if weights is None:
                loss = coordinate["loss"]
            else:
                weights = weights.to(device, dtype=history.dtype)
                if not torch.isfinite(weights).all() or torch.any(weights <= 0):
                    raise ValueError("Invalid target-to-origin loss weights")
                loss = ((coordinate["scene_ade"] + 0.5 * coordinate["scene_fde"]) * weights).mean()
            if args.model == "llm":
                targets = model.llm.future_token_ids(history, future)
                if weights is None:
                    tokens = token_loss(output["token_logits"], targets, mask)
                else:
                    logits = output["token_logits"]
                    ce = torch.nn.functional.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                        reduction="none",
                    ).reshape_as(targets)
                    valid = mask[:, None, :].expand_as(targets)
                    per_sample = (ce * valid).sum((1, 2)) / valid.sum((1, 2))
                    tokens = (per_sample * weights).mean()
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
            if stop_requested or time.perf_counter() - started > args.max_seconds:
                checkpoint(epoch, step + 1, loss_sum, seen)
                status = "PAUSED_SIGNAL" if stop_requested else "PAUSED_BUDGET"
                summary(status)
                print(status, flush=True)
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
