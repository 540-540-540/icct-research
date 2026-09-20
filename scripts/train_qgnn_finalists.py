"""Resumable train/validation harness for QGNN finalists. Never opens prediction test."""
from __future__ import annotations

import argparse
import datetime as dt
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from frontend.sind_prediction_dataset import SinDPredictionDataset
from prediction.q0.metrics import trajectory_metrics
from prediction.qgnn_finalists.model import build_model
from scripts.evaluate_q0_motion_llm_val import interaction_stats
from scripts.train_q0_motion_llm import token_loss


def atomic_json(path: Path, payload) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def human_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "warming up"
    seconds = max(0, int(round(seconds)))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def estimated_finish(remaining: float | None) -> str:
    if remaining is None:
        return "warming up"
    return (dt.datetime.now().astimezone() + dt.timedelta(seconds=max(0.0, remaining))).isoformat(timespec="seconds")


def gpu_memory(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {name: 0.0 for name in ("allocated_gib", "reserved_gib", "peak_allocated_gib", "peak_reserved_gib")}
    return {
        "allocated_gib": torch.cuda.memory_allocated(device) / 2**30,
        "reserved_gib": torch.cuda.memory_reserved(device) / 2**30,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
    }


def saved_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    skipped = ("llm.base.gpt2.", "llm.gpt2.")
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if not (key.startswith(skipped) and "lora_" not in key)
    }


@torch.no_grad()
def evaluate(model, loader, device: torch.device):
    model.eval()
    rows = []
    batch_seconds = []
    index = 0
    for batch in loader:
        tick = time.perf_counter()
        history = batch["history_state"].to(device, non_blocking=True).float()
        mask = batch["vehicle_mask"].to(device, non_blocking=True).bool()
        future = batch["future_state"].to(device, non_blocking=True).float()
        timestamps = batch["history_timestamp"].to(device, non_blocking=True)
        out = model(history, mask, timestamps)
        metric = trajectory_metrics(out["prediction"], future, mask)
        stats = interaction_stats(history, mask)
        for row in range(history.shape[0]):
            rows.append(
                {
                    "index": index + row,
                    "scene_id": int(batch["scene_id"][row]),
                    "start_frame": int(batch["start_frame"][row]),
                    "ADE": float(metric["scene_ade"][row]),
                    "FDE": float(metric["scene_fde"][row]),
                    "closing_pairs_30m_gt_0p5": int(stats["closing_pairs_30m_gt_0p5"][row]),
                }
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        batch_seconds.append(time.perf_counter() - tick)
        index += history.shape[0]
    ade = float(np.mean([row["ADE"] for row in rows]))
    fde = float(np.mean([row["FDE"] for row in rows]))
    return {"ADE": ade, "FDE": fde, "J": ade + 0.5 * fde, "scenes": len(rows)}, rows, float(np.mean(batch_seconds))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["trc", "rtcn", "toj", "trtgn"], required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--snr", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--correction-cap", type=float, default=16.0)
    parser.add_argument("--rtcn-width", "--classical-width", dest="classical_width", type=int, choices=[64, 128], default=64)
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=0.0, help="Optional wall-clock pause; 0 disables it.")
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--progress-steps", type=int, default=25)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.save_steps < 1 or args.progress_steps < 1:
        raise ValueError("epochs, batch size, save steps, and progress steps must be positive")

    torch.set_num_threads(4)
    seed_all(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    outdir = ROOT / args.run_dir
    outdir.mkdir(parents=True, exist_ok=True)
    if (outdir / "last.pt").exists() and not args.resume:
        raise FileExistsError("Use --resume or a new run directory")

    config_args = vars(args).copy()
    source_files = list((ROOT / "prediction/qgnn_finalists").glob("*.py")) + [Path(__file__)]
    source_hash = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}
    token_hash = hashlib.sha256((ROOT / "configs/qgnn_final_tokens.json").read_bytes()).hexdigest()
    config = config_args | {
        "source_sha256": source_hash,
        "token_sha256": token_hash,
        "pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "test_set_used": False,
    }
    if not args.resume:
        atomic_json(outdir / "config.json", config)

    model = build_model(args.kind, args.seed, args.correction_cap, args.classical_width, not args.no_checkpoint).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    atomic_json(outdir / "parameters.json", model.parameter_summary())

    digest = hashlib.sha256()
    for name, parameter in model.llm.named_parameters():
        if parameter.requires_grad and ("base.gpt2" in name or "motion_embedding" in name or "history_adapter" in name or "future_queries" in name or "token_head" in name):
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().numpy().tobytes())
    shared_hash = digest.hexdigest()

    train = SinDPredictionDataset("train", args.snr, ROOT, True)
    validation = SinDPredictionDataset("val", args.snr, ROOT, True)
    validation_indices = np.arange(len(validation), dtype=np.int64)
    if args.val_limit:
        validation_indices = np.random.default_rng(args.seed + 991).permutation(len(validation))[: args.val_limit]
        validation = Subset(validation, validation_indices.tolist())
    selected = np.random.default_rng(args.seed).permutation(len(train))
    if args.train_limit:
        selected = selected[: args.train_limit]
    train = Subset(train, selected.tolist())
    train_indices_hash = hashlib.sha256(selected.tobytes()).hexdigest()
    validation_indices_hash = hashlib.sha256(validation_indices.tobytes()).hexdigest()
    total_batches = (len(train) + args.batch_size - 1) // args.batch_size
    total_global_steps = args.epochs * total_batches
    validation_batches = (len(validation) + args.batch_size - 1) // args.batch_size
    val_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")

    groups = [
        {"params": [value for name, value in model.named_parameters() if value.requires_grad and "lora_" not in name], "lr": args.lr},
        {"params": [value for name, value in model.named_parameters() if value.requires_grad and "lora_" in name], "lr": args.lr * 0.25},
    ]
    optimizer = torch.optim.AdamW(groups, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.08)

    best = None
    best_epoch = 0
    records = []
    start_epoch = 1
    skip_batches = 0
    global_step = 0
    elapsed_before = 0.0
    resumed_loss = 0.0
    resumed_seen = 0
    smoothed_train_step = None
    smoothed_validation_batch = None
    if args.resume:
        saved = torch.load(outdir / "last.pt", map_location="cpu", weights_only=False)
        if saved["source_sha256"] != source_hash or saved["token_sha256"] != token_hash:
            raise ValueError("Resume code/token hash mismatch")
        for key in ("kind", "seed", "snr", "epochs", "batch_size", "train_limit", "val_limit", "lr", "correction_cap", "classical_width", "no_checkpoint"):
            if saved["config"][key] != config_args[key]:
                raise ValueError("Resume config mismatch: " + key)
        model.load_state_dict(saved["model_state"], strict=False)
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
        best_epoch = saved["best_epoch"]
        records = saved["records"]
        elapsed_before = saved["elapsed_seconds"]
        resumed_loss = saved["epoch_loss_sum"]
        resumed_seen = saved["epoch_seen"]
        smoothed_train_step = saved.get("smoothed_train_step_seconds")
        smoothed_validation_batch = saved.get("smoothed_validation_batch_seconds")

    started = time.perf_counter()

    def elapsed() -> float:
        return elapsed_before + time.perf_counter() - started

    def smooth(previous: float | None, value: float) -> float:
        return value if previous is None else 0.15 * value + 0.85 * previous

    def remaining(validation_passes: int) -> float | None:
        if smoothed_train_step is None or global_step < min(3, total_global_steps):
            return None
        estimate = max(total_global_steps - global_step, 0) * smoothed_train_step
        if smoothed_validation_batch is not None:
            estimate += max(validation_passes, 0) * validation_batches * smoothed_validation_batch
        return estimate

    def checkpoint_save(next_epoch: int, next_batch: int, epoch_loss_sum: float = 0.0, epoch_seen: int = 0) -> None:
        data = {
            "model_state": saved_state_dict(model),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": config_args,
            "source_sha256": source_hash,
            "token_sha256": token_hash,
            "next_epoch": next_epoch,
            "next_batch": next_batch,
            "global_step": global_step,
            "best": best,
            "best_epoch": best_epoch,
            "records": records,
            "elapsed_seconds": elapsed(),
            "epoch_loss_sum": epoch_loss_sum,
            "epoch_seen": epoch_seen,
            "smoothed_train_step_seconds": smoothed_train_step,
            "smoothed_validation_batch_seconds": smoothed_validation_batch,
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }
        tmp = outdir / "last.tmp"
        torch.save(data, tmp)
        tmp.replace(outdir / "last.pt")

    def write_summary(status: str) -> None:
        atomic_json(
            outdir / "summary.json",
            {
                "status": status,
                "kind": args.kind,
                "best_validation": best,
                "best_epoch": best_epoch,
                "epochs_completed": len(records),
                "global_step": global_step,
                "total_global_steps": total_global_steps,
                "elapsed_seconds": elapsed(),
                "parameters": model.parameter_summary(),
                "shared_initialization_sha256": shared_hash,
                "train_indices_sha256": train_indices_hash,
                "validation_indices_sha256": validation_indices_hash,
                "train_samples": len(train),
                "validation_samples": len(validation),
                "test_set_used": False,
            },
        )

    def write_heartbeat(status: str, payload: dict) -> None:
        heartbeat = {
            "status": status,
            "pid": os.getpid(),
            "kind": args.kind,
            "elapsed_seconds": elapsed(),
            "run_dir": str(outdir),
            "test_set_used": False,
        } | payload
        atomic_json(outdir / "heartbeat.json", heartbeat)

    if not args.resume:
        checkpoint_save(1, 0)
    atomic_json(outdir / "training.json", records)
    write_summary("RUNNING")
    write_heartbeat("RUNNING", {"global_step": global_step, "total_global_steps": total_global_steps, "memory": gpu_memory(device)})
    print(f"[START] model={args.kind} device={device} epochs={args.epochs} batch_size={args.batch_size} train={len(train)} val={len(validation)} total_global_steps={total_global_steps} run_dir={outdir}", flush=True)
    if args.resume:
        print(f"[RESUME] epoch={start_epoch}/{args.epochs} next_step={skip_batches + 1}/{total_batches} global_step={global_step}/{total_global_steps}", flush=True)

    if start_epoch > args.epochs:
        write_summary("COMPLETED")
        print(f"[COMPLETED] model={args.kind} best_epoch={best_epoch} best_J={best['J'] if best else None} run_dir={outdir}", flush=True)
        return

    for epoch in range(start_epoch, args.epochs + 1):
        order = torch.randperm(len(train), generator=torch.Generator().manual_seed(args.seed + 1009 * epoch)).tolist()
        skip = skip_batches if epoch == start_epoch else 0
        if skip > total_batches:
            raise ValueError("Resume batch exceeds epoch length")
        epoch_data = Subset(train, order[skip * args.batch_size :])
        loader = DataLoader(epoch_data, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
        loss_sum = resumed_loss if epoch == start_epoch else 0.0
        seen = resumed_seen if epoch == start_epoch else 0
        epoch_started = time.perf_counter()
        model.train()
        for step, batch in enumerate(loader, start=skip):
            tick = time.perf_counter()
            history = batch["history_state"].to(device, non_blocking=True).float()
            mask = batch["vehicle_mask"].to(device, non_blocking=True).bool()
            future = batch["future_state"].to(device, non_blocking=True).float()
            timestamps = batch["history_timestamp"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out = model(history, mask, timestamps)
            metric = trajectory_metrics(out["prediction"], future, mask)
            tokens = token_loss(out["token_logits"], model.llm.future_token_ids(history, future), mask)
            loss = metric["loss"] + 0.035 * tokens
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("Nonfinite gradient")
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            global_step += 1
            loss_sum += float(loss.detach()) * len(history)
            seen += len(history)
            step_seconds = time.perf_counter() - tick
            smoothed_train_step = smooth(smoothed_train_step, step_seconds)
            current_step = step + 1
            eta = remaining(args.epochs - epoch + 1)
            first_step = current_step == skip + 1
            last_step = current_step == total_batches
            progress = first_step or last_step or current_step % args.progress_steps == 0
            memory = gpu_memory(device)
            if progress:
                payload = {
                    "epoch": epoch,
                    "total_epochs": args.epochs,
                    "epoch_step": current_step,
                    "total_epoch_steps": total_batches,
                    "global_step": global_step,
                    "total_global_steps": total_global_steps,
                    "loss": float(loss.detach()),
                    "ADE": float(metric["ADE"].detach()),
                    "FDE": float(metric["FDE"].detach()),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "lora_learning_rate": optimizer.param_groups[1]["lr"],
                    "step_seconds": step_seconds,
                    "smoothed_step_seconds": smoothed_train_step,
                    "estimated_remaining_seconds": eta,
                    "estimated_finish_time": estimated_finish(eta),
                    "memory": memory,
                }
                write_heartbeat("TRAINING", payload)
                print(
                    f"[TRAIN] model={args.kind} epoch={epoch}/{args.epochs} step={current_step}/{total_batches} "
                    f"global={global_step}/{total_global_steps} loss={payload['loss']:.6f} ADE={payload['ADE']:.6f} FDE={payload['FDE']:.6f} "
                    f"lr={payload['learning_rate']:.3e} step_time={step_seconds:.3f}s smooth={smoothed_train_step:.3f}s "
                    f"elapsed={human_seconds(elapsed())} ETA={human_seconds(eta)} finish={payload['estimated_finish_time']} "
                    f"GPU alloc/resv={memory['allocated_gib']:.2f}/{memory['reserved_gib']:.2f}GiB "
                    f"peak={memory['peak_allocated_gib']:.2f}/{memory['peak_reserved_gib']:.2f}GiB",
                    flush=True,
                )
            if first_step or last_step or global_step % args.save_steps == 0:
                checkpoint_save(epoch, current_step, loss_sum, seen)
            if args.max_seconds > 0 and elapsed() >= args.max_seconds:
                checkpoint_save(epoch, current_step, loss_sum, seen)
                write_summary("PAUSED_BUDGET")
                write_heartbeat("PAUSED_BUDGET", {"epoch": epoch, "epoch_step": current_step, "global_step": global_step, "total_global_steps": total_global_steps, "memory": memory})
                print(f"[PAUSED_BUDGET] model={args.kind} global_step={global_step}/{total_global_steps} run_dir={outdir}", flush=True)
                return

        scheduler.step()
        print(f"[VALIDATION] model={args.kind} epoch={epoch}/{args.epochs} starting batches={validation_batches}", flush=True)
        validation_result, rows, validation_batch_seconds = evaluate(model, val_loader, device)
        smoothed_validation_batch = smooth(smoothed_validation_batch, validation_batch_seconds)
        record = {
            "epoch": epoch,
            "train_loss": loss_sum / max(seen, 1),
            "validation": validation_result,
            "training_seconds": time.perf_counter() - epoch_started,
            "smoothed_train_step_seconds": smoothed_train_step,
            "validation_batch_seconds": validation_batch_seconds,
            "smoothed_validation_batch_seconds": smoothed_validation_batch,
        }
        records.append(record)
        if best is None or validation_result["J"] < best["J"]:
            best = validation_result
            best_epoch = epoch
            tmp = outdir / "best.tmp"
            torch.save(
                {
                    "model_state": saved_state_dict(model),
                    "validation": best,
                    "epoch": epoch,
                    "config": config_args,
                    "source_sha256": source_hash,
                    "token_sha256": token_hash,
                },
                tmp,
            )
            tmp.replace(outdir / "best.pt")
            atomic_json(outdir / "best_validation_rows.json", {"summary": validation_result, "rows": rows, "test_set_used": False})
        checkpoint_save(epoch + 1, 0)
        atomic_json(outdir / "training.json", records)
        write_summary("RUNNING")
        eta = remaining(args.epochs - epoch)
        memory = gpu_memory(device)
        write_heartbeat(
            "VALIDATED",
            {
                "epoch": epoch,
                "total_epochs": args.epochs,
                "global_step": global_step,
                "total_global_steps": total_global_steps,
                "validation": validation_result,
                "best_epoch": best_epoch,
                "best_J": best["J"],
                "estimated_remaining_seconds": eta,
                "estimated_finish_time": estimated_finish(eta),
                "memory": memory,
            },
        )
        print(
            f"[VALIDATION] model={args.kind} epoch={epoch}/{args.epochs} ADE={validation_result['ADE']:.6f} "
            f"FDE={validation_result['FDE']:.6f} J={validation_result['J']:.6f} best_epoch={best_epoch} best_J={best['J']:.6f} "
            f"elapsed={human_seconds(elapsed())} ETA={human_seconds(eta)} finish={estimated_finish(eta)}",
            flush=True,
        )
        skip_batches = 0
        resumed_loss = 0.0
        resumed_seen = 0

    write_summary("COMPLETED")
    write_heartbeat("COMPLETED", {"global_step": global_step, "total_global_steps": total_global_steps, "best_validation": best, "best_epoch": best_epoch, "memory": gpu_memory(device)})
    print(f"[COMPLETED] model={args.kind} best_epoch={best_epoch} best_J={best['J']:.6f} elapsed={human_seconds(elapsed())} run_dir={outdir}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        traceback.print_exc()
        if "--run-dir" in sys.argv:
            directory = ROOT / sys.argv[sys.argv.index("--run-dir") + 1]
            directory.mkdir(parents=True, exist_ok=True)
            atomic_json(directory / "failure.json", {"status": "FAILED", "error": repr(exc), "traceback": traceback.format_exc()})
        raise
