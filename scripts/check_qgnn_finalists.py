"""Synthetic correctness and optional RTX 4090 profile for QGNN finalists; never opens prediction test."""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from prediction.q0.metrics import trajectory_metrics
from prediction.qgnn_finalists.common import to_patch_indices
from prediction.qgnn_finalists.model import build_model
from prediction.qgnn_finalists.toj import TOJQGNNCore, TRTGNCore
from prediction.qgnn_finalists.trc import RTCNCore, TRCQuantumCore
from scripts.train_q0_motion_llm import token_loss


CORE_BUILDERS = {
    "trc": TRCQuantumCore,
    "rtcn": RTCNCore,
    "toj": TOJQGNNCore,
    "trtgn": TRTGNCore,
}
EXPECTED_SHAPES = {
    "trc": [1, 8, 2, 4, 35],
    "rtcn": [1, 8, 2, 4, 35],
    "toj": [1, 8, 4, 31],
    "trtgn": [1, 8, 4, 31],
}


def memory(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {"peak_allocated_gib": 0.0, "peak_reserved_gib": 0.0}
    return {
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
    }


def synthetic_history(batch: int, device: torch.device, dtype=torch.float32):
    history = torch.randn(batch, 20, 8, 4, device=device, dtype=dtype)
    history[..., 2:] *= 0.4
    mask = torch.ones(batch, 8, dtype=torch.bool, device=device)
    timestamps = torch.arange(20, device=device, dtype=torch.float64)[None].expand(batch, -1) * 0.1001001
    future = torch.empty(batch, 20, 8, 4, device=device, dtype=dtype)
    future[..., :2] = history[:, -1:, :, :2] + torch.randn(batch, 20, 8, 2, device=device, dtype=dtype) * 0.1
    future[..., 2:] = history[:, -1:, :, 2:] + torch.randn(batch, 20, 8, 2, device=device, dtype=dtype) * 0.02
    return history, mask, timestamps, future


def nonzero_finite_gradients(module: torch.nn.Module):
    gradients = [parameter.grad for parameter in module.parameters() if parameter.requires_grad]
    good = sum(int(gradient is not None and torch.isfinite(gradient).all() and float(gradient.abs().sum()) > 0.0) for gradient in gradients)
    return len(gradients), good


def core_check(name: str, device: torch.device):
    torch.manual_seed(97)
    history, mask, timestamps, _ = synthetic_history(1, device)
    permutation = torch.tensor([3, 0, 7, 2, 6, 1, 5, 4], device=device)
    model = CORE_BUILDERS[name]().to(device).train()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    tick = time.perf_counter()
    out = model(history, mask, timestamps)
    loss = out["readout"].square().mean()
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    total_grads, nonzero_grads = nonzero_finite_gradients(model)
    model.eval()
    with torch.no_grad():
        base = model(history, mask, timestamps)["readout"]
        permuted = model(history[:, :, permutation], mask[:, permutation], timestamps)["readout"]
        padded_mask = mask.clone()
        padded_mask[:, 7] = False
        padded_history = history.clone()
        padded_history[:, :, 7] = 0.0
        pad_base = model(padded_history, padded_mask, timestamps)["readout"]
        padded_history[:, :, 7] = 9999.0 * torch.randn_like(padded_history[:, :, 7])
        pad_changed = model(padded_history, padded_mask, timestamps)["readout"]
        empty = model(torch.zeros_like(history), torch.zeros_like(mask), timestamps)["readout"]
    model64 = CORE_BUILDERS[name]().to(device).double().eval()
    with torch.no_grad():
        output64 = model64(history.double(), mask, timestamps)["readout"]
    row = {
        "shape": list(out["readout"].shape),
        "shape_expected": EXPECTED_SHAPES[name],
        "finite_forward_backward": bool(torch.isfinite(out["readout"]).all() and torch.isfinite(loss)),
        "trainable_gradient_tensors": total_grads,
        "nonzero_finite_gradient_tensors": nonzero_grads,
        "permutation_max_abs": float((permuted - base[:, permutation]).abs().max()),
        "padding_active_max_abs": float((pad_base[:, :7] - pad_changed[:, :7]).abs().max()),
        "all_padding_max_abs": float(empty.abs().max()),
        "float64_forward_finite": bool(torch.isfinite(output64).all()),
        "seconds_including_backward_and_checks": time.perf_counter() - tick,
    } | memory(device)
    del model, model64, out, loss
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def selector_check(device: torch.device):
    history = torch.zeros(1, 20, 3, 4, device=device)
    mask = torch.ones(1, 3, dtype=torch.bool, device=device)
    risk = torch.zeros(1, 4, 3, 3, device=device)
    risk[:, 0, 0, 1] = 1.0
    risk[:, :, 0, 2] = 0.8
    distance = torch.ones(1, 4, 3, 3, device=device)
    slots, valid = to_patch_indices(risk, {"d": distance}, mask, cap=2, history=history)
    return {
        "root_slot_is_root": bool(slots[0, 0, 0] == 0),
        "final_stage_weighted_selection": int(slots[0, 0, 1]),
        "expected_selected_neighbor": 2,
        "selected_slot_valid": bool(valid[0, 0, 1]),
    }


def compact_state(model: torch.nn.Module):
    skipped = ("llm.base.gpt2.", "llm.gpt2.")
    return {key: value.detach().cpu() for key, value in model.state_dict().items() if not (key.startswith(skipped) and "lora_" not in key)}


def model_check(kind: str, device: torch.device):
    torch.manual_seed(137)
    history, mask, timestamps, future = synthetic_history(1, device)
    model = build_model(kind, seed=2026, classical_width=128 if kind == "rtcn" else 64).to(device).train()
    out = model(history, mask, timestamps)
    metrics = trajectory_metrics(out["prediction"], future, mask)
    loss = metrics["loss"] + 0.035 * token_loss(out["token_logits"], model.llm.future_token_ids(history, future), mask)
    loss.backward()
    total, nonzero = nonzero_finite_gradients(model.core)
    model.eval()
    with torch.no_grad():
        before = model(history, mask, timestamps)["prediction"]
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "model.pt"
        torch.save({"model_state": compact_state(model)}, checkpoint)
        restored = build_model(kind, seed=2026, classical_width=128 if kind == "rtcn" else 64).to(device).eval()
        restored.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model_state"], strict=False)
        with torch.no_grad():
            after = restored(history, mask, timestamps)["prediction"]
    row = {
        "prediction_shape": list(out["prediction"].shape),
        "token_logits_shape": list(out["token_logits"].shape),
        "finite_forward_backward": bool(torch.isfinite(loss) and torch.isfinite(out["prediction"]).all() and torch.isfinite(out["token_logits"]).all()),
        "core_trainable_gradient_tensors": total,
        "core_nonzero_finite_gradient_tensors": nonzero,
        "checkpoint_load_prediction_max_abs": float((before - after).abs().max()),
        "parameters": model.parameter_summary(),
    }
    del model, restored, out, loss
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def train_step(model, optimizer, history, mask, timestamps, future):
    optimizer.zero_grad(set_to_none=True)
    out = model(history, mask, timestamps)
    metrics = trajectory_metrics(out["prediction"], future, mask)
    loss = metrics["loss"] + 0.035 * token_loss(out["token_logits"], model.llm.future_token_ids(history, future), mask)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
    optimizer.step()
    return loss


def profile(kind: str, batch_sizes: list[int], warmup_steps: int, measured_steps: int, device: torch.device):
    rows = {}
    for batch_size in batch_sizes:
        torch.manual_seed(1000 + batch_size)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)
        model = build_model(kind, seed=2026, classical_width=128 if kind == "rtcn" else 64).to(device).train()
        optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=3e-4, weight_decay=2e-4)
        history, mask, timestamps, future = synthetic_history(batch_size, device)
        for _ in range(warmup_steps):
            train_step(model, optimizer, history, mask, timestamps, future)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        steps = []
        for _ in range(measured_steps):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            tick = time.perf_counter()
            loss = train_step(model, optimizer, history, mask, timestamps, future)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            steps.append(time.perf_counter() - tick)
        rows[str(batch_size)] = {
            "batch_size": batch_size,
            "trainable_parameters": model.parameter_summary()["trainable"],
            "core_trainable_parameters": model.parameter_summary()["core_trainable"],
            "step_seconds_forward_backward_optimizer_mean": float(sum(steps) / len(steps)),
            "step_seconds_forward_backward_optimizer_min": float(min(steps)),
            "step_seconds_forward_backward_optimizer_max": float(max(steps)),
            "final_loss_finite": bool(torch.isfinite(loss)),
        } | memory(device)
        del model, optimizer, history, mask, timestamps, future, loss
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="reports/qgnn/finalists_preflight/preflight_20260920.json")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--measured-steps", type=int, default=2)
    args = parser.parse_args()
    if args.warmup_steps < 0 or args.measured_steps < 1 or any(batch < 1 for batch in args.batch_sizes):
        raise ValueError("Profile sizes must be positive, with at least one measured step")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(97)
    core = {name: core_check(name, device) for name in CORE_BUILDERS}
    selector = selector_check(device)
    models = {name: model_check(name, device) for name in CORE_BUILDERS}
    profiles = {name: profile(name, args.batch_sizes, args.warmup_steps, args.measured_steps, device) for name in CORE_BUILDERS} if args.profile else {}
    core_ok = all(
        row["shape"] == row["shape_expected"]
        and row["finite_forward_backward"]
        and row["trainable_gradient_tensors"] == row["nonzero_finite_gradient_tensors"]
        and row["permutation_max_abs"] < 1e-5
        and row["padding_active_max_abs"] < 1e-7
        and row["all_padding_max_abs"] < 1e-7
        and row["float64_forward_finite"]
        for row in core.values()
    )
    model_ok = all(
        row["finite_forward_backward"]
        and row["core_trainable_gradient_tensors"] == row["core_nonzero_finite_gradient_tensors"]
        and row["checkpoint_load_prediction_max_abs"] < 1e-6
        for row in models.values()
    )
    selector_ok = selector["root_slot_is_root"] and selector["selected_slot_valid"] and selector["final_stage_weighted_selection"] == selector["expected_selected_neighbor"]
    profile_ok = all(row["final_loss_finite"] for arm in profiles.values() for row in arm.values())
    payload = {
        "status": "PASS" if core_ok and model_ok and selector_ok and profile_ok else "FAIL",
        "device": str(device),
        "test_set_used": False,
        "core_checks": core,
        "toj_selector_check": selector,
        "end_to_end_model_checks": models,
        "profiles": profiles,
        "profile_definition": "Synthetic full-model train step: forward + ADE/FDE loss + token loss + backward + gradient clip + AdamW update.",
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if payload["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
