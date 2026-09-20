#!/usr/bin/env python3
"""P2 gate for the Raj QGNN -> LLM residual interface.

This check uses train histories only. It performs no optimizer step and never
opens future labels, validation data, or test data.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prediction.qgnn_raj_pennylane import build_model


def train_history_sample(root: Path, snr_db: float = 0.0):
    sample_path = root / "data/sind/splits/train/samples.npz"
    cache_path = root / "data/sind/isac/train/sensing_cache.npz"
    with np.load(sample_path, allow_pickle=False) as samples:
        masks = samples["vehicle_mask"]
        candidates = np.flatnonzero(masks.sum(axis=1) >= 3)
        if not len(candidates):
            raise RuntimeError("No SinD train sample has at least three vehicles")
        index = int(candidates[0])
        scene = int(samples["scene_id"][index])
        start = int(samples["start_frame"][index])
        vehicle_ids = samples["vehicle_ids"][index]
        mask = masks[index].astype(np.bool_)

    with np.load(cache_path, allow_pickle=False) as cache:
        hits = np.flatnonzero(np.isclose(cache["snr_levels_db"], snr_db))
        if len(hits) != 1:
            raise ValueError(f"SinD train cache has no unique SNR {snr_db}")
        state_hat = cache["state_hat"][int(hits[0])]
        lookup = {
            (int(scene_id), int(frame), int(vehicle_id)): row
            for row, (scene_id, frame, vehicle_id) in enumerate(
                zip(cache["scene_id"], cache["frame"], cache["vehicle_id"])
            )
        }
        history = np.zeros((20, 8, 4), dtype=np.float32)
        for step in range(20):
            for slot in range(8):
                if mask[slot]:
                    key = (scene, start + step, int(vehicle_ids[slot]))
                    history[step, slot] = state_hat[lookup[key]]
    return index, history[None], mask[None]


def count_trainable(module):
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def grad_summary(named_parameters):
    rows = []
    for name, parameter in named_parameters:
        if parameter.grad is None:
            continue
        rows.append(
            {
                "name": name,
                "finite": bool(torch.isfinite(parameter.grad).all()),
                "sum_abs": float(parameter.grad.abs().sum()),
            }
        )
    return {
        "tensors": len(rows),
        "all_finite": bool(rows) and all(row["finite"] for row in rows),
        "nonzero_tensors": sum(row["sum_abs"] > 1e-16 for row in rows),
        "sum_abs": sum(row["sum_abs"] for row in rows),
    }


def record(checks, name, passed, **evidence):
    checks[name] = {"passed": bool(passed), **evidence}
    print(json.dumps({"check": name, **checks[name]}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="reports/qgnn/raj_pennylane_p2/preflight_20260920.json",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("P2 Raj interface gate requires CUDA")

    device = torch.device("cuda:0")
    checks = {}
    report = {
        "schema": "raj_pennylane_p2_residual_v1",
        "scope": {
            "optimizer_steps": 0,
            "train_histories_only": True,
            "labels_opened": False,
            "validation_opened": False,
            "test_opened": False,
            "ade_fde_claim": False,
        },
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(0),
        },
    }

    torch.manual_seed(20260920)
    model = build_model(phase="self").to(device)
    interface_parameters = model.llm.interface_parameter_count()
    record(
        checks,
        "small_auxiliary_interface",
        interface_parameters <= 50000,
        interface_parameters=interface_parameters,
        limit=50000,
        query_dim=model.llm.query_dim,
        raj_readout_dim=model.llm.readout_dim,
    )

    phase_rows = {}
    for phase in ("self", "interaction", "joint"):
        model.set_training_phase(phase)
        lora_trainable = sum(
            parameter.numel()
            for name, parameter in model.llm.base.named_parameters()
            if parameter.requires_grad
            and name.endswith(("lora_A", "lora_B"))
        )
        phase_rows[phase] = {
            "core": count_trainable(model.core),
            "base": count_trainable(model.llm.base),
            "self_head": count_trainable(model.llm.self_head),
            "interface": sum(
                parameter.numel()
                for name, parameter in model.llm.named_parameters()
                if parameter.requires_grad
                and not name.startswith(("base.", "self_head."))
            ),
            "lora": lora_trainable,
        }
    phase_contract = (
        phase_rows["self"]["core"] == 0
        and phase_rows["self"]["self_head"] > 0
        and phase_rows["self"]["interface"] == 0
        and phase_rows["interaction"]["core"] > 0
        and phase_rows["interaction"]["base"] == 0
        and phase_rows["interaction"]["self_head"] == 0
        and phase_rows["interaction"]["interface"] == interface_parameters
        and phase_rows["joint"]["core"] > 0
        and phase_rows["joint"]["self_head"] == 0
        and phase_rows["joint"]["interface"] == interface_parameters
        and phase_rows["joint"]["base"] == phase_rows["joint"]["lora"] > 0
    )
    record(checks, "three_phase_trainability", phase_contract, phases=phase_rows)

    sample_index, history_np, mask_np = train_history_sample(ROOT)
    history = torch.as_tensor(history_np, device=device)
    mask = torch.as_tensor(mask_np, device=device)

    calls = {"count": 0}

    def count_call(_module, _inputs):
        calls["count"] += 1

    hook = model.core.register_forward_pre_hook(count_call)
    model.eval()
    model.set_training_phase("self")
    with torch.no_grad():
        self_output = model(history, mask)
    hook.remove()
    record(
        checks,
        "self_phase_bypasses_quantum_core",
        calls["count"] == 0
        and float(self_output["interaction_gate"].abs().max()) == 0.0,
        core_calls=calls["count"],
        gate_max_abs=float(self_output["interaction_gate"].abs().max()),
        prediction_shape=list(self_output["prediction"].shape),
    )

    active_readout = torch.randn(
        history.shape[0], history.shape[2], model.core.readout_dim, device=device
    )
    zero_readout = torch.zeros_like(active_readout)
    with torch.no_grad():
        left = model.llm(history, mask, zero_readout, torch.zeros_like(mask))
        right = model.llm(history, mask, active_readout, mask)
    self_difference = float((left["prediction"] - right["prediction"]).abs().max())
    record(
        checks,
        "self_branch_graph_exclusive",
        self_difference == 0.0
        and float(left["interaction_gate"].abs().max()) == 0.0
        and float(right["interaction_gate"].abs().max()) == 0.0,
        prediction_max_abs_difference=self_difference,
    )

    model.train()
    model.set_training_phase("interaction")
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    output = model(history, mask)
    weight = torch.randn_like(output["prediction"])
    loss = (output["prediction"] * weight).mean()
    loss.backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    core_grad = grad_summary(model.core.named_parameters())
    interface_grad = grad_summary(
        (name, parameter)
        for name, parameter in model.llm.named_parameters()
        if not name.startswith(("base.", "self_head."))
    )
    frozen_grad = grad_summary(
        (name, parameter)
        for name, parameter in model.llm.named_parameters()
        if name.startswith(("base.", "self_head."))
    )
    active = mask[:, None, :, None].expand_as(output["interaction_gate"])
    inactive_gate = output["interaction_gate"].masked_select(~active)
    interaction_ok = (
        output["prediction"].shape == (1, 20, 8, 2)
        and output["interaction_readout"].shape == (1, 8, 64)
        and bool(torch.isfinite(output["prediction"]).all())
        and bool(torch.isfinite(loss))
        and core_grad["all_finite"]
        and core_grad["nonzero_tensors"] > 0
        and interface_grad["all_finite"]
        and interface_grad["nonzero_tensors"] > 0
        and frozen_grad["tensors"] == 0
        and (inactive_gate.numel() == 0 or float(inactive_gate.abs().max()) == 0.0)
    )
    report["actual_train_history"] = {
        "dataset": "SinD",
        "split": "train",
        "snr_db": 0.0,
        "sample_index": sample_index,
        "active_vehicles": int(mask.sum()),
        "labels_opened": False,
    }
    report["profile"] = {
        "batch": 1,
        "seconds_forward_backward": elapsed,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    record(
        checks,
        "actual_train_history_forward_backward",
        interaction_ok,
        loss=float(loss.detach()),
        core_gradients=core_grad,
        interface_gradients=interface_grad,
        frozen_llm_gradients=frozen_grad,
        profile=report["profile"],
        prediction_shape=list(output["prediction"].shape),
        readout_shape=list(output["interaction_readout"].shape),
    )

    report["checks"] = checks
    report["passed"] = all(row["passed"] for row in checks.values())
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "check_count": len(checks),
                "output": str(output_path),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
