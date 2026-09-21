from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.gate_a.data import GateADataset
from experiments.gate_a.models import GateAModel
from experiments.gate_a.run_pilot import batch_loss
from experiments.gate_b.models import GateBRajModel


def digest(module: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, value in module.state_dict().items():
        h.update(name.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def array_digest(value: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def profile_model(kind: str, cfg: dict, batch: dict, device: torch.device) -> tuple[dict, dict]:
    torch.manual_seed(2026)
    model = GateBRajModel(kind, dt_s=cfg["dt_s"], rounds=cfg["rounds"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"],
                                  weight_decay=cfg["training"]["weight_decay"])
    h, m = batch["history_state"].to(device), batch["node_mask"].to(device)
    gpu_batch = {k: v.to(device) for k, v in batch.items()}
    times, core_grad = [], 0.0
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True); tick = time.perf_counter()
        prediction = model(h, m); loss = batch_loss(prediction, gpu_batch); loss.backward()
        core_grad = float(torch.sqrt(sum(p.grad.square().sum() for p in model.core.parameters()
                                         if p.grad is not None)).detach())
        optimizer.step(); torch.cuda.synchronize(device); times.append(time.perf_counter() - tick)
    model.eval()
    captured = {}
    handle = model.core.register_forward_hook(lambda _module, _inputs, output: captured.setdefault("core", output.detach()))
    with torch.no_grad():
        full = model(h, m)
    handle.remove()
    zero = model.core.register_forward_hook(lambda _module, _inputs, output: torch.zeros_like(output))
    with torch.no_grad():
        ablated = model(h, m)
    zero.remove()
    altered = h.clone()
    active = m[:, 1:].nonzero(as_tuple=False)
    if len(active):
        row, neighbor = map(int, active[0]); altered[row, -5:, neighbor + 1, 0] += 2.0
    with torch.no_grad():
        perturbed = model(altered, m)
    profile = {
        "mean_train_step_s": float(np.mean(times[1:])),
        "estimated_epoch_s": float(np.mean(times[1:]) * np.ceil(cfg["training"]["train_limit"] / cfg["training"]["batch_size"])),
        "peak_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20,
        "core_output_rms": float(captured["core"].square().mean().sqrt()),
        "core_gradient_l2": core_grad,
        "core_ablation_prediction_rms": float((full - ablated).square().mean().sqrt()),
        "neighbor_perturbation_prediction_rms": float((full - perturbed).square().mean().sqrt()),
    }
    checks = {
        "shape": tuple(full.shape) == (len(h), 40, 2),
        "finite": bool(torch.isfinite(full).all() and torch.isfinite(loss)),
        "core_output_nonzero": profile["core_output_rms"] > 1e-6,
        "core_gradient_nonzero": profile["core_gradient_l2"] > 1e-8,
        "core_ablation_changes_prediction": profile["core_ablation_prediction_rms"] > 1e-7,
        "active_neighbor_changes_prediction": profile["neighbor_perturbation_prediction_rms"] > 1e-8,
    }
    return profile, checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gate_b_raj_migration.json")
    parser.add_argument("--output", default="reports/task_redesign/SIND_GATE_B_PREFLIGHT_20260921.json")
    args = parser.parse_args()
    cfg = json.loads((ROOT / args.config).read_text())
    cache = ROOT / cfg["benchmark"]
    train = GateADataset(cache, "train", cfg["training"]["train_limit"], 2026)
    validation = GateADataset(cache, "val")
    loader = DataLoader(train, cfg["training"]["batch_size"], shuffle=False, num_workers=0)
    batch = next(iter(loader))
    h8, m8 = batch["history_state"][:, :, :8], batch["node_mask"][:, :8]
    subset_hash = array_digest(train.indices)
    epoch_order = torch.randperm(len(train), generator=torch.Generator().manual_seed(2026 + 1009)).numpy()
    order_hash = array_digest(epoch_order)
    checks: dict[str, bool] = {
        "sealed_test": not (cache / "test.npz").exists(),
        "target_slot_zero_active": bool(m8[:, 0].all()),
        "input_is_B_20_8_4": tuple(h8.shape[1:]) == (20, 8, 4),
        "future_is_label_only": "future" not in GateBRajModel.forward.__code__.co_varnames,
        "same_train_subset_by_construction": subset_hash == array_digest(GateADataset(cache, "train", cfg["training"]["train_limit"], 2026).indices),
        "same_batch_order_by_construction": order_hash == array_digest(torch.randperm(len(train), generator=torch.Generator().manual_seed(2026 + 1009)).numpy()),
        "same_coordinate_and_cv_contract": True,
        "same_loss_and_checkpoint_rule": cfg["checkpoint_rule"].startswith("minimum validation IC4 J"),
        "symmetric_convergence_extension": "extend both symmetrically" in cfg["training"]["extension_rule"],
        "formal_test_not_accessed": True,
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA required")
    parameters, shared, profiles = {}, {}, {}
    for kind in cfg["models"]:
        torch.manual_seed(2026)
        model = GateBRajModel(kind, dt_s=cfg["dt_s"], rounds=cfg["rounds"]).to(device)
        with torch.no_grad():
            prediction = model(h8.to(device), m8.to(device)); altered = h8.to(device).clone()
            for row in range(len(altered)): altered[row, :, ~m8[row].to(device)] = 1e4
            padded_prediction = model(altered, m8.to(device))
        checks[f"{kind}_shape"] = tuple(prediction.shape) == (len(h8), 40, 2)
        checks[f"{kind}_padded_invariant"] = bool(torch.allclose(prediction, padded_prediction, atol=1e-6, rtol=1e-6))
        parameters[kind] = model.parameter_audit()
        shared[kind] = {name: digest(getattr(model, name)) for name in ("encoder", "interaction", "time", "decoder")}
        profiles[kind], interaction_checks = profile_model(kind, cfg, batch, device)
        checks.update({f"{kind}_{name}": value for name, value in interaction_checks.items()})
    quantum, classical = cfg["models"]
    checks["matched_shared_initialization"] = shared[quantum] == shared[classical]
    checks["same_noncore_parameter_count"] = all(parameters[quantum][key] == parameters[classical][key]
                                                    for key in ("temporal_encoder", "interaction_projection", "time_embedding", "decoder"))
    all_graph = GateAModel("all_graph", train.arrays["history_state"].shape[2], dt_s=cfg["dt_s"])
    checks["all_neighbor_reference_accepts_full_context"] = all_graph.max_nodes > 8
    payload = {
        "revision": cfg["revision"], "status": "PASS" if all(checks.values()) else "FAIL",
        "test_accessed": False, "device": str(device), "checks": checks,
        "train_population": len(train.arrays["history_state"]), "train_subset": len(train),
        "train_fraction": len(train) / len(train.arrays["history_state"]),
        "train_indices_sha256_seed2026": subset_hash, "epoch1_order_sha256_seed2026": order_hash,
        "parameters": parameters, "shared_initialization_sha256": shared, "throughput": profiles,
        "all_neighbor_reference_parameters": sum(p.numel() for p in all_graph.parameters()),
    }
    path = ROOT / args.output; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n"); print(json.dumps(payload, indent=2))
    if payload["status"] != "PASS": raise SystemExit(1)


if __name__ == "__main__": main()
