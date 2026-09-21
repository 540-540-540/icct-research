from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.gate_a.data import GateADataset
from experiments.gate_a.models import GateAModel
from experiments.gate_b_plus.models import GateBPlusQuantumModel
from experiments.gate_b_plus.run_screen import screen_loss

HORIZONS = tuple(range(5, 41, 5))


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def atomic_json(path: Path, payload: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)


def decode(model: GateBPlusQuantumModel, history: torch.Tensor, temporal: torch.Tensor,
           interaction: torch.Tensor) -> torch.Tensor:
    target = torch.cat((temporal, interaction), -1)
    steps = torch.arange(40, device=history.device)
    times = (steps.to(history.dtype) + 1) * model.dt_s
    cv = history[:, -1, 0, 2:4][:, None] * times[None, :, None]
    features = torch.cat((target[:, None].expand(-1, 40, -1),
                          model.time(steps)[None].expand(len(history), -1, -1),
                          cv / 20.0), -1)
    return cv + model.decoder(features) * 10.0


def forward_parts(model: GateBPlusQuantumModel, history: torch.Tensor,
                  node_mask: torch.Tensor) -> dict[str, torch.Tensor]:
    history = history[:, :, :8]
    node_mask = node_mask[:, :8]
    history = torch.where(node_mask[:, None, :, None], history, 0.0)
    j2 = model.core.j2(history, node_mask)[:, 0]
    j3 = model.core.j3(history, node_mask)[:, 0]
    quantum = torch.cat((j2, j3), -1)
    interaction = model.interaction(quantum)
    temporal = model.encoder(history)[:, 0]
    prediction = decode(model, history, temporal, interaction)
    return {"history": history, "j2": j2, "j3": j3, "quantum": quantum,
            "interaction": interaction, "temporal": temporal, "prediction": prediction}


def variant_prediction(model: GateBPlusQuantumModel, parts: dict[str, torch.Tensor], variant: str) -> torch.Tensor:
    j2, j3 = parts["j2"], parts["j3"]
    if variant == "full":
        quantum = torch.cat((j2, j3), -1)
    elif variant == "j2_only":
        quantum = torch.cat((j2, torch.zeros_like(j3)), -1)
    elif variant == "j3_only":
        quantum = torch.cat((torch.zeros_like(j2), j3), -1)
    elif variant == "no_quantum":
        quantum = torch.zeros_like(parts["quantum"])
    elif variant == "j2_permuted":
        quantum = torch.cat((j2.roll(1, 0) if len(j2) > 1 else torch.zeros_like(j2), j3), -1)
    elif variant == "j3_permuted":
        quantum = torch.cat((j2, j3.roll(1, 0) if len(j3) > 1 else torch.zeros_like(j3)), -1)
    else:
        raise ValueError(variant)
    return decode(model, parts["history"], parts["temporal"], model.interaction(quantum))


def add_errors(store: dict, name: str, prediction: torch.Tensor, batch: dict) -> None:
    error = torch.linalg.vector_norm(prediction.detach().cpu() - batch["future_xy"], dim=-1)
    mask = batch["future_mask"]
    origin = batch["origin_id"]
    critical = batch["k"] >= 1
    for scope, select in (("full", torch.ones_like(critical, dtype=torch.bool)), ("ic", critical)):
        for horizon in HORIZONS:
            endpoint = horizon - 1
            for i in torch.nonzero(select, as_tuple=True)[0].tolist():
                key = (name, scope, horizon, int(origin[i]))
                valid = mask[i, :horizon]
                if bool(valid.any()):
                    store["ade"][key].append(float((error[i, :horizon] * valid).sum() / valid.sum()))
                if bool(mask[i, endpoint]):
                    store["fde"][key].append(float(error[i, endpoint]))


def summarize_errors(store: dict) -> dict:
    out = {}
    keys = sorted({(name, scope, horizon) for metric in store.values()
                   for name, scope, horizon, _ in metric})
    for name, scope, horizon in keys:
        row = {}
        for metric in ("ade", "fde"):
            origin_values = [statistics_mean(values) for (n, s, h, _), values in store[metric].items()
                             if (n, s, h) == (name, scope, horizon)]
            row[f"{metric}_m"] = statistics_mean(origin_values)
            row[f"{metric}_origins"] = len(origin_values)
        row["J_m"] = row["ade_m"] + 0.5 * row["fde_m"]
        out.setdefault(name, {}).setdefault(scope, {})[f"{horizon / 10:g}s"] = row
    return out


def statistics_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def spectrum_stats(x: np.ndarray) -> dict:
    x = x.astype(np.float64)
    x = x - x.mean(0, keepdims=True)
    covariance = x.T @ x / max(len(x) - 1, 1)
    eigenvalues = np.linalg.eigvalsh(covariance).clip(0)[::-1]
    total = float(eigenvalues.sum())
    probability = eigenvalues / max(total, 1e-12)
    nz = probability[probability > 1e-12]
    effective_rank = float(np.exp(-(nz * np.log(nz)).sum()))
    participation = float(total * total / max(float(np.square(eigenvalues).sum()), 1e-12))
    return {"samples": len(x), "width": x.shape[1], "total_variance": total,
            "mean_feature_variance": float(np.var(x, axis=0, ddof=1).mean()),
            "effective_rank": effective_rank, "participation_ratio": participation,
            "rank_99pct": int(np.searchsorted(np.cumsum(probability), 0.99) + 1),
            "top_eigenvalue_fraction": float(probability[0]) if len(probability) else float("nan")}


def relation_stats(x: np.ndarray, y: np.ndarray) -> dict:
    x = x.astype(np.float64); y = y.astype(np.float64)
    cosine = (x * y).sum(1) / np.maximum(np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1), 1e-12)
    xc = x - x.mean(0, keepdims=True); yc = y - y.mean(0, keepdims=True)
    cross = xc.T @ yc
    cka = float(np.square(cross).sum() /
                max(math.sqrt(float(np.square(xc.T @ xc).sum()) * float(np.square(yc.T @ yc).sum())), 1e-12))
    xs = xc / np.maximum(xc.std(0, ddof=1, keepdims=True), 1e-12)
    ys = yc / np.maximum(yc.std(0, ddof=1, keepdims=True), 1e-12)
    matched_corr = (xs * ys).sum(0) / max(len(x) - 1, 1)
    return {"cosine_mean": float(cosine.mean()), "cosine_std": float(cosine.std(ddof=1)),
            "cosine_p05": float(np.quantile(cosine, 0.05)),
            "cosine_p95": float(np.quantile(cosine, 0.95)),
            "linear_cka": cka, "matched_feature_abs_correlation_mean": float(np.abs(matched_corr).mean())}


def update_distance(seed: int, trained: GateBPlusQuantumModel, cfg: dict) -> dict:
    seed_all(seed)
    initial = GateBPlusQuantumModel("full_multiscale", dt_s=cfg["dt_s"], rounds=cfg["rounds"])
    result = {}
    for branch in ("j2", "j3"):
        init = dict(getattr(initial.core, branch).named_parameters())
        numerator = denominator = 0.0
        per_parameter = []
        for name, parameter in getattr(trained.core, branch).named_parameters():
            delta = parameter.detach().cpu() - init[name].detach().cpu()
            dnorm = float(torch.linalg.vector_norm(delta))
            pnorm = float(torch.linalg.vector_norm(init[name].detach().cpu()))
            numerator += dnorm * dnorm; denominator += pnorm * pnorm
            per_parameter.append(dnorm / max(pnorm, 1e-12))
        result[branch] = {"relative_l2_update": math.sqrt(numerator / max(denominator, 1e-24)),
                          "median_parameter_relative_update": float(np.median(per_parameter))}
    return result


def gradient_diagnostics(model: GateBPlusQuantumModel, loader: DataLoader, device: torch.device) -> dict:
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()
    branch_rows = []
    horizon_rows = defaultdict(list)
    for batch_no, batch_cpu in enumerate(loader):
        if batch_no >= 4:
            break
        batch = {key: value.to(device) for key, value in batch_cpu.items()}
        model.zero_grad(set_to_none=True)
        parts = forward_parts(model, batch["history_state"], batch["node_mask"])
        loss = screen_loss(parts["prediction"], batch, 0.75)
        loss.backward()
        row = {}
        for name, module in (("j2", model.core.j2), ("j3", model.core.j3),
                             ("projection", model.interaction), ("temporal", model.encoder),
                             ("decoder", model.decoder)):
            grads = [parameter.grad.detach() for parameter in module.parameters() if parameter.grad is not None]
            count = sum(value.numel() for value in grads)
            norm = math.sqrt(sum(float(value.square().sum()) for value in grads))
            row[name] = {"l2": norm, "rms_per_parameter": norm / math.sqrt(max(count, 1))}
        branch_rows.append(row)

        parts = forward_parts(model, batch["history_state"], batch["node_mask"])
        error = torch.linalg.vector_norm(parts["prediction"] - batch["future_xy"], dim=-1)
        for horizon in HORIZONS:
            valid = batch["future_mask"][:, horizon - 1] & (batch["k"] >= 1)
            if not valid.any():
                continue
            objective = error[valid, horizon - 1].mean()
            gradients = torch.autograd.grad(objective,
                                            (parts["j2"], parts["j3"], parts["interaction"], parts["temporal"]),
                                            retain_graph=True, allow_unused=False)
            for name, gradient in zip(("j2", "j3", "interaction", "temporal"), gradients):
                horizon_rows[(horizon, name)].append(float(torch.linalg.vector_norm(gradient) / math.sqrt(gradient.numel())))
    aggregate = {}
    for name in branch_rows[0]:
        aggregate[name] = {
            key: {"mean": float(np.mean([row[name][key] for row in branch_rows])),
                  "cv": float(np.std([row[name][key] for row in branch_rows], ddof=1) /
                              max(np.mean([row[name][key] for row in branch_rows]), 1e-12))}
            for key in ("l2", "rms_per_parameter")
        }
    horizons = {f"{horizon / 10:g}s": {name: float(np.mean(horizon_rows[(horizon, name)]))
                                        for name in ("j2", "j3", "interaction", "temporal")}
                for horizon in HORIZONS}
    return {"loss_gradient_by_module": aggregate,
            "j3_to_j2_gradient_rms_ratio": (aggregate["j3"]["rms_per_parameter"]["mean"] /
                                             max(aggregate["j2"]["rms_per_parameter"]["mean"], 1e-12)),
            "ic_endpoint_error_sensitivity_rms": horizons}


def run_seed(seed: int, cfg: dict, loader: DataLoader, device: torch.device) -> dict:
    q_path = ROOT / f"reports/task_redesign/sind_gate_b_plus_robust_ema999_fw075_seed{seed}/best.pt"
    strong_graph_path = ROOT / f"reports/task_redesign/sind_gate_b_plus_robust_graph_seed{seed}/best.pt"
    all_graph_path = ROOT / f"reports/task_redesign/sind_gate_b_plus_robust_all_graph_seed{seed}/best.pt"
    q_state = torch.load(q_path, map_location=device, weights_only=True)
    q_model = GateBPlusQuantumModel("full_multiscale", dt_s=cfg["dt_s"], rounds=cfg["rounds"]).to(device)
    q_model.load_state_dict(q_state["model"]); q_model.eval()
    strong_graph_state = torch.load(strong_graph_path, map_location=device, weights_only=True)
    strong_graph = GateAModel("graph", 33, dt_s=cfg["dt_s"]).to(device)
    strong_graph.load_state_dict(strong_graph_state["model"]); strong_graph.eval()
    all_graph_state = torch.load(all_graph_path, map_location=device, weights_only=True)
    all_graph = GateAModel("all_graph", 33, dt_s=cfg["dt_s"]).to(device)
    all_graph.load_state_dict(all_graph_state["model"]); all_graph.eval()

    variants = ("full", "j2_only", "j3_only", "no_quantum", "j2_permuted", "j3_permuted")
    errors = {"ade": defaultdict(list), "fde": defaultdict(list)}
    latents = defaultdict(list)
    critical_flags = []
    with torch.no_grad():
        for batch in loader:
            history = batch["history_state"].to(device); mask = batch["node_mask"].to(device)
            parts = forward_parts(q_model, history, mask)
            for variant in variants:
                add_errors(errors, f"raj_{variant}", variant_prediction(q_model, parts, variant), batch)
            add_errors(errors, "strong_graph_n8", strong_graph(history, mask), batch)
            add_errors(errors, "all_graph_budget_ablation", all_graph(history, mask), batch)
            for name in ("j2", "j3", "quantum", "interaction", "temporal"):
                latents[name].append(parts[name].cpu().numpy())
            critical_flags.append((batch["k"] >= 1).numpy())
    latents = {name: np.concatenate(values) for name, values in latents.items()}
    critical = np.concatenate(critical_flags)
    representations = {}
    for scope, selection in (("full", np.ones(len(critical), dtype=bool)), ("ic", critical)):
        representations[scope] = {
            name: spectrum_stats(latents[name][selection])
            for name in ("j2", "j3", "quantum", "interaction", "temporal")
        }
        representations[scope]["j2_vs_j3"] = relation_stats(latents["j2"][selection], latents["j3"][selection])
        representations[scope]["quantum_vs_temporal_linear_cka"] = relation_stats(
            latents["quantum"][selection], latents["temporal"][selection]
        )["linear_cka"]
    return {"seed": seed, "checkpoint_epoch": q_state["epoch"], "checkpoint_score": q_state["score"],
            "horizon_and_ablation_metrics": summarize_errors(errors),
            "representations": representations,
            "branch_parameter_update": update_distance(seed, q_model.cpu(), cfg),
            "gradients": gradient_diagnostics(q_model.to(device), loader, device)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="reports/task_redesign/SIND_GATE_B_FULL_MULTISCALE_DIAGNOSTIC_20260921.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    cfg = json.loads((ROOT / "configs/gate_b_plus_screen.json").read_text())
    validation = GateADataset(ROOT / cfg["benchmark"], "val")
    loader = DataLoader(validation, batch_size=32, shuffle=False, num_workers=2)
    device = torch.device("cuda:0")
    result = {"status": "DEVELOPMENT_MECHANISM_DIAGNOSTIC_NOT_FORMAL_EVIDENCE",
              "test_accessed": False, "model": "Full-Multiscale Raj j2||j3",
              "seeds": {str(seed): run_seed(seed, cfg, loader, device) for seed in (2026, 2027)}}
    atomic_json(ROOT / args.output, result)
    print(json.dumps({"status": result["status"],
                      "seeds": {seed: {"checkpoint_epoch": row["checkpoint_epoch"],
                                       "checkpoint_score": row["checkpoint_score"]}
                                for seed, row in result["seeds"].items()}}, indent=2))


if __name__ == "__main__":
    main()
