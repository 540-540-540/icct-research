from __future__ import annotations

import json
import math
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
from experiments.gate_b_plus.diagnose_full_multiscale import (
    add_errors,
    relation_stats,
    spectrum_stats,
    summarize_errors,
)
from experiments.gate_b_plus.models import GateBPlusQuantumModel
from experiments.gate_b_plus.run_screen import screen_loss


def atomic_json(path: Path, payload: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)


def forward_parts(model: GateBPlusQuantumModel, history: torch.Tensor, node_mask: torch.Tensor,
                  keep_j2: bool = True, keep_j3: bool = True) -> dict[str, torch.Tensor]:
    history, node_mask = history[:, :, :8], node_mask[:, :8]
    history = torch.where(node_mask[:, None, :, None], history, 0.0)
    j2 = model.core.j2(history, node_mask)[:, 0]
    j3 = model.core.j3(history, node_mask)[:, 0]
    used_j2 = j2 if keep_j2 else torch.zeros_like(j2)
    used_j3 = j3 if keep_j3 else torch.zeros_like(j3)
    interaction = model.interaction(torch.cat((used_j2, used_j3), -1))
    q2 = model.branch_projections[0](model.branch_norms[0](used_j2)) if keep_j2 else interaction.new_zeros(interaction.shape)
    q3 = model.branch_projections[1](model.branch_norms[1](used_j3)) if keep_j3 else interaction.new_zeros(interaction.shape)
    temporal = model.encoder(history)[:, 0]
    steps = torch.arange(40, device=history.device)
    times = (steps.to(history.dtype) + 1) * model.dt_s
    cv = history[:, -1, 0, 2:4][:, None] * times[None, :, None]
    time_embedding = model.time(steps)
    gates = torch.softmax(model.horizon_gate(time_embedding), -1)
    residual = (gates[None, :, :1] * model.branch_scales[0] * q2[:, None]
                + gates[None, :, 1:] * model.branch_scales[1] * q3[:, None])
    target = torch.cat((temporal[:, None].expand(-1, 40, -1), interaction[:, None] + residual), -1)
    decoded = torch.cat((target, time_embedding[None].expand(len(history), -1, -1), cv / 20.0), -1)
    prediction = cv + model.decoder(decoded) * 10.0
    return {"history": history, "j2": j2, "j3": j3, "interaction": interaction,
            "q2": q2, "q3": q3, "temporal": temporal, "gates": gates, "prediction": prediction}


def gradient_stats(model: GateBPlusQuantumModel, loader: DataLoader, device: torch.device) -> dict:
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()
    rows = []
    for batch_no, batch in enumerate(loader):
        if batch_no == 4:
            break
        batch = {key: value.to(device) for key, value in batch.items()}
        model.zero_grad(set_to_none=True)
        prediction = model(batch["history_state"], batch["node_mask"])
        screen_loss(prediction, batch, 0.75).backward()
        row = {}
        modules = (("j2", model.core.j2), ("j3", model.core.j3),
                   ("branch2_projection", model.branch_projections[0]),
                   ("branch3_projection", model.branch_projections[1]),
                   ("horizon_gate", model.horizon_gate), ("decoder", model.decoder))
        for name, module in modules:
            gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
            count = sum(value.numel() for value in gradients)
            norm = math.sqrt(sum(float(value.square().sum()) for value in gradients))
            row[name] = norm / math.sqrt(max(count, 1))
        rows.append(row)
    means = {name: float(np.mean([row[name] for row in rows])) for name in rows[0]}
    means["j3_to_j2_rms_ratio"] = means["j3"] / max(means["j2"], 1e-12)
    means["branch3_to_branch2_projection_rms_ratio"] = (
        means["branch3_projection"] / max(means["branch2_projection"], 1e-12)
    )
    return means


def load_quantum(seed: int, device: torch.device) -> GateBPlusQuantumModel:
    path = ROOT / f"reports/task_redesign/sind_gate_b_plus_horizon_multiscale_fw075_seed{seed}/best.pt"
    state = torch.load(path, map_location=device, weights_only=True)
    model = GateBPlusQuantumModel("horizon_multiscale").to(device)
    model.load_state_dict(state["model"])
    return model


def run_seed(seed: int, loader: DataLoader, device: torch.device) -> dict:
    raj = load_quantum(seed, device).eval()
    graph_state = torch.load(ROOT / f"reports/task_redesign/sind_gate_b_plus_primary_strong_graph_seed{seed}/best.pt",
                             map_location=device, weights_only=True)
    graph = GateAModel("graph", 33).to(device)
    graph.load_state_dict(graph_state["model"]); graph.eval()
    controls = {}
    for name, core in (("matched_johnson", "johnson"), ("large_johnson", "johnson_large")):
        state = torch.load(ROOT / f"reports/task_redesign/sind_gate_b_plus_primary_{name}_seed{seed}/best.pt",
                           map_location=device, weights_only=True)
        model = GateBPlusQuantumModel("horizon_multiscale", core_kind=core).to(device)
        model.load_state_dict(state["model"]); model.eval(); controls[name] = model

    errors = {"ade": defaultdict(list), "fde": defaultdict(list)}
    latents = defaultdict(list)
    critical = []
    with torch.no_grad():
        for batch in loader:
            history, mask = batch["history_state"].to(device), batch["node_mask"].to(device)
            full = forward_parts(raj, history, mask)
            add_errors(errors, "raj_full", full["prediction"], batch)
            add_errors(errors, "raj_j2_only", forward_parts(raj, history, mask, keep_j3=False)["prediction"], batch)
            add_errors(errors, "raj_j3_only", forward_parts(raj, history, mask, keep_j2=False)["prediction"], batch)
            add_errors(errors, "raj_no_quantum", forward_parts(raj, history, mask, False, False)["prediction"], batch)
            add_errors(errors, "strong_graph_n8", graph(history, mask), batch)
            for name, model in controls.items():
                add_errors(errors, name, model(history, mask), batch)
            for name in ("j2", "j3", "q2", "q3", "interaction", "temporal"):
                latents[name].append(full[name].cpu().numpy())
            critical.append((batch["k"] >= 1).numpy())
    latents = {name: np.concatenate(values) for name, values in latents.items()}
    selection = np.concatenate(critical)
    representations = {name: spectrum_stats(value[selection]) for name, value in latents.items()}
    representations["j2_vs_j3"] = relation_stats(latents["j2"][selection], latents["j3"][selection])
    representations["q2_vs_q3"] = relation_stats(latents["q2"][selection], latents["q3"][selection])
    steps = torch.arange(40, device=device)
    with torch.no_grad():
        gates = torch.softmax(raj.horizon_gate(raj.time(steps)), -1)
    gate_rows = {f"{step / 10:g}s": [float(value) for value in gates[step - 1]] for step in range(5, 41, 5)}
    scales = [float(value) for value in raj.branch_scales.detach().cpu()]
    return {"horizon_metrics": summarize_errors(errors), "representations_ic": representations,
            "gate_by_horizon_j2_j3": gate_rows, "branch_scales_j2_j3": scales,
            "gradients": gradient_stats(raj, loader, device)}


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    cfg = json.loads((ROOT / "configs/gate_b_plus_screen.json").read_text())
    validation = GateADataset(ROOT / cfg["benchmark"], "val")
    loader = DataLoader(validation, batch_size=32, shuffle=False, num_workers=2)
    device = torch.device("cuda:0")
    result = {"status": "DEVELOPMENT_MECHANISM_DIAGNOSTIC_NOT_FORMAL_EVIDENCE",
              "test_accessed": False,
              "primary_baselines": ["strong_graph_n8", "matched_johnson", "large_johnson"],
              "all_graph_role": "neighborhood_budget_ablation_only_not_a_primary_hurdle",
              "seeds": {str(seed): run_seed(seed, loader, device) for seed in (2026, 2027)}}
    output = ROOT / "reports/task_redesign/SIND_GATE_B_HORIZON_MULTISCALE_DIAGNOSTIC_20260921.json"
    atomic_json(output, result)
    print(json.dumps({"output": str(output), "seeds": list(result["seeds"])}, indent=2))


if __name__ == "__main__":
    main()
