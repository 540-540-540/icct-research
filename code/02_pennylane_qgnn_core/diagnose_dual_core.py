"""Layer-wise representation diagnostics for matched trained two-core models."""
import json
import math
from pathlib import Path

import numpy as np
import torch

import experiment_utils as r
import run as retained
from dual_core_model import build_dual_graph


ROOT = Path("/home/js_cn/sensing")
BASE = Path(__file__).resolve().parent
RESULTS = BASE / "dual_core_seed2026_v1"
SEED = 2026


def distribution(values):
    values = values.double()
    return {name: float(function(values)) for name, function in (
        ("mean", torch.mean), ("std", torch.std), ("min", torch.min), ("max", torch.max)
    )}


def representation_stats(values):
    values = values.double()
    centered = values - values.mean(0, keepdim=True)
    covariance = centered.T @ centered / max(1, len(centered) - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    total = eigenvalues.sum()
    effective_rank = total.square() / eigenvalues.square().sum().clamp_min(1e-30)
    standard = centered.std(0).clamp_min(1e-12)
    correlation = covariance / standard[:, None] / standard[None, :]
    mask = ~torch.eye(correlation.shape[0], dtype=torch.bool)
    return {
        "samples": int(len(values)),
        "dimension": int(values.shape[1]),
        "effective_rank_participation": float(effective_rank),
        "dimension_std_min": float(values.std(0).min()),
        "dimension_std_median": float(values.std(0).median()),
        "dimension_std_max": float(values.std(0).max()),
        "mean_abs_off_diagonal_correlation": float(correlation[mask].abs().mean()),
        "fraction_abs_gt_0_95": float((values.abs() > 0.95).double().mean()),
    }


@torch.no_grad()
def inspect_arm(arm, bank):
    kind = arm[:-5]
    r.set_seed(SEED)
    model = build_dual_graph(arm).cuda().eval()
    initial = [
        {name: value.detach().cpu().clone() for name, value in layer.core.named_parameters()}
        for layer in model.graph_layers
    ]
    checkpoint = torch.load(RESULTS / f"{arm}_graph_selected.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state"], strict=True)
    captures = [
        {"angles": [], "raw": [], "normalized": [], "score": [], "gate_logit": []}
        for _ in model.graph_layers
    ]
    handles = []
    for index, layer in enumerate(model.graph_layers):
        bucket = captures[index]
        handles.extend([
            layer.core.register_forward_hook(
                lambda module, inputs, output, target=bucket: target["angles"].append(inputs[0].detach().cpu())
            ),
            layer.core.register_forward_hook(
                lambda module, inputs, output, target=bucket: target["raw"].append(output.detach().cpu())
            ),
            layer.latent_norm.register_forward_hook(
                lambda module, inputs, output, target=bucket: target["normalized"].append(output.detach().cpu())
            ),
            layer.score.register_forward_hook(
                lambda module, inputs, output, target=bucket: target["score"].append(output.detach().cpu())
            ),
            layer.gate.register_forward_hook(
                lambda module, inputs, output, target=bucket: target["gate_logit"].append(output.detach().cpu())
            ),
        ])
    for snr in (5, 10, 15, 20):
        generator = torch.Generator(device="cuda").manual_seed(SEED + 100000)
        for start in range(0, len(bank["history"]), 24):
            rows = torch.arange(start, min(start + 24, len(bank["history"])), device="cuda")
            history, _, mask = retained.make_batch(bank, rows, snr, generator)
            model(history, mask)
    for handle in handles:
        handle.remove()
    layers = []
    for index, (layer, bucket) in enumerate(zip(model.graph_layers, captures)):
        tensors = {key: torch.cat(value) for key, value in bucket.items()}
        angles = tensors["angles"]
        gates = 2.0 * torch.sigmoid(tensors["gate_logit"])
        parameter_change = {}
        for name, value in layer.core.named_parameters():
            delta = value.detach().cpu() - initial[index][name]
            parameter_change[name] = {"max_abs": float(delta.abs().max()), "l2": float(delta.double().norm())}
        layers.append({
            "layer": index + 1,
            "active_edge_evaluations": int(len(angles)),
            "angles": {
                **distribution(angles),
                "fraction_abs_gt_0_9pi": float((angles.abs() > 0.9 * math.pi).double().mean()),
                "fraction_abs_gt_0_99pi": float((angles.abs() > 0.99 * math.pi).double().mean()),
            },
            "raw_core_output": representation_stats(tensors["raw"]),
            "normalized_core_output": representation_stats(tensors["normalized"]),
            "score_logits": distribution(tensors["score"]),
            "message_gates": {
                **distribution(gates),
                "fraction_lt_0_2_or_gt_1_8": float(((gates < 0.2) | (gates > 1.8)).double().mean()),
            },
            "core_parameter_change_from_initialization": parameter_change,
        })
    return {"selected_epoch": int(checkpoint["epoch"]), "layers": layers}


def main():
    with np.load(BASE / "retained_inputs/circuit_split_indices.npz", allow_pickle=False) as split:
        dev_indices = split["circuit_dev_indices"].copy()
    selection_indices = r.select_profile_indices(dev_indices, 128)
    with np.load(ROOT / "data/multitarget_lankershim_v1.npz", allow_pickle=False) as data:
        states = data["train_states"][selection_indices].copy()
        masks = data["train_mask"][selection_indices].copy()
    bank = {
        "history": torch.from_numpy(states[:, :20]).float().cuda(),
        "future": torch.from_numpy(states[:, 20:]).float().cuda(),
        "mask": torch.from_numpy(masks).bool().cuda(),
    }
    retained.NOISE = r.load_snr_noise_map(ROOT / "results/multitarget_snr/snr_calibration.json")
    report = {
        "seed": SEED,
        "selection_scenes": int(len(selection_indices)),
        "snrs": [5, 10, 15, 20],
        "classical_dual": inspect_arm("classical_dual", bank),
        "quantum_dual": inspect_arm("quantum_dual", bank),
        "no_test": True,
    }
    output = RESULTS / "layerwise_mechanism_diagnostics.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
