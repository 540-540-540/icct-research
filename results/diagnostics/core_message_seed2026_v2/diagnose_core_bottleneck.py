"""Diagnose why the trained PennyLane core does not exceed the matched MLP core."""
import json
import math
from pathlib import Path

import numpy as np
import torch

import experiment_utils as r
import run as retained
from model import build_graph


ROOT = Path("/home/js_cn/sensing")
BASE = Path(__file__).resolve().parent
RESULTS = BASE / "converged_protocol_seed2026_v1"
SEED = 2026


def distribution(values):
    values = values.double()
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def representation_stats(values):
    values = values.double()
    centered = values - values.mean(0, keepdim=True)
    covariance = centered.T @ centered / max(1, len(centered) - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    total = eigenvalues.sum()
    effective_rank = total.square() / eigenvalues.square().sum().clamp_min(1e-30)
    standard = centered.std(0).clamp_min(1e-12)
    correlation = (centered.T @ centered) / max(1, len(centered) - 1)
    correlation = correlation / standard[:, None] / standard[None, :]
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
def inspect_arm(arm, bank, noise):
    backend = "pennylane" if arm == "quantum" else "torch"
    r.set_seed(SEED)
    model = build_graph(arm, quantum_backend=backend).cuda().eval()
    layer = model.graph_layers[1]
    initial_core = {name: value.detach().cpu().clone() for name, value in layer.core.named_parameters()}
    checkpoint = torch.load(RESULTS / f"{arm}_graph_selected.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state"], strict=True)

    captured = {"angles": [], "raw": [], "normalized": [], "score": [], "gate_logit": [], "attention_entropy": []}
    def capture_core(module, inputs, output):
        captured["angles"].append(inputs[0].detach().cpu())
        captured["raw"].append(output.detach().cpu())

    handles = [
        layer.core.register_forward_hook(capture_core),
        layer.latent_norm.register_forward_hook(
            lambda module, inputs, output: captured["normalized"].append(output.detach().cpu())
        ),
        layer.score.register_forward_hook(
            lambda module, inputs, output: captured["score"].append(output.detach().cpu())
        ),
        layer.gate.register_forward_hook(
            lambda module, inputs, output: captured["gate_logit"].append(output.detach().cpu())
        ),
    ]

    for snr in (5, 10, 15, 20):
        generator = torch.Generator(device="cuda").manual_seed(SEED + 100000)
        for start in range(0, len(bank["history"]), 24):
            indices = torch.arange(start, min(start + 24, len(bank["history"])), device="cuda")
            history, _, mask = retained.make_batch(bank, indices, snr, generator)
            output = model(history, mask)
            attention = output["attention"].double()
            adjacency = output["adjacency"]
            degree = adjacency.sum(2)
            entropy = -(attention.clamp_min(1e-12).log() * attention * adjacency).sum(2)
            valid = degree > 1
            normalized_entropy = entropy[valid] / degree[valid].double().log()
            captured["attention_entropy"].append(normalized_entropy.cpu())
    for handle in handles:
        handle.remove()

    tensors = {key: torch.cat(value) for key, value in captured.items()}
    angles = tensors["angles"]
    gates = 2.0 * torch.sigmoid(tensors["gate_logit"])
    parameter_change = {}
    for name, value in layer.core.named_parameters():
        delta = value.detach().cpu() - initial_core[name]
        parameter_change[name] = {
            "max_abs": float(delta.abs().max()),
            "l2": float(delta.double().norm()),
        }
    return {
        "selected_epoch": int(checkpoint["epoch"]),
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
        "attention_normalized_entropy": distribution(tensors["attention_entropy"]),
        "core_parameter_change_from_initialization": parameter_change,
    }


def main():
    with np.load(BASE / "retained_inputs/circuit_split_indices.npz", allow_pickle=False) as split:
        dev_indices = split["circuit_dev_indices"].copy()
    selection_indices = r.select_profile_indices(dev_indices, 256)
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
        "classical": inspect_arm("classical", bank, retained.NOISE),
        "quantum": inspect_arm("quantum", bank, retained.NOISE),
    }
    output = RESULTS / "core_bottleneck_diagnostics.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
