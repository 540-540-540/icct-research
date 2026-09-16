"""Paired error and representation diagnostics for the 0.5-pi experiment."""
import json
import math
from pathlib import Path

import numpy as np
import torch

import experiment_utils as r
import run as retained
from analyze_value_modulation import load_arrays, paired_bootstrap
from angle_scaled_model import build_angle_scaled_graph
from diagnose_core_bottleneck import distribution, representation_stats


ROOT = Path("/home/js_cn/sensing")
BASE = Path(__file__).resolve().parent
NEW = BASE / "angle_scale_0p5_seed2026_v1"
OLD = BASE / "converged_protocol_seed2026_v1"
ARM = "quantum_angle_0p5"
SEED = 2026


@torch.no_grad()
def inspect_representation(bank):
    r.set_seed(SEED)
    model = build_angle_scaled_graph().cuda().eval()
    checkpoint = torch.load(NEW / f"{ARM}_graph_selected.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state"], strict=True)
    layer = model.graph_layers[1]
    captured = {"angles": [], "raw": [], "normalized": [], "score": [], "gate": [], "entropy": []}

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
            lambda module, inputs, output: captured["gate"].append(output.detach().cpu())
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
            captured["entropy"].append((entropy[valid] / degree[valid].double().log()).cpu())
    for handle in handles:
        handle.remove()
    tensors = {name: torch.cat(values) for name, values in captured.items()}
    angles = tensors["angles"]
    gates = 2.0 * torch.sigmoid(tensors["gate"])
    return {
        "selected_epoch": int(checkpoint["epoch"]),
        "active_edge_evaluations": int(len(angles)),
        "angles": {
            **distribution(angles),
            "fraction_abs_gt_0_45pi": float((angles.abs() > 0.45 * math.pi).double().mean()),
            "fraction_abs_gt_0_49pi": float((angles.abs() > 0.49 * math.pi).double().mean()),
            "fraction_abs_gt_0_9pi": float((angles.abs() > 0.9 * math.pi).double().mean()),
        },
        "raw_core_output": representation_stats(tensors["raw"]),
        "normalized_core_output": representation_stats(tensors["normalized"]),
        "score_logits": distribution(tensors["score"]),
        "message_gates": {
            **distribution(gates),
            "fraction_lt_0_2_or_gt_1_8": float(((gates < 0.2) | (gates > 1.8)).double().mean()),
        },
        "attention_normalized_entropy": distribution(tensors["entropy"]),
    }


def main():
    indices, scaled = load_arrays(NEW, ARM)
    old_indices, old_quantum = load_arrays(OLD, "quantum")
    classical_indices, old_classical = load_arrays(OLD, "classical")
    assert np.array_equal(indices, old_indices) and np.array_equal(indices, classical_indices)
    with np.load(ROOT / "data/multitarget_lankershim_v1.npz", allow_pickle=False) as data:
        blocks = (data["train_start_index"][indices] // 200).astype(np.int64)
        states = data["train_states"].copy()
        masks = data["train_mask"].copy()
    with np.load(BASE / "retained_inputs/circuit_split_indices.npz", allow_pickle=False) as split:
        dev_indices = split["circuit_dev_indices"].copy()
    selection_indices = r.select_profile_indices(dev_indices, 256)
    bank = {
        "history": torch.from_numpy(states[selection_indices, :20]).float().cuda(),
        "future": torch.from_numpy(states[selection_indices, 20:]).float().cuda(),
        "mask": torch.from_numpy(masks[selection_indices]).bool().cuda(),
    }
    retained.NOISE = r.load_snr_noise_map(ROOT / "results/multitarget_snr/snr_calibration.json")
    old_diagnostics = json.loads((OLD / "core_bottleneck_diagnostics.json").read_text(encoding="utf-8"))
    report = {
        "seed": SEED,
        "scenes": int(len(indices)),
        "unique_time_blocks": int(len(np.unique(blocks))),
        "bootstrap_repetitions": 10000,
        "warning": "Intervals condition on one training seed and measure paired scene/time-block variation only.",
        "comparisons": {
            "angle_0p5_vs_retained_quantum": paired_bootstrap(old_quantum, scaled, blocks),
            "angle_0p5_vs_retained_classical": paired_bootstrap(old_classical, scaled, blocks),
        },
        "representation": {
            "angle_0p5": inspect_representation(bank),
            "retained_quantum": old_diagnostics["quantum"],
        },
        "no_llm": True,
        "no_test": True,
    }
    r.atomic_json(NEW / "analysis.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
