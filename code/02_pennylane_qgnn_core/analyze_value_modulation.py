"""Paired uncertainty and mechanism checks for the value-modulation experiment."""
import json
from pathlib import Path

import numpy as np
import torch

import experiment_utils as r
import run as retained
from value_modulated_model import build_value_graph


ROOT = Path("/home/js_cn/sensing")
BASE = Path(__file__).resolve().parent
NEW = BASE / "value_modulation_seed2026_v1"
OLD = BASE / "converged_protocol_seed2026_v1"
SEED = 2026
SNRS = (5, 10, 15, 20)


def distribution(values):
    values = values.double()
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "fraction_lt_0_8_or_gt_1_2": float(((values < 0.8) | (values > 1.2)).double().mean()),
    }


def load_arrays(directory, arm):
    with np.load(directory / f"{arm}_graph_full_dev.npz") as payload:
        return payload["indices"].copy(), {snr: payload[str(snr)].copy() for snr in SNRS}


def macro_metrics(arrays, rows):
    per_snr = []
    for snr in SNRS:
        values = arrays[snr][rows]
        denominator = values[:, 2].sum()
        per_snr.append([values[:, 0].sum() / (20 * denominator), values[:, 1].sum() / denominator])
    return np.asarray(per_snr).mean(0)


def paired_bootstrap(reference, candidate, blocks, repetitions=10000):
    point_ref = macro_metrics(reference, np.arange(len(blocks)))
    point_candidate = macro_metrics(candidate, np.arange(len(blocks)))
    point = 100 * (point_ref - point_candidate) / point_ref
    unique = np.unique(blocks)
    rng = np.random.default_rng(20260907)
    draws = []
    for _ in range(repetitions):
        chosen = rng.choice(unique, len(unique), replace=True)
        rows = np.concatenate([np.flatnonzero(blocks == block) for block in chosen])
        ref = macro_metrics(reference, rows)
        cand = macro_metrics(candidate, rows)
        draws.append(100 * (ref - cand) / ref)
    draws = np.asarray(draws)
    return {
        "positive_means_candidate_better": True,
        "reference_metrics": {"ade_m": float(point_ref[0]), "fde_m": float(point_ref[1])},
        "candidate_metrics": {"ade_m": float(point_candidate[0]), "fde_m": float(point_candidate[1])},
        "reduction_percent": {"ade": float(point[0]), "fde": float(point[1])},
        "block_bootstrap_95_percentile_ci": {
            "ade": np.percentile(draws[:, 0], [2.5, 97.5]).tolist(),
            "fde": np.percentile(draws[:, 1], [2.5, 97.5]).tolist(),
        },
    }


@torch.no_grad()
def inspect_channel(arm, bank):
    r.set_seed(SEED)
    model = build_value_graph(arm).cuda().eval()
    checkpoint = torch.load(NEW / f"{arm}_graph_selected.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state"], strict=True)
    layer = model.graph_layers[1]
    logits = []
    handle = layer.value_channel.register_forward_hook(
        lambda module, inputs, output: logits.append(output.detach().cpu())
    )
    trained, _ = retained.evaluate(model, bank)
    handle.remove()
    factors = 2.0 * torch.sigmoid(torch.cat(logits))

    saved = layer.value_channel.weight.detach().clone()
    layer.value_channel.weight.zero_()
    neutral, _ = retained.evaluate(model, bank)
    layer.value_channel.weight.copy_(saved)
    trained_aggregate = trained["aggregate"]
    neutral_aggregate = neutral["aggregate"]
    return {
        "selected_epoch": int(checkpoint["epoch"]),
        "value_channel_weight_l2": float(saved.double().norm()),
        "value_channel_weight_max_abs": float(saved.abs().max()),
        "channel_factors": distribution(factors),
        "full_dev_trained": trained,
        "full_dev_neutralized_at_inference": neutral,
        "trained_reduction_vs_neutralized_percent": {
            metric: 100 * (neutral_aggregate[metric] - trained_aggregate[metric]) / neutral_aggregate[metric]
            for metric in ("ade_m", "fde_m")
        },
    }


def main():
    indices, classical_value = load_arrays(NEW, "classical_value")
    q_indices, quantum_value = load_arrays(NEW, "quantum_value")
    old_indices, classical_old = load_arrays(OLD, "classical")
    oq_indices, quantum_old = load_arrays(OLD, "quantum")
    assert np.array_equal(indices, q_indices) and np.array_equal(indices, old_indices)
    assert np.array_equal(indices, oq_indices)
    with np.load(ROOT / "data/multitarget_lankershim_v1.npz", allow_pickle=False) as data:
        starts = data["train_start_index"][indices]
        states = data["train_states"][indices].copy()
        masks = data["train_mask"][indices].copy()
    blocks = (starts // 200).astype(np.int64)
    bank = {
        "history": torch.from_numpy(states[:, :20]).float().cuda(),
        "future": torch.from_numpy(states[:, 20:]).float().cuda(),
        "mask": torch.from_numpy(masks).bool().cuda(),
    }
    retained.NOISE = r.load_snr_noise_map(ROOT / "results/multitarget_snr/snr_calibration.json")
    report = {
        "seed": SEED,
        "scenes": int(len(indices)),
        "unique_time_blocks": int(len(np.unique(blocks))),
        "bootstrap_repetitions": 10000,
        "warning": "Intervals condition on one training seed and measure paired scene/time-block variation only.",
        "comparisons": {
            "quantum_value_vs_classical_value": paired_bootstrap(classical_value, quantum_value, blocks),
            "quantum_value_vs_retained_quantum": paired_bootstrap(quantum_old, quantum_value, blocks),
            "classical_value_vs_retained_classical": paired_bootstrap(classical_old, classical_value, blocks),
        },
        "mechanism": {
            "classical_value": inspect_channel("classical_value", bank),
            "quantum_value": inspect_channel("quantum_value", bank),
        },
        "no_llm": True,
        "no_test": True,
    }
    r.atomic_json(NEW / "analysis.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
