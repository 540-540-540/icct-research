"""Matched single-seed test of classical versus quantum value modulation."""
import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch

import experiment_utils as r
import run as retained
from model import build_graph
from run_converged_training import train_stage
from run_multitarget_experiment import prediction_loss
from value_modulated_model import build_value_graph


ROOT = Path("/home/js_cn/sensing")
BASE = Path(__file__).resolve().parent
SEED = 2026
ARMS = ("classical_value", "quantum_value")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def matching_tensor_hash(model, excluded=("graph_layers.1.core.", "graph_layers.1.value_channel.")):
    return r.tensor_mapping_sha256(
        {
            name: value.detach().cpu()
            for name, value in model.named_parameters()
            if not any(name.startswith(prefix) for prefix in excluded)
        }
    )


def banks():
    cache = ROOT / "data/multitarget_lankershim_v1.npz"
    with np.load(BASE / "retained_inputs/circuit_split_indices.npz", allow_pickle=False) as split:
        train_indices = split["circuit_train_indices"].copy()
        dev_indices = split["circuit_dev_indices"].copy()
    with np.load(cache, allow_pickle=False) as data:
        states = data["train_states"].copy()
        masks = data["train_mask"].copy()
    selection_indices = r.select_profile_indices(dev_indices, 256)

    def bank(indices):
        return {
            "history": torch.from_numpy(states[indices, :20]).float().cuda(),
            "future": torch.from_numpy(states[indices, 20:]).float().cuda(),
            "mask": torch.from_numpy(masks[indices]).bool().cuda(),
        }

    return bank(train_indices), bank(selection_indices), bank(dev_indices), train_indices, selection_indices, dev_indices


def smoke(output_dir, train):
    evidence = {}
    common_hash = None
    for arm in ARMS:
        r.set_seed(SEED)
        model = build_value_graph(arm).cuda()
        current_hash = matching_tensor_hash(model)
        if common_hash is None:
            common_hash = current_hash
        assert current_hash == common_hash

        # The zero initialized channel path must preserve the retained arm's
        # epoch-zero prediction exactly.
        baseline = build_graph(
            arm[:-6], quantum_backend="pennylane" if arm.startswith("quantum") else "torch"
        ).cuda().eval()
        model.eval()
        history = train["history"][:2]
        future = train["future"][:2]
        mask = train["mask"][:2]
        with torch.no_grad():
            old_output = baseline(history, mask)["future_position"]
            new_output = model(history, mask)["future_position"]
        max_initial_difference = float((old_output - new_output).abs().max())
        assert max_initial_difference < 1e-6

        model.train()
        output = model(history, mask)
        loss = prediction_loss(output, future, mask, graph_weighting=True)
        loss.backward()
        layer = model.graph_layers[1]
        channel_gradient = float(layer.value_channel.weight.grad.abs().max())
        core_gradient = max(
            float(parameter.grad.abs().max())
            for parameter in layer.core.parameters()
            if parameter.grad is not None
        )
        assert channel_gradient > 0 and core_gradient > 0
        evidence[arm] = {
            "loss": float(loss.detach()),
            "max_epoch_zero_difference_from_retained_arm": max_initial_difference,
            "value_channel_gradient_max": channel_gradient,
            "core_gradient_max": core_gradient,
            "parameters": sum(p.numel() for p in model.parameters()),
        }
        del baseline, model
        torch.cuda.empty_cache()
    r.atomic_json(output_dir / "smoke.json", evidence)
    return evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-name", default="value_modulation_seed2026_v1")
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    assert Path.cwd() == ROOT
    assert platform.node() == "jscn"
    assert torch.cuda.is_available() and "4090" in torch.cuda.get_device_name(0)
    torch.set_num_threads(4)
    r.set_seed(SEED)

    output_dir = BASE / args.output_name
    output_dir.mkdir(exist_ok=False)
    train, selection, full_dev, train_indices, selection_indices, dev_indices = banks()
    retained.NOISE = r.load_snr_noise_map(ROOT / "results/multitarget_snr/snr_calibration.json")
    evidence = smoke(output_dir, train)
    if args.smoke_only:
        r.atomic_json(output_dir / "completed.json", {"status": "smoke_completed", "evidence": evidence})
        print(json.dumps(evidence, ensure_ascii=False), flush=True)
        return

    source_paths = [
        BASE / "value_modulated_model.py",
        BASE / "run_value_modulation_experiment.py",
        BASE / "model.py",
        BASE / "run_converged_training.py",
        BASE / "pennylane_core.py",
        BASE / "quantum_core.py",
        ROOT / "target_interaction_graph.py",
    ]
    source_hashes = {str(path): sha256(path) for path in source_paths}
    protocol = {
        "seed": SEED,
        "arms": list(ARMS),
        "intervention": "pairwise channel-wise value modulation: 2*sigmoid(W_value*z)",
        "value_channel_initialization": "zeros; exact neutral factor 1 at epoch zero",
        "quantum_backend": "PennyLane default.qubit analytic statevector",
        "graph": {"max_epochs": 40, "patience": 6, "min_delta": 1e-4},
        "batch_size": 24,
        "selection": "four-SNR macro ADE + 0.35 FDE on fixed 256-scene selection split",
        "train_indices": train_indices.tolist(),
        "selection_indices": selection_indices.tolist(),
        "full_dev_indices": dev_indices.tolist(),
        "same_input_hashes_across_arms_required": True,
        "matched_noncore_initialization_required": True,
        "no_llm": True,
        "no_test": True,
        "gpu": torch.cuda.get_device_name(0),
        "python": sys.executable,
        "source_hashes": source_hashes,
    }
    r.atomic_json(output_dir / "protocol.json", protocol)

    input_hashes = {}
    summaries = {}
    common_hash = None
    for arm in ARMS:
        r.set_seed(SEED)
        model = build_value_graph(arm).cuda()
        current_hash = matching_tensor_hash(model)
        if common_hash is None:
            common_hash = current_hash
        assert current_hash == common_hash
        audit = train_stage(model, arm, "graph", 40, 6, train, selection, output_dir, input_hashes)
        full_metrics, arrays = retained.evaluate(model, full_dev, collect=True)
        np.savez_compressed(output_dir / f"{arm}_graph_full_dev.npz", **arrays, indices=dev_indices)
        summaries[arm] = {
            "graph": full_metrics,
            "selected_epoch": audit["selected_epoch"],
            "executed_epochs": audit["executed_epochs"],
            "stop_reason": audit["stop_reason"],
            "parameters": sum(p.numel() for p in model.parameters()),
        }
        r.atomic_json(output_dir / "summary.json", summaries)
        del model
        torch.cuda.empty_cache()

    for path, digest in source_hashes.items():
        assert sha256(path) == digest
    completed = {
        "status": "completed",
        "seed": SEED,
        "summaries": summaries,
        "source_hashes_verified": True,
        "same_input_hashes_across_arms_verified": True,
        "matched_noncore_initialization_verified": True,
        "no_llm": True,
        "no_test": True,
    }
    r.atomic_json(output_dir / "completed.json", completed)
    r.atomic_json(output_dir / "progress.json", completed)
    print(json.dumps(completed, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
