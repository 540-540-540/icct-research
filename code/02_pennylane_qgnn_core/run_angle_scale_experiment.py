"""From-scratch single-seed test of a fixed 0.5-pi quantum encoding range."""
import argparse
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import torch

import experiment_utils as r
import run as retained
from angle_scaled_model import build_angle_scaled_graph
from model import build_graph
from run_converged_training import train_stage
from run_multitarget_experiment import prediction_loss


ROOT = Path("/home/js_cn/sensing")
BASE = Path(__file__).resolve().parent
OLD = BASE / "converged_protocol_seed2026_v1"
SEED = 2026
ARM = "quantum_angle_0p5"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_hash(model):
    return r.tensor_mapping_sha256({name: value.detach().cpu() for name, value in model.named_parameters()})


def make_banks():
    with np.load(BASE / "retained_inputs/circuit_split_indices.npz", allow_pickle=False) as split:
        train_indices = split["circuit_train_indices"].copy()
        dev_indices = split["circuit_dev_indices"].copy()
    with np.load(ROOT / "data/multitarget_lankershim_v1.npz", allow_pickle=False) as data:
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


def expected_input_hashes():
    expected = {}
    with (OLD / "quantum_graph.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            expected[f"graph_{record['epoch']}"] = record["input_sha256"]
    assert len(expected) == 40
    return expected


def smoke(output_dir, train):
    r.set_seed(SEED)
    baseline = build_graph("quantum", quantum_backend="pennylane").cuda().eval()
    r.set_seed(SEED)
    scaled = build_angle_scaled_graph().cuda().eval()
    assert state_hash(baseline) == state_hash(scaled)
    history = train["history"][:2]
    future = train["future"][:2]
    mask = train["mask"][:2]
    captured = []
    handle = scaled.graph_layers[1].core.register_forward_pre_hook(
        lambda module, inputs: captured.append(inputs[0].detach().cpu())
    )
    baseline_output = baseline(history, mask)["future_position"]
    scaled_output = scaled(history, mask)["future_position"]
    handle.remove()
    assert captured and float(torch.cat(captured).abs().max()) <= 0.5 * math.pi + 1e-6
    assert float((baseline_output - scaled_output).abs().max()) > 0
    scaled.train()
    loss = prediction_loss(scaled(history, mask), future, mask, graph_weighting=True)
    loss.backward()
    layer = scaled.graph_layers[1]
    evidence = {
        "initial_parameter_hash_matches_retained_quantum": True,
        "max_encoded_angle": float(torch.cat(captured).abs().max()),
        "max_initial_prediction_difference": float((baseline_output - scaled_output).abs().max()),
        "loss": float(loss.detach()),
        "encoder_gradient_max": float(layer.encoder.weight.grad.abs().max()),
        "core_gradient_max": float(layer.core.weights.grad.abs().max()),
        "parameters": sum(parameter.numel() for parameter in scaled.parameters()),
    }
    assert evidence["encoder_gradient_max"] > 0 and evidence["core_gradient_max"] > 0
    r.atomic_json(output_dir / "smoke.json", evidence)
    return evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-name", default="angle_scale_0p5_seed2026_v1")
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    assert Path.cwd() == ROOT
    assert platform.node() == "jscn"
    assert torch.cuda.is_available() and "4090" in torch.cuda.get_device_name(0)
    torch.set_num_threads(4)
    r.set_seed(SEED)
    output_dir = BASE / args.output_name
    output_dir.mkdir(exist_ok=False)
    train, selection, full_dev, train_indices, selection_indices, dev_indices = make_banks()
    retained.NOISE = r.load_snr_noise_map(ROOT / "results/multitarget_snr/snr_calibration.json")
    smoke_evidence = smoke(output_dir, train)
    if args.smoke_only:
        r.atomic_json(output_dir / "completed.json", {"status": "smoke_completed", "evidence": smoke_evidence})
        print(json.dumps(smoke_evidence, ensure_ascii=False), flush=True)
        return

    source_paths = [
        BASE / "angle_scaled_model.py",
        BASE / "run_angle_scale_experiment.py",
        BASE / "model.py",
        BASE / "run_converged_training.py",
        BASE / "pennylane_core.py",
        BASE / "quantum_core.py",
        ROOT / "target_interaction_graph.py",
    ]
    source_hashes = {str(path): sha256(path) for path in source_paths}
    r.atomic_json(
        output_dir / "protocol.json",
        {
            "seed": SEED,
            "arm": ARM,
            "intervention": "fixed encoding angle range 0.5*pi*tanh(x)",
            "retained_reference": "fixed encoding angle range pi*tanh(x)",
            "quantum_backend": "PennyLane default.qubit analytic statevector",
            "n_qubits": 6,
            "graph": {"max_epochs": 40, "patience": 6, "min_delta": 1e-4},
            "batch_size": 24,
            "selection": "four-SNR macro ADE + 0.35 FDE on fixed 256-scene selection split",
            "train_indices": train_indices.tolist(),
            "selection_indices": selection_indices.tolist(),
            "full_dev_indices": dev_indices.tolist(),
            "same_initial_parameters_as_retained_quantum_verified": True,
            "same_input_hashes_as_retained_quantum_required": True,
            "no_llm": True,
            "no_test": True,
            "gpu": torch.cuda.get_device_name(0),
            "python": sys.executable,
            "source_hashes": source_hashes,
        },
    )
    r.set_seed(SEED)
    model = build_angle_scaled_graph().cuda()
    audit = train_stage(
        model, ARM, "graph", 40, 6, train, selection, output_dir, expected_input_hashes()
    )
    full_metrics, arrays = retained.evaluate(model, full_dev, collect=True)
    np.savez_compressed(output_dir / f"{ARM}_graph_full_dev.npz", **arrays, indices=dev_indices)
    summary = {
        ARM: {
            "graph": full_metrics,
            "selected_epoch": audit["selected_epoch"],
            "executed_epochs": audit["executed_epochs"],
            "stop_reason": audit["stop_reason"],
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        }
    }
    r.atomic_json(output_dir / "summary.json", summary)
    for path, digest in source_hashes.items():
        assert sha256(path) == digest
    completed = {
        "status": "completed",
        "seed": SEED,
        "summary": summary,
        "source_hashes_verified": True,
        "same_initial_parameters_as_retained_quantum_verified": True,
        "same_input_hashes_as_retained_quantum_verified": True,
        "no_llm": True,
        "no_test": True,
    }
    r.atomic_json(output_dir / "completed.json", completed)
    r.atomic_json(output_dir / "progress.json", completed)
    print(json.dumps(completed, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
