"""Convergence-controlled Classical x2 versus Quantum x2 graph experiment."""
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
from dual_core_model import build_dual_graph
from model import build_graph
from run_converged_training import train_stage
from run_multitarget_experiment import prediction_loss


ROOT = Path("/home/js_cn/sensing")
BASE = Path(__file__).resolve().parent
OLD = BASE / "converged_protocol_seed2026_v1"
SEED = 2026
ARMS = ("classical_dual", "quantum_dual")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_hash(mapping):
    return r.tensor_mapping_sha256({name: value.detach().cpu() for name, value in mapping.items()})


def banks():
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


def expected_inputs():
    values = {}
    with (OLD / "quantum_graph.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            values[f"graph_{record['epoch']}"] = record["input_sha256"]
    assert len(values) == 40
    return values


def smoke(output_dir, train):
    evidence = {}
    shared_outside = None
    matched_noncore = None
    history = train["history"][:2]
    future = train["future"][:2]
    mask = train["mask"][:2]
    permutation = torch.arange(history.shape[2] - 1, -1, -1, device="cuda")
    for arm in ARMS:
        kind = arm[:-5]
        r.set_seed(SEED)
        model = build_dual_graph(arm).cuda()
        outside = tensor_hash(
            {name: value for name, value in model.named_parameters() if not name.startswith("graph_layers.")}
        )
        if shared_outside is None:
            shared_outside = outside
        assert outside == shared_outside
        noncore = tensor_hash(
            {
                name: value
                for name, value in model.named_parameters()
                if name.startswith("graph_layers.") and ".core." not in name
            }
        )
        if matched_noncore is None:
            matched_noncore = noncore
        assert noncore == matched_noncore

        # Layer 2 is an exact initialization match to the retained single-core arm.
        r.set_seed(SEED)
        retained = build_graph(
            kind, quantum_backend="pennylane" if kind == "quantum" else "torch"
        ).cuda()
        assert tensor_hash(dict(model.graph_layers[1].named_parameters())) == tensor_hash(
            dict(retained.graph_layers[1].named_parameters())
        )
        assert tensor_hash(dict(model.graph_layers[0].core.named_parameters())) != tensor_hash(
            dict(model.graph_layers[1].core.named_parameters())
        )

        model.train()
        output = model(history, mask)
        loss = prediction_loss(output, future, mask, graph_weighting=True)
        loss.backward()
        core_gradients = []
        for layer in model.graph_layers:
            core_gradients.append(
                max(
                    float(parameter.grad.abs().max())
                    for parameter in layer.core.parameters()
                    if parameter.grad is not None
                )
            )
        assert min(core_gradients) > 0
        model.eval()
        with torch.no_grad():
            original = model(history, mask)["future_position"]
            permuted = model(history[:, :, permutation], mask[:, permutation])["future_position"]
        equivariance_error = float((permuted - original[:, :, permutation]).abs().max())
        assert equivariance_error < 2e-5
        evidence[arm] = {
            "loss": float(loss.detach()),
            "core_gradient_max_by_layer": core_gradients,
            "permutation_equivariance_max_error": equivariance_error,
            "layer_2_initialization_matches_retained_single_core": True,
            "layer_core_initializations_are_independent": True,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        }
        del retained, model
        torch.cuda.empty_cache()
    r.atomic_json(output_dir / "smoke.json", evidence)
    return evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-name", default="dual_core_seed2026_v1")
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
        BASE / "dual_core_model.py",
        BASE / "run_dual_core_experiment.py",
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
            "arms": list(ARMS),
            "intervention": "replace both of the two graph layers with matched message cores",
            "layer_2_initialization": "exactly matches retained single-core arm",
            "layer_1_initialization": "independent deterministic core and readouts",
            "quantum_backend": "PennyLane default.qubit analytic statevector",
            "n_qubits_per_quantum_layer": 6,
            "graph": {"max_epochs": 40, "patience": 6, "min_delta": 1e-4},
            "batch_size": 24,
            "selection": "four-SNR macro ADE + 0.35 FDE on fixed 256-scene selection split",
            "interaction_stratification_protocol": str(OLD / "interaction_stratification_protocol.json"),
            "train_indices": train_indices.tolist(),
            "selection_indices": selection_indices.tolist(),
            "full_dev_indices": dev_indices.tolist(),
            "matched_noncore_initialization_required": True,
            "same_input_hashes_as_retained_quantum_required": True,
            "no_llm": True,
            "no_test": True,
            "gpu": torch.cuda.get_device_name(0),
            "python": sys.executable,
            "source_hashes": source_hashes,
        },
    )
    summaries = {}
    for arm in ARMS:
        r.set_seed(SEED)
        model = build_dual_graph(arm).cuda()
        audit = train_stage(
            model, arm, "graph", 40, 6, train, selection, output_dir, expected_inputs()
        )
        full_metrics, arrays = retained.evaluate(model, full_dev, collect=True)
        np.savez_compressed(output_dir / f"{arm}_graph_full_dev.npz", **arrays, indices=dev_indices)
        summaries[arm] = {
            "graph": full_metrics,
            "selected_epoch": audit["selected_epoch"],
            "executed_epochs": audit["executed_epochs"],
            "stop_reason": audit["stop_reason"],
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
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
        "matched_noncore_initialization_verified": True,
        "same_input_hashes_as_retained_quantum_verified": True,
        "no_llm": True,
        "no_test": True,
    }
    r.atomic_json(output_dir / "completed.json", completed)
    r.atomic_json(output_dir / "progress.json", completed)
    print(json.dumps(completed, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
