"""Single-quantum-core experiment on outcome-blind high two-hop sparse scenes."""
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
from discover_quantum_scenarios import scene_features
from model import CoreMessageLayer
from run_converged_training import train_stage
from run_multitarget_experiment import prediction_loss
from target_interaction_graph import ForecasterConfig, TargetInteractionGNN


ROOT = Path("/home/js_cn/sensing")
BASE = Path(__file__).resolve().parent
DISCOVERY = BASE / "quantum_scenario_discovery_seed2026_v1"
DATA = ROOT / "data/multitarget_lankershim_v1.npz"
SPLIT = BASE / "retained_inputs/circuit_split_indices.npz"
SEED = 2026
RADIUS_M = 20.0
TWO_HOP_THRESHOLD = 0.25
ARMS = ("plain_radius20", "classical_radius20", "quantum_radius20")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_model(arm):
    if arm not in ARMS:
        raise ValueError(arm)
    torch.manual_seed(SEED)
    config = ForecasterConfig(graph_radius_m=RADIUS_M)
    model = TargetInteractionGNN(config)
    if arm != "plain_radius20":
        kind = arm.split("_")[0]
        torch.manual_seed(SEED + 1)
        model.graph_layers[1] = CoreMessageLayer(
            config, kind, quantum_backend="pennylane" if kind == "quantum" else "torch"
        )
    return model


def bank(states, masks, indices):
    return {
        "history": torch.from_numpy(states[indices, :20]).float().cuda(),
        "future": torch.from_numpy(states[indices, 20:]).float().cuda(),
        "mask": torch.from_numpy(masks[indices]).bool().cuda(),
    }


def tensor_hash(mapping):
    return r.tensor_mapping_sha256({name: value.detach().cpu() for name, value in mapping.items()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-name", default="two_hop_sparse_radius20_seed2026_v1")
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    assert Path.cwd() == ROOT
    assert platform.node() == "jscn"
    assert torch.cuda.is_available() and "4090" in torch.cuda.get_device_name(0)
    output = BASE / args.output_name
    output.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    r.set_seed(SEED)

    scenario_protocol_path = DISCOVERY / "scenario_protocol_frozen.json"
    scenario_protocol = json.loads(scenario_protocol_path.read_text(encoding="utf-8"))
    assert scenario_protocol["status"] == "frozen_before_loading_model_outcomes"
    assert scenario_protocol["thresholds"]["two_hop_pair_fraction"] == TWO_HOP_THRESHOLD
    with np.load(SPLIT, allow_pickle=False) as split:
        train_indices = split["circuit_train_indices"].copy()
        dev_indices = split["circuit_dev_indices"].copy()
    with np.load(DATA, allow_pickle=False) as data:
        states = data["train_states"].copy()
        masks = data["train_mask"].copy()
        starts = data["train_start_index"].copy()

    train_feature = scene_features(states[train_indices, :20], masks[train_indices])["two_hop_pair_fraction"]
    dev_feature = scene_features(states[dev_indices, :20], masks[dev_indices])["two_hop_pair_fraction"]
    selected_train_indices = train_indices[train_feature >= TWO_HOP_THRESHOLD]
    discovery_rows = np.asarray(scenario_protocol["discovery_rows"], dtype=np.int64)
    confirmation_rows = np.asarray(scenario_protocol["confirmation_rows"], dtype=np.int64)
    selection_indices = dev_indices[discovery_rows[dev_feature[discovery_rows] >= TWO_HOP_THRESHOLD]]
    confirmation_indices = dev_indices[confirmation_rows[dev_feature[confirmation_rows] >= TWO_HOP_THRESHOLD]]
    assert len(selected_train_indices) >= 500
    assert len(selection_indices) >= 100 and len(confirmation_indices) >= 100

    train_bank = bank(states, masks, selected_train_indices)
    selection_bank = bank(states, masks, selection_indices)
    confirmation_bank = bank(states, masks, confirmation_indices)
    retained.NOISE = r.load_snr_noise_map(ROOT / "results/multitarget_snr/snr_calibration.json")

    shared_nonreplacement = None
    shared_replacement_noncore = None
    smoke = {}
    history = train_bank["history"][:2]
    future = train_bank["future"][:2]
    mask = train_bank["mask"][:2]
    for arm in ARMS:
        r.set_seed(SEED)
        model = build_model(arm).cuda()
        if arm != "plain_radius20":
            outside = tensor_hash({
                name: value for name, value in model.named_parameters()
                if not name.startswith("graph_layers.1.")
            })
            noncore = tensor_hash({
                name: value for name, value in model.graph_layers[1].named_parameters()
                if not name.startswith("core.")
            })
            if shared_nonreplacement is None:
                shared_nonreplacement = outside
                shared_replacement_noncore = noncore
            assert outside == shared_nonreplacement
            assert noncore == shared_replacement_noncore
        model.train()
        prediction = model(history, mask)
        loss = prediction_loss(prediction, future, mask, graph_weighting=True)
        loss.backward()
        gradient = max(float(p.grad.abs().max()) for p in model.parameters() if p.grad is not None)
        core_gradient = None
        if arm != "plain_radius20":
            core_gradient = max(
                float(p.grad.abs().max()) for p in model.graph_layers[1].core.parameters() if p.grad is not None
            )
            assert core_gradient > 0
        smoke[arm] = {
            "loss": float(loss.detach()),
            "gradient_max": gradient,
            "core_gradient_max": core_gradient,
            "parameters": sum(p.numel() for p in model.parameters()),
        }
        del model
        torch.cuda.empty_cache()

    source_paths = [
        Path(__file__), BASE / "discover_quantum_scenarios.py", BASE / "model.py",
        BASE / "run_converged_training.py", BASE / "pennylane_core.py",
        ROOT / "target_interaction_graph.py",
    ]
    protocol = {
        "status": "frozen_before_training",
        "seed": SEED,
        "arms": list(ARMS),
        "graph_radius_m": RADIUS_M,
        "high_two_hop_threshold": TWO_HOP_THRESHOLD,
        "training_filter": "history-only two_hop_pair_fraction >= 0.25",
        "selection": "discovery time blocks and high-two-hop scenes only",
        "final_evaluation": "confirmation time blocks and high-two-hop scenes only",
        "train_scenes": int(len(selected_train_indices)),
        "selection_scenes": int(len(selection_indices)),
        "confirmation_scenes": int(len(confirmation_indices)),
        "train_indices": selected_train_indices.tolist(),
        "selection_indices": selection_indices.tolist(),
        "confirmation_indices": confirmation_indices.tolist(),
        "graph": {"max_epochs": 40, "patience": 6, "min_delta": 1e-4},
        "batch_size": 24,
        "scenario_protocol": str(scenario_protocol_path),
        "scenario_protocol_sha256": sha256(scenario_protocol_path),
        "matched_initialization_required": True,
        "smoke": smoke,
        "no_llm": True,
        "no_test": True,
        "gpu": torch.cuda.get_device_name(0),
        "python": sys.executable,
        "source_hashes": {str(path): sha256(path) for path in source_paths},
    }
    r.atomic_json(output / "protocol.json", protocol)
    if args.smoke_only:
        r.atomic_json(output / "completed.json", {"status": "smoke_completed", "protocol": protocol})
        print(json.dumps(protocol, ensure_ascii=False), flush=True)
        return

    input_hashes = {}
    summaries = {}
    for arm in ARMS:
        r.set_seed(SEED)
        model = build_model(arm).cuda()
        audit = train_stage(
            model, arm, "graph", 40, 6, train_bank, selection_bank, output, input_hashes
        )
        confirmation_metrics, arrays = retained.evaluate(model, confirmation_bank, collect=True)
        np.savez_compressed(
            output / f"{arm}_graph_confirmation.npz", **arrays, indices=confirmation_indices
        )
        summaries[arm] = {
            "confirmation": confirmation_metrics,
            "selected_epoch": audit["selected_epoch"],
            "executed_epochs": audit["executed_epochs"],
            "stop_reason": audit["stop_reason"],
            "parameters": sum(p.numel() for p in model.parameters()),
        }
        r.atomic_json(output / "summary.json", summaries)
        del model
        torch.cuda.empty_cache()
    for path, digest in protocol["source_hashes"].items():
        assert sha256(path) == digest
    completed = {
        "status": "completed",
        "seed": SEED,
        "summaries": summaries,
        "source_hashes_verified": True,
        "same_input_hashes_across_arms_verified": True,
        "confirmation_outcomes_used_only_after_each_arm_selection": True,
        "no_llm": True,
        "no_test": True,
    }
    r.atomic_json(output / "completed.json", completed)
    r.atomic_json(output / "progress.json", completed)
    print(json.dumps(completed, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
