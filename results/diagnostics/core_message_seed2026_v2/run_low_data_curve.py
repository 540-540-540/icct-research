"""Predeclared low-data learning curve for matched classical and quantum graph cores."""
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch

import experiment_utils as r
import run as retained
import run_two_hop_sparse_experiment as sparse
from run_converged_training import train_stage


ROOT = Path("/home/js_cn/sensing")
BASE = Path(__file__).resolve().parent
REFERENCE = BASE / "two_hop_sparse_radius20_seed2026_v1"
OUTPUT = BASE / "two_hop_low_data_curve_seed2026_v1"
DATA = ROOT / "data/multitarget_lankershim_v1.npz"
SEED = 2026
FRACTIONS = (0.10, 0.25, 0.50)
ARMS = ("classical", "quantum")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def block_balanced_subset(indices, starts, fraction):
    rng = np.random.default_rng(SEED)
    chosen = []
    blocks = starts[indices] // 200
    for block in np.unique(blocks):
        members = indices[blocks == block].copy()
        rng.shuffle(members)
        count = max(1, int(round(len(members) * fraction)))
        chosen.extend(members[:count].tolist())
    return np.asarray(sorted(chosen), dtype=np.int64)


def main():
    assert Path.cwd() == ROOT and platform.node() == "jscn"
    assert torch.cuda.is_available() and "4090" in torch.cuda.get_device_name(0)
    OUTPUT.mkdir(exist_ok=False)
    reference_protocol_path = REFERENCE / "protocol.json"
    reference = json.loads(reference_protocol_path.read_text(encoding="utf-8"))
    full_train_indices = np.asarray(reference["train_indices"], dtype=np.int64)
    selection_indices = np.asarray(reference["selection_indices"], dtype=np.int64)
    confirmation_indices = np.asarray(reference["confirmation_indices"], dtype=np.int64)
    with np.load(DATA, allow_pickle=False) as data:
        states = data["train_states"].copy()
        masks = data["train_mask"].copy()
        starts = data["train_start_index"].copy()
    subsets = {
        str(fraction): block_balanced_subset(full_train_indices, starts, fraction)
        for fraction in FRACTIONS
    }
    selection = sparse.bank(states, masks, selection_indices)
    confirmation = sparse.bank(states, masks, confirmation_indices)
    retained.NOISE = r.load_snr_noise_map(ROOT / "results/multitarget_snr/snr_calibration.json")

    source_paths = [Path(__file__), BASE / "run_two_hop_sparse_experiment.py", BASE / "model.py",
                    BASE / "run_converged_training.py", BASE / "pennylane_core.py",
                    ROOT / "target_interaction_graph.py"]
    protocol = {
        "status": "frozen_before_training",
        "seed": SEED,
        "fractions": list(FRACTIONS),
        "arms": list(ARMS),
        "subset_rule": "nested deterministic block-balanced sample without outcome access",
        "full_training_scenes": int(len(full_train_indices)),
        "subset_indices": {key: value.tolist() for key, value in subsets.items()},
        "selection_indices": selection_indices.tolist(),
        "confirmation_indices": confirmation_indices.tolist(),
        "graph": {"max_epochs": 40, "patience": 6, "min_delta": 1e-4},
        "reference_protocol": str(reference_protocol_path),
        "reference_protocol_sha256": sha256(reference_protocol_path),
        "confirmation_reported_for_all_predeclared_fractions": True,
        "no_llm": True,
        "no_test": True,
        "gpu": torch.cuda.get_device_name(0),
        "python": sys.executable,
        "source_hashes": {str(path): sha256(path) for path in source_paths},
    }
    r.atomic_json(OUTPUT / "protocol.json", protocol)

    results = {}
    for fraction in FRACTIONS:
        fraction_key = str(fraction)
        train_indices = subsets[fraction_key]
        train = sparse.bank(states, masks, train_indices)
        input_hashes = {}
        results[fraction_key] = {"training_scenes": int(len(train_indices))}
        for arm in ARMS:
            arm_key = f"{arm}_fraction_{fraction_key.replace('.', 'p')}"
            r.set_seed(SEED)
            model = sparse.build_model(f"{arm}_radius20").cuda()
            audit = train_stage(model, arm_key, "graph", 40, 6, train, selection, OUTPUT, input_hashes)
            confirmation_metric, arrays = retained.evaluate(model, confirmation, collect=True)
            np.savez_compressed(
                OUTPUT / f"{arm_key}_confirmation.npz",
                **arrays, indices=confirmation_indices, starts=starts[confirmation_indices]
            )
            results[fraction_key][arm] = {
                "confirmation": confirmation_metric,
                "selected_epoch": audit["selected_epoch"],
                "executed_epochs": audit["executed_epochs"],
                "stop_reason": audit["stop_reason"],
                "parameters": sum(p.numel() for p in model.parameters()),
            }
            r.atomic_json(OUTPUT / "progress.json", {
                "status": "running", "completed_fraction": fraction_key,
                "completed_arm": arm, "results": results,
            })
            del model
            torch.cuda.empty_cache()
    for path, digest in protocol["source_hashes"].items():
        assert sha256(path) == digest
    completed = {
        "status": "completed", "seed": SEED, "results": results,
        "source_hashes_verified": True,
        "same_inputs_within_each_fraction_verified": True,
        "confirmation_reported_for_all_predeclared_fractions": True,
        "no_llm": True, "no_test": True,
    }
    r.atomic_json(OUTPUT / "completed.json", completed)
    r.atomic_json(OUTPUT / "progress.json", completed)
    print(json.dumps(completed, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
