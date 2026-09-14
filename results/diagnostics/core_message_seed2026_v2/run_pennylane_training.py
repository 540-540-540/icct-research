"""Retrain the retained quantum arm with the PennyLane circuit backend.

The data split, seed, losses, optimizer settings, checkpoint selection and epoch
budgets are inherited from run.py.  Results are written to a new directory and
the retained PyTorch-backend experiment is never overwritten.
"""
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

import experiment_utils as r
import run as retained
from model import CoreGraphLLM, build_graph, compact_state
from run_multitarget_experiment import prediction_loss


ROOT = Path("/home/js_cn/sensing")
BASE = Path(__file__).resolve().parent
SEED = 2026


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    r.atomic_json(path, value)


def load_expected_input_hashes():
    expected = {}
    for phase in ("graph", "llm"):
        audit_path = BASE / "development" / f"quantum_{phase}_audit.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        for record in audit["records"]:
            if record["epoch"] > 0:
                expected[f"{phase}_{record['epoch']}"] = record["input_sha256"]
    return expected


def make_banks():
    cache = ROOT / "data/multitarget_lankershim_v1.npz"
    with np.load(BASE / "retained_inputs/circuit_split_indices.npz", allow_pickle=False) as split:
        train_idx = split["circuit_train_indices"].copy()
        dev_idx = split["circuit_dev_indices"].copy()
    with np.load(cache, allow_pickle=False) as data:
        states = data["train_states"].copy()
        masks = data["train_mask"].copy()
    selection_idx = r.select_profile_indices(dev_idx, 256)

    def bank(indices):
        return {
            "history": torch.from_numpy(states[indices, :20]).float().cuda(),
            "future": torch.from_numpy(states[indices, 20:]).float().cuda(),
            "mask": torch.from_numpy(masks[indices]).bool().cuda(),
        }

    return bank(train_idx), bank(selection_idx), bank(dev_idx), train_idx, dev_idx


def smoke(train, output_dir):
    r.set_seed(SEED)
    model = build_graph("quantum", quantum_backend="pennylane").cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-4)
    generator = torch.Generator(device="cuda").manual_seed(SEED + 10001)
    indices = torch.arange(2, device="cuda")
    history, future, mask = retained.make_batch(train, indices, 10, generator)
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    prediction = model(history, mask)
    loss = prediction_loss(prediction, future, mask, graph_weighting=True)
    loss.backward()
    gradient = model.graph_layers[1].core.weights.grad
    optimizer.step()
    torch.cuda.synchronize()
    result = {
        "status": "passed",
        "seed": SEED,
        "backend": model.graph_layers[1].core.metadata(),
        "batch_size": 2,
        "loss": float(loss.detach().cpu()),
        "core_gradient_max_abs": float(gradient.detach().abs().max().cpu()),
        "seconds": time.time() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
    }
    assert np.isfinite(result["loss"]) and result["core_gradient_max_abs"] > 0
    atomic_json(output_dir / "smoke.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def train(output_dir, train_bank, selection_bank, full_dev_bank, train_idx, dev_idx):
    expected_inputs = load_expected_input_hashes()
    source_paths = [
        BASE / "model.py",
        BASE / "run.py",
        BASE / "run_pennylane_training.py",
        BASE / "quantum_core.py",
        BASE / "pennylane_core.py",
        ROOT / "target_interaction_graph.py",
        ROOT / "MultiTargetTimeLLM.py",
    ]
    source_hashes = {str(path): sha256(path) for path in source_paths}
    atomic_json(
        output_dir / "protocol.json",
        {
            "seed": SEED,
            "arm": "quantum",
            "quantum_backend": "PennyLane default.qubit analytic statevector",
            "graph_epochs": 12,
            "llm_epochs": 6,
            "batch_size": 24,
            "precision": "FP32 surrounding model; PennyLane expectations cast back to input dtype",
            "train_indices": train_idx.tolist(),
            "full_dev_indices": dev_idx.tolist(),
            "checkpoint_selection": "four-SNR macro ADE + 0.35 FDE; checkpoints 0..budget",
            "same_input_hashes_required": True,
            "no_test": True,
            "gpu": torch.cuda.get_device_name(0),
            "python": sys.executable,
            "source_hashes": source_hashes,
        },
    )

    r.set_seed(SEED)
    graph = build_graph("quantum", quantum_backend="pennylane").cuda()
    graph_result = retained.train_stage(
        graph, "quantum", "graph", 12, train_bank, selection_bank, output_dir, expected_inputs
    )
    graph_full, arrays = retained.evaluate(graph, full_dev_bank, collect=True)
    np.savez_compressed(output_dir / "quantum_graph_full_dev.npz", **arrays, indices=dev_idx)

    r.set_seed(SEED)
    model = CoreGraphLLM(graph).cuda()
    frozen = {name: value.detach().cpu().clone() for name, value in model.named_parameters() if not value.requires_grad}
    frozen_hash = r.tensor_mapping_sha256(frozen)
    del frozen
    llm_result = retained.train_stage(
        model, "quantum", "llm", 6, train_bank, selection_bank, output_dir, expected_inputs
    )
    current_frozen_hash = r.tensor_mapping_sha256(
        {name: value.detach().cpu() for name, value in model.named_parameters() if not value.requires_grad}
    )
    assert frozen_hash == current_frozen_hash
    llm_full, arrays = retained.evaluate(model, full_dev_bank, collect=True)
    np.savez_compressed(output_dir / "quantum_llm_full_dev.npz", **arrays, indices=dev_idx)

    for path, digest in source_hashes.items():
        assert sha256(path) == digest
    summary = {
        "seed": SEED,
        "backend": model.graph_backbone.graph_layers[1].core.metadata(),
        "graph": graph_full,
        "llm": llm_full,
        "graph_selected_epoch": graph_result["selected_epoch"],
        "llm_selected_epoch": llm_result["selected_epoch"],
        "frozen_gpt_verified": True,
        "source_hashes_verified": True,
        "same_training_input_hashes_verified": True,
        "no_test": True,
    }
    atomic_json(output_dir / "summary.json", summary)
    atomic_json(output_dir / "completed.json", {"status": "completed", **summary})
    atomic_json(output_dir / "progress.json", {"status": "completed"})
    print(json.dumps({"status": "COMPLETED", "summary": summary}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-name", default="pennylane_retrain_v1")
    args = parser.parse_args()
    assert Path.cwd() == ROOT
    assert platform.node() == "jscn"
    assert torch.cuda.is_available() and "4090" in torch.cuda.get_device_name(0)
    torch.set_num_threads(4)
    r.set_seed(SEED)

    output_name = args.output_name + ("_smoke" if args.smoke else "")
    output_dir = BASE / output_name
    output_dir.mkdir(exist_ok=False)
    train_bank, selection_bank, full_dev_bank, train_idx, dev_idx = make_banks()
    calibration = ROOT / "results/multitarget_snr/snr_calibration.json"
    retained.NOISE = retained.r.load_snr_noise_map(calibration)
    if args.smoke:
        smoke(train_bank, output_dir)
    else:
        train(output_dir, train_bank, selection_bank, full_dev_bank, train_idx, dev_idx)


if __name__ == "__main__":
    main()
