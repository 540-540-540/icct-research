"""Audited recovery: keep classical result, restart only quantum after disk-full."""
import json
from pathlib import Path

import numpy as np
import torch

import experiment_utils as r
import run as retained
from dual_core_model import build_dual_graph
from run_converged_training import train_stage
from run_dual_core_experiment import BASE, ROOT, SEED, expected_inputs, sha256


OUTPUT = BASE / "dual_core_seed2026_v1"
ARM = "quantum_dual"


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

    return bank(train_indices), bank(selection_indices), bank(dev_indices), dev_indices


def main():
    assert Path.cwd() == ROOT and OUTPUT.is_dir()
    assert not (OUTPUT / "completed.json").exists()
    assert (OUTPUT / "classical_dual_graph_audit.json").exists()
    protocol = json.loads((OUTPUT / "protocol.json").read_text(encoding="utf-8"))
    for path, digest in protocol["source_hashes"].items():
        assert sha256(path) == digest

    interrupted_log = OUTPUT / f"{ARM}_graph.jsonl"
    records = [json.loads(line) for line in interrupted_log.read_text(encoding="utf-8").splitlines()]
    assert records and records[-1]["epoch"] == 6
    archived_log = OUTPUT / f"{ARM}_graph_interrupted_disk_full_epoch6.jsonl"
    assert not archived_log.exists()
    interrupted_log.rename(archived_log)
    corrupted_checkpoint = OUTPUT / f"{ARM}_graph_selected.pt"
    assert corrupted_checkpoint.exists()
    corrupted_checkpoint.unlink()
    recovery = {
        "reason": "disk full while writing quantum selected checkpoint after epoch 6",
        "action": "discard corrupted checkpoint and restart quantum arm from identical seed; retain completed classical arm",
        "interrupted_records": len(records),
        "last_completed_epoch_in_interrupted_log": records[-1]["epoch"],
        "archived_interrupted_log": archived_log.name,
        "quantum_restart_from_epoch_zero": True,
        "classical_not_retrained": True,
    }
    r.atomic_json(OUTPUT / "recovery_audit.json", recovery)

    torch.set_num_threads(4)
    r.set_seed(SEED)
    train, selection, full_dev, dev_indices = banks()
    retained.NOISE = r.load_snr_noise_map(ROOT / "results/multitarget_snr/snr_calibration.json")
    r.set_seed(SEED)
    model = build_dual_graph(ARM).cuda()
    audit = train_stage(
        model, ARM, "graph", 40, 6, train, selection, OUTPUT, expected_inputs()
    )
    full_metrics, arrays = retained.evaluate(model, full_dev, collect=True)
    np.savez_compressed(OUTPUT / f"{ARM}_graph_full_dev.npz", **arrays, indices=dev_indices)
    summaries = json.loads((OUTPUT / "summary.json").read_text(encoding="utf-8"))
    summaries[ARM] = {
        "graph": full_metrics,
        "selected_epoch": audit["selected_epoch"],
        "executed_epochs": audit["executed_epochs"],
        "stop_reason": audit["stop_reason"],
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    r.atomic_json(OUTPUT / "summary.json", summaries)
    for path, digest in protocol["source_hashes"].items():
        assert sha256(path) == digest
    completed = {
        "status": "completed",
        "seed": SEED,
        "summaries": summaries,
        "source_hashes_verified": True,
        "matched_noncore_initialization_verified": True,
        "same_input_hashes_as_retained_quantum_verified": True,
        "recovery": recovery,
        "no_llm": True,
        "no_test": True,
    }
    r.atomic_json(OUTPUT / "completed.json", completed)
    r.atomic_json(OUTPUT / "progress.json", completed)
    print(json.dumps(completed, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
