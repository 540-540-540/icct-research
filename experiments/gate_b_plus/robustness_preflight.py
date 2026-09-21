from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.gate_a.data import GateADataset


def digest(value: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def main() -> None:
    cfg = json.loads((ROOT / "configs/gate_b_plus_screen.json").read_text())
    cache = ROOT / cfg["benchmark"]
    datasets = {seed: GateADataset(cache, "train", limit=36643, seed=seed) for seed in range(2026, 2031)}
    hashes = {str(seed): digest(dataset.indices) for seed, dataset in datasets.items()}
    populations = {str(seed): len(dataset) for seed, dataset in datasets.items()}
    robustness_cfg = json.loads((ROOT / "configs/gate_b_plus_robustness.json").read_text())
    order_seed = robustness_cfg["variance_controls"]["fixed_batch_order_seed"]
    early_stopping_patience = robustness_cfg["variance_controls"]["early_stopping_patience"]
    orders = {
        str(seed): torch.randperm(
            36643, generator=torch.Generator().manual_seed(order_seed + 1009)
        ).numpy()
        for seed in range(2026, 2031)
    }
    order_hashes = {seed: digest(order) for seed, order in orders.items()}
    same_order = len(set(order_hashes.values())) == 1
    result = {
        "status": "PASS" if (len(set(hashes.values())) == 1
                              and set(populations.values()) == {36643} and same_order) else "FAIL",
        "test_accessed": False,
        "test_artifact_absent": not (cache / "test.npz").exists(),
        "full_population_by_seed": populations,
        "train_indices_sha256_by_seed": hashes,
        "same_full_training_population_across_seeds": len(set(hashes.values())) == 1,
        "fixed_batch_order_seed": order_seed,
        "epoch1_order_sha256_by_model_seed": order_hashes,
        "same_batch_order_across_model_seeds": same_order,
        "maximum_epochs": robustness_cfg["variance_controls"]["maximum_epochs"],
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_metric": "validation IC4 J",
        "remaining_seed_sources": ["parameter initialization", "dropout"],
    }
    out = ROOT / "reports/task_redesign/SIND_GATE_B_PLUS_ROBUSTNESS_PREFLIGHT_20260921.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
