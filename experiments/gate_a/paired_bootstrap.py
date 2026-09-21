from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.gate_a.data import GateADataset
from experiments.gate_a.models import GateAModel


@torch.no_grad()
def errors(model, loader, device):
    model.eval(); values = []
    for batch in loader:
        history = batch["history_state"].to(device); node_mask = batch["node_mask"].to(device)
        prediction = model(history, node_mask).cpu()
        distance = torch.linalg.vector_norm(prediction - batch["future_xy"], dim=-1)
        mask = batch["future_mask"]
        ade = (distance * mask).sum(1) / mask.sum(1).clamp_min(1)
        last = (mask.to(torch.int64).sum(1) - 1).clamp_min(0)
        fde = distance.gather(1, last[:, None]).squeeze(1)
        values.append(torch.stack([ade, fde, ade + 0.5 * fde], 1).numpy())
    return np.concatenate(values)


def origin_values(values, times, selection):
    result = []
    for time in np.unique(times[selection]):
        use = selection & (times == time)
        result.append(values[use].mean(0))
    return np.asarray(result)


def interval(delta, seed):
    rng = np.random.default_rng(seed); n = len(delta)
    samples = delta[rng.integers(0, n, size=(5000, n))].mean(1)
    return {"origins": n, "mean_m": float(delta.mean()),
            "ci95_m": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
            "probability_positive": float(np.mean(samples > 0))}


def main():
    cache = ROOT / "data/task_redesign/lankershim_gate_a_v1"
    dataset = GateADataset(cache, "val")
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = {"unit": "validation prediction origin", "resamples": 5000, "seeds": {}}
    for seed in (2026, 2027):
        model_errors = {}
        for name in ("self", "own", "pool", "graph", "all_graph"):
            model = GateAModel(name, dataset.arrays["history_state"].shape[2]).to(device)
            checkpoint = torch.load(ROOT / f"reports/task_redesign/gate_a_seed{seed}/{name}.pt",
                                    map_location=device, weights_only=True)
            model.load_state_dict(checkpoint["model"])
            model_errors[name] = errors(model, loader, device)
        times, critical = dataset.arrays["time_ms"], dataset.arrays["k"] >= 1
        result = {}
        for cohort, selection in (("full4", np.ones(len(times), bool)), ("ic4", critical)):
            by_model = {name: origin_values(value, times, selection) for name, value in model_errors.items()}
            result[cohort] = {}
            for reference, candidate in (("self", "graph"), ("self", "all_graph"),
                                         ("own", "graph"), ("pool", "graph"),
                                         ("pool", "all_graph"), ("graph", "all_graph")):
                result[cohort][f"{reference}_minus_{candidate}"] = {
                    metric: interval(by_model[reference][:, i] - by_model[candidate][:, i], seed + i)
                    for i, metric in enumerate(("ade", "fde", "J"))}
        output["seeds"][str(seed)] = result
    path = ROOT / "reports/task_redesign/GATE_A_PAIRED_BOOTSTRAP_20260921.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__": main()
