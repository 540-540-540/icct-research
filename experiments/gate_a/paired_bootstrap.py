from __future__ import annotations

import argparse
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
    model.eval()
    output = {20: [], 40: []}
    for batch in loader:
        prediction = model(batch["history_state"].to(device), batch["node_mask"].to(device)).cpu()
        distance = torch.linalg.vector_norm(prediction - batch["future_xy"], dim=-1)
        mask = batch["future_mask"]
        for horizon in (20, 40):
            hmask = mask[:, :horizon]
            ade = ((distance[:, :horizon] * hmask).sum(1) / hmask.sum(1).clamp_min(1)).numpy()
            ade[~hmask.any(1).numpy()] = np.nan
            fde = distance[:, horizon - 1].numpy()
            fde[~mask[:, horizon - 1].numpy()] = np.nan
            output[horizon].append(np.column_stack([ade, fde, ade + 0.5 * fde]))
    return {horizon: np.concatenate(parts) for horizon, parts in output.items()}


def origin_values(values, origins, selection):
    result = []
    for origin in np.unique(origins[selection]):
        selected = selection & (origins == origin)
        result.append([np.nanmean(values[selected, i]) if np.isfinite(values[selected, i]).any() else np.nan
                       for i in range(values.shape[1])])
    return np.asarray(result)


def interval(reference, candidate, seed):
    valid = np.isfinite(reference) & np.isfinite(candidate)
    delta = reference[valid] - candidate[valid]
    rng = np.random.default_rng(seed)
    samples = delta[rng.integers(0, len(delta), size=(5000, len(delta)))].mean(1)
    return {"origins": len(delta), "mean_m": float(delta.mean()),
            "ci95_m": [float(np.quantile(samples, .025)), float(np.quantile(samples, .975))],
            "probability_positive": float(np.mean(samples > 0))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gate_a_sind_ic4.json")
    parser.add_argument("--output", default="reports/task_redesign/SIND_GATE_A_PAIRED_BOOTSTRAP_20260921.json")
    args = parser.parse_args()
    cfg = json.loads((ROOT / args.config).read_text())
    dataset = GateADataset(ROOT / cfg["cache"], "val")
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    origins = dataset.arrays["origin_id"]
    critical = dataset.arrays["k"] >= 1
    output = {"unit": "validation prediction origin", "resamples": 5000,
              "fixed_endpoint_fde": True, "seeds": {}}
    comparisons = (("self", "graph"), ("self", "all_graph"), ("own", "graph"),
                   ("pool", "graph"), ("pool", "all_graph"), ("graph", "all_graph"))
    for seed in (2026, 2027):
        model_errors = {}
        for name in ("self", "own", "pool", "graph", "all_graph"):
            model = GateAModel(name, dataset.arrays["history_state"].shape[2], dt_s=cfg["dt_s"]).to(device)
            checkpoint = torch.load(ROOT / f"reports/task_redesign/sind_gate_a_seed{seed}/{name}.pt",
                                    map_location=device, weights_only=True)
            model.load_state_dict(checkpoint["model"])
            model_errors[name] = errors(model, loader, device)
        seed_result = {}
        for horizon in (20, 40):
            for cohort, selection in (("full", np.ones(len(origins), bool)), ("ic", critical)):
                key = f"{cohort}{horizon // 10}"
                by_model = {name: origin_values(values[horizon], origins, selection)
                            for name, values in model_errors.items()}
                seed_result[key] = {}
                for reference, candidate in comparisons:
                    seed_result[key][f"{reference}_minus_{candidate}"] = {
                        metric: interval(by_model[reference][:, i], by_model[candidate][:, i], seed + i + horizon)
                        for i, metric in enumerate(("ade", "fde", "J"))
                    }
        output["seeds"][str(seed)] = seed_result
    path = ROOT / args.output
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
