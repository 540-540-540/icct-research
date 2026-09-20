"""One-shot test evaluation for the frozen two-seed exact-match RAJ models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluate_raj_cross_snr import ROOT, evaluate, load_model, metrics, gains
from frontend.sind_prediction_dataset import SinDPredictionDataset


def run_model(kind, run_dir, dataset, device, batch_size):
    model = load_model(kind, run_dir / "best.pt", device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    summary, rows = evaluate(model, loader, device)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary, rows


def aggregate(entries):
    result = {}
    for key in ("ADE", "FDE", "J"):
        q = float(np.mean([entry["quantum"][key] for entry in entries]))
        c = float(np.mean([entry["classical"][key] for entry in entries]))
        per_seed = [100.0 * (entry["classical"][key] - entry["quantum"][key]) / entry["classical"][key]
                    for entry in entries]
        result[key] = {"quantum_mean": q, "classical_mean": c,
                       "gain_pct": 100.0 * (c - q) / c,
                       "seed_gain_mean": float(np.mean(per_seed)),
                       "seed_gain_std": float(np.std(per_seed, ddof=1))}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed2026-quantum", required=True)
    parser.add_argument("--seed2026-classical", required=True)
    parser.add_argument("--seed2027-quantum", required=True)
    parser.add_argument("--seed2027-classical", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--snr", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Thresholds are frozen from validation inputs before any test predictions are inspected.
    validation = SinDPredictionDataset("val", args.snr, ROOT, True)
    probe = load_model("quantum", ROOT / args.seed2026_quantum / "best.pt", device)
    _, validation_rows = evaluate(probe, DataLoader(validation, batch_size=args.batch_size,
                                                     shuffle=False, num_workers=0), device)
    del probe
    risk_val = np.asarray([row["max_pair_risk"] for row in validation_rows])
    thresholds = {"top20": float(np.quantile(risk_val, 0.80)),
                  "top10": float(np.quantile(risk_val, 0.90))}

    test = SinDPredictionDataset("test", args.snr, ROOT, True)
    runs = {
        2026: {"quantum": ROOT / args.seed2026_quantum,
               "classical": ROOT / args.seed2026_classical},
        2027: {"quantum": ROOT / args.seed2027_quantum,
               "classical": ROOT / args.seed2027_classical},
    }
    per_seed = {}
    strata_entries = {"all": [], "top20": [], "top10": []}
    for seed, pair in runs.items():
        qsummary, qrows = run_model("quantum", pair["quantum"], test, device, args.batch_size)
        csummary, crows = run_model("matched_classical", pair["classical"], test, device, args.batch_size)
        if len(qrows) != len(crows) or any(q["index"] != c["index"] for q, c in zip(qrows, crows)):
            raise ValueError(f"Unpaired test rows for seed {seed}")
        risk = np.asarray([row["max_pair_risk"] for row in qrows])
        seed_result = {"all": {"quantum": qsummary, "classical": csummary,
                               "gain_pct": gains(qsummary, csummary)}}
        strata_entries["all"].append(seed_result["all"])
        for name, threshold in thresholds.items():
            ids = np.flatnonzero(risk >= threshold).tolist()
            qm, cm = metrics(qrows, ids), metrics(crows, ids)
            seed_result[name] = {"quantum": qm, "classical": cm, "gain_pct": gains(qm, cm)}
            strata_entries[name].append(seed_result[name])
        per_seed[str(seed)] = seed_result

    result = {
        "protocol": {"split": "test", "one_shot": True, "snr_db": args.snr,
                     "seeds": [2026, 2027], "threshold_source": "0 dB validation inputs",
                     "thresholds": thresholds},
        "per_seed": per_seed,
        "two_seed": {name: aggregate(entries) for name, entries in strata_entries.items()},
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
