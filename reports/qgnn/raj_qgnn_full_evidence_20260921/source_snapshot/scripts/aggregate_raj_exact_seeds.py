"""Aggregate the frozen seed-2026 and seed-2027 exact-match RAJ results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(path: Path):
    return json.loads(path.read_text())


def gain(q, c):
    return 100.0 * (c - q) / c


def aggregate(entries):
    out = {}
    for metric in ("ADE", "FDE", "J"):
        q = float(np.mean([entry["quantum"][metric] for entry in entries]))
        c = float(np.mean([entry["classical"][metric] for entry in entries]))
        seed_gains = [gain(entry["quantum"][metric], entry["classical"][metric]) for entry in entries]
        out[metric] = {"quantum_mean": q, "classical_mean": c, "gain_pct": gain(q, c),
                       "seed_gain_mean": float(np.mean(seed_gains)),
                       "seed_gain_std": float(np.std(seed_gains, ddof=1))}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed2026-main", required=True)
    parser.add_argument("--seed2027-main", required=True)
    parser.add_argument("--seed2026-snr", required=True)
    parser.add_argument("--seed2027-snr", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    main26, main27 = load(root / args.seed2026_main), load(root / args.seed2027_main)
    snr26, snr27 = load(root / args.seed2026_snr), load(root / args.seed2027_snr)
    m26 = main26["sizes"]["4096"]
    result = {
        "protocol": {"seeds": [2026, 2027], "train_samples": 4096, "test_set_used": False,
                     "classical_control": "Johnson hidden=42; graph-core parameter gap 0.66%"},
        "overall": aggregate([
            {"quantum": m26["quantum"], "classical": m26["classical"]},
            main27["all"],
        ]),
        "interaction_strata": {},
        "cross_snr": {},
    }
    for stratum in ("top20", "top10"):
        result["interaction_strata"][stratum] = aggregate([
            m26["interaction_strata"][stratum], main27["strata"][stratum]
        ])
    for snr in ("-10", "-5", "0", "5", "10"):
        result["cross_snr"][snr] = {}
        for stratum in ("all", "top20", "top10"):
            result["cross_snr"][snr][stratum] = aggregate([
                snr26["snr"][snr]["strata"][stratum],
                snr27["snr"][snr]["strata"][stratum],
            ])
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
