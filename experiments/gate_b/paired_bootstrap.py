from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
QUANTUM = "raj_weighted_multij_quantum"
CLASSICAL = "raj_multij_johnson"


def origin_means(rows: list[dict], horizon: int, critical: bool, metric: str) -> dict[int, float]:
    grouped: dict[int, list[float]] = {}
    for row in rows:
        if row["horizon"] != horizon or (critical and row["k"] < 1): continue
        value = row[metric]
        if np.isfinite(value): grouped.setdefault(row["origin"], []).append(value)
    return {origin: float(np.mean(values)) for origin, values in grouped.items()}


def interval(reference: dict[int, float], candidate: dict[int, float], seed: int) -> dict:
    origins = sorted(set(reference) & set(candidate))
    delta = np.asarray([reference[key] - candidate[key] for key in origins])
    rng = np.random.default_rng(seed)
    samples = delta[rng.integers(0, len(delta), size=(5000, len(delta)))].mean(1)
    return {"origins": len(delta), "mean_m": float(delta.mean()),
            "ci95_m": [float(np.quantile(samples, .025)), float(np.quantile(samples, .975))],
            "probability_positive": float(np.mean(samples > 0))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="reports/task_redesign/SIND_GATE_B_PAIRED_BOOTSTRAP_20260921.json")
    args = parser.parse_args()
    result = {"comparison": "matched Johnson minus Raj QGNN", "unit": "validation prediction origin",
              "resamples": 5000, "fixed_endpoint_fde": True, "seeds": {}}
    for seed in (2026, 2027):
        root = ROOT / f"reports/task_redesign/sind_gate_b_seed{seed}"
        rows = {kind: json.loads((root / f"{kind}_validation_rows.json").read_text())["rows"]
                for kind in (CLASSICAL, QUANTUM)}
        seed_result = {}
        for horizon in (20, 40):
            for cohort, critical in (("full", False), ("ic", True)):
                view = f"{cohort}{horizon // 10}"; seed_result[view] = {}
                for index, metric in enumerate(("ade", "fde")):
                    reference = origin_means(rows[CLASSICAL], horizon, critical, metric)
                    candidate = origin_means(rows[QUANTUM], horizon, critical, metric)
                    seed_result[view][metric] = interval(reference, candidate, seed + horizon + index)
                # J uses origins where both endpoint-aware ADE and FDE are defined.
                for kind in (CLASSICAL, QUANTUM):
                    ade = origin_means(rows[kind], horizon, critical, "ade")
                    fde = origin_means(rows[kind], horizon, critical, "fde")
                    rows[kind + "_J"] = {origin: ade[origin] + .5 * fde[origin] for origin in set(ade) & set(fde)}
                seed_result[view]["J"] = interval(rows[CLASSICAL + "_J"], rows[QUANTUM + "_J"], seed + horizon + 2)
        result["seeds"][str(seed)] = seed_result
    path = ROOT / args.output; path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({seed: values["ic4"] for seed, values in result["seeds"].items()}, indent=2))


if __name__ == "__main__": main()
