"""Analyze one frozen, exactly parameter-matched RAJ validation pair."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(path: Path):
    return json.loads(path.read_text())


def metrics(rows, indices):
    chosen = [rows[i] for i in indices]
    ade = float(np.mean([row["ADE"] for row in chosen]))
    fde = float(np.mean([row["FDE"] for row in chosen]))
    return {"ADE": ade, "FDE": fde, "J": ade + 0.5 * fde, "scenes": len(chosen)}


def gains(q, c):
    return {key: 100.0 * (c[key] - q[key]) / c[key] for key in ("ADE", "FDE", "J")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantum", required=True)
    parser.add_argument("--classical", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    qdir, cdir = root / args.quantum, root / args.classical
    qs, cs = load(qdir / "summary.json"), load(cdir / "summary.json")
    if qs["status"] != "COMPLETED" or cs["status"] != "COMPLETED":
        raise RuntimeError("Both paired runs must be completed")
    for key in ("train_indices_sha256", "train_samples", "validation_samples"):
        if qs[key] != cs[key]:
            raise ValueError(f"Pair mismatch: {key}")
    qrows = load(qdir / "best_validation_rows.json")["rows"]
    crows = load(cdir / "best_validation_rows.json")["rows"]
    if len(qrows) != len(crows) or any(q["index"] != c["index"] for q, c in zip(qrows, crows)):
        raise ValueError("Validation rows are not paired")
    risk = np.asarray([row["max_pair_risk"] for row in qrows])
    if not np.allclose(risk, [row["max_pair_risk"] for row in crows], atol=1e-7):
        raise ValueError("Interaction scores differ across the pair")
    result = {
        "protocol": {"seed": args.seed, "test_set_used": False, "train_samples": qs["train_samples"]},
        "parameters": {"quantum_graph": qs["parameters"]["graph_trainable"],
                       "classical_graph": cs["parameters"]["graph_trainable"]},
        "all": {"quantum": qs["best_validation"], "classical": cs["best_validation"],
                "gain_pct": gains(qs["best_validation"], cs["best_validation"])},
        "strata": {},
    }
    for name, quantile in (("top20", 0.80), ("top10", 0.90)):
        threshold = float(np.quantile(risk, quantile))
        indices = np.flatnonzero(risk >= threshold).tolist()
        qm, cm = metrics(qrows, indices), metrics(crows, indices)
        result["strata"][name] = {"threshold": threshold, "quantum": qm, "classical": cm,
                                  "gain_pct": gains(qm, cm)}
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
