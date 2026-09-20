"""Aggregate paired RAJ learning curves and input-defined interaction strata."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(path):
    return json.loads(path.read_text())


def gain(q, c):
    return 100.0 * (c - q) / c


def means(rows, ids):
    selected = [rows[i] for i in ids]
    ade = float(np.mean([r["ADE"] for r in selected]))
    fde = float(np.mean([r["FDE"] for r in selected]))
    return {"ADE": ade, "FDE": fde, "J": ade + 0.5 * fde, "scenes": len(selected)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    campaign = root / args.campaign
    result = {"protocol": {"seed": 2026, "test_set_used": False,
              "primary_stratum": "top 20% max_pair_risk, threshold fixed from inputs"}, "sizes": {}}
    completed=[]
    for samples in (1000,2000,4096,8000):
        paths=[campaign/f"{samples}_{kind}"/"summary.json" for kind in ("quantum","matched_classical")]
        if all(p.exists() and load(p).get("status")=="COMPLETED" for p in paths):completed.append(samples)
    if not completed:raise RuntimeError("No completed paired sample sizes")
    for samples in completed:
        pair = {}
        rowsets = {}
        for kind in ("quantum", "matched_classical"):
            run = campaign / f"{samples}_{kind}"
            summary = load(run / "summary.json")
            if summary["status"] != "COMPLETED":
                raise RuntimeError(f"Incomplete run: {run}")
            pair[kind] = summary["best_validation"]
            rowsets[kind] = load(run / "best_validation_rows.json")["rows"]
        qrows, crows = rowsets["quantum"], rowsets["matched_classical"]
        if len(qrows) != len(crows) or any(q["index"] != c["index"] for q, c in zip(qrows, crows)):
            raise ValueError(f"Unpaired validation rows at {samples}")
        risk = np.asarray([r["max_pair_risk"] for r in qrows])
        if not np.allclose(risk, [r["max_pair_risk"] for r in crows], atol=1e-7):
            raise ValueError(f"Interaction scores differ at {samples}")
        thresholds = {"top20": float(np.quantile(risk, 0.80)), "top10": float(np.quantile(risk, 0.90))}
        strata = {}
        for name, threshold in thresholds.items():
            ids = np.flatnonzero(risk >= threshold).tolist()
            qm, cm = means(qrows, ids), means(crows, ids)
            strata[name] = {"threshold": threshold, "quantum": qm, "classical": cm,
                            "gain_pct": {k: gain(qm[k], cm[k]) for k in ("ADE", "FDE", "J")}}
        result["sizes"][str(samples)] = {
            "quantum": pair["quantum"], "classical": pair["matched_classical"],
            "gain_pct": {k: gain(pair["quantum"][k], pair["matched_classical"][k]) for k in ("ADE", "FDE", "J")},
            "interaction_strata": strata,
        }
    out = campaign / "learning_curve_analysis.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
