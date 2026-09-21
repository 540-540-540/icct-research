#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
python=/home/dell/YrM/envs/ICCT/bin/python
base=results/senior_raj_hybrid_matched
mkdir -p "$base"

run_seed() {
  local gpu="$1" seed="$2"
  CUDA_VISIBLE_DEVICES="$gpu" "$python" -u scripts/run_senior_raj_hybrid_matched.py \
    --seed "$seed" --output-dir "$base/seed$seed" \
    >"$base/seed$seed.console.log" 2>&1
}

run_seed 0 2026 & pid0=$!
run_seed 1 2027 & pid1=$!
wait "$pid0"
run_seed 0 2028 & pid2=$!
wait "$pid1"
wait "$pid2"

"$python" - <<'PY'
import json
from pathlib import Path
import numpy as np

root = Path("results/senior_raj_hybrid_matched")
runs = [json.loads((root / f"seed{s}" / "results.json").read_text()) for s in (2026, 2027, 2028)]
summary = {"seeds": [2026, 2027, 2028], "parameters": runs[0]["parameters"]}
for arm in ("target_interaction_gnn", "raj_hybrid_qgnn"):
    summary[arm] = {}
    for metric in ("ade_m", "fde_m"):
        values = [run["test"][arm][metric] for run in runs]
        summary[arm][metric] = {
            "values": values,
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
        }
summary["per_seed_qgnn_gain_percent"] = [run["qgnn_gain_over_gnn_percent"] for run in runs]
summary["qgnn_wins_both_metrics_all_seeds"] = all(
    run["test"]["raj_hybrid_qgnn"]["ade_m"] < run["test"]["target_interaction_gnn"]["ade_m"]
    and run["test"]["raj_hybrid_qgnn"]["fde_m"] < run["test"]["target_interaction_gnn"]["fde_m"]
    for run in runs
)
for metric in ("ade_m", "fde_m"):
    classical = summary["target_interaction_gnn"][metric]["mean"]
    quantum = summary["raj_hybrid_qgnn"][metric]["mean"]
    summary.setdefault("mean_qgnn_gain_percent", {})[metric] = 100.0 * (classical - quantum) / classical
(root / "three_seed_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY
