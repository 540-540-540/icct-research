#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
python=/home/dell/YrM/envs/ICCT/bin/python
base=results/senior_raj_matched
mkdir -p "$base"

run_one() {
  local gpu="$1" seed="$2"
  CUDA_VISIBLE_DEVICES="$gpu" "$python" -u scripts/run_senior_raj_matched.py \
    --seed "$seed" --output-dir "$base/quantum_seed${seed}" \
    >"$base/quantum_seed${seed}.console.log" 2>&1
}

run_one 0 2026 & pid0=$!
run_one 1 2027 & pid1=$!
wait "$pid0"
run_one 0 2028 & pid2=$!
wait "$pid1"
wait "$pid2"

"$python" - <<'PY'
import json
from pathlib import Path
import numpy as np

root = Path("results/senior_raj_matched")
runs = {
    seed: json.loads((root / f"quantum_seed{seed}" / "results.json").read_text())
    for seed in (2026, 2027, 2028)
}
summary = {"seeds": [2026, 2027, 2028], "raj_qgnn": {}}
for metric in ("ade_m", "fde_m"):
    values = [runs[seed]["test"][metric] for seed in summary["seeds"]]
    summary["raj_qgnn"][metric] = {
        "values": values,
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
    }
phase1 = json.loads(Path("reports/senior_r0_reproduction/phase1/phase1_results.json").read_text())
phase2 = json.loads(Path("reports/senior_r0_reproduction/phase2/phase2_results.json").read_text())
summary["senior_reproduced_references"] = {
    "target_interaction_gnn": phase1["test"]["target_interaction_gnn"],
    "graph_motion_token_gpt2": phase2["test"]["graph_motion_token_gpt2"],
}
(root / "three_seed_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY
