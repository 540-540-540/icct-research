#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
python=/home/dell/YrM/envs/ICCT/bin/python
base=results/senior_raj_stack
mkdir -p "$base"

run_seed() {
  local gpu="$1" seed="$2"
  CUDA_VISIBLE_DEVICES="$gpu" "$python" -u scripts/run_senior_raj_stack.py \
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

root = Path("results/senior_raj_stack")
runs = [json.loads((root / f"seed{s}" / "results.json").read_text()) for s in (2026, 2027, 2028)]
summary = {"seeds": [2026, 2027, 2028]}
for arm in ("raj_qgnn", "raj_graph_motion_token_gpt2"):
    summary[arm] = {}
    for metric in ("ade_m", "fde_m"):
        values = [run[arm]["test"][metric] for run in runs]
        summary[arm][metric] = {"values": values, "mean": float(np.mean(values)), "std": float(np.std(values, ddof=1))}
reference = json.loads(
    Path("reports/senior_r0_reproduction/phase2/phase2_results.json").read_text()
)["test"]["graph_motion_token_gpt2"]
summary["senior_seed2026_reference"] = reference
for metric in ("ade_m", "fde_m"):
    candidate = summary["raj_graph_motion_token_gpt2"][metric]["mean"]
    summary.setdefault("mean_gain_vs_senior_seed2026_reference_percent", {})[metric] = 100 * (reference[metric] - candidate) / reference[metric]
(root / "three_seed_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY
