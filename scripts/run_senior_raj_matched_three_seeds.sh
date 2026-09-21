#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
python=/home/dell/YrM/envs/ICCT/bin/python
base=results/senior_raj_matched
mkdir -p "$base"

run_one() {
  local gpu="$1" kind="$2" seed="$3"
  CUDA_VISIBLE_DEVICES="$gpu" "$python" -u scripts/run_senior_raj_matched.py \
    --kind "$kind" --seed "$seed" --output-dir "$base/${kind}_seed${seed}" \
    >"$base/${kind}_seed${seed}.console.log" 2>&1
}

run_pair() {
  local gpu="$1" seed="$2"
  run_one "$gpu" quantum "$seed"
  run_one "$gpu" classical "$seed"
}

run_pair 0 2026 & pid0=$!
run_pair 1 2027 & pid1=$!
wait "$pid0"
run_pair 0 2028 & pid2=$!
wait "$pid1"
wait "$pid2"

"$python" - <<'PY'
import json
from pathlib import Path
import numpy as np

root = Path("results/senior_raj_matched")
runs = {
    kind: {
        seed: json.loads((root / f"{kind}_seed{seed}" / "results.json").read_text())
        for seed in (2026, 2027, 2028)
    }
    for kind in ("quantum", "classical")
}
for seed in runs["quantum"]:
    assert runs["quantum"][seed]["llm_initialization_sha256"] == runs["classical"][seed]["llm_initialization_sha256"]
summary = {"seeds": [2026, 2027, 2028], "paired_llm_initialization_equal": True}
for kind in runs:
    summary[kind] = {}
    for metric in ("ade_m", "fde_m"):
        values = [runs[kind][seed]["test"][metric] for seed in summary["seeds"]]
        summary[kind][metric] = {
            "values": values,
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
        }
summary["quantum_gain_percent_by_seed"] = {}
for seed in summary["seeds"]:
    q = runs["quantum"][seed]["test"]
    c = runs["classical"][seed]["test"]
    summary["quantum_gain_percent_by_seed"][str(seed)] = {
        "ade_m": 100 * (c["ade_m"] - q["ade_m"]) / c["ade_m"],
        "fde_m": 100 * (c["fde_m"] - q["fde_m"]) / c["fde_m"],
    }
summary["quantum_gain_from_three_seed_means_percent"] = {
    metric: 100 * (summary["classical"][metric]["mean"] - summary["quantum"][metric]["mean"]) / summary["classical"][metric]["mean"]
    for metric in ("ade_m", "fde_m")
}
(root / "three_seed_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY
