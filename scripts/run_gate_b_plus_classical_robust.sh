#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-/home/dell/YrM/envs/ICCT/bin/python}
pids=(); index=0
for seed in 2026 2027; do
  for kind in graph all_graph; do
    gpu=$((index % 2)); out="reports/task_redesign/sind_gate_b_plus_robust_${kind}_seed${seed}"
    CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" -u experiments/gate_b_plus/run_classical_robust.py \
      --kind "$kind" --seed "$seed" --resume --output "$out" > "${out}.log" 2>&1 &
    pids+=("$!"); index=$((index + 1))
  done
done
status=0; for pid in "${pids[@]}"; do wait "$pid" || status=1; done; exit "$status"
