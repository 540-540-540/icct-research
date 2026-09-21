#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-/home/dell/YrM/envs/ICCT/bin/python}
pids=()
index=0
for seed in 2026 2027; do
  for spec in "075:0.75" "100:1.0"; do
    tag=${spec%%:*}; weight=${spec##*:}; gpu=$((index % 2))
    output="reports/task_redesign/sind_gate_b_plus_screen_full_multiscale_fw${tag}_seed${seed}"
    CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" -u experiments/gate_b_plus/run_screen.py \
      --mode full_multiscale --seed "$seed" --fde-weight "$weight" --resume --output "$output" \
      > "${output}.log" 2>&1 &
    pids+=("$!"); index=$((index + 1))
  done
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
