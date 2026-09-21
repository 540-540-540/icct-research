#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-/home/dell/YrM/envs/ICCT/bin/python}
pids=()
for seed in 2026 2027; do
  gpu=$((seed - 2026))
  output="reports/task_redesign/sind_gate_b_plus_screen_full_multiscale_seed${seed}"
  CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" -u experiments/gate_b_plus/run_screen.py \
    --mode full_multiscale --seed "$seed" --epochs 30 --resume --output "$output" \
    > "${output}_extension.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
