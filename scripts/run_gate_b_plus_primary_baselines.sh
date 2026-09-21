#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-/home/dell/YrM/envs/ICCT/bin/python}
pids=()

launch_graph() {
  local seed=$1 gpu=$2
  local output="reports/task_redesign/sind_gate_b_plus_primary_strong_graph_seed${seed}"
  CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" -u experiments/gate_b_plus/run_classical_robust.py \
    --kind graph --seed "$seed" --order-seed 20260921 --epochs 50 --patience 8 --fde-weight 0.75 \
    --output "$output" > "${output}.log" 2>&1 &
  pids+=("$!")
}

launch_johnson() {
  local label=$1 core=$2 seed=$3 gpu=$4
  local output="reports/task_redesign/sind_gate_b_plus_primary_${label}_seed${seed}"
  CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" -u experiments/gate_b_plus/run_screen.py \
    --mode horizon_multiscale --core-kind "$core" --seed "$seed" --order-seed 20260921 \
    --epochs 50 --patience 8 --train-limit 36643 --ema-decay 0.999 --fde-weight 0.75 \
    --output "$output" > "${output}.log" 2>&1 &
  pids+=("$!")
}

launch_graph 2026 0
launch_graph 2027 1
launch_johnson matched_johnson johnson 2026 0
launch_johnson matched_johnson johnson 2027 1
launch_johnson large_johnson johnson_large 2026 0
launch_johnson large_johnson johnson_large 2027 1

status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
