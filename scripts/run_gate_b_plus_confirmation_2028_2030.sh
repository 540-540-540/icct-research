#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-/home/dell/YrM/envs/ICCT/bin/python}
pids=()

run_raj() {
  local gpu=$1 seed=$2
  local name="sind_gate_b_plus_confirmation_stochastic_raj_seed${seed}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u experiments/gate_b_plus/run_screen.py \
    --mode stochastic_multiscale_quantum_attention \
    --seed "$seed" --order-seed 20260921 --dropout-seed 20260921 \
    --epochs 40 --patience 8 --train-limit 36643 --ema-decay 0.999 \
    --fde-weight 0.75 --branch-drop-probability 0.2 \
    --output "reports/task_redesign/${name}" \
    > "reports/task_redesign/${name}.log" 2>&1 &
  pids+=("$!")
}

run_graph() {
  local gpu=$1 seed=$2
  local name="sind_gate_b_plus_confirmation_strong_graph_seed${seed}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u experiments/gate_b_plus/run_classical_robust.py \
    --kind graph --seed "$seed" --order-seed 20260921 \
    --epochs 50 --patience 8 --fde-weight 0.75 \
    --output "reports/task_redesign/${name}" \
    > "reports/task_redesign/${name}.log" 2>&1 &
  pids+=("$!")
}

run_raj 0 2028
run_raj 1 2029
run_raj 0 2030
run_graph 1 2028
run_graph 0 2029
run_graph 1 2030

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
