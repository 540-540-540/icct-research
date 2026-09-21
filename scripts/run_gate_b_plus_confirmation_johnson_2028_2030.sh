#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-/home/dell/YrM/envs/ICCT/bin/python}
pids=()

run_control() {
  local gpu=$1 label=$2 core=$3 seed=$4
  local name="sind_gate_b_plus_confirmation_stochastic_${label}_seed${seed}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u experiments/gate_b_plus/run_screen.py \
    --mode stochastic_multiscale_quantum_attention --core-kind "$core" \
    --seed "$seed" --order-seed 20260921 --dropout-seed 20260921 \
    --epochs 40 --patience 8 --train-limit 36643 --ema-decay 0.999 \
    --fde-weight 0.75 --branch-drop-probability 0.2 \
    --output "reports/task_redesign/${name}" \
    > "reports/task_redesign/${name}.log" 2>&1 &
  pids+=("$!")
}

run_control 0 matched_johnson johnson 2028
run_control 1 matched_johnson johnson 2029
run_control 0 matched_johnson johnson 2030
run_control 1 large_johnson johnson_large 2028
run_control 0 large_johnson johnson_large 2029
run_control 1 large_johnson johnson_large 2030

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
