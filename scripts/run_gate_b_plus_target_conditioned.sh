#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-/home/dell/YrM/envs/ICCT/bin/python}

run_seed() {
  local gpu=$1 seed=$2
  local name="sind_gate_b_plus_target_conditioned_quantum_attention_seed${seed}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u experiments/gate_b_plus/run_screen.py \
    --mode target_conditioned_quantum_attention \
    --seed "$seed" --order-seed 20260921 --dropout-seed 20260921 \
    --epochs 40 --patience 8 --train-limit 36643 --ema-decay 0.999 \
    --fde-weight 0.75 --target-loss-weight 0.25 \
    --output "reports/task_redesign/${name}" \
    > "reports/task_redesign/${name}.log" 2>&1
}

run_seed 0 2026 & p1=$!
run_seed 1 2027 & p2=$!
wait "$p1"
wait "$p2"
