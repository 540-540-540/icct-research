#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-/home/dell/YrM/envs/ICCT/bin/python}

run_one() {
  local gpu=$1 scale=$2 tag=$3 seed=$4
  local name="sind_gate_b_plus_multiscale_quantum_attention_near_identity_${tag}_seed${seed}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u experiments/gate_b_plus/run_screen.py \
    --mode multiscale_quantum_attention \
    --seed "$seed" --order-seed 20260921 --dropout-seed 20260921 \
    --epochs 40 --patience 8 --train-limit 36643 --ema-decay 0.999 \
    --fde-weight 0.75 --quantum-init near_identity --quantum-init-scale "$scale" \
    --output "reports/task_redesign/${name}" \
    > "reports/task_redesign/${name}.log" 2>&1
}

run_one 0 0.05 scale005 2026 & p1=$!
run_one 0 0.10 scale010 2027 & p2=$!
run_one 1 0.05 scale005 2027 & p3=$!
run_one 1 0.10 scale010 2026 & p4=$!
wait "$p1"
wait "$p2"
wait "$p3"
wait "$p4"
