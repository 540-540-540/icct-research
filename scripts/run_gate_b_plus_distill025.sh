#!/usr/bin/env bash
set -euo pipefail

pids=()
for seed in 2026 2027; do
  gpu=$((seed - 2026))
  teacher="reports/task_redesign/sind_gate_b_plus_robust_all_graph_seed${seed}/best.pt"
  out="reports/task_redesign/sind_gate_b_plus_robust_distill025_fw075_seed${seed}"
  CUDA_VISIBLE_DEVICES="$gpu" /home/dell/YrM/envs/ICCT/bin/python -u \
    experiments/gate_b_plus/run_screen.py \
    --mode full_multiscale \
    --seed "$seed" \
    --order-seed 20260921 \
    --epochs 30 \
    --train-limit 36643 \
    --ema-decay 0.999 \
    --fde-weight 0.75 \
    --warmstart-classical "$teacher" \
    --distill-teacher "$teacher" \
    --distill-weight 0.25 \
    --resume \
    --output "$out" > "${out}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
