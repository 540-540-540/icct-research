#!/usr/bin/env bash
set -euo pipefail

pids=()
for seed in 2026 2027; do
  gpu=$((seed - 2026))
  out="reports/task_redesign/sind_gate_b_plus_robust_quantum_latent_attention_fw075_seed${seed}"
  CUDA_VISIBLE_DEVICES="$gpu" /home/dell/YrM/envs/ICCT/bin/python -u \
    experiments/gate_b_plus/run_screen.py \
    --mode quantum_latent_attention \
    --seed "$seed" \
    --epochs 50 \
    --train-limit 36643 \
    --ema-decay 0.999 \
    --fde-weight 0.75 \
    --resume \
    --output "$out" >> "${out}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
