#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-/home/dell/YrM/envs/ICCT/bin/python}
pids=()
labels=()
index=0
for seed in 2026 2027; do
  for mode in fused_multiscale cross_order_multiscale; do
    gpu=$((index % 2))
    output="reports/task_redesign/sind_gate_b_plus_screen_${mode}_seed${seed}"
    CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" -u experiments/gate_b_plus/run_screen.py \
      --mode "$mode" --seed "$seed" --resume --output "$output" \
      > "${output}.log" 2>&1 &
    pids+=("$!"); labels+=("${mode}_seed${seed}_gpu${gpu}"); index=$((index + 1))
  done
done
status=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then echo "PASS ${labels[$i]}"; else echo "FAIL ${labels[$i]}" >&2; status=1; fi
done
exit "$status"
