#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-/home/dell/YrM/envs/ICCT/bin/python}
pids=()
index=0
for seed in 2026 2027; do
  for spec in "raw:0.0" "ema999:0.999"; do
    tag=${spec%%:*}; decay=${spec##*:}; gpu=$((index % 2))
    output="reports/task_redesign/sind_gate_b_plus_robust_${tag}_seed${seed}"
    CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" -u experiments/gate_b_plus/run_screen.py \
      --mode full_multiscale --seed "$seed" --epochs 30 --train-limit 36643 --ema-decay "$decay" \
      --resume --output "$output" > "${output}.log" 2>&1 &
    pids+=("$!"); index=$((index + 1))
  done
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
