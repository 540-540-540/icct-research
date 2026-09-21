#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-/home/dell/YrM/envs/ICCT/bin/python}
pids=()
for seed_gpu in "2026:0" "2027:1"; do
  seed=${seed_gpu%%:*}
  gpu=${seed_gpu##*:}
  source="reports/task_redesign/sind_gate_b_plus_stage2_restart_control_seed${seed}/best.pt"
  for mode in horizon_multiscale horizon_kinematic; do
    label=$([ "$mode" = horizon_multiscale ] && echo fixed_dropout_control || echo quantum_kinematic)
    output="reports/task_redesign/sind_gate_b_plus_stage3_${label}_seed${seed}"
    CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" -u experiments/gate_b_plus/run_screen.py \
      --mode "$mode" --seed "$seed" --order-seed 20260921 --dropout-seed 20260921 \
      --epochs 12 --patience 6 --train-limit 36643 --ema-decay 0.999 --fde-weight 0.75 \
      --warmstart-quantum "$source" --output "$output" > "${output}.log" 2>&1 &
    pids+=("$!")
  done
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
