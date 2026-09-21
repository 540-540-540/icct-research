#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-/home/dell/YrM/envs/ICCT/bin/python}
pids=()
for seed_gpu in "2026:0" "2027:1"; do
  seed=${seed_gpu%%:*}
  gpu=${seed_gpu##*:}
  output="reports/task_redesign/sind_gate_b_plus_residual_multiscale_attention_control_avg5_seed${seed}"
  CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" -u experiments/gate_b_plus/run_screen.py \
    --mode residual_multiscale_quantum_attention --seed "$seed" \
    --order-seed 20260921 --dropout-seed 20260921 --epochs 40 --patience 8 \
    --checkpoint-average-k 5 --train-limit 36643 --ema-decay 0.999 \
    --fde-weight 0.75 --latent-rank-weight 0.0 --output "$output" \
    > "${output}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
