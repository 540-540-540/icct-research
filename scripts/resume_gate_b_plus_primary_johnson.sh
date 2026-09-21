#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-/home/dell/YrM/envs/ICCT/bin/python}
pids=()
for spec in "matched_johnson:johnson:2026:0" "matched_johnson:johnson:2027:1" \
            "large_johnson:johnson_large:2026:0" "large_johnson:johnson_large:2027:1"; do
  IFS=: read -r label core seed gpu <<< "$spec"
  output="reports/task_redesign/sind_gate_b_plus_primary_${label}_seed${seed}"
  CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" -u experiments/gate_b_plus/run_screen.py \
    --mode horizon_multiscale --core-kind "$core" --seed "$seed" --order-seed 20260921 \
    --epochs 50 --patience 8 --train-limit 36643 --ema-decay 0.999 --fde-weight 0.75 \
    --resume --output "$output" > "${output}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
