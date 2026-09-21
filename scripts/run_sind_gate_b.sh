#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-/home/dell/YrM/envs/ICCT/bin/python}
"$PYTHON" -u experiments/gate_b/preflight.py
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u experiments/gate_b/run_pilot.py --seed 2026 --resume \
  --output reports/task_redesign/sind_gate_b_seed2026 \
  > reports/task_redesign/sind_gate_b_seed2026.log 2>&1 &
pid2026=$!
CUDA_VISIBLE_DEVICES=1 "$PYTHON" -u experiments/gate_b/run_pilot.py --seed 2027 --resume \
  --output reports/task_redesign/sind_gate_b_seed2027 \
  > reports/task_redesign/sind_gate_b_seed2027.log 2>&1 &
pid2027=$!
status=0
wait "$pid2026" || status=1
wait "$pid2027" || status=1
if [[ "$status" -ne 0 ]]; then
  echo "Gate B training failed; inspect seed logs and resume with this same command." >&2
  exit "$status"
fi
"$PYTHON" -u experiments/gate_b/paired_bootstrap.py
"$PYTHON" -u experiments/gate_b/summarize.py
