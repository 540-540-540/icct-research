#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
python=/home/dell/YrM/envs/ICCT/bin/python
cd "$root"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
if [[ ! -f data/task_redesign/lankershim_gate_a_v1/manifest.json ]]; then
  "$python" -u experiments/gate_a/build_benchmark.py
fi
"$python" -u experiments/gate_a/preflight.py
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$python" -u experiments/gate_a/run_pilot.py --seed 2026
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$python" -u experiments/gate_a/run_pilot.py --seed 2027
"$python" -u experiments/gate_a/summarize.py
