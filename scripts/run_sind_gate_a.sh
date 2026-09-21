#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-/home/dell/YrM/envs/ICCT/bin/python}
"$PYTHON" -u experiments/gate_a/build_sind_benchmark.py
"$PYTHON" -u experiments/gate_a/preflight.py
"$PYTHON" -u experiments/gate_a/run_pilot.py --seed 2026 --output reports/task_redesign/sind_gate_a_seed2026
"$PYTHON" -u experiments/gate_a/run_pilot.py --seed 2027 --epochs 40 --output reports/task_redesign/sind_gate_a_seed2027
"$PYTHON" -u experiments/gate_a/paired_bootstrap.py
"$PYTHON" -u experiments/gate_a/summarize.py
