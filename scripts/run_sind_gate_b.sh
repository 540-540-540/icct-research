#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-/home/dell/YrM/envs/ICCT/bin/python}
"$PYTHON" -u experiments/gate_b/preflight.py
"$PYTHON" -u experiments/gate_b/run_pilot.py --seed 2026 --output reports/task_redesign/sind_gate_b_seed2026
"$PYTHON" -u experiments/gate_b/run_pilot.py --seed 2027 --output reports/task_redesign/sind_gate_b_seed2027
