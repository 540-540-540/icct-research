#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
pipeline_pid=${1:?pipeline PID required}
while kill -0 "$pipeline_pid" 2>/dev/null; do sleep 30; done
if compgen -G "reports/task_redesign/sind_gate_b_plus_ranknorm_*_seed*/summary.json" >/dev/null; then
  exit 0
fi
if compgen -G "reports/task_redesign/sind_gate_b_plus_ranknorm_*_seed*" >/dev/null; then
  echo "ranknorm output exists without completed summary"
  exit 1
fi
exec bash scripts/run_gate_b_plus_ranknorm.sh
