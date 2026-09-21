#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
rank_runner_pid=${1:?rank runner PID required}
while kill -0 "$rank_runner_pid" 2>/dev/null; do sleep 30; done
for directory in reports/task_redesign/sind_gate_b_plus_rank_{control,rank001,rank005}_seed{2026,2027}; do
  if [[ ! -f "$directory/summary.json" ]]; then
    echo "missing $directory/summary.json"
    exit 1
  fi
done
exec bash scripts/run_gate_b_plus_next_mechanisms.sh
