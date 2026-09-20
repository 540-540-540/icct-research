#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/js_cn/sensing/venv/bin/python}"
CAMPAIGN="${CAMPAIGN:-runs/raj_learning_curve_20260920}"
SEED="${SEED:-2026}"

cd "$ROOT"
mkdir -p "$CAMPAIGN"

for samples in 1000 2000 4096 8000; do
  for kind in quantum matched_classical; do
    run_dir="$CAMPAIGN/${samples}_${kind}"
    mkdir -p "$run_dir"
    if [ -f "$run_dir/summary.json" ] && grep -q '"status": "COMPLETED"' "$run_dir/summary.json"; then
      echo "SKIP completed $samples $kind"
      continue
    fi
    resume=()
    if [ -f "$run_dir/last.pt" ]; then resume=(--resume); fi
    echo "RUN samples=$samples kind=$kind seed=$SEED"
    "$PYTHON" scripts/train_raj_qgnn.py \
      --kind "$kind" --classical-hidden 42 --train-limit "$samples" \
      --epochs 20 --batch-size 32 --seed "$SEED" --max-seconds 7200 \
      --run-dir "$run_dir" "${resume[@]}" 2>&1 | tee -a "$run_dir/console.log"
  done
done

"$PYTHON" scripts/analyze_raj_learning_curve.py --campaign "$CAMPAIGN"
