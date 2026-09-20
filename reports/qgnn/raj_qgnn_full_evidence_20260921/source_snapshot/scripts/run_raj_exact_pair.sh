#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/js_cn/sensing/venv/bin/python}"
SEED="${SEED:-2027}"
SAMPLES="${SAMPLES:-4096}"
EPOCHS="${EPOCHS:-20}"
OUT="${OUT:-runs/raj_seed${SEED}_exact}"

cd "$ROOT"
mkdir -p "$OUT"

for kind in quantum matched_classical; do
  run_dir="$OUT/$kind"
  mkdir -p "$run_dir"
  "$PYTHON" scripts/train_raj_qgnn.py \
    --kind "$kind" --classical-hidden 42 --train-limit "$SAMPLES" \
    --epochs "$EPOCHS" --batch-size 32 --seed "$SEED" --max-seconds 7200 \
    --run-dir "$run_dir" 2>&1 | tee "$OUT/$kind.log"
done
