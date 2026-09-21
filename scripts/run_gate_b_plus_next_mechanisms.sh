#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-/home/dell/YrM/envs/ICCT/bin/python}
rank_args=()
for label in control rank001 rank005; do
  for seed in 2026 2027; do
    rank_args+=(--run "${label}_${seed}=reports/task_redesign/sind_gate_b_plus_rank_${label}_seed${seed}/best.pt")
  done
done
CUDA_VISIBLE_DEVICES=0 "$PYTHON" experiments/gate_b_plus/diagnose_latent_rank.py \
  "${rank_args[@]}" --output reports/task_redesign/SIND_GATE_B_LATENT_RANK_DIAGNOSTIC_20260921.json || exit 1
pids=()
for runner in \
  scripts/run_gate_b_plus_quantum_latent_attention_fixed.sh \
  scripts/run_gate_b_plus_multiscale_quantum_attention.sh \
  scripts/run_gate_b_plus_horizon_adaptive.sh; do
  bash "$runner" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if [[ "$status" -ne 0 ]]; then exit "$status"; fi
bash scripts/run_gate_b_plus_rank_finetune.sh || exit 1
exec bash scripts/run_gate_b_plus_ranknorm.sh
