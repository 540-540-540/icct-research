#!/usr/bin/env bash
# User-started training only; never part of the short preflight.
set -euo pipefail
cd "$(dirname "$0")/.."
python=/home/dell/YrM/envs/ICCT/bin/python
mode="${1:-start}"
run_root=results/qgnn/self_repair_v1
run_self() {
    local gpu="$1" dataset="$2" model="$3" batch="$4"
    local run_dir="$run_root/${dataset}_${model}_ego_v1_0db_seed2026"
    local -a command=(
        "$python" -u scripts/train_raj_residual_self.py
        --dataset "$dataset" --model "$model" --self-frame ego_v1
        --seed 2026 --snr 0 --epochs 20 --batch-size "$batch"
        --lr 3e-4 --token-weight 0.035 --save-steps 100
        --run-dir "$run_dir" --max-seconds 14400
    )
    if [[ "$mode" == resume && -f "$run_dir/last.pt" ]]; then
        command+=(--resume)
    fi
    printf '\nStarting GPU %s: %s / %s\n' "$gpu" "$dataset" "$model"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONDONTWRITEBYTECODE=1 "${command[@]}" \
        2>&1 | tee -a "$run_root/launch_logs/${dataset}_${model}.log"
    "$python" -c 'import json,sys; s=json.load(open(sys.argv[1])); print(s["status"],s["selected_checkpoint"]); sys.exit(0 if s["status"]=="COMPLETED" else 1)' \
        "$run_dir/summary.json"
}
if [[ "$mode" == queue0 || "$mode" == queue1 ]]; then
    queue="$mode"
    mode="$2"
    if [[ "$queue" == queue0 ]]; then
        run_self 0 legacy llm 32
        run_self 0 target llm 256
    else
        run_self 1 target tcn 256
        run_self 1 target transformer 256
        run_self 1 target lstm 256
    fi
    exit
fi
if [[ "$mode" != start && "$mode" != resume ]]; then
    echo "Usage: bash scripts/run_sind_self_repair.sh [start|resume]" >&2
    exit 2
fi
mkdir -p "$run_root/launch_logs"
exec 9>"$run_root/.launch.lock"
if ! flock -n 9; then
    echo "This experiment queue is already running. Check its terminal/logs." >&2
    exit 1
fi
# Each queue gets its own process group so interruption cannot hit unrelated jobs.
setsid bash scripts/run_sind_self_repair.sh queue0 "$mode" &
gpu0_job=$!
setsid bash scripts/run_sind_self_repair.sh queue1 "$mode" &
gpu1_job=$!
stop_queues() {
    trap - INT TERM
    echo "Stopping both queues; each trainer saves at the end of its current batch."
    kill -TERM -- "-$gpu0_job" "-$gpu1_job" 2>/dev/null || true
    wait "$gpu0_job" 2>/dev/null || true
    wait "$gpu1_job" 2>/dev/null || true
    exit 130
}
trap stop_queues INT TERM
failed=0
wait "$gpu0_job" || failed=1
wait "$gpu1_job" || failed=1
trap - INT TERM
if [[ "$failed" == 0 ]]; then
    echo "All Self runs completed. Review selected checkpoints before Interaction training."
else
    echo "A queue stopped. Inspect launch_logs and summary.json before resuming." >&2
fi
exit "$failed"
