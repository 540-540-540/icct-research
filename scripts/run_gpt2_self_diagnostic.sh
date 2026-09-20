#!/usr/bin/env bash
# Five independent diagnostic runs. User starts long training with this command.
set -euo pipefail
cd "$(dirname "$0")/.."
python=/home/dell/YrM/envs/ICCT/bin/python
run_root=results/qgnn/gpt2_self_diagnostic_v1
mode="${1:-run}"
command_for() {
    local name="$1" kind="$2" frame="$3" init="$4" adaptation="$5"
    command=("$python" -u -m experiments.gpt2_self_diagnostic.run
        --model "$kind" --frame "$frame" --initialization "$init" --adaptation "$adaptation"
        --seed 2026 --epochs 20 --batch-size 256 --lr 3e-4 --backbone-lr 7.5e-5
        --save-steps 500 --max-seconds 21600 --run-dir "$run_root/$name")
    if [[ -f "$run_root/$name/last.pt" ]]; then command+=(--resume); fi
}
run_worker() {
    local gpu="$1" name="$2"
    shift 2
    command_for "$name" "$@"
    echo "GPU $gpu: $name"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONDONTWRITEBYTECODE=1 "${command[@]}" \
        2>&1 | tee -a "$run_root/launch_logs/$name.log"
    "$python" -c 'import json,sys; s=json.load(open(sys.argv[1])); print(s["status"],s["selected_checkpoint"]); sys.exit(0 if s["status"]=="COMPLETED" else 1)' "$run_root/$name/summary.json"
}
if [[ "$mode" == worker ]]; then shift; run_worker "$@"; exit; fi
if [[ "$mode" != run && "$mode" != --dry-run ]]; then
    echo "Usage: bash scripts/run_gpt2_self_diagnostic.sh [--dry-run]" >&2; exit 2
fi
# All five start together; two GPT jobs per 4090. TCN uses the spare margin.
jobs=(
    "0 tcn_ego tcn ego pretrained lora"
    "0 gpt_heading_pretrained_lora llm ego_heading pretrained lora"
    "0 gpt_heading_pretrained_full llm ego_heading pretrained full"
    "1 gpt_heading_random_lora llm ego_heading random lora"
    "1 gpt_heading_random_full llm ego_heading random full"
)
if [[ "$mode" == --dry-run ]]; then
    for job in "${jobs[@]}"; do
        read -r gpu name kind frame init adaptation <<< "$job"
        command_for "$name" "$kind" "$frame" "$init" "$adaptation"
        printf 'CUDA_VISIBLE_DEVICES=%s ' "$gpu"; printf '%q ' "${command[@]}"; printf '\n'
    done
    exit
fi
mkdir -p "$run_root/launch_logs"
exec 9>"$run_root/.launch.lock"
if ! flock -n 9; then echo "This diagnostic batch is already running or saving." >&2; exit 1; fi
pids=()
stop_jobs() {
    trap - INT TERM
    echo "Stopping workers after their current batch and saving full checkpoints."
    for pid in "${pids[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done
    for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
    exit 130
}
trap stop_jobs INT TERM
for job in "${jobs[@]}"; do
    read -r gpu name kind frame init adaptation <<< "$job"
    setsid bash scripts/run_gpt2_self_diagnostic.sh worker "$gpu" "$name" "$kind" "$frame" "$init" "$adaptation" &
    pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
trap - INT TERM
if [[ "$failed" != 0 ]]; then
    echo "A run stopped or failed. Inspect launch_logs; the same command resumes saved runs." >&2
    exit 1
fi
"$python" - <<'PY'
import json
from pathlib import Path
root = Path("results/qgnn")
rows = [
    ("reference_tcn_global", root/"self_repair_v1/target_tcn_ego_v1_0db_seed2026/summary.json"),
    ("reference_gpt_ego_lora", root/"self_repair_v1/target_llm_ego_v1_0db_seed2026/summary.json"),
]
rows += [(p.parent.name, p) for p in sorted((root/"gpt2_self_diagnostic_v1").glob("*/summary.json"))]
print("\nSelected checkpoints: origin-macro ADE / FDE in meters; reused validation diagnosis.")
for name,p in rows:
    s=json.loads(p.read_text()); b=s["selected_checkpoint"]
    print(f"{name:32s} {s['status']:10s} epoch={b['epoch']:2d} ADE={b['ADE']:.6f} FDE={b['FDE']:.6f}")
PY
