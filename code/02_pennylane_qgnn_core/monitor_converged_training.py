"""Compact live terminal monitor for the convergence-controlled experiment."""
import argparse
import json
import subprocess
import time
from pathlib import Path


BASE = Path(__file__).resolve().parent


def gpu_status():
    command = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        values = subprocess.check_output(command, text=True, timeout=3).strip().split(", ")
        return f"GPU {values[0]}% | VRAM {values[1]}/{values[2]} MiB | {values[3]} C | {values[4]} W"
    except Exception as error:
        return f"GPU status unavailable: {error}"


def render(output_dir):
    progress_path = output_dir / "progress.json"
    if not progress_path.exists():
        return f"Waiting for {progress_path}\n{gpu_status()}"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    lines = [
        "QGNN convergence-controlled training (seed 2026)",
        time.strftime("Updated: %Y-%m-%d %H:%M:%S"),
        gpu_status(),
        "-" * 78,
    ]
    if progress.get("status") == "completed":
        lines.append("STATUS: COMPLETED")
        for arm, result in progress.get("summaries", {}).items():
            graph = result["graph"]["aggregate"]
            llm = result["llm"]["aggregate"]
            lines.append(
                f"{arm:9s} graph ADE/FDE {graph['ade_m']:.6f}/{graph['fde_m']:.6f} | "
                f"LLM ADE/FDE {llm['ade_m']:.6f}/{llm['fde_m']:.6f}"
            )
        return "\n".join(lines)
    last = progress.get("last", {})
    metrics = last.get("metrics", {}).get("aggregate", {})
    lines.extend(
        [
            f"STATUS   : {progress.get('status', 'unknown')}",
            f"ARM/PHASE: {last.get('arm', '-')} / {last.get('phase', '-')}",
            f"EPOCH    : {last.get('epoch', 0)}/{last.get('max_epochs', '-')}  "
            f"early-stop wait {last.get('wait', 0)}/{last.get('patience', '-')}",
            f"LOSS     : {last.get('train_loss', float('nan')):.6f}",
            f"DEV ADE  : {metrics.get('ade_m', float('nan')):.6f} m",
            f"DEV FDE  : {metrics.get('fde_m', float('nan')):.6f} m",
            f"SCORE    : {last.get('score', float('nan')):.6f}",
            f"BEST     : epoch {last.get('best_epoch', '-')} | score {last.get('best_score', float('nan')):.6f}",
            f"EPOCH TIME: {last.get('seconds', float('nan')):.1f} s",
        ]
    )
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        lines.extend(["-" * 78, "COMPLETED ARMS"])
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for arm, result in summary.items():
            llm = result["llm"]["aggregate"]
            lines.append(f"{arm:9s} LLM ADE {llm['ade_m']:.6f} | FDE {llm['fde_m']:.6f}")
    lines.append("Ctrl+C exits this monitor only; remote training continues.")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-name", default="converged_protocol_seed2026_v1")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=2.0)
    args = parser.parse_args()
    output_dir = BASE / args.output_name
    while True:
        print("\033[2J\033[H" + render(output_dir), flush=True)
        if not args.watch:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
