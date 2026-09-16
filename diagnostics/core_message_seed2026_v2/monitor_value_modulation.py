"""Compact live status for the value-modulation experiment."""
import json
import time
from pathlib import Path

base = Path(__file__).resolve().parent / "value_modulation_seed2026_v1"
while True:
    progress = base / "progress.json"
    completed = base / "completed.json"
    if progress.exists():
        payload = json.loads(progress.read_text(encoding="utf-8"))
        last = payload.get("last")
        if last:
            metrics = last["metrics"]["aggregate"]
            print(
                f"{last['arm']:17s} epoch {last['epoch']:02d}/{last['max_epochs']:02d} "
                f"loss={last['train_loss']:.6f} ADE={metrics['ade_m']:.6f} "
                f"FDE={metrics['fde_m']:.6f} best={last['best_epoch']:02d} "
                f"wait={last['wait']}/{last['patience']} time={last['seconds']:.1f}s",
                flush=True,
            )
    if completed.exists():
        print(completed.read_text(encoding="utf-8"), flush=True)
        break
    time.sleep(10)
