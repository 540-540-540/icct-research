from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.gate_a.data import GateADataset
from experiments.gate_b_plus.diagnose_full_multiscale import relation_stats, spectrum_stats
from experiments.gate_b_plus.models import GateBPlusQuantumModel


def atomic_json(path: Path, payload: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)


def diagnose(checkpoint: Path, loader: DataLoader, device: torch.device) -> dict:
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model = GateBPlusQuantumModel(state["mode"], core_kind=state.get("core_kind", "quantum")).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    values: dict[str, list[np.ndarray]] = defaultdict(list)
    with torch.no_grad():
        for batch in loader:
            _, latents = model(batch["history_state"].to(device), batch["node_mask"].to(device),
                               return_aux=True)
            select = (batch["k"] >= 1).numpy()
            for name, value in latents.items():
                values[name].append(value.cpu().numpy()[select])
    merged = {name: np.concatenate(rows) for name, rows in values.items()}
    return {
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_epoch": state["epoch"],
        "representations_ic": {name: spectrum_stats(value) for name, value in merged.items()},
        "relations_ic": {
            "j2_vs_j3": relation_stats(merged["j2"], merged["j3"]),
            "q2_vs_q3": relation_stats(merged["q2"], merged["q3"]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=CHECKPOINT")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    cfg = json.loads((ROOT / "configs/gate_b_plus_screen.json").read_text())
    loader = DataLoader(GateADataset(ROOT / cfg["benchmark"], "val"), batch_size=32,
                        shuffle=False, num_workers=2)
    result = {"status": "DEVELOPMENT_MECHANISM_DIAGNOSTIC_NOT_FORMAL_EVIDENCE",
              "test_accessed": False, "runs": {}}
    for spec in args.run:
        label, path = spec.split("=", 1)
        result["runs"][label] = diagnose(ROOT / path, loader, torch.device("cuda:0"))
    output = ROOT / args.output
    atomic_json(output, result)
    print(json.dumps({"output": str(output), "runs": list(result["runs"])}, indent=2))


if __name__ == "__main__":
    main()
