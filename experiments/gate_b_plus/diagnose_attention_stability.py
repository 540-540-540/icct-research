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
from experiments.gate_b_plus.diagnose_full_multiscale import add_errors, decode, relation_stats, summarize_errors
from experiments.gate_b_plus.models import GateBPlusQuantumModel


def inspect(checkpoint: Path, loader: DataLoader, device: torch.device) -> dict:
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model = GateBPlusQuantumModel(state["mode"]).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    errors = {"ade": defaultdict(list), "fde": defaultdict(list)}
    stats = {f"{order}_{name}": [] for order in ("j2", "j3")
             for name in ("entropy", "max_weight", "attention_mean_cosine")}
    pooled = {name: [] for name in ("j2_attention", "j3_attention")}
    with torch.no_grad():
        for batch_cpu in loader:
            history = batch_cpu["history_state"].to(device)[:, :, :8]
            mask = batch_cpu["node_mask"].to(device)[:, :8]
            history = torch.where(mask[:, None, :, None], history, 0.0)
            branch = []
            for order, nodes, message, attention in zip(
                    ("j2", "j3"), (model.core.j2(history, mask), model.core.j3(history, mask)),
                    model.branch_messages, model.branch_attentions):
                neighbor_mask = mask[:, 1:]
                neighbors = nodes[:, 1:]
                target = nodes[:, :1].expand_as(neighbors)
                messages = message(torch.cat((target, neighbors), -1))
                scores = attention(messages).squeeze(-1).masked_fill(~neighbor_mask, -1e4)
                weights = torch.softmax(scores, 1) * neighbor_mask
                attention_pool = (messages * weights[..., None]).sum(1)
                count = neighbor_mask.sum(1, keepdim=True).clamp_min(1)
                mean_pool = (messages * neighbor_mask[..., None]).sum(1) / count
                valid = (batch_cpu["k"] >= 1).numpy()
                n = count.squeeze(1)
                entropy = -(weights * weights.clamp_min(1e-8).log()).sum(1)
                entropy = torch.where(n > 1, entropy / n.float().log(), torch.ones_like(entropy))
                cosine = torch.nn.functional.cosine_similarity(attention_pool, mean_pool)
                stats[f"{order}_entropy"].extend(entropy.cpu().numpy()[valid].tolist())
                stats[f"{order}_max_weight"].extend(weights.max(1).values.cpu().numpy()[valid].tolist())
                stats[f"{order}_attention_mean_cosine"].extend(cosine.cpu().numpy()[valid].tolist())
                pooled[f"{order}_attention"].append(attention_pool.cpu().numpy()[valid])
                branch.append((attention_pool, mean_pool))
            temporal = model.encoder(history)[:, 0]
            variants = {
                "attention": (branch[0][0], branch[1][0]),
                "j2_mean": (branch[0][1], branch[1][0]),
                "j3_mean": (branch[0][0], branch[1][1]),
                "both_mean": (branch[0][1], branch[1][1]),
            }
            for name, pools in variants.items():
                prediction = decode(model, history, temporal, model.interaction(torch.cat(pools, -1)))
                add_errors(errors, name, prediction, batch_cpu)
    j2 = np.concatenate(pooled["j2_attention"])
    j3 = np.concatenate(pooled["j3_attention"])
    return {
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_epoch": state["epoch"],
        "ic4_ablation": {name: values["ic"]["4s"] for name, values in summarize_errors(errors).items()},
        "attention": {name: {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1))}
                      for name, values in stats.items()},
        "j2_j3_pooled_relation": relation_stats(j2, j3),
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
    result = {"status": "DEVELOPMENT_DIAGNOSTIC_NOT_FORMAL_EVIDENCE", "test_accessed": False,
              "runs": {}}
    for spec in args.run:
        label, path = spec.split("=", 1)
        result["runs"][label] = inspect(ROOT / path, loader, torch.device("cuda:0"))
    output = ROOT / args.output
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(output), "runs": list(result["runs"])}, indent=2))


if __name__ == "__main__":
    main()
