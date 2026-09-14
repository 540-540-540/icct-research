"""Evaluate an interrupted attention-routed run from its selected checkpoint."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/js_cn/sensing")
BASE = ROOT / "diagnostics/core_message_seed2026_v2"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(BASE))

import experiment_utils as utils
import run as retained
from run_attention_routed_fusion import build_pair, load_banks, restore_routed


def main():
    output_dir = BASE / "attention_routed_qgnn_llm_seed2026_v1"
    retained.NOISE = utils.load_snr_noise_map(ROOT / "results/multitarget_snr/snr_calibration.json")
    _, _, dev_bank, _, _, dev_indices = load_banks()
    _, model, _, _ = build_pair("quantum")
    checkpoint = torch.load(output_dir / "attention_routed_selected.pt", map_location="cpu", weights_only=True)
    restore_routed(model, checkpoint["state"])
    model = model.cuda().eval()
    model.attention_ablation_mode = "learned"
    learned, arrays = retained.evaluate(model, dev_bank, collect=True)
    np.savez_compressed(output_dir / "attention_routed_full_dev.npz", **arrays, indices=dev_indices)
    model.attention_ablation_mode = "uniform"; uniform, _ = retained.evaluate(model, dev_bank)
    model.attention_ablation_mode = "off"; off, _ = retained.evaluate(model, dev_bank)
    summary = {
        "status": "stopped_after_epoch_3_and_evaluated_selected",
        "selected_epoch": int(checkpoint["epoch"]),
        "selection": checkpoint["metrics"],
        "full_dev_learned_attention": learned,
        "full_dev_uniform_attention": uniform,
        "full_dev_interaction_off": off,
        "no_test": True,
    }
    utils.atomic_json(output_dir / "summary.json", summary)
    utils.atomic_json(output_dir / "completed.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
