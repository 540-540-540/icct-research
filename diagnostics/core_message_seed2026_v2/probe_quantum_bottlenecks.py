"""Causal inference probes for the trained PennyLane graph checkpoint.

The 256 checkpoint-selection scenes and the remaining 735 development scenes
are reported separately.  No model is retrained and no test data are read.
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

import experiment_utils as r
import run as retained
from model import build_graph


ROOT = Path("/home/js_cn/sensing")
BASE = Path(__file__).resolve().parent
RESULTS = BASE / "converged_protocol_seed2026_v1"
SEED = 2026


class InputScaleCore(nn.Module):
    def __init__(self, core, scale):
        super().__init__()
        self.core = core
        self.scale = float(scale)

    def forward(self, angles):
        return self.core(angles * self.scale)


class OutputMaskCore(nn.Module):
    def __init__(self, core, keep):
        super().__init__()
        self.core = core
        self.register_buffer("keep", torch.tensor(keep, dtype=torch.float32))

    def forward(self, angles):
        output = self.core(angles)
        return output * self.keep.to(dtype=output.dtype, device=output.device)


class AffineReadout(nn.Module):
    def __init__(self, readout, scale=1.0, shift=0.0):
        super().__init__()
        self.readout = readout
        self.scale = float(scale)
        self.shift = float(shift)

    def forward(self, features):
        return self.readout(features) * self.scale + self.shift


def make_bank(states, masks, indices):
    return {
        "history": torch.from_numpy(states[indices, :20]).float().cuda(),
        "future": torch.from_numpy(states[indices, 20:]).float().cuda(),
        "mask": torch.from_numpy(masks[indices]).bool().cuda(),
    }


def build_variant(name):
    r.set_seed(SEED)
    model = build_graph("quantum", quantum_backend="pennylane").cuda()
    checkpoint = torch.load(
        RESULTS / "quantum_graph_selected.pt", map_location="cpu", weights_only=True
    )
    model.load_state_dict(checkpoint["state"], strict=True)
    layer = model.graph_layers[1]
    if name == "baseline":
        pass
    elif name == "angle_scale_0.50":
        layer.core = InputScaleCore(layer.core, 0.50)
    elif name == "angle_scale_0.75":
        layer.core = InputScaleCore(layer.core, 0.75)
    elif name == "uniform_attention":
        layer.score = AffineReadout(layer.score, scale=0.0)
    elif name == "score_scale_2":
        layer.score = AffineReadout(layer.score, scale=2.0)
    elif name == "score_scale_4":
        layer.score = AffineReadout(layer.score, scale=4.0)
    elif name == "neutral_gate":
        layer.gate = AffineReadout(layer.gate, scale=0.0)
    elif name == "gate_shift_minus_0.5":
        layer.gate = AffineReadout(layer.gate, shift=-0.5)
    elif name == "gate_shift_minus_1.0":
        layer.gate = AffineReadout(layer.gate, shift=-1.0)
    elif name == "only_single_Z":
        layer.core = OutputMaskCore(layer.core, [1] * 6 + [0] * 6)
    elif name == "only_neighbor_ZZ":
        layer.core = OutputMaskCore(layer.core, [0] * 6 + [1] * 6)
    elif name == "zero_quantum_features":
        layer.core = OutputMaskCore(layer.core, [0] * 12)
    else:
        raise ValueError(name)
    return model.eval(), int(checkpoint["epoch"])


def reduction(reference, candidate):
    return {
        metric: 100.0 * (reference[metric] - candidate[metric]) / reference[metric]
        for metric in ("ade_m", "fde_m")
    }


def main():
    with np.load(BASE / "retained_inputs/circuit_split_indices.npz", allow_pickle=False) as split:
        dev_indices = split["circuit_dev_indices"].copy()
    selection_indices = r.select_profile_indices(dev_indices, 256)
    selection_set = set(selection_indices.tolist())
    holdout_indices = np.asarray(
        [index for index in dev_indices.tolist() if index not in selection_set], dtype=np.int64
    )
    assert len(selection_indices) == 256 and len(holdout_indices) == len(dev_indices) - 256
    with np.load(ROOT / "data/multitarget_lankershim_v1.npz", allow_pickle=False) as data:
        states = data["train_states"].copy()
        masks = data["train_mask"].copy()
    banks = {
        "selection_256": make_bank(states, masks, selection_indices),
        "holdout_735": make_bank(states, masks, holdout_indices),
    }
    retained.NOISE = r.load_snr_noise_map(ROOT / "results/multitarget_snr/snr_calibration.json")
    variants = (
        "baseline",
        "angle_scale_0.50",
        "angle_scale_0.75",
        "uniform_attention",
        "score_scale_2",
        "score_scale_4",
        "neutral_gate",
        "gate_shift_minus_0.5",
        "gate_shift_minus_1.0",
        "only_single_Z",
        "only_neighbor_ZZ",
        "zero_quantum_features",
    )
    report = {
        "seed": SEED,
        "checkpoint": "quantum_graph_selected.pt",
        "checkpoint_selection_scenes": int(len(selection_indices)),
        "probe_holdout_scenes": int(len(holdout_indices)),
        "no_retraining": True,
        "no_test": True,
        "variants": {},
    }
    for name in variants:
        model, epoch = build_variant(name)
        values = {}
        for split_name, bank in banks.items():
            evaluation, _ = retained.evaluate(model, bank)
            values[split_name] = evaluation
        report["variants"][name] = {"checkpoint_epoch": epoch, **values}
        print(json.dumps({"variant": name, **values}, ensure_ascii=False), flush=True)
        del model
        torch.cuda.empty_cache()
    for name, values in report["variants"].items():
        if name == "baseline":
            continue
        values["reduction_vs_baseline_percent"] = {
            split_name: reduction(
                report["variants"]["baseline"][split_name]["aggregate"],
                values[split_name]["aggregate"],
            )
            for split_name in banks
        }
    output = RESULTS / "quantum_bottleneck_causal_probes.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": "completed", "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
