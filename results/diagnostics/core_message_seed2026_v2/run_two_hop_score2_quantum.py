"""Score-sharpened quantum run on the frozen high-two-hop protocol."""
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import torch

import experiment_utils as r
import run as retained
from model import CoreMessageLayer
from run_converged_training import train_stage
from run_multitarget_experiment import prediction_loss
from target_interaction_graph import ForecasterConfig, TargetInteractionGNN


ROOT = Path("/home/js_cn/sensing")
BASE = Path(__file__).resolve().parent
REFERENCE = BASE / "two_hop_sparse_radius20_seed2026_v1"
OUTPUT = BASE / "two_hop_score2_quantum_seed2026_v1"
DATA = ROOT / "data/multitarget_lankershim_v1.npz"
SEED = 2026


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Score2QuantumLayer(CoreMessageLayer):
    def __init__(self, config):
        super().__init__(config, "quantum", quantum_backend="pennylane")

    def forward(self, nodes, edge_features, adjacency):
        b, n, h = nodes.shape
        pairs = torch.cat([
            nodes[:, :, None, :].expand(-1, -1, n, -1),
            nodes[:, None, :, :].expand(-1, n, -1, -1),
            edge_features,
        ], -1)
        selected = adjacency.reshape(-1).nonzero().squeeze(-1)
        x = pairs.reshape(-1, 2 * h + 7).index_select(0, selected)
        angles = math.pi * torch.tanh(self.encoder(self.input_norm(x)))
        z = self.latent_norm(self.core(angles))
        value = self.value(self.value_norm(nodes)).view(b, n, self.heads, self.head_dim)
        edge_value = self.edge_value(edge_features).view(b, n, n, self.heads, self.head_dim)
        base_msg = value[:, None, :, :, :] + edge_value
        score = nodes.new_zeros(b * n * n, self.heads).index_copy(
            0, selected, 2.0 * self.score(z)
        ).view(b, n, n, self.heads)
        gate = nodes.new_zeros(b * n * n, self.heads).index_copy(
            0, selected, self.gate(z)
        ).view(b, n, n, self.heads)
        gate = 2.0 * torch.sigmoid(gate)
        attention = torch.softmax(score.masked_fill(~adjacency[..., None], -1e4), dim=2)
        aggregate = (self.dropout(attention)[..., None] * gate[..., None] * base_msg).sum(2).reshape(b, n, h)
        nodes = self.output_norm(nodes + self.dropout(aggregate))
        nodes = self.final_norm(nodes + self.dropout(self.ffn(nodes)))
        return nodes, attention.mean(-1)


def build_model(score2):
    torch.manual_seed(SEED)
    config = ForecasterConfig(graph_radius_m=20.0)
    model = TargetInteractionGNN(config)
    torch.manual_seed(SEED + 1)
    model.graph_layers[1] = Score2QuantumLayer(config) if score2 else CoreMessageLayer(
        config, "quantum", quantum_backend="pennylane"
    )
    return model


def bank(states, masks, indices):
    return {
        "history": torch.from_numpy(states[indices, :20]).float().cuda(),
        "future": torch.from_numpy(states[indices, 20:]).float().cuda(),
        "mask": torch.from_numpy(masks[indices]).bool().cuda(),
    }


def main():
    assert Path.cwd() == ROOT and platform.node() == "jscn"
    assert torch.cuda.is_available() and "4090" in torch.cuda.get_device_name(0)
    OUTPUT.mkdir(exist_ok=False)
    reference_protocol = json.loads((REFERENCE / "protocol.json").read_text(encoding="utf-8"))
    train_indices = np.asarray(reference_protocol["train_indices"], dtype=np.int64)
    selection_indices = np.asarray(reference_protocol["selection_indices"], dtype=np.int64)
    confirmation_indices = np.asarray(reference_protocol["confirmation_indices"], dtype=np.int64)
    with np.load(DATA, allow_pickle=False) as data:
        states = data["train_states"].copy()
        masks = data["train_mask"].copy()
    train = bank(states, masks, train_indices)
    selection = bank(states, masks, selection_indices)
    confirmation = bank(states, masks, confirmation_indices)
    retained.NOISE = r.load_snr_noise_map(ROOT / "results/multitarget_snr/snr_calibration.json")

    # Prove that this is a strict score-only intervention: the quantum core is identical.
    r.set_seed(SEED)
    baseline = build_model(False).cuda().eval()
    r.set_seed(SEED)
    candidate = build_model(True).cuda().eval()
    angles = torch.linspace(-math.pi, math.pi, 66, device="cuda").reshape(11, 6)
    with torch.no_grad():
        parity_error = float((baseline.graph_layers[1].core(angles)
                              - candidate.graph_layers[1].core(angles)).abs().max())
    assert parity_error == 0.0
    del baseline
    candidate.train()
    prediction = candidate(train["history"][:2], train["mask"][:2])
    loss = prediction_loss(prediction, train["future"][:2], train["mask"][:2], graph_weighting=True)
    loss.backward()
    core_gradient = max(
        float(p.grad.abs().max())
        for p in candidate.graph_layers[1].core.parameters() if p.grad is not None
    )
    assert core_gradient > 0
    smoke = {
        "baseline_core_initial_parity_max_error": parity_error,
        "core_gradient_max": core_gradient,
        "loss": float(loss.detach()),
        "parameters": sum(p.numel() for p in candidate.parameters()),
    }
    del candidate
    torch.cuda.empty_cache()

    source_paths = [Path(__file__), BASE / "model.py",
                    BASE / "run_converged_training.py", ROOT / "target_interaction_graph.py"]
    protocol = {
        "status": "frozen_before_training",
        "seed": SEED,
        "reference_protocol": str(REFERENCE / "protocol.json"),
        "reference_protocol_sha256": sha256(REFERENCE / "protocol.json"),
        "same_train_selection_confirmation_indices_as_reference": True,
        "intervention": {"attention_score_scale": 2.0, "only_change": True},
        "graph": {"max_epochs": 40, "patience": 6, "min_delta": 1e-4},
        "smoke": smoke,
        "no_llm": True,
        "no_test": True,
        "gpu": torch.cuda.get_device_name(0),
        "python": sys.executable,
        "source_hashes": {str(path): sha256(path) for path in source_paths},
    }
    r.atomic_json(OUTPUT / "protocol.json", protocol)

    expected = {}
    with (REFERENCE / "quantum_radius20_graph.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            expected[f"graph_{record['epoch']}"] = record["input_sha256"]
    r.set_seed(SEED)
    model = build_model(True).cuda()
    audit = train_stage(model, "quantum_score2_radius20", "graph", 40, 6, train, selection, OUTPUT, expected)
    confirmation_metrics, arrays = retained.evaluate(model, confirmation, collect=True)
    np.savez_compressed(
        OUTPUT / "quantum_score2_radius20_graph_confirmation.npz",
        **arrays, indices=confirmation_indices,
    )
    for path, digest in protocol["source_hashes"].items():
        assert sha256(path) == digest
    completed = {
        "status": "completed",
        "seed": SEED,
        "confirmation": confirmation_metrics,
        "selected_epoch": audit["selected_epoch"],
        "executed_epochs": audit["executed_epochs"],
        "stop_reason": audit["stop_reason"],
        "parameters": sum(p.numel() for p in model.parameters()),
        "source_hashes_verified": True,
        "same_input_hashes_as_reference_verified": True,
        "no_llm": True,
        "no_test": True,
    }
    r.atomic_json(OUTPUT / "completed.json", completed)
    r.atomic_json(OUTPUT / "progress.json", completed)
    print(json.dumps(completed, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
