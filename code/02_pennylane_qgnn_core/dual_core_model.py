"""Matched two-layer classical and PennyLane message-core graph models."""
import torch
from torch import nn

from model import CoreMessageLayer
from target_interaction_graph import ForecasterConfig, TargetInteractionGNN


def reinitialize_first_core(layer, kind, seed=2029):
    """Give the added first-layer core independent deterministic parameters."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if kind == "quantum":
            nn.init.uniform_(layer.core.weights, -0.08, 0.08)
        elif kind == "classical":
            for module in layer.core.modules():
                if isinstance(module, nn.Linear):
                    module.reset_parameters()
        else:
            raise ValueError(kind)


def build_dual_graph(arm, seed=2026):
    if arm not in ("classical_dual", "quantum_dual"):
        raise ValueError(arm)
    kind = arm[:-5]
    backend = "pennylane" if kind == "quantum" else "torch"
    torch.manual_seed(seed)
    config = ForecasterConfig()
    model = TargetInteractionGNN(config)
    assert len(model.graph_layers) == 2

    # Construct layer 2 first with exactly the retained one-layer seed so its
    # complete initialization matches the existing experiment.
    torch.manual_seed(seed + 1)
    second = CoreMessageLayer(config, kind, quantum_backend=backend)

    # The added layer gets independent readouts and an independently seeded core.
    torch.manual_seed(seed + 3)
    first = CoreMessageLayer(config, kind, quantum_backend=backend)
    reinitialize_first_core(first, kind, seed=seed + 3)
    model.graph_layers[0] = first
    model.graph_layers[1] = second
    return model
