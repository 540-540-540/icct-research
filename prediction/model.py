"""A04 primary model entry points, displayed only as QGNN and GNN."""
import json
from pathlib import Path
from torch import nn
from .quantum import QuantumGraph
from .classical import GNNGraph
from .temporal import TrajectoryPredictor

ROOT = Path(__file__).resolve().parents[1]


class QGNN(nn.Module):
    display_name = 'QGNN'

    def __init__(self, normalization, depth=3):
        super().__init__()
        self.graph = QuantumGraph(normalization['quantum_scale'], depth=depth)
        self.temporal = TrajectoryPredictor(24)

    def forward(self, state_hat, standardized_state, track_exists, detected):
        features = self.graph.forward_history(state_hat, track_exists)
        return self.temporal(state_hat, standardized_state, track_exists, detected, features)


class GNN(nn.Module):
    display_name = 'GNN'

    def __init__(self):
        super().__init__()
        self.graph = GNNGraph()
        self.temporal = TrajectoryPredictor(128)

    def forward(self, state_hat, standardized_state, track_exists, detected):
        features = self.graph(state_hat, standardized_state, track_exists, detected)
        return self.temporal(state_hat, standardized_state, track_exists, detected, features)


def build_model(name, normalization=None, device='cuda:0', depth=3):
    """The two canonical primary models; legacy comparators are never selected here."""
    if name.lower() == 'qgnn':
        if normalization is None:
            normalization = json.loads((ROOT/'data/f01d/normalization.json').read_text())
        model = QGNN(normalization, depth)
    elif name.lower() == 'gnn':
        model = GNN()
    else:
        raise ValueError('Primary model must be QGNN or GNN')
    return model.to(device)
