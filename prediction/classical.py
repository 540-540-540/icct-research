"""GNN baseline: original attention core with a shared-input adapter."""
import torch
from torch import nn


def mlp(in_features, out_features=128):
    return nn.Sequential(nn.Linear(in_features, 128), nn.SiLU(), nn.Linear(128, out_features))


class ClassicalGNN(nn.Module):
    def __init__(self, layers=3):
        super().__init__()
        if layers not in (2, 3):
            raise ValueError('Protocol allows two or three graph layers')
        self.input = mlp(6)
        self.messages = nn.ModuleList(mlp(260) for _ in range(layers))
        self.updates = nn.ModuleList(mlp(256) for _ in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(128) for _ in range(layers))

    def forward(self, state_hat, standardized_state, track_exists, detected):
        if state_hat.shape != standardized_state.shape or state_hat.shape[-1] != 4:
            raise ValueError('States must have matching [...,N,4] shapes')
        if track_exists.shape != state_hat.shape[:-1] or detected.shape != track_exists.shape:
            raise ValueError('Mask shape mismatch')
        n = state_hat.shape[-2]
        if not 1 <= n <= 8:
            raise ValueError('One to eight slots are required')
        m = track_exists.bool()
        clean = torch.where(m[..., None], state_hat, 0)
        z = torch.where(m[..., None], standardized_state, 0)
        # Axis order is target i, source j throughout.
        delta = clean.unsqueeze(-3) - clean.unsqueeze(-2)
        edge = delta / delta.new_tensor([45., 45., 15., 15.])
        pair = m.unsqueeze(-1) & m.unsqueeze(-2) & ~torch.eye(n, device=m.device, dtype=torch.bool)
        weight = torch.exp(-delta[..., :2].square().sum(-1) / (2 * 45**2)) * pair
        h = self.input(torch.cat((z, detected[..., None].to(z), m[..., None].to(z)), -1))
        h = h * m[..., None]
        for message, update, norm in zip(self.messages, self.updates, self.norms):
            target = h.unsqueeze(-2).expand(*h.shape[:-2], n, n, 128)
            source = h.unsqueeze(-3).expand_as(target)
            u = (message(torch.cat((target, source, edge), -1)) * weight[..., None]).sum(-2) / 7
            h = norm(h + update(torch.cat((h, u), -1))) * m[..., None]
        return h

# The primary graph arm reuses the archived attention layers and SI edge builder.
# The A02 ClassicalGNN above remains an optional matched GNN+LLM comparator.
import importlib.util
import sys
from pathlib import Path

PLAIN_SOURCE = Path(__file__).resolve().parents[1] / 'code/00_remote_shared_dependencies/target_interaction_graph.py'
_source_name = '_icct_archived_plain_gnn'
if _source_name not in sys.modules:
    _spec = importlib.util.spec_from_file_location(_source_name, PLAIN_SOURCE)
    _source = importlib.util.module_from_spec(_spec)
    sys.modules[_source_name] = _source
    _spec.loader.exec_module(_source)
_source = sys.modules[_source_name]


class GNNGraph(nn.Module):
    """Primary graph arm: original attention core with the common six-input front end.

    Reuses the source's two 128-wide/four-head graph layers and SI edge features.
    The historical GRU, node projection, time embedding and decoder are excluded;
    both current graph arms use the same downstream TrajectoryPredictor.
    """
    def __init__(self):
        super().__init__()
        self.input = mlp(6)
        self.graph_layers = nn.ModuleList(_source.DenseEdgeGraphAttention(128, 4, .10) for _ in range(2))

    def forward(self, state_hat, standardized_state, track_exists, detected):
        if state_hat.shape != standardized_state.shape or state_hat.ndim < 3 or state_hat.shape[-1] != 4:
            raise ValueError('Expected matching [...,N,4] state tensors')
        if track_exists.shape != state_hat.shape[:-1] or detected.shape != track_exists.shape:
            raise ValueError('Mask shape mismatch')
        n = state_hat.shape[-2]
        if not 1 <= n <= 8:
            raise ValueError('One to eight slots are required')
        mask = track_exists.bool()
        physical = torch.where(mask[..., None], state_hat, 0)
        z = torch.where(mask[..., None], standardized_state, 0)
        if not torch.isfinite(physical).all() or not torch.isfinite(z).all():
            raise ValueError('Existing states must be finite')
        nodes = self.input(torch.cat((z, detected[..., None].to(z), mask[..., None].to(z)), -1))
        nodes = (nodes * mask[..., None]).reshape(-1, n, 128)
        physical = physical.reshape(-1, n, 4)
        current = mask.reshape(-1, n)
        edges, distances = _source.build_edge_features(physical[..., :2], physical[..., 2:])
        pairs = current[:, :, None] & current[:, None, :]
        adjacency = pairs & (distances <= 45.)
        adjacency |= torch.eye(n, device=state_hat.device, dtype=torch.bool)[None] & pairs
        for layer in self.graph_layers:
            nodes, _ = layer(nodes, edges, adjacency)
            nodes = nodes * current[..., None]
        return nodes.reshape(*state_hat.shape[:-1], 128)

