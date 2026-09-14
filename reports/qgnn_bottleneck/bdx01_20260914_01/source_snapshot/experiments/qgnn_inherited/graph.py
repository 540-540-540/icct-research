"""Archived dual QGNN layers adapted to the current per-frame graph interface."""
import importlib.util
import sys
from pathlib import Path

import torch
from torch import nn
import pennylane as qml


ROOT = Path(__file__).resolve().parents[2]


def _load(name, path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(name, None)
            raise
    return sys.modules[name]


_base = _load('_icct_inherited_graph_base', ROOT / 'code/00_remote_shared_dependencies/target_interaction_graph.py')
# The archive hard-codes its former server paths. Bind its actual dependency only
# during import and restore both global import surfaces immediately afterwards.
_previous = sys.modules.get('target_interaction_graph')
_paths = sys.path[:]
try:
    sys.modules['target_interaction_graph'] = _base
    _archive = _load('_icct_inherited_dual_qgnn', ROOT / 'code/02_pennylane_qgnn_core/physics_aligned_dual_model.py')
finally:
    sys.path[:] = _paths
    if _previous is None:
        sys.modules.pop('target_interaction_graph', None)
    else:
        sys.modules['target_interaction_graph'] = _previous


class _ReplicaSafePQC(_archive.DualAxisPennyLanePQC):
    """Retain the archived weights/forward, with a private simulator per replica."""

    def __init__(self, original_core):
        # Adopt the existing parameter without drawing or changing any weights.
        nn.Module.__init__(self)
        self.n_qubits = original_core.n_qubits
        self.depth = original_core.depth
        self.weights = original_core.weights
        self._original_function = original_core._circuit.func
        self._make_circuit()

    def _make_circuit(self):
        original = self._original_function
        n_qubits = self.n_qubits

        def circuit(angles, weights):
            # The usual |000000> state must start on the input device for CUDA.
            state = torch.zeros(2 ** n_qubits, dtype=torch.complex128, device=angles.device)
            state[0] = 1
            qml.StatePrep(state, wires=range(n_qubits))
            return original(angles, weights)

        # Analytic execution has no sampling; explicit seed avoids consuming
        # process-global NumPy RNG when DataParallel constructs each replica.
        self._device = qml.device('default.qubit', wires=n_qubits, shots=None, seed=0)
        self._circuit = qml.QNode(circuit, self._device, interface='torch', diff_method='backprop')

    def _replicate_for_data_parallel(self):
        replica = super()._replicate_for_data_parallel()
        replica._make_circuit()
        return replica

    def forward(self, angles):
        # PennyLane 0.45 may convert rotation parameters to CPU under no_grad.
        # Keep its Torch backprop execution path, then discard the temporary graph
        # at this core boundary. No backward or parameter update occurs here.
        if not torch.is_grad_enabled():
            with torch.enable_grad():
                return super().forward(angles).detach()
        return super().forward(angles)


class InheritedQGNNGraph(nn.Module):
    """Same six-input front end as GNNGraph; two unchanged archived quantum layers.

    Each frame is a separate graph. The original GRU and decoder are excluded.
    Layer/core seeds follow the archive: seed+3/+103 and seed+1/+101.
    """

    def __init__(self, seed=2026):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(6, 128), nn.SiLU(), nn.Linear(128, 128))
        config = _base.ForecasterConfig()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 3)
            physical = _archive.PhysicsAlignedMessageLayer(config, 'quantum', 'physical', seed + 103)
            torch.manual_seed(seed + 1)
            contextual = _archive.PhysicsAlignedMessageLayer(config, 'quantum', 'contextual', seed + 101)
        self.graph_layers = nn.ModuleList([physical, contextual])
        for layer in self.graph_layers:
            layer.core = _ReplicaSafePQC(layer.core)

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
        detection = torch.where(mask, detected, 0).to(z)
        if not torch.isfinite(physical).all() or not torch.isfinite(z).all() or not torch.isfinite(detection).all():
            raise ValueError('Existing states and detection flags must be finite')
        nodes = self.input(torch.cat((z, detection[..., None], mask[..., None].to(z)), -1))
        current = mask.reshape(-1, n)
        nodes = (nodes * mask[..., None]).reshape(-1, n, 128)
        active = current.any(-1).nonzero().squeeze(-1)
        # Never call a broadcast quantum circuit with zero edges.
        if active.numel() == 0:
            return nodes.reshape(*state_hat.shape[:-1], 128)
        live = current.index_select(0, active)
        h = nodes.index_select(0, active)
        physical = physical.reshape(-1, n, 4).index_select(0, active)
        edges, distances = _base.build_edge_features(physical[..., :2], physical[..., 2:])
        pairs = live[:, :, None] & live[:, None, :]
        adjacency = pairs & (distances <= 45.)
        adjacency |= torch.eye(n, device=state_hat.device, dtype=torch.bool)[None] & pairs
        for layer in self.graph_layers:
            h, _ = layer(h, edges, adjacency)
            h = h * live[..., None]
        nodes = nodes.index_copy(0, active, h)
        return nodes.reshape(*state_hat.shape[:-1], 128)
