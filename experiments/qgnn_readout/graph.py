"""Inherited two-layer QGNN with zero-initialized single-X readout increments."""
import pennylane as qml
import torch
from torch import nn
from torch.nn import functional as F

from experiments.qgnn_inherited.graph import InheritedQGNNGraph, _ReplicaSafePQC, _archive


class _XReadoutPQC(_ReplicaSafePQC):
    def __init__(self, original_core):
        nn.Module.__init__(self)
        self.n_qubits = original_core.n_qubits
        self.depth = original_core.depth
        self.weights = original_core.weights
        # Use the archived gate function, not the wrapper containing StatePrep.
        self._original_function = original_core._original_function
        self._make_circuit()

    def _make_circuit(self):
        original = self._original_function
        n_qubits = self.n_qubits

        def circuit(angles, weights):
            state = torch.zeros(2 ** n_qubits, dtype=torch.complex128, device=angles.device)
            state[0] = 1
            qml.StatePrep(state, wires=range(n_qubits))
            old_readouts = original(angles, weights)
            return old_readouts + [qml.expval(qml.PauliX(q)) for q in range(n_qubits)]

        self._device = qml.device('default.qubit', wires=n_qubits, shots=None, seed=0)
        self._circuit = qml.QNode(circuit, self._device, interface='torch', diff_method='backprop')


class _XReadoutMessageLayer(_archive.PhysicsAlignedMessageLayer):
    def __init__(self, original):
        nn.Module.__init__(self)
        for name in ('kind', 'role', 'heads', 'head_dim', 'context_gain'):
            setattr(self, name, getattr(original, name))
        # Adopt every old module and direct parameter, without reinitializing.
        for name, module in original.named_children():
            self.add_module(name, module)
        for name, parameter in original.named_parameters(recurse=False):
            self.register_parameter(name, parameter)
        self.core = _XReadoutPQC(original.core)
        self.x_score = nn.Parameter(self.score.weight.new_zeros(self.heads, self.core.n_qubits))
        self.x_gate = nn.Parameter(self.gate.weight.new_zeros(self.heads, self.core.n_qubits))

    def forward(self, nodes, edge_features, adjacency):
        b, n, h = nodes.shape
        relations = self.relation_features(nodes, edge_features)
        selected = adjacency.reshape(-1).nonzero().squeeze(-1)
        angles = self.encode_angles(relations.reshape(-1, 12).index_select(0, selected))
        readouts = self.core(angles)
        latent = self.latent_norm(readouts[:, :12])
        x = readouts[:, 12:]

        values = self.value(self.value_norm(nodes)).view(b, n, self.heads, self.head_dim)
        edge_values = self.edge_value(edge_features).view(b, n, n, self.heads, self.head_dim)
        base_message = values[:, None, :, :, :] + edge_values
        score_values = self.score(latent) + F.linear(x, self.x_score)
        gate_values = self.gate(latent) + F.linear(x, self.x_gate)
        score = nodes.new_zeros(b*n*n, self.heads).index_copy(0, selected, score_values).view(b, n, n, self.heads)
        gate = nodes.new_zeros(b*n*n, self.heads).index_copy(0, selected, gate_values).view(b, n, n, self.heads)
        gate = 2.0 * torch.sigmoid(gate)
        attention = torch.softmax(score.masked_fill(~adjacency[..., None], -1e4), dim=2)
        aggregate = (self.dropout(attention)[..., None] * gate[..., None] * base_message).sum(2).reshape(b, n, h)
        residual_strength = torch.sigmoid(self.residual_logit)
        nodes = self.output_norm(nodes + residual_strength * self.dropout(aggregate))
        nodes = self.final_norm(nodes + self.dropout(self.ffn(nodes)))
        return nodes, attention.mean(-1)


class XReadoutQGNNGraph(InheritedQGNNGraph):
    """Same graph contract and old parameter names; exactly 96 extra parameters."""

    def __init__(self, seed=2026):
        super().__init__(seed=seed)
        self.graph_layers = nn.ModuleList([_XReadoutMessageLayer(layer) for layer in self.graph_layers])
