"""Physics-aligned hierarchical dual-core GNN for a matched PennyLane experiment."""
import math
import sys
from pathlib import Path

import pennylane as qml
import torch
from torch import nn

ROOT = Path('/home/js_cn/sensing')
BASE = ROOT / 'diagnostics/core_message_seed2026_v2'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BASE))

from target_interaction_graph import ForecasterConfig, TargetInteractionGNN


class DualAxisPennyLanePQC(nn.Module):
    """Encode twelve relation features on six qubits using RY and RZ."""

    def __init__(self, depth=3):
        super().__init__()
        self.n_qubits = 6
        self.depth = int(depth)
        self.weights = nn.Parameter(torch.empty(self.depth, self.n_qubits, 3))
        nn.init.uniform_(self.weights, -0.08, 0.08)
        device = qml.device('default.qubit', wires=self.n_qubits, shots=None)

        @qml.qnode(device, interface='torch', diff_method='backprop')
        def circuit(angles, weights):
            for layer in range(self.depth):
                for qubit in range(self.n_qubits):
                    qml.RY(angles[..., qubit], wires=qubit)
                    qml.RZ(angles[..., qubit + self.n_qubits], wires=qubit)
                for qubit in range(self.n_qubits):
                    phi, theta, omega = weights[layer, qubit]
                    qml.RZ(phi, wires=qubit)
                    qml.RY(theta, wires=qubit)
                    qml.RZ(omega, wires=qubit)
                for control in range(self.n_qubits):
                    qml.CNOT(wires=[control, (control + 1) % self.n_qubits])
            single = [qml.expval(qml.PauliZ(q)) for q in range(self.n_qubits)]
            pairs = [qml.expval(qml.PauliZ(q) @ qml.PauliZ((q + 1) % self.n_qubits))
                     for q in range(self.n_qubits)]
            return single + pairs

        self._device = device
        self._circuit = circuit

    def forward(self, angles):
        if angles.ndim != 2 or angles.shape[-1] != 12:
            raise ValueError('Expected [edges, 12] encoded angles.')
        result = self._circuit(angles, self.weights)
        if not isinstance(result, torch.Tensor):
            result = torch.stack(tuple(result), dim=-1)
        return result.to(dtype=angles.dtype, device=angles.device)


class LowRankClassicalCore(nn.Module):
    """Slightly larger matched classical core: 60 versus 54 core parameters."""

    def __init__(self):
        super().__init__()
        self.down = nn.Linear(12, 2, bias=False)  # 24
        self.up = nn.Linear(2, 12, bias=True)     # 36

    def forward(self, angles):
        return torch.tanh(self.up(torch.nn.functional.silu(self.down(angles))))


class PhysicsAlignedMessageLayer(nn.Module):
    """Quantum/classical message layer fed by bounded structured relations."""

    def __init__(self, config, kind, role, core_seed):
        super().__init__()
        if kind not in ('quantum', 'classical'):
            raise ValueError(kind)
        if role not in ('physical', 'contextual'):
            raise ValueError(role)
        h = config.hidden_dim
        self.kind = kind
        self.role = role
        self.heads = config.graph_heads
        self.head_dim = h // self.heads
        self.context_gain = 0.35 if role == 'physical' else 1.0
        self.node_norm = nn.LayerNorm(h)
        # Per-channel calibration followed by 2*atan keeps angles in (-pi, pi).
        init_scale = (1.0 - 0.5) / 3.5
        init_raw = math.log(init_scale / (1.0 - init_scale))
        self.angle_scale_raw = nn.Parameter(torch.full((12,), init_raw))
        self.angle_bias = nn.Parameter(torch.zeros(12))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(core_seed)
            self.core = DualAxisPennyLanePQC(depth=3) if kind == 'quantum' else LowRankClassicalCore()
        self.latent_norm = nn.LayerNorm(12)
        self.score = nn.Linear(12, self.heads, bias=False)
        self.gate = nn.Linear(12, self.heads, bias=False)
        nn.init.zeros_(self.gate.weight)  # 2*sigmoid(0)=1: neutral message strength.
        self.value_norm = nn.LayerNorm(h)
        self.value = nn.Linear(h, h, bias=False)
        self.edge_value = nn.Sequential(nn.Linear(7, h), nn.SiLU(), nn.Linear(h, h, bias=False))
        self.residual_logit = nn.Parameter(torch.tensor(-1.3862944))  # sigmoid = 0.2
        self.output_norm = nn.LayerNorm(h)
        self.ffn = nn.Sequential(nn.Linear(h, 2*h), nn.SiLU(), nn.Dropout(config.dropout), nn.Linear(2*h, h))
        self.final_norm = nn.LayerNorm(h)
        self.dropout = nn.Dropout(config.dropout)

    def relation_features(self, nodes, edge_features):
        normalized = self.node_norm(nodes)
        b, n, h = normalized.shape
        receiver = normalized[:, :, None, :].expand(-1, -1, n, -1)
        sender = normalized[:, None, :, :].expand(-1, n, -1, -1)
        products = (receiver * sender).reshape(b, n, n, 4, h // 4).mean(-1)
        rms_difference = (receiver - sender).square().mean(-1, keepdim=True).add(1e-6).sqrt()
        context = torch.cat([products, rms_difference], dim=-1)
        return torch.cat([torch.tanh(edge_features), self.context_gain * torch.tanh(context)], dim=-1)

    def encode_angles(self, features):
        scale = 0.5 + 3.5 * torch.sigmoid(self.angle_scale_raw)
        shifted = scale * features + 0.25 * torch.tanh(self.angle_bias)
        return 2.0 * torch.atan(shifted)

    def forward(self, nodes, edge_features, adjacency):
        b, n, h = nodes.shape
        relations = self.relation_features(nodes, edge_features)
        selected = adjacency.reshape(-1).nonzero().squeeze(-1)
        angles = self.encode_angles(relations.reshape(-1, 12).index_select(0, selected))
        latent = self.latent_norm(self.core(angles))
        # Keep the complete edge-level relation representation for downstream
        # QGNN--LLM interfaces.  Non-edges remain exactly zero.
        self.last_relation_latent = nodes.new_zeros(b*n*n, 12).index_copy(
            0, selected, latent
        ).view(b, n, n, 12)

        values = self.value(self.value_norm(nodes)).view(b, n, self.heads, self.head_dim)
        edge_values = self.edge_value(edge_features).view(b, n, n, self.heads, self.head_dim)
        base_message = values[:, None, :, :, :] + edge_values
        score = nodes.new_zeros(b*n*n, self.heads).index_copy(0, selected, self.score(latent)).view(b, n, n, self.heads)
        gate = nodes.new_zeros(b*n*n, self.heads).index_copy(0, selected, self.gate(latent)).view(b, n, n, self.heads)
        gate = 2.0 * torch.sigmoid(gate)
        attention = torch.softmax(score.masked_fill(~adjacency[..., None], -1e4), dim=2)
        # Preserve the actual edge influence used in message passing.  The
        # original public return stays unchanged for checkpoint compatibility.
        self.last_effective_relation = (attention * gate).mean(-1)
        aggregate = (self.dropout(attention)[..., None] * gate[..., None] * base_message).sum(2).reshape(b, n, h)
        residual_strength = torch.sigmoid(self.residual_logit)
        nodes = self.output_norm(nodes + residual_strength * self.dropout(aggregate))
        nodes = self.final_norm(nodes + self.dropout(self.ffn(nodes)))
        return nodes, attention.mean(-1)


def build_physics_aligned_dual_graph(arm, seed=2026):
    if arm not in ('physics_classical_dual', 'physics_quantum_dual'):
        raise ValueError(arm)
    kind = 'quantum' if arm == 'physics_quantum_dual' else 'classical'
    torch.manual_seed(seed)
    config = ForecasterConfig()
    model = TargetInteractionGNN(config)
    # Independent layer seeds; shared non-core initialization stays identical across arms.
    torch.manual_seed(seed + 3)
    model.graph_layers[0] = PhysicsAlignedMessageLayer(config, kind, 'physical', seed + 103)
    torch.manual_seed(seed + 1)
    model.graph_layers[1] = PhysicsAlignedMessageLayer(config, kind, 'contextual', seed + 101)
    return model
