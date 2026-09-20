from __future__ import annotations

import math

import pennylane as qml
import torch
from torch import nn

from prediction.qgnn_final.common import patch_inputs, physical_graph
from prediction.qgnn_paper_native.raj_subset import SubsetFeatureBuilder

from .common import (
    EMBED_PAIRS,
    EMBED_QUBITS,
    EMBED_RING,
    NODE_PAIRS,
    NODE_QUBITS,
    OBS_PER_NODE,
    TOTAL_QUBITS,
    branch_agent_mask,
    directed_pair_features,
    pair_valid_mask,
    pool_subset_features,
    uniform_joint_state,
)


def _mlp(in_dim: int, hidden: int, out_dim: int):
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, out_dim))


class PennyLaneRajBranch(nn.Module):
    """One hardware-valid Raj j-subset branch expressed entirely as a PennyLane QNode.

    The 8 vehicle qubits live in the fixed-Hamming-weight j sector, which is the
    Johnson/subset register. Six embedding qubits provide the D=6 workspace.
    All trainable quantum evolution is expressed with standard unitary gates.
    The output contains only expectation values of Pauli observables.
    """

    def __init__(
        self,
        j: int,
        rounds: int = 3,
        encoder_hidden: int = 64,
        phase_hidden: int = 32,
        readout_hidden: int = 96,
        device_name: str = "lightning.gpu",
        diff_method: str = "adjoint",
    ):
        super().__init__()
        if j not in (2, 3):
            raise ValueError("Raj branch supports j=2 or j=3")
        self.j = int(j)
        self.rounds = int(rounds)
        self.builder = SubsetFeatureBuilder(32)
        feat_dim = self.builder.out_dim

        self.node_phase = nn.ModuleList(_mlp(feat_dim, phase_hidden, 1) for _ in range(rounds))
        self.graph_angle = nn.ModuleList(_mlp(32, 32, 1) for _ in range(rounds))
        self.embed_angle = nn.ModuleList(_mlp(feat_dim, encoder_hidden, len(EMBED_PAIRS)) for _ in range(rounds))
        self.cross_strength = nn.ModuleList(_mlp(feat_dim, phase_hidden, 1) for _ in range(rounds))
        self.cross_basis = nn.Parameter(torch.randn(rounds, EMBED_QUBITS) * 0.05)

        self.readout = nn.Sequential(
            nn.LayerNorm(OBS_PER_NODE),
            nn.Linear(OBS_PER_NODE, readout_hidden),
            nn.SiLU(),
            nn.Linear(readout_hidden, 64),
            nn.LayerNorm(64),
        )

        self.device_name = str(device_name)
        self.diff_method = str(diff_method)
        self._device = qml.device(self.device_name, wires=TOTAL_QUBITS, shots=None)
        self._observables = self._build_observables()
        self._circuit = self._build_qnode()

    @staticmethod
    def _build_observables():
        obs = []
        for i in range(NODE_QUBITS):
            zi = qml.PauliZ(i)
            obs.append(zi)
            for e in range(EMBED_QUBITS):
                obs.append(zi @ qml.PauliZ(NODE_QUBITS + e))
            for p, q in EMBED_RING:
                wp, wq = NODE_QUBITS + p, NODE_QUBITS + q
                obs.append(zi @ qml.PauliX(wp) @ qml.PauliX(wq))
                obs.append(zi @ qml.PauliY(wp) @ qml.PauliY(wq))
                obs.append(zi @ qml.PauliX(wp) @ qml.PauliY(wq))
                obs.append(zi @ qml.PauliY(wp) @ qml.PauliX(wq))
        return tuple(obs)

    def _build_qnode(self):
        device = self._device
        observables = self._observables
        rounds = self.rounds

        @qml.qnode(device, interface="torch", diff_method=self.diff_method)
        def circuit(state, node_phase, graph_angles, embed_angles, cross_angles):
            qml.StatePrep(state, wires=range(TOTAL_QUBITS))
            for layer in range(rounds):
                for i in range(NODE_QUBITS):
                    qml.RZ(node_phase[layer, i], wires=i)
                for pair_index, (a, b) in enumerate(NODE_PAIRS):
                    qml.SingleExcitation(graph_angles[layer, pair_index], wires=[a, b])
                for pair_index, (p, q) in enumerate(EMBED_PAIRS):
                    qml.SingleExcitation(
                        embed_angles[layer, pair_index],
                        wires=[NODE_QUBITS + p, NODE_QUBITS + q],
                    )
                for i in range(NODE_QUBITS):
                    for e in range(EMBED_QUBITS):
                        qml.IsingZZ(
                            cross_angles[layer, i, e],
                            wires=[i, NODE_QUBITS + e],
                        )
            return tuple(qml.expval(op) for op in observables)

        return circuit

    def _angles(self, history, mask, feat, valid):
        edge, _ = physical_graph(history, mask)
        node_feat, scene_feat = pool_subset_features(feat, valid, self.j)
        pair_feat = directed_pair_features(edge)
        pair_mask = pair_valid_mask(mask)

        node_rows, graph_rows, embed_rows, cross_rows = [], [], [], []
        for layer in range(self.rounds):
            node = math.pi * torch.tanh(self.node_phase[layer](node_feat).squeeze(-1))
            node = node * mask.to(node.dtype)
            graph = (math.pi / 2) * torch.tanh(self.graph_angle[layer](pair_feat).squeeze(-1))
            graph = graph * pair_mask.to(graph.dtype)
            embed = (math.pi / 2) * torch.tanh(self.embed_angle[layer](scene_feat))
            strength = torch.tanh(self.cross_strength[layer](node_feat).squeeze(-1))
            strength = strength * mask.to(strength.dtype)
            basis = torch.tanh(self.cross_basis[layer])[None, None, :]
            cross = (math.pi / 4) * strength[..., None] * basis

            node_rows.append(node)
            graph_rows.append(graph)
            embed_rows.append(embed)
            cross_rows.append(cross)

        return (
            torch.stack(node_rows, 1),
            torch.stack(graph_rows, 1),
            torch.stack(embed_rows, 1),
            torch.stack(cross_rows, 1),
        )

    def forward_patch(self, history: torch.Tensor, mask: torch.Tensor):
        _, feat, valid, _ = self.builder(history, mask, self.j)
        node_phase, graph_angles, embed_angles, cross_angles = self._angles(
            history, mask, feat, valid
        )
        active = branch_agent_mask(mask, self.j)
        rows = []

        for batch_index in range(history.shape[0]):
            if not bool(active[batch_index].any()):
                rows.append(history.new_zeros(NODE_QUBITS, OBS_PER_NODE))
                continue

            state = uniform_joint_state(mask[batch_index], self.j)
            values = self._circuit(
                state,
                node_phase[batch_index].to(torch.float64),
                graph_angles[batch_index].to(torch.float64),
                embed_angles[batch_index].to(torch.float64),
                cross_angles[batch_index].to(torch.float64),
            )
            raw = torch.stack(values).reshape(NODE_QUBITS, OBS_PER_NODE)
            rows.append(raw.to(history.dtype))

        raw = torch.stack(rows, 0) * active[..., None].to(history.dtype)
        return self.readout(raw) * active[..., None]


class PennyLaneRajMultiJCore(nn.Module):
    """Raj j=2+j=3 interaction core with two explicit quantum tokens per agent."""

    readout_dim = 64
    interaction_tokens = 2

    def __init__(
        self,
        rounds: int = 3,
        device_name: str = "lightning.gpu",
        diff_method: str = "adjoint",
    ):
        super().__init__()
        self.j2 = PennyLaneRajBranch(2, rounds, device_name=device_name, diff_method=diff_method)
        self.j3 = PennyLaneRajBranch(3, rounds, device_name=device_name, diff_method=diff_method)

    def forward(self, history: torch.Tensor, mask: torch.Tensor, timestamps=None):
        del timestamps
        if history.ndim != 4 or history.shape[1] != 20 or history.shape[-1] != 4:
            raise ValueError("Expected history [B,20,N,4]")
        if mask.shape != (history.shape[0], history.shape[2]) or mask.dtype != torch.bool:
            raise ValueError("Expected boolean mask [B,N]")

        history = torch.where(mask[:, None, :, None], history, 0.0)
        hp, mp, restore = patch_inputs(history, mask)
        z2 = self.j2.forward_patch(hp, mp)
        z3 = self.j3.forward_patch(hp, mp)
        tokens = torch.stack((z2, z3), 2)
        token_mask = torch.stack(
            (branch_agent_mask(mp, 2), branch_agent_mask(mp, 3)), 2
        )
        if restore is not None:
            b, n = restore
            tokens = tokens[:, 0].reshape(b, n, 2, 64)
            token_mask = token_mask[:, 0].reshape(b, n, 2)

        return {
            "readout": tokens,
            "interaction_mask": token_mask,
        }
