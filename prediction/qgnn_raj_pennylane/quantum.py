from __future__ import annotations

import math
from typing import Iterable

import pennylane as qml
import torch
from torch import nn

from prediction.qgnn_final.common import patch_inputs, physical_graph
from prediction.qgnn_paper_native.raj_subset import SubsetFeatureBuilder

from .common import (
    EMBED_PAIRS,
    EMBED_QUBITS,
    NODE_PAIRS,
    NODE_QUBITS,
    OBS_PER_NODE,
    TOTAL_QUBITS,
    branch_agent_mask,
    directed_pair_features,
    pad_vehicle_register,
    pair_valid_mask,
    pool_subset_features,
    uniform_joint_state,
)


def _mlp(in_dim: int, hidden: int, out_dim: int):
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, out_dim))


def _count(module: nn.Module) -> int:
    if isinstance(module, torch.Tensor):
        return int(module.numel()) if module.requires_grad else 0
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


class PennyLaneRajBranch(nn.Module):
    """One native Raj higher-order branch on an 8+6 fixed-weight register.

    The vehicle register starts in a fixed superposition of valid j-subsets and
    the embedding register starts in a fixed weight-k=3 superposition. Every
    feature injection below is a gate angle inside the PennyLane QNode. The
    branch never exposes qml.state() to the model readout.
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
        if rounds < 1:
            raise ValueError("rounds must be positive")
        self.j = int(j)
        self.rounds = int(rounds)
        self.builder = SubsetFeatureBuilder(32)
        feat_dim = self.builder.out_dim

        # History/physical feature -> unitary gate-angle generators.
        self.node_phase = nn.ModuleList(
            _mlp(feat_dim, phase_hidden, 1) for _ in range(rounds)
        )
        self.graph_angle = nn.ModuleList(
            _mlp(32, 32, 1) for _ in range(rounds)
        )
        self.embed_angle = nn.ModuleList(
            _mlp(feat_dim, encoder_hidden, len(EMBED_PAIRS))
            for _ in range(rounds)
        )
        self.cross_strength = nn.ModuleList(
            _mlp(feat_dim, phase_hidden, 1) for _ in range(rounds)
        )

        # Explicit trainable circuit parameters. They are shared over vehicle
        # slots; no absolute vehicle ID is encoded here.
        self.node_bias = nn.Parameter(torch.zeros(rounds))
        self.embed_bias = nn.Parameter(torch.zeros(rounds, len(EMBED_PAIRS)))
        self.cross_bias = nn.Parameter(torch.zeros(rounds, EMBED_QUBITS))
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
        self._qnodes: dict[tuple[tuple[int, int, int], ...], qml.QNode] = {}

    @staticmethod
    def _build_observables():
        observables = []
        for i in range(NODE_QUBITS):
            zi = qml.PauliZ(i)
            observables.append(zi)
            for e in range(EMBED_QUBITS):
                observables.append(zi @ qml.PauliZ(NODE_QUBITS + e))
            for p, q in EMBED_PAIRS:
                wp, wq = NODE_QUBITS + p, NODE_QUBITS + q
                observables.extend(
                    (
                        zi @ qml.PauliX(wp) @ qml.PauliX(wq),
                        zi @ qml.PauliY(wp) @ qml.PauliY(wq),
                        zi @ qml.PauliX(wp) @ qml.PauliY(wq),
                        zi @ qml.PauliY(wp) @ qml.PauliX(wq),
                    )
                )
        return tuple(observables)

    def _build_qnode(self, pair_order: tuple[tuple[int, int, int], ...]):
        """Build one QNode for a canonical history-only occupation schedule."""
        device = self._device
        observables = self._observables
        rounds = self.rounds

        @qml.qnode(device, interface="torch", diff_method=self.diff_method)
        def circuit(state, node_phase, graph_angles, embed_angles, cross_angles):
            qml.StatePrep(state, wires=range(TOTAL_QUBITS))
            for layer in range(rounds):
                for i in range(NODE_QUBITS):
                    qml.RZ(node_phase[layer, i], wires=i)
                for pair_index, a, b in pair_order:
                    qml.SingleExcitation(
                        graph_angles[layer, pair_index], wires=[a, b]
                    )
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
            return tuple(qml.expval(observable) for observable in observables)

        return circuit

    def qnode_for_order(self, pair_order: Iterable[tuple[int, int, int]]):
        key = tuple((int(index), int(a), int(b)) for index, a, b in pair_order)
        if key not in self._qnodes:
            self._qnodes[key] = self._build_qnode(key)
        return self._qnodes[key]

    @staticmethod
    def _canonical_pair_order(
        edge: torch.Tensor, mask: torch.Tensor, node_key: torch.Tensor
    ):
        """Build a permutation-equivariant oriented non-commuting schedule.

        Pair ordering uses only symmetric physical edge keys and unordered
        node-key pairs. The orientation of SingleExcitation is chosen from the
        node keys, not from the vehicle slot index. Exact ties are skipped
        because no index-free order/orientation exists for non-commuting gates.
        """
        entries = []
        active = mask.detach().cpu().tolist()
        node_keys = [
            tuple(float(value) for value in row.detach().cpu())
            for row in node_key
        ]
        for pair_index, (a, b) in enumerate(NODE_PAIRS):
            if not (active[a] and active[b]) or node_keys[a] == node_keys[b]:
                continue
            key_a = tuple(float(value) for value in edge[a, b].detach().cpu())
            key_b = tuple(float(value) for value in edge[b, a].detach().cpu())
            pair_key = (min(key_a, key_b), max(key_a, key_b))
            if node_keys[a] > node_keys[b]:
                first, second = a, b
            else:
                first, second = b, a
            node_pair_key = (
                max(node_keys[a], node_keys[b]),
                min(node_keys[a], node_keys[b]),
            )
            entries.append(
                (pair_index, (first, second), (pair_key, node_pair_key))
            )
        entries.sort(key=lambda item: item[2], reverse=True)
        ordered = []
        cursor = 0
        while cursor < len(entries):
            end = cursor + 1
            while end < len(entries) and entries[end][2] == entries[cursor][2]:
                end += 1
            if end - cursor == 1:
                pair_index, (first, second), _ = entries[cursor]
                ordered.append((pair_index, first, second))
            cursor = end
        return tuple(ordered)
    def _angles(self, history, mask, feat, valid):
        edge, _ = physical_graph(history, mask)
        node_feat, scene_feat = pool_subset_features(feat, valid, self.j)
        pair_feat = directed_pair_features(edge)
        pair_mask = pair_valid_mask(mask)

        node_rows, graph_rows, embed_rows, cross_rows = [], [], [], []
        for layer in range(self.rounds):
            node = math.pi * torch.tanh(
                self.node_phase[layer](node_feat).squeeze(-1) + self.node_bias[layer]
            )
            node = node * mask.to(node.dtype)
            graph = (math.pi / 2) * torch.tanh(
                self.graph_angle[layer](pair_feat).squeeze(-1)
            )
            graph = graph * pair_mask.to(graph.dtype)
            embed = (math.pi / 2) * torch.tanh(
                self.embed_angle[layer](scene_feat) + self.embed_bias[layer]
            )
            strength = torch.tanh(
                self.cross_strength[layer](node_feat).squeeze(-1)
            )
            strength = strength * mask.to(strength.dtype)
            basis = torch.tanh(self.cross_basis[layer])[None, None, :]
            cross = (math.pi / 4) * (
                strength[..., None] * basis
                + torch.tanh(self.cross_bias[layer])[None, None, :]
            )
            cross = cross * mask[..., None].to(cross.dtype)

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

    def forward_patch(
        self,
        history: torch.Tensor,
        mask: torch.Tensor,
        *,
        interaction_scale: float = 1.0,
        johnson_scale: float = 1.0,
        entangling_scale: float = 1.0,
        return_raw: bool = False,
    ):
        history, mask = pad_vehicle_register(history, mask)
        _, feat, valid, _ = self.builder(history, mask, self.j)
        node_feat, _ = pool_subset_features(feat, valid, self.j)
        edge, _ = physical_graph(history, mask)
        node_phase, graph_angles, embed_angles, cross_angles = self._angles(
            history, mask, feat, valid
        )
        graph_angles = graph_angles * float(interaction_scale) * float(johnson_scale)
        cross_angles = cross_angles * float(interaction_scale) * float(entangling_scale)
        active = branch_agent_mask(mask, self.j)
        rows, raw_rows = [], []

        for batch_index in range(history.shape[0]):
            if not bool(active[batch_index].any()):
                raw = history.new_zeros(NODE_QUBITS, OBS_PER_NODE)
            else:
                pair_order = self._canonical_pair_order(
                    edge[batch_index], mask[batch_index], node_feat[batch_index]
                )
                state = uniform_joint_state(mask[batch_index], self.j)
                values = self.qnode_for_order(pair_order)(
                    state,
                    node_phase[batch_index].to(torch.float64),
                    graph_angles[batch_index].to(torch.float64),
                    embed_angles[batch_index].to(torch.float64),
                    cross_angles[batch_index].to(torch.float64),
                )
                raw = torch.stack(values).reshape(NODE_QUBITS, OBS_PER_NODE)
                raw = raw.to(history.dtype)
            raw_rows.append(raw)
            rows.append(raw * active[batch_index, :, None].to(raw.dtype))

        raw = torch.stack(raw_rows, 0)
        output = self.readout(torch.stack(rows, 0)) * active[..., None].to(history.dtype)
        if return_raw:
            return output, raw
        return output

    def parameter_breakdown(self):
        feature_blocks = (
            "builder",
            "node_phase",
            "graph_angle",
            "embed_angle",
            "cross_strength",
        )
        quantum_blocks = ("node_bias", "embed_bias", "cross_bias", "cross_basis")
        feature = sum(_count(getattr(self, name)) for name in feature_blocks)
        quantum = sum(_count(getattr(self, name)) for name in quantum_blocks)
        return {
            "history_feature_and_angle_generation": feature,
            "quantum_specific_circuit": quantum,
            "post_measurement_readout": _count(self.readout),
            "total": sum(parameter.numel() for parameter in self.parameters()),
        }


class MultiJFusion(nn.Module):
    """Per-agent j=2/j=3 fusion with no new cross-agent message passing."""

    def __init__(self):
        super().__init__()
        self.gate = _mlp(128, 32, 1)
        self.delta = _mlp(128, 64, 64)
        self.norm = nn.LayerNorm(64)

    def forward(self, z2, z3, mask2, mask3):
        both = torch.cat((z2, z3), -1)
        available = mask2 | mask3
        both_available = mask2 & mask3
        gate = torch.sigmoid(self.gate(both))
        gate = torch.where(
            mask2[..., None] & ~mask3[..., None],
            torch.zeros_like(gate),
            gate,
        )
        gate = torch.where(
            mask3[..., None] & ~mask2[..., None],
            torch.ones_like(gate),
            gate,
        )
        base = (1.0 - gate) * z2 + gate * z3
        delta = self.delta(both) * both_available[..., None].to(both.dtype)
        return self.norm(base + delta) * available[..., None].to(base.dtype)


class PennyLaneRajMultiJCore(nn.Module):
    """Raj Weighted Multi-j core with a single [B,N,64] interaction output."""

    readout_dim = 64
    interaction_tokens = 1

    def __init__(
        self,
        rounds: int = 3,
        device_name: str = "lightning.gpu",
        diff_method: str = "adjoint",
    ):
        super().__init__()
        self.rounds = int(rounds)
        self.j2 = PennyLaneRajBranch(
            2, rounds, device_name=device_name, diff_method=diff_method
        )
        self.j3 = PennyLaneRajBranch(
            3, rounds, device_name=device_name, diff_method=diff_method
        )
        self.fusion = MultiJFusion()

    def forward(
        self,
        history: torch.Tensor,
        mask: torch.Tensor,
        timestamps=None,
        *,
        branch_scales=(1.0, 1.0),
        interaction_scale: float = 1.0,
        johnson_scale: float = 1.0,
        entangling_scale: float = 1.0,
    ):
        del timestamps
        if history.ndim != 4 or history.shape[1] != 20 or history.shape[-1] != 4:
            raise ValueError("Expected history [B,20,N,4]")
        if mask.shape != (history.shape[0], history.shape[2]) or mask.dtype != torch.bool:
            raise ValueError("Expected boolean mask [B,N]")
        if history.shape[2] < 1:
            raise ValueError("At least one vehicle slot is required")
        history = torch.where(mask[:, None, :, None], history, 0.0)
        original_n = history.shape[2]
        hp, mp, restore = patch_inputs(history, mask)
        hp, mp = pad_vehicle_register(hp, mp)
        scale2, scale3 = (float(branch_scales[0]), float(branch_scales[1]))
        z2 = self.j2.forward_patch(
            hp,
            mp,
            interaction_scale=interaction_scale,
            johnson_scale=johnson_scale,
            entangling_scale=entangling_scale,
        ) * scale2
        z3 = self.j3.forward_patch(
            hp,
            mp,
            interaction_scale=interaction_scale,
            johnson_scale=johnson_scale,
            entangling_scale=entangling_scale,
        ) * scale3
        mask2 = branch_agent_mask(mp, 2) & (scale2 != 0.0)
        mask3 = branch_agent_mask(mp, 3) & (scale3 != 0.0)
        fused = self.fusion(z2, z3, mask2, mask3)
        branch_tokens = torch.stack((z2, z3), 2)
        branch_masks = torch.stack((mask2, mask3), 2)

        if restore is not None:
            batch_size, vehicle_count = restore
            readout = fused[:, 0].reshape(batch_size, vehicle_count, 64)
            tokens = branch_tokens[:, 0].reshape(batch_size, vehicle_count, 2, 64)
            interaction_mask = (
                (mask2[:, 0] | mask3[:, 0]).reshape(batch_size, vehicle_count)
            )
            branch_masks = branch_masks[:, 0].reshape(
                batch_size, vehicle_count, 2
            )
        else:
            readout = fused[:, :original_n]
            tokens = branch_tokens[:, :original_n]
            interaction_mask = (mask2 | mask3)[:, :original_n]
            branch_masks = branch_masks[:, :original_n]

        return {
            "readout": readout,
            "interaction_mask": interaction_mask,
            "branch_readout": tokens,
            "branch_mask": branch_masks,
        }

    def parameter_breakdown(self):
        j2 = self.j2.parameter_breakdown()
        j3 = self.j3.parameter_breakdown()
        fusion = {"multi_j_fusion": _count(self.fusion)}
        total = sum(parameter.numel() for parameter in self.parameters())
        return {
            "j2": j2,
            "j3": j3,
            "multi_j_fusion": fusion["multi_j_fusion"],
            "total_interaction_core": total,
        }

    def circuit_metadata(self):
        return {
            "vehicle_qubits": NODE_QUBITS,
            "embedding_qubits": EMBED_QUBITS,
            "total_qubits_per_branch": TOTAL_QUBITS,
            "j2_vehicle_weight": 2,
            "j3_vehicle_weight": 3,
            "embedding_hamming_weight": 3,
            "vehicle_pair_count": len(NODE_PAIRS),
            "embedding_pair_count": len(EMBED_PAIRS),
            "cross_register_pair_count": NODE_QUBITS * EMBED_QUBITS,
            "observables_per_vehicle": OBS_PER_NODE,
            "observable_count": len(self.j2._observables),
            "rounds": self.rounds,
            "gate_families": [
                "StatePrep(fixed initial state)",
                "RZ(history/feature phase encoding)",
                "SingleExcitation(weighted Johnson occupation transfer)",
                "SingleExcitation(embedding weight-k evolution)",
                "IsingZZ(vehicle-embedding entangling conditioning)",
            ],
            "data_reupload": "every round: RZ + node SingleExcitation + embedding SingleExcitation + IsingZZ",
            "formal_readout": "PauliZ, Z_i Z_e, and Z_i {XX,YY,XY,YX}_{e_p,e_q} expectations",
        }
