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
    NODE_PAIRS,
    NODE_QUBITS,
    OBS_PER_NODE,
    TOTAL_QUBITS,
    branch_agent_mask,
    canonical_joint_state,
    canonical_pack,
    directed_pair_features,
    node_subsets,
    pair_valid_mask,
    pool_subset_features,
    restore_canonical,
)
from .observables import (
    conditional_embedding_rdm_features,
    formal_observables,
    normalize_formal_raw,
)


def _mlp(in_dim: int, hidden: int, out_dim: int):
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, out_dim))


def _count(module) -> int:
    if isinstance(module, torch.Tensor):
        return int(module.numel()) if module.requires_grad else 0
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


class PennyLaneRajBranch(nn.Module):
    """One Raj higher-order branch on an 8-vehicle + 6-embedding register.

    Training uses one fixed PennyLane QNode returning qml.state(). Amplitudes are
    never exposed as model features: the state is only an exact-simulator
    implementation detail used to evaluate a fixed set of measurable Hermitian
    observables efficiently. The formal expval path evaluates the same
    observables directly with qml.expval for numerical validation.
    """

    def __init__(
        self,
        j: int,
        rounds: int = 3,
        encoder_hidden: int = 64,
        phase_hidden: int = 32,
        readout_hidden: int = 96,
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

        self.subset_phase = nn.ModuleList(
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

        self.subset_bias = nn.Parameter(torch.zeros(rounds))
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

        self._state_device = qml.device("default.qubit", wires=TOTAL_QUBITS, shots=None)
        self._state_qnode = self._build_state_qnode()
        self._formal_qnodes = {}
        self.last_group_count = 0

    def _apply_circuit(self, subset_phase, graph_angles, embed_angles, cross_angles):
        subsets = node_subsets(self.j)
        for layer in range(self.rounds):
            for subset_index, subset in enumerate(subsets):
                angle = subset_phase[:, layer, subset_index]
                if self.j == 2:
                    qml.ControlledPhaseShift(angle, wires=list(subset))
                else:
                    a, b, target = subset
                    qml.ctrl(qml.PhaseShift, control=[a, b])(
                        angle, wires=target
                    )

            for pair_index, (a, b) in enumerate(NODE_PAIRS):
                qml.SingleExcitation(
                    graph_angles[:, layer, pair_index], wires=[a, b]
                )

            for pair_index, (p, q) in enumerate(EMBED_PAIRS):
                qml.SingleExcitation(
                    embed_angles[:, layer, pair_index],
                    wires=[NODE_QUBITS + p, NODE_QUBITS + q],
                )

            for i in range(NODE_QUBITS):
                for e in range(EMBED_QUBITS):
                    qml.IsingZZ(
                        cross_angles[:, layer, i, e],
                        wires=[i, NODE_QUBITS + e],
                    )

    def _build_state_qnode(self):
        device = self._state_device

        @qml.qnode(device, interface="torch", diff_method="backprop")
        def circuit(state, subset_phase, graph_angles, embed_angles, cross_angles):
            qml.StatePrep(state, wires=range(TOTAL_QUBITS))
            self._apply_circuit(subset_phase, graph_angles, embed_angles, cross_angles)
            return qml.state()

        return circuit

    def _build_formal_qnode(self, device_name: str, diff_method: str):
        key = (str(device_name), str(diff_method))
        if key in self._formal_qnodes:
            return self._formal_qnodes[key]
        device = qml.device(device_name, wires=TOTAL_QUBITS, shots=None)
        observables = formal_observables()

        @qml.qnode(device, interface="torch", diff_method=diff_method)
        def circuit(state, subset_phase, graph_angles, embed_angles, cross_angles):
            qml.StatePrep(state, wires=range(TOTAL_QUBITS))
            self._apply_circuit(subset_phase, graph_angles, embed_angles, cross_angles)
            return tuple(qml.expval(observable) for observable in observables)

        self._formal_qnodes[key] = circuit
        return circuit

    def _angles(self, history, mask, feat, valid):
        edge, _ = physical_graph(history, mask)
        node_feat, scene_feat = pool_subset_features(feat, valid, self.j)
        pair_feat = directed_pair_features(edge)
        pair_mask = pair_valid_mask(mask)

        subset_rows, graph_rows, embed_rows, cross_rows = [], [], [], []
        for layer in range(self.rounds):
            subset = math.pi * torch.tanh(
                self.subset_phase[layer](feat).squeeze(-1) + self.subset_bias[layer]
            )
            subset = subset * valid.to(subset.dtype)

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

            subset_rows.append(subset)
            graph_rows.append(graph)
            embed_rows.append(embed)
            cross_rows.append(cross)

        return (
            torch.stack(subset_rows, 1),
            torch.stack(graph_rows, 1),
            torch.stack(embed_rows, 1),
            torch.stack(cross_rows, 1),
        )

    def _statevector_raw(
        self,
        history,
        mask,
        *,
        interaction_scale=1.0,
        johnson_scale=1.0,
        entangling_scale=1.0,
        subset_scale=1.0,
    ):
        _, feat, valid, _ = self.builder(history, mask, self.j)
        subset_phase, graph_angles, embed_angles, cross_angles = self._angles(
            history, mask, feat, valid
        )
        subset_phase = subset_phase * float(subset_scale)
        graph_angles = graph_angles * float(interaction_scale) * float(johnson_scale)
        cross_angles = cross_angles * float(interaction_scale) * float(entangling_scale)

        active = branch_agent_mask(mask, self.j)
        counts = mask.sum(-1)
        raw = history.new_zeros(history.shape[0], NODE_QUBITS, OBS_PER_NODE)
        group_count = 0

        for count in sorted(set(int(v) for v in counts.detach().cpu().tolist())):
            indices = (counts == count).nonzero(as_tuple=True)[0]
            if not indices.numel() or count < self.j:
                continue
            group_count += 1
            state = canonical_joint_state(count, self.j).to(
                device=history.device, dtype=torch.complex128
            )
            psi = self._state_qnode(
                state,
                subset_phase[indices].to(torch.float64),
                graph_angles[indices].to(torch.float64),
                embed_angles[indices].to(torch.float64),
                cross_angles[indices].to(torch.float64),
            )
            if psi.ndim == 1:
                psi = psi.unsqueeze(0)
            features = conditional_embedding_rdm_features(
                psi, active[indices]
            ).to(history.dtype)
            raw = raw.index_copy(0, indices, features)

        self.last_group_count = group_count
        return raw, active

    def forward_packed(
        self,
        history: torch.Tensor,
        mask: torch.Tensor,
        *,
        interaction_scale: float = 1.0,
        johnson_scale: float = 1.0,
        entangling_scale: float = 1.0,
        subset_scale: float = 1.0,
        return_raw: bool = False,
    ):
        raw, active = self._statevector_raw(
            history,
            mask,
            interaction_scale=interaction_scale,
            johnson_scale=johnson_scale,
            entangling_scale=entangling_scale,
            subset_scale=subset_scale,
        )
        output = self.readout(raw) * active[..., None].to(raw.dtype)
        if return_raw:
            return output, raw
        return output

    def formal_expval_features(
        self,
        history: torch.Tensor,
        mask: torch.Tensor,
        *,
        device_name: str = "default.qubit",
        diff_method: str = "backprop",
    ):
        """Slow validation path for direct qml.expval equivalence."""
        _, feat, valid, _ = self.builder(history, mask, self.j)
        subset_phase, graph_angles, embed_angles, cross_angles = self._angles(
            history, mask, feat, valid
        )
        active = branch_agent_mask(mask, self.j)
        counts = mask.sum(-1)
        raw = history.new_zeros(history.shape[0], NODE_QUBITS, OBS_PER_NODE)
        qnode = self._build_formal_qnode(device_name, diff_method)

        for count in sorted(set(int(v) for v in counts.detach().cpu().tolist())):
            indices = (counts == count).nonzero(as_tuple=True)[0]
            if not indices.numel() or count < self.j:
                continue
            state = canonical_joint_state(count, self.j).to(
                device=history.device, dtype=torch.complex128
            )
            values = qnode(
                state,
                subset_phase[indices].to(torch.float64),
                graph_angles[indices].to(torch.float64),
                embed_angles[indices].to(torch.float64),
                cross_angles[indices].to(torch.float64),
            )
            stacked = torch.stack(values, -1)
            if stacked.ndim == 1:
                stacked = stacked.unsqueeze(0)
            stacked = stacked.reshape(len(indices), NODE_QUBITS, OBS_PER_NODE)
            normalized = normalize_formal_raw(stacked, active[indices]).to(
                device=history.device, dtype=history.dtype
            )
            raw = raw.index_copy(0, indices, normalized)
        return raw

    def parameter_breakdown(self):
        feature_blocks = (
            "builder",
            "subset_phase",
            "graph_angle",
            "embed_angle",
            "cross_strength",
        )
        quantum_blocks = ("subset_bias", "embed_bias", "cross_bias", "cross_basis")
        feature = sum(_count(getattr(self, name)) for name in feature_blocks)
        quantum = sum(_count(getattr(self, name)) for name in quantum_blocks)
        return {
            "history_feature_and_angle_generation": feature,
            "quantum_specific_circuit": quantum,
            "post_measurement_readout": _count(self.readout),
            "total": sum(parameter.numel() for parameter in self.parameters()),
        }

    def circuit_metadata(self):
        return {
            "j": self.j,
            "vehicle_qubits": NODE_QUBITS,
            "embedding_qubits": EMBED_QUBITS,
            "total_qubits": TOTAL_QUBITS,
            "rounds": self.rounds,
            "fixed_topology": True,
            "scene_specific_qnode_cache": False,
            "training_device": "default.qubit with Torch/CUDA tensors",
            "training_interface": "torch",
            "training_diff_method": "backprop",
            "training_measurement": "qml.state exact-simulator optimization",
            "formal_measurement": "37 Hermitian observables/vehicle via qml.expval",
            "state_amplitudes_used_as_features": False,
            "subset_phase_gates_per_round": len(node_subsets(self.j)),
            "johnson_single_excitation_per_round": len(NODE_PAIRS),
            "embedding_single_excitation_per_round": len(EMBED_PAIRS),
            "cross_isingzz_per_round": NODE_QUBITS * EMBED_QUBITS,
            "logical_stateprep_undecomposed": True,
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
    """Batched PennyLane-native Raj Weighted Multi-j interaction core."""

    readout_dim = 64
    interaction_tokens = 2

    def __init__(self, rounds: int = 3):
        super().__init__()
        self.rounds = int(rounds)
        self.j2 = PennyLaneRajBranch(2, rounds)
        self.j3 = PennyLaneRajBranch(3, rounds)
        self.fusion = MultiJFusion()

    def _forward_fixed_register(
        self,
        history,
        mask,
        *,
        preserve_first: bool,
        branch_scales,
        interaction_scale,
        johnson_scale,
        entangling_scale,
        subset_scale,
    ):
        packed_history, packed_mask, order = canonical_pack(
            history, mask, preserve_first=preserve_first
        )
        scale2, scale3 = float(branch_scales[0]), float(branch_scales[1])
        z2 = self.j2.forward_packed(
            packed_history,
            packed_mask,
            interaction_scale=interaction_scale,
            johnson_scale=johnson_scale,
            entangling_scale=entangling_scale,
            subset_scale=subset_scale,
        ) * scale2
        z3 = self.j3.forward_packed(
            packed_history,
            packed_mask,
            interaction_scale=interaction_scale,
            johnson_scale=johnson_scale,
            entangling_scale=entangling_scale,
            subset_scale=subset_scale,
        ) * scale3
        mask2 = branch_agent_mask(packed_mask, 2) & (scale2 != 0.0)
        mask3 = branch_agent_mask(packed_mask, 3) & (scale3 != 0.0)
        fused = self.fusion(z2, z3, mask2, mask3)
        tokens = torch.stack((z2, z3), 2)
        token_mask = torch.stack((mask2, mask3), 2)

        return (
            restore_canonical(fused, order),
            restore_canonical(tokens, order),
            restore_canonical(token_mask, order),
        )

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
        subset_scale: float = 1.0,
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
        fused, tokens, token_mask = self._forward_fixed_register(
            hp,
            mp,
            preserve_first=restore is not None,
            branch_scales=branch_scales,
            interaction_scale=interaction_scale,
            johnson_scale=johnson_scale,
            entangling_scale=entangling_scale,
            subset_scale=subset_scale,
        )

        if restore is not None:
            batch_size, vehicle_count = restore
            readout = fused[:, 0].reshape(batch_size, vehicle_count, 64)
            tokens = tokens[:, 0].reshape(batch_size, vehicle_count, 2, 64)
            token_mask = token_mask[:, 0].reshape(batch_size, vehicle_count, 2)
        else:
            readout = fused[:, :original_n]
            tokens = tokens[:, :original_n]
            token_mask = token_mask[:, :original_n]

        interaction_mask = token_mask.any(-1)
        return {
            "readout": readout,
            "interaction_mask": interaction_mask,
            "branch_readout": tokens,
            "branch_mask": token_mask,
        }

    def parameter_breakdown(self):
        j2 = self.j2.parameter_breakdown()
        j3 = self.j3.parameter_breakdown()
        fusion = _count(self.fusion)
        total = sum(parameter.numel() for parameter in self.parameters())
        return {
            "j2": j2,
            "j3": j3,
            "multi_j_fusion": fusion,
            "total_interaction_core": total,
        }

    def circuit_metadata(self):
        return {
            "j2": self.j2.circuit_metadata(),
            "j3": self.j3.circuit_metadata(),
            "output": "[B,N,64]",
            "branch_tokens": "[B,N,2,64]",
            "interaction_tokens": 2,
            "canonicalization": (
                "raw first/middle/last history lexicographic packing; "
                "no trainable feature controls circuit topology"
            ),
        }
