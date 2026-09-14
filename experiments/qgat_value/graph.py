"""Relation D with an independent physical-relation rotation of each value state."""
import math
import torch
from torch import nn
import pennylane as qml

from experiments.qgat_factorial.graph import two_unitary, pauli_observables
from experiments.qgat_relation.graph import (
    RelationGraph, build_edge_features, relation_rotation, relation_matrix,
)


class RelationValueGraph(RelationGraph):
    def __init__(self):
        super().__init__()
        # Independent of the score encoder; preserve all following initialization.
        with torch.random.fork_rng(devices=[]):
            self.value_edge_encoder = nn.Linear(7, 4, bias=False)
        nn.init.zeros_(self.value_edge_encoder.weight)

        @qml.qnode(qml.device('default.qubit', wires=7, shots=None),
                   interface='torch', diff_method='backprop')
        def full_circuit(angles, query, index_state, theta, score_phi, value_phi):
            qml.StatePrep(index_state, wires=[0, 1, 2])
            two_unitary(query, theta[0], [3, 4])
            for j in range(8):
                controls = [(j >> bit) & 1 for bit in (2, 1, 0)]
                qml.ctrl(relation_rotation, control=[0, 1, 2], control_values=controls)(
                    score_phi[..., j, :], [3, 4])
                qml.ctrl(qml.adjoint(two_unitary), control=[0, 1, 2], control_values=controls)(
                    angles[..., j, :], theta[1], [3, 4])
                qml.ctrl(two_unitary, control=[0, 1, 2], control_values=controls)(
                    angles[..., j, :], theta[2], [5, 6])
                # Value rotation follows V preparation on the value register only.
                qml.ctrl(relation_rotation, control=[0, 1, 2], control_values=controls)(
                    value_phi[..., j, :], [5, 6])
            return qml.state()
        self.full_circuit = full_circuit

    def value_relation_angles(self, physical, pairs):
        """Use cleaned physical states and the parent's neighbor mask; self Rv=I."""
        edges, _ = build_edge_features(physical[..., :2], physical[..., 2:])
        nonself = ~torch.eye(physical.shape[-2], device=physical.device, dtype=torch.bool)
        active = pairs & nonself[None]
        edges = torch.where(active[..., None], edges, 0)
        phi = math.pi * torch.tanh(self.value_edge_encoder(edges.to(self.value_edge_encoder.weight.dtype)))
        return torch.where(active[..., None], phi, 0).to(torch.float64)

    def features(self, state_hat, standardized_state, track_exists, detected, reference=False):
        shape = standardized_state.shape
        if shape[-1] != 4 or not 1 <= shape[-2] <= 8 or state_hat.shape != shape:
            raise ValueError('Expected matching physical/standardized [...,1..8,4] states')
        if track_exists.shape != shape[:-1] or detected.shape != track_exists.shape:
            raise ValueError('Mask shape mismatch')
        m = track_exists.bool()
        z = torch.where(m[..., None], standardized_state, 0)
        physical = torch.where(m[..., None], state_hat, 0)
        if not torch.isfinite(z).all() or not torch.isfinite(physical).all():
            raise ValueError('Nonfinite valid state')
        if (detected.bool() & ~m).any():
            raise ValueError('Detected without track')
        inp = torch.cat((z, m[..., None].to(z), detected[..., None].to(z)), -1)
        a = math.pi * torch.tanh(self.encoder(inp.to(self.encoder.weight.dtype)))
        n = shape[-2]
        a = a.reshape(-1, n, 6).to(torch.float64)
        m = m.reshape(-1, n)
        physical = physical.reshape(-1, n, 4)
        score_phi, pairs = self.relation_angles(physical, m)
        value_phi = self.value_relation_angles(physical, pairs)
        theta = self.theta.to(torch.float64)
        batch = len(a)
        if reference is False:
            q, k, v = [self.local_state(a.reshape(-1, 6), t).reshape(batch, n, 4) for t in theta]
            rotated_query = torch.einsum('bijde,bie->bijd', relation_matrix(score_phi), q)
            gamma = torch.einsum('bjd,bijd->bij', k.conj(), rotated_query)
            # i receives from j: only the value content gains this second rotation.
            rotated_value = torch.einsum('bijde,bje->bijd', relation_matrix(value_phi), v)
            unnormalized = gamma[..., None] * rotated_value * pairs[..., None]
            unnormalized = unnormalized / pairs.sum(-1).to(a.dtype).clamp_min(1).sqrt()[..., None, None]
        elif reference == 'full':
            neighbors = torch.nn.functional.pad(a, (0, 0, 0, 8-n))[:, None].expand(batch, n, 8, 6)
            score_rotations = torch.nn.functional.pad(score_phi, (0, 0, 0, 8-n))
            value_rotations = torch.nn.functional.pad(value_phi, (0, 0, 0, 8-n))
            masks = torch.nn.functional.pad(pairs, (0, 8-n))
            prep = masks.to(a.dtype)
            prep = prep + (~masks.any(-1))[..., None] * torch.nn.functional.one_hot(
                torch.tensor(0, device=a.device), 8)
            prep = prep / prep.sum(-1, keepdim=True).sqrt()
            state = self.full_circuit(neighbors.reshape(-1, 8, 6), a.reshape(-1, 6),
                                      prep.reshape(-1, 8), theta, score_rotations.reshape(-1, 8, 4),
                                      value_rotations.reshape(-1, 8, 4))
            unnormalized = state.reshape(batch, n, 8, 4, 4)[:, :, :n, 0, :] * pairs[..., None]
        else:
            raise ValueError('reference must be False or full')
        probability = unnormalized.abs().square().sum((-1, -2))
        usable = probability > 1e-12
        amplitude = unnormalized / torch.where(usable, probability, 1).sqrt()[..., None, None]
        result = pauli_observables(amplitude, pairs, probability)
        result = torch.where(usable[..., None], result, 0) * m[..., None]
        return result.reshape(*shape[:-1], self.raw_dim)
