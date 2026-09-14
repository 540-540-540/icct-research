"""Exact differentiable six-qubit statevector core retained by the final model."""
from dataclasses import asdict, dataclass
import torch
from torch import nn


@dataclass(frozen=True)
class QuantumCircuitConfig:
    n_qubits: int = 6
    depth: int = 3
    data_reuploading: bool = True
    entanglement: bool = True


class BatchedStatevectorPQC(nn.Module):
    def __init__(self, config: QuantumCircuitConfig):
        super().__init__()
        if config.n_qubits < 2:
            raise ValueError('At least two qubits are required.')
        self.config = config
        self.dimension = 1 << config.n_qubits
        self.weights = nn.Parameter(torch.empty(config.depth, config.n_qubits, 3))
        nn.init.uniform_(self.weights, -0.08, 0.08)
        basis = torch.arange(self.dimension, dtype=torch.long)
        zeros, ones, signs = [], [], []
        for qubit in range(config.n_qubits):
            z = basis[((basis >> qubit) & 1) == 0]
            zeros.append(z); ones.append(z | (1 << qubit))
            signs.append(1.0 - 2.0 * ((basis >> qubit) & 1).to(torch.float32))
        self.register_buffer('zero_indices', torch.stack(zeros), persistent=False)
        self.register_buffer('one_indices', torch.stack(ones), persistent=False)
        z_table = torch.stack(signs)
        zz_table = torch.stack([z_table[q] * z_table[(q + 1) % config.n_qubits]
                                for q in range(config.n_qubits)])
        self.register_buffer('observable_table', torch.cat([z_table, zz_table]), persistent=False)
        permutations = []
        for control in range(config.n_qubits):
            target = (control + 1) % config.n_qubits
            permutation = basis.clone()
            active = ((permutation >> control) & 1).bool()
            permutation[active] ^= 1 << target
            permutations.append(permutation)
        self.register_buffer('cnot_permutations', torch.stack(permutations), persistent=False)
        initial = torch.zeros(self.dimension, dtype=torch.complex64); initial[0] = 1.0 + 0.0j
        self.register_buffer('initial_state', initial, persistent=False)

    @staticmethod
    def _column(theta, batch):
        if theta.ndim == 0:
            theta = theta.expand(batch)
        if theta.ndim != 1 or theta.shape[0] != batch:
            raise ValueError('Gate angle shape mismatch.')
        return theta[:, None]

    def _ry(self, state, theta, qubit):
        theta = self._column(theta, state.shape[0])
        c, s = torch.cos(theta * .5), torch.sin(theta * .5)
        iz, io = self.zero_indices[qubit], self.one_indices[qubit]
        az, ao = state.index_select(1, iz), state.index_select(1, io)
        output = state.index_copy(1, iz, c * az - s * ao)
        return output.index_copy(1, io, s * az + c * ao)

    def _rz(self, state, theta, qubit):
        half = self._column(theta, state.shape[0]) * .5
        pz = torch.complex(torch.cos(half), -torch.sin(half))
        po = torch.complex(torch.cos(half), torch.sin(half))
        iz, io = self.zero_indices[qubit], self.one_indices[qubit]
        output = state.index_copy(1, iz, state.index_select(1, iz) * pz)
        return output.index_copy(1, io, state.index_select(1, io) * po)

    def forward(self, angles):
        if angles.ndim != 2 or angles.shape[1] != self.config.n_qubits:
            raise ValueError('Encoded-angle shape mismatch.')
        if angles.dtype not in (torch.float32, torch.float64):
            angles = angles.float()
        dtype = torch.complex128 if angles.dtype == torch.float64 else torch.complex64
        state = self.initial_state.to(dtype=dtype).expand(angles.shape[0], -1)
        for layer in range(self.config.depth):
            if layer == 0 or self.config.data_reuploading:
                for qubit in range(self.config.n_qubits):
                    state = self._ry(state, angles[:, qubit], qubit)
            for qubit in range(self.config.n_qubits):
                phi, theta, omega = self.weights[layer, qubit]
                state = self._rz(state, phi, qubit)
                state = self._ry(state, theta, qubit)
                state = self._rz(state, omega, qubit)
            if self.config.entanglement:
                for edge in range(self.config.n_qubits):
                    state = state.index_select(1, self.cnot_permutations[edge])
        probability = state.real.square() + state.imag.square()
        return probability @ self.observable_table.to(probability.dtype).T

    def metadata(self):
        return {**asdict(self.config), 'state_dimension': self.dimension,
                'trainable_rotation_parameters': self.weights.numel(),
                'simulator': 'batched analytic PyTorch statevector', 'shots': None}
