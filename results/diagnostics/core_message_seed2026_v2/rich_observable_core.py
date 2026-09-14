"""PennyLane core with a baseline-preserving trainable rich-observable readout."""
from dataclasses import asdict

import pennylane as qml
import torch
from torch import nn

from quantum_core import QuantumCircuitConfig


class RichObservablePQC(nn.Module):
    """Return 12 learned mixtures of 36 local and nearest-pair observables.

    The projection starts as the exact retained [Z, ZZ] readout, so the added
    observables can only enter through training rather than changing the initial
    quantum representation arbitrarily.
    """

    def __init__(self, config: QuantumCircuitConfig):
        super().__init__()
        self.config = config
        self.weights = nn.Parameter(torch.empty(config.depth, config.n_qubits, 3))
        nn.init.uniform_(self.weights, -0.08, 0.08)
        self.readout = nn.Linear(6 * config.n_qubits, 2 * config.n_qubits, bias=False)
        with torch.no_grad():
            self.readout.weight.zero_()
            self.readout.weight[:, : 2 * config.n_qubits].copy_(
                torch.eye(2 * config.n_qubits, dtype=self.readout.weight.dtype)
            )
        device = qml.device("default.qubit", wires=config.n_qubits, shots=None)

        dimension = 1 << config.n_qubits
        basis = torch.arange(dimension, dtype=torch.long)
        specs = []
        # PennyLane orders wire 0 as the most-significant computational bit.
        for family in ("Z", "ZZ", "X", "Y", "XX", "YY"):
            for qubit in range(config.n_qubits):
                wires = (qubit,) if len(family) == 1 else (qubit, (qubit + 1) % config.n_qubits)
                paulis = family if len(family) == len(wires) else family * len(wires)
                permutation = basis.clone()
                phase = torch.ones(dimension, dtype=torch.complex64)
                for pauli, wire in zip(paulis, wires):
                    bit = 1 << (config.n_qubits - 1 - wire)
                    occupied = ((basis & bit) != 0)
                    if pauli == "X":
                        permutation ^= bit
                    elif pauli == "Y":
                        permutation ^= bit
                        phase *= torch.where(
                            occupied,
                            torch.full_like(phase, -1j),
                            torch.full_like(phase, 1j),
                        )
                    elif pauli == "Z":
                        phase *= torch.where(occupied, -torch.ones_like(phase), torch.ones_like(phase))
                    else:
                        raise ValueError(pauli)
                specs.append((permutation, phase))
        self.register_buffer("observable_permutations", torch.stack([item[0] for item in specs]), persistent=False)
        self.register_buffer("observable_phases", torch.stack([item[1] for item in specs]), persistent=False)

        @qml.qnode(device, interface="torch", diff_method="backprop")
        def circuit(angles, weights):
            for layer in range(config.depth):
                if layer == 0 or config.data_reuploading:
                    for qubit in range(config.n_qubits):
                        qml.RY(angles[..., qubit], wires=qubit)
                for qubit in range(config.n_qubits):
                    phi, theta, omega = weights[layer, qubit]
                    qml.RZ(phi, wires=qubit)
                    qml.RY(theta, wires=qubit)
                    qml.RZ(omega, wires=qubit)
                if config.entanglement:
                    for control in range(config.n_qubits):
                        qml.CNOT(wires=[control, (control + 1) % config.n_qubits])
            return qml.state()

        self._device = device
        self._circuit = circuit

    def forward(self, angles):
        state = self._circuit(angles, self.weights)
        permutations = self.observable_permutations.to(state.device)
        phases = self.observable_phases.to(device=state.device, dtype=state.dtype)
        # <psi|P|psi> = sum_j conj(psi[P(j)]) phase_P(j) psi[j].
        transformed = state[:, permutations]
        values = (transformed.conj() * phases[None, :, :] * state[:, None, :]).sum(-1).real
        values = values.to(dtype=angles.dtype, device=angles.device)
        return self.readout(values)

    def metadata(self):
        return {
            **asdict(self.config),
            "raw_observables": 6 * self.config.n_qubits,
            "projected_observables": 2 * self.config.n_qubits,
            "observable_families": ["Z", "ZZ", "X", "Y", "XX", "YY"],
            "baseline_preserving_initialization": True,
            "framework": f"PennyLane {qml.__version__}",
            "device": "default.qubit",
            "shots": None,
            "diff_method": "backprop",
        }
