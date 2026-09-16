"""PennyLane implementation of the retained six-qubit message core.

This module deliberately keeps the same parameter tensor and output ordering as
``BatchedStatevectorPQC`` so an existing checkpoint can be loaded unchanged.
"""
from dataclasses import asdict

import pennylane as qml
import torch
from torch import nn

from quantum_core import QuantumCircuitConfig


class PennyLaneStatevectorPQC(nn.Module):
    """Six-qubit analytic circuit implemented with PennyLane ``default.qubit``."""

    def __init__(
        self,
        config: QuantumCircuitConfig,
        device_name: str = "default.qubit",
        shots=None,
        diff_method: str = "backprop",
    ):
        super().__init__()
        if config.n_qubits < 2:
            raise ValueError("At least two qubits are required.")
        self.config = config
        self.device_name = device_name
        self.shots = shots
        self.diff_method = diff_method
        self.weights = nn.Parameter(torch.empty(config.depth, config.n_qubits, 3))
        nn.init.uniform_(self.weights, -0.08, 0.08)

        device = qml.device(device_name, wires=config.n_qubits, shots=shots)

        @qml.qnode(device, interface="torch", diff_method=diff_method)
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
            single = [qml.expval(qml.PauliZ(qubit)) for qubit in range(config.n_qubits)]
            pairs = [
                qml.expval(qml.PauliZ(qubit) @ qml.PauliZ((qubit + 1) % config.n_qubits))
                for qubit in range(config.n_qubits)
            ]
            return single + pairs

        self._device = device
        self._circuit = circuit

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        if angles.ndim != 2 or angles.shape[1] != self.config.n_qubits:
            raise ValueError("Encoded-angle shape mismatch.")
        if angles.dtype not in (torch.float32, torch.float64):
            angles = angles.float()
        result = self._circuit(angles, self.weights)
        if not isinstance(result, torch.Tensor):
            result = torch.stack(tuple(result), dim=-1)
        # ``default.qubit`` 0.31 returns analytic expectation values in
        # float64.  The surrounding retained graph uses the input dtype.
        return result.to(dtype=angles.dtype, device=angles.device)

    def metadata(self):
        return {
            **asdict(self.config),
            "state_dimension": 1 << self.config.n_qubits,
            "trainable_rotation_parameters": self.weights.numel(),
            "framework": f"PennyLane {qml.__version__}",
            "device": self.device_name,
            "simulator": "PennyLane analytic statevector" if self.shots is None else "PennyLane shot-based simulator",
            "shots": self.shots,
            "diff_method": self.diff_method,
        }
