"""H-initialized Z/Y encoding; an explicit new circuit, with the inherited graph."""
import pennylane as qml
import torch

from experiments.qgnn_inherited.graph import InheritedQGNNGraph, _ReplicaSafePQC


class _HZYPQC(_ReplicaSafePQC):
    def _make_circuit(self):
        n_qubits, depth = self.n_qubits, self.depth

        def circuit(angles, weights):
            # Preserve the inherited CUDA-safe state initialization.
            state = torch.zeros(2 ** n_qubits, dtype=torch.complex128, device=angles.device)
            state[0] = 1
            qml.StatePrep(state, wires=range(n_qubits))
            for qubit in range(n_qubits):
                qml.Hadamard(wires=qubit)
            for layer in range(depth):
                for qubit in range(n_qubits):
                    qml.RZ(angles[..., qubit], wires=qubit)
                    qml.RY(angles[..., qubit + n_qubits], wires=qubit)
                for qubit in range(n_qubits):
                    phi, theta, omega = weights[layer, qubit]
                    qml.RZ(phi, wires=qubit)
                    qml.RY(theta, wires=qubit)
                    qml.RZ(omega, wires=qubit)
                for control in range(n_qubits):
                    qml.CNOT(wires=[control, (control + 1) % n_qubits])
            single = [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]
            pairs = [qml.expval(qml.PauliZ(q) @ qml.PauliZ((q + 1) % n_qubits))
                     for q in range(n_qubits)]
            return single + pairs

        self._device = qml.device('default.qubit', wires=n_qubits, shots=None, seed=0)
        self._circuit = qml.QNode(circuit, self._device, interface='torch', diff_method='backprop')


class HZYQGNNGraph(InheritedQGNNGraph):
    """Six qubits, H once, three ZY/ZYZ/CNOT rounds, unchanged Z/ZZ readout.

    Historical checkpoint shapes match, but their results do not validate this
    circuit. Construct this model afresh and use a dedicated result directory.
    """

    circuit_revision = 'h-once-rz-ry-rz-ry-rz-ring-z-zz-v1'

    def __init__(self, seed=2026):
        super().__init__(seed=seed)
        for layer in self.graph_layers:
            layer.core = _HZYPQC(layer.core)
