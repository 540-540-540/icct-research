"""Predeclared quantum-circuit mechanism ablations for core-message v2."""
import torch

from model import CoreMessageLayer
from target_interaction_graph import TargetInteractionGNN, ForecasterConfig
from quantum_core import BatchedStatevectorPQC, QuantumCircuitConfig


ABLATION_CONFIGS = {
    'quantum_no_entanglement': QuantumCircuitConfig(n_qubits=6, depth=3, data_reuploading=True, entanglement=False),
    'quantum_depth1': QuantumCircuitConfig(n_qubits=6, depth=1, data_reuploading=True, entanglement=True),
    'quantum_no_reupload': QuantumCircuitConfig(n_qubits=6, depth=3, data_reuploading=False, entanglement=True),
}


def build_ablation_graph(arm, seed=2026):
    if arm not in ABLATION_CONFIGS:
        raise ValueError(arm)
    torch.manual_seed(seed)
    config = ForecasterConfig()
    model = TargetInteractionGNN(config)
    torch.manual_seed(seed + 1)
    layer = CoreMessageLayer(config, 'quantum')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2028)
        layer.core = BatchedStatevectorPQC(ABLATION_CONFIGS[arm])
    model.graph_layers[1] = layer
    return model
