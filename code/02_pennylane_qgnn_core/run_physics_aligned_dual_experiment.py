"""Seed-2026 Graph-only matched test of the physics-aligned hierarchical dual-QGNN."""
import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/home/js_cn/sensing')
BASE = ROOT / 'diagnostics/core_message_seed2026_v2'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BASE))

import experiment_utils as r
import run as retained
from physics_aligned_dual_model import build_physics_aligned_dual_graph
from run_converged_training import train_stage
from run_multitarget_experiment import prediction_loss

SEED = 2026
ARMS = ('physics_classical_dual', 'physics_quantum_dual')
REFERENCE = BASE / 'converged_protocol_seed2026_v1'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_hash(mapping):
    return r.tensor_mapping_sha256({name: value.detach().cpu() for name, value in mapping.items()})


def banks():
    with np.load(BASE / 'retained_inputs/circuit_split_indices.npz', allow_pickle=False) as split:
        train_indices = split['circuit_train_indices'].copy()
        dev_indices = split['circuit_dev_indices'].copy()
    with np.load(ROOT / 'data/multitarget_lankershim_v1.npz', allow_pickle=False) as data:
        states = data['train_states'].copy()
        masks = data['train_mask'].copy()
    selection_indices = r.select_profile_indices(dev_indices, 256)

    def bank(indices):
        return {'history': torch.from_numpy(states[indices, :20]).float().cuda(),
                'future': torch.from_numpy(states[indices, 20:]).float().cuda(),
                'mask': torch.from_numpy(masks[indices]).bool().cuda()}
    return bank(train_indices), bank(selection_indices), bank(dev_indices), train_indices, selection_indices, dev_indices


def expected_inputs():
    values = {}
    with (REFERENCE / 'quantum_graph.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            record = json.loads(line)
            values[f"graph_{record['epoch']}"] = record['input_sha256']
    assert len(values) == 40
    return values


def smoke(output_dir, train):
    evidence = {}
    common_outside = None
    common_noncore = None
    history, future, mask = train['history'][:2], train['future'][:2], train['mask'][:2]
    permutation = torch.arange(history.shape[2]-1, -1, -1, device='cuda')
    for arm in ARMS:
        r.set_seed(SEED)
        model = build_physics_aligned_dual_graph(arm).cuda()
        outside = tensor_hash({n:p for n,p in model.named_parameters() if not n.startswith('graph_layers.')})
        noncore = tensor_hash({n:p for n,p in model.named_parameters() if n.startswith('graph_layers.') and '.core.' not in n})
        if common_outside is None: common_outside = outside
        if common_noncore is None: common_noncore = noncore
        assert outside == common_outside and noncore == common_noncore
        assert model.graph_layers[0].role == 'physical' and model.graph_layers[1].role == 'contextual'
        model.train()
        output = model(history, mask)
        loss = prediction_loss(output, future, mask, graph_weighting=True)
        loss.backward()
        core_gradients = [max(float(p.grad.abs().max()) for p in layer.core.parameters() if p.grad is not None)
                          for layer in model.graph_layers]
        encoder_gradients = [max(float(layer.angle_scale_raw.grad.abs().max()), float(layer.angle_bias.grad.abs().max()))
                             for layer in model.graph_layers]
        assert min(core_gradients) > 0 and min(encoder_gradients) > 0
        model.eval()
        with torch.no_grad():
            original = model(history, mask)['future_position']
            permuted = model(history[:, :, permutation], mask[:, permutation])['future_position']
            layer_angles = []
            nodes = torch.randn(2, history.shape[2], 128, device='cuda')
            edges = torch.randn(2, history.shape[2], history.shape[2], 7, device='cuda')
            for layer in model.graph_layers:
                encoded = layer.encode_angles(layer.relation_features(nodes, edges))
                layer_angles.append({'minimum': float(encoded.min()), 'maximum': float(encoded.max())})
        equivariance_error = float((permuted-original[:, :, permutation]).abs().max())
        assert equivariance_error < 2e-5
        assert all(-np.pi < a['minimum'] <= a['maximum'] < np.pi for a in layer_angles)
        evidence[arm] = {'loss': float(loss.detach()), 'core_gradient_max_by_layer': core_gradients,
                         'frontend_gradient_max_by_layer': encoder_gradients,
                         'permutation_equivariance_max_error': equivariance_error,
                         'angle_range_by_layer': layer_angles,
                         'total_parameters': sum(p.numel() for p in model.parameters()),
                         'core_parameters_by_layer': [sum(p.numel() for p in x.core.parameters()) for x in model.graph_layers]}
        del model
        torch.cuda.empty_cache()
    r.atomic_json(output_dir / 'smoke.json', evidence)
    return evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-name', default='physics_aligned_dual_seed2026_v1')
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    assert Path.cwd() == ROOT and platform.node() == 'jscn'
    assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name(0)
    torch.set_num_threads(4)
    output_dir = BASE / args.output_name
    output_dir.mkdir(exist_ok=False)
    train, selection, full_dev, train_idx, select_idx, dev_idx = banks()
    retained.NOISE = r.load_snr_noise_map(ROOT / 'results/multitarget_snr/snr_calibration.json')
    evidence = smoke(output_dir, train)
    if args.smoke_only:
        r.atomic_json(output_dir / 'completed.json', {'status':'smoke_completed','evidence':evidence})
        print(json.dumps(evidence, ensure_ascii=False), flush=True)
        return
    sources = [BASE/'physics_aligned_dual_model.py', BASE/'run_physics_aligned_dual_experiment.py',
               BASE/'run_converged_training.py', ROOT/'target_interaction_graph.py']
    source_hashes = {str(p):sha256(p) for p in sources}
    r.atomic_json(output_dir/'protocol.json', {
        'seed':SEED, 'arms':list(ARMS), 'hypothesis':'physics-aligned hierarchical dual message cores',
        'quantum_backend':'PennyLane default.qubit analytic statevector', 'shots':None,
        'layer_roles':['physical direct interaction','contextual higher-order interaction'],
        'relation_features':'7 bounded physical edge features + 4 grouped node correlations + 1 RMS node difference',
        'encoding':'12 features -> six RY plus six RZ angles using learnable-scale 2*atan',
        'quantum_circuit':'6 qubits, depth 3, data reuploading, directed ring CNOT, 54 parameters per layer',
        'classical_control':'12->2->12 low-rank SiLU MLP, 60 parameters per layer',
        'graph':{'max_epochs':40,'patience':6,'min_delta':1e-4}, 'batch_size':24,
        'selection':'four-SNR macro ADE + 0.35 FDE on fixed 256-scene selection split',
        'train_indices':train_idx.tolist(), 'selection_indices':select_idx.tolist(), 'full_dev_indices':dev_idx.tolist(),
        'same_input_hashes_as_converged_protocol_required':True, 'matched_noncore_initialization_required':True,
        'no_llm':True, 'no_test':True, 'gpu':torch.cuda.get_device_name(0), 'python':sys.executable,
        'source_hashes':source_hashes})
    summaries = {}
    for arm in ARMS:
        r.set_seed(SEED)
        model = build_physics_aligned_dual_graph(arm).cuda()
        audit = train_stage(model, arm, 'graph', 40, 6, train, selection, output_dir, expected_inputs())
        metrics, arrays = retained.evaluate(model, full_dev, collect=True)
        np.savez_compressed(output_dir/f'{arm}_graph_full_dev.npz', **arrays, indices=dev_idx)
        summaries[arm] = {'graph':metrics, 'selected_epoch':audit['selected_epoch'],
                          'executed_epochs':audit['executed_epochs'], 'stop_reason':audit['stop_reason'],
                          'parameters':sum(p.numel() for p in model.parameters())}
        r.atomic_json(output_dir/'summary.json', summaries)
        del model
        torch.cuda.empty_cache()
    assert all(sha256(p)==digest for p,digest in source_hashes.items())
    done={'status':'completed','seed':SEED,'summaries':summaries,'source_hashes_verified':True,
          'matched_noncore_initialization_verified':True,'same_input_hashes_verified':True,
          'no_llm':True,'no_test':True}
    r.atomic_json(output_dir/'completed.json',done);r.atomic_json(output_dir/'progress.json',done)
    print(json.dumps(done,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
