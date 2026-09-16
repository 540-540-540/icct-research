"""Parameter, storage, memory, and fixed-batch inference comparison."""
import json
import platform
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/home/js_cn/sensing')
BASE = ROOT / 'diagnostics/core_message_seed2026_v2'
sys.path.insert(0, str(BASE))
import run as exp
from model import build_graph, CoreGraphLLM, restore_compact

ARMS = ('plain', 'classical', 'quantum')
SEED = 2026
BATCH = 24
WARMUP = 10
REPEATS = 30


def parameters(model):
    return {'total': sum(p.numel() for p in model.parameters()),
            'trainable': sum(p.numel() for p in model.parameters() if p.requires_grad)}


@torch.no_grad()
def benchmark(model, history, mask):
    model.eval()
    for _ in range(WARMUP):
        model(history, mask)
    torch.cuda.synchronize()
    times = []
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    for _ in range(REPEATS):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(); model(history, mask); end.record(); torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))
    peak = torch.cuda.max_memory_allocated() - baseline
    return {'batch_size': BATCH, 'warmup': WARMUP, 'repeats': REPEATS,
            'latency_ms_mean': statistics.mean(times), 'latency_ms_std': statistics.stdev(times),
            'latency_ms_median': statistics.median(times),
            'throughput_scenes_per_second': BATCH * 1000.0 / statistics.mean(times),
            'incremental_peak_memory_mib': peak / 2**20}


def main():
    assert Path.cwd() == ROOT and platform.node() == 'jscn'
    assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name(0)
    out = BASE / 'complexity_v1'
    if out.exists():
        raise FileExistsError(out)
    out.mkdir()
    with np.load(ROOT / 'data/multitarget_lankershim_v1.npz', allow_pickle=False) as z:
        history = torch.from_numpy(z['test_states'][:BATCH, :20]).float().cuda()
        mask = torch.from_numpy(z['test_mask'][:BATCH]).bool().cuda()
    noise = exp.r.load_snr_noise_map(ROOT / 'results/multitarget_snr/snr_calibration.json')[10]
    generator = torch.Generator(device='cuda').manual_seed(SEED + 100000)
    history = exp.r.add_noise(history, mask, noise['position_sigma_m'], noise['velocity_sigma_mps'], generator)
    results = {}
    for arm in ARMS:
        exp.r.set_seed(SEED)
        graph = build_graph(arm).cuda()
        graph_ck = BASE / 'development' / (arm + '_graph_selected.pt')
        graph.load_state_dict(torch.load(graph_ck, map_location='cpu', weights_only=True)['state'], strict=True)
        graph_info = {'parameters': parameters(graph), 'checkpoint_mib': graph_ck.stat().st_size / 2**20,
                      'inference': benchmark(graph, history, mask)}
        del graph
        torch.cuda.empty_cache()

        exp.r.set_seed(SEED)
        llm = CoreGraphLLM(build_graph(arm)).cuda()
        llm_ck = BASE / 'development' / (arm + '_llm_selected.pt')
        restore_compact(llm, torch.load(llm_ck, map_location='cpu', weights_only=True)['state'])
        llm_info = {'parameters': parameters(llm), 'compact_checkpoint_mib': llm_ck.stat().st_size / 2**20,
                    'inference': benchmark(llm, history, mask)}
        results[arm] = {'graph': graph_info, 'llm': llm_info}
        print(json.dumps({'arm': arm, 'graph_ms': graph_info['inference']['latency_ms_mean'],
                          'llm_ms': llm_info['inference']['latency_ms_mean']}, ensure_ascii=False), flush=True)
        del llm
        torch.cuda.empty_cache()
    results['quantum_circuit'] = {
        'n_qubits': 6, 'depth': 3, 'statevector_complex_amplitudes': 64,
        'trainable_parameters': 54, 'data_encoding_ry_gates_per_edge': 18,
        'trainable_single_qubit_rotation_gates_per_edge': 54,
        'ring_cnot_gates_per_edge': 18, 'measured_expectations_per_edge': 12,
        'backend': 'analytic differentiable PyTorch statevector; shots=None',
        'warning': 'Gate and statevector counts are circuit-complexity descriptors, not hardware runtime or FLOP equivalence.'}
    exp.r.atomic_json(out / 'complexity.json', {'status': 'completed', 'device': torch.cuda.get_device_name(0),
                                                'precision': 'FP32/complex64 quantum state', 'results': results})
    print('COMPLETED', flush=True)


if __name__ == '__main__':
    main()
