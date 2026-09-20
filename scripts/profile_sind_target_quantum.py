"""Short train-data throughput diagnostic; invoke with external timeout 150s. No optimizer updates."""
import gc
import os
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from frontend.sind_target_dataset import SinDTargetPredictionDataset
from prediction.qgnn_raj_pennylane.quantum import PennyLaneRajMultiJCore

OUT = ROOT / 'reports/qgnn/sind_target_self_repair/quantum_current_train_throughput.json'
torch.set_num_threads(4)
torch.manual_seed(20260920)
device = torch.device('cuda:0')
data = SinDTargetPredictionDataset('train')
ids = np.random.default_rng(20260920).choice(len(data), size=32, replace=False)
model = PennyLaneRajMultiJCore(rounds=3).to(device)
model.train()
result = {
    'scope': 'current target_views_v1 train only; no optimizer and zero weight updates',
    'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
    'test_set_used': False,
    'optimizer_updates': 0,
    'device': torch.cuda.get_device_name(device),
    'torch': torch.__version__,
    'rounds': 3,
    'sample_indices': ids.tolist(),
    'train_targets': len(data),
    'objective': 'mean batch of center-slot readout dotted with fixed linspace weights',
    'hard_timeout_seconds': 150,
    'measurements': [],
}

def save():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + '\n')

def measure(batch_size, warmup=False):
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    samples = [data[int(i)] for i in ids[:batch_size]]
    history = torch.stack([s['history_state'] for s in samples]).to(device)
    mask = torch.stack([s['vehicle_mask'] for s in samples]).to(device)
    weights = torch.linspace(-0.9, 1.1, 64, device=device)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    before = torch.cuda.memory_allocated(device)
    started = time.perf_counter()
    output = model(history, mask)
    loss = (output['readout'][:, 0] * weights).sum(-1).mean()
    torch.cuda.synchronize(device)
    after_forward = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize(device)
    stopped = time.perf_counter()
    gradients = [(name, p.grad) for name, p in model.named_parameters() if p.requires_grad]
    present = [(name, grad) for name, grad in gradients if grad is not None]
    entry = {
        'batch': batch_size, 'warmup': warmup,
        'full_context_counts': mask.sum(-1).cpu().tolist(),
        'forward_seconds': after_forward - started,
        'backward_seconds': stopped - after_forward,
        'forward_backward_seconds': stopped - started,
        'target_graphs_per_second': batch_size / (stopped - started),
        'baseline_allocated_gib': before / (1024**3),
        'peak_allocated_gib': torch.cuda.max_memory_allocated(device) / (1024**3),
        'peak_reserved_gib': torch.cuda.max_memory_reserved(device) / (1024**3),
        'loss': float(loss.detach()),
        'finite_output': bool(torch.isfinite(output['readout']).all()),
        'gradient_parameter_tensors': len(gradients),
        'gradient_tensors_present': len(present),
        'all_gradients_finite': all(bool(torch.isfinite(g).all()) for _, g in present),
        'nonzero_gradient_tensors': sum(bool(torch.any(g != 0)) for _, g in present),
        'gradient_l2': float(torch.stack([g.detach().double().square().sum() for _, g in present]).sum().sqrt()),
        'status': 'OK',
    }
    result['measurements'].append(entry)
    save()
    print(json.dumps(entry), flush=True)
    del output, loss, history, mask, weights, samples
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()

save()
for size, warmup in [(1, True), (8, False), (16, False), (32, False)]:
    try:
        measure(size, warmup)
    except torch.OutOfMemoryError as error:
        result['measurements'].append({'batch': size, 'warmup': warmup, 'status': 'OOM', 'error': str(error)[:600]})
        save()
        print('OOM batch=' + str(size), flush=True)
        break
result['completed'] = True
save()
print('RESULT ' + str(OUT), flush=True)
