"""Read-only A02 diagnostics on TRAIN inputs; no optimizer or label access.

Run from the server project using its validated ICCT environment. Production
QuantumGraph is imported unchanged. Alternate gates exist only in this probe.
"""
import argparse
import inspect
import json
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import prediction.quantum as production
from frontend.symbol_dataset import SharedPredictionInputs


class GateProbe(production.QuantumGraph):
    def graph_rotate(self, state, j, i, layer, angle):
        ops = {2*j: 'Z', 2*i+1: 'X'}
        if layer == 1:
            if self.variant == 'VP_ZX':
                ops = {2*j+1: 'Z', 2*i: 'X'}
            elif self.variant == 'VV_ZX_serial':
                ops = {2*j+1: 'Z', 2*i+1: 'X'}
            elif self.variant == 'VV_XX':
                ops = {2*j+1: 'X', 2*i+1: 'X'}
        return production.pauli_rotation(state, ops, angle)


# Preserve the validated forward literally except for the gate dispatch.
source = textwrap.dedent(inspect.getsource(production.QuantumGraph.forward))
needle = "pauli_rotation(state,{2*j:'Z',2*i+1:'X'},angles[...,l,j,i])"
if source.count(needle) != 1:
    raise RuntimeError('Production forward changed; review probe before using it')
namespace = dict(vars(production))
exec(source.replace(needle, 'self.graph_rotate(state,j,i,l,angles[...,l,j,i])'), namespace)
GateProbe.forward = namespace['forward']


def features(details, target=0):
    q = details['readout'][..., target, :]
    n = details['readout'].shape[-2]
    neighbors = [j for j in range(n) if j != target]
    # Diagnostic only: ordered per-edge moments are NOT an equivariant model
    # replacement. Source order stays fixed when taking a local derivative.
    sc = details['shallow']['pair_connected'][..., neighbors, target, 1:]
    dc = details['deep']['pair_connected'][..., neighbors, target, :]
    extended = torch.cat((q, sc.flatten(-2), dc.flatten(-2)), -1)
    return q, extended


def spectrum(jac):
    singular = torch.linalg.svdvals(jac)
    peak = max(singular[0].item(), 1e-30)
    return {'singular_values': singular.cpu().tolist(),
            'rank_by_relative_tolerance': {str(t): int((singular > peak*t).sum())
                                            for t in (1e-4, 1e-6, 1e-8)}}


def jacobians(model, x, mask, step):
    # Neighbor-only dimensionless perturbations; own state is fixed.
    directions = []
    for j in range(1, len(x)):
        for c in range(4):
            d = torch.zeros_like(x)
            d[j, c] = model.scales[c]
            directions.append(d)
    directions = torch.stack(directions)
    plus = model(x + step*directions, mask.expand(len(directions), -1), return_details=True)
    qp, ep = features(plus)
    del plus
    minus = model(x - step*directions, mask.expand(len(directions), -1), return_details=True)
    qm, em = features(minus)
    return ((qp-qm)/(2*step)).T, ((ep-em)/(2*step)).T, directions


def run(args):
    start = time.perf_counter()
    device = torch.device(args.device)
    torch.set_num_threads(2)
    loader = SharedPredictionInputs(ROOT/'data/f01d/inputs/train_snr_20.npz')
    xall, mall = loader.arrays['state_hat'], loader.arrays['track_exists']
    result = {'purpose': 'connection/order and local readout diagnostic',
              'optimizer_steps': 0, 'labels_opened': False,
              'validation_or_test_opened': False, 'split': 'train', 'snr_db': 20,
              'input_hash_from_existing_loader': loader.input_hash,
              'device': str(device), 'dtype': 'float64/complex128',
              'parameter_status': 'zero initialization and fixed UNTRAINED perturbations',
              'limits': ['No ADE/FDE comparison and no model selection.',
                         'Local frame Jacobian does not establish full-history sufficiency.',
                         'Expanded per-edge readout is a diagnostic, not a production architecture.',
                         'VV_XX changes both connection support and operator axis.',
                         'VV_ZX_serial intentionally tests unsafe order dependence.'],
              'samples': []}
    for n in (3, 8):
        locations = np.argwhere(mall.sum(-1) == n)
        if not len(locations):
            result['samples'].append({'n': n, 'skipped': 'no training frame with this count'})
            continue
        sample, frame = map(int, locations[len(locations)//2])
        indices = np.flatnonzero(mall[sample, frame])
        x = torch.as_tensor(xall[sample, frame, indices], dtype=torch.float64, device=device)
        mask = torch.ones(n, dtype=torch.bool, device=device)
        item = {'n': n, 'sample_index': sample, 'frame': frame,
                'active_slots': indices.tolist(), 'parameter_trials': []}
        result['samples'].append(item)
        for seed in (0, 2023, 2024):
            base = GateProbe(loader.normalization['quantum_scale']).to(device)
            base.variant = 'PV_ZX'
            if seed:
                generator = torch.Generator(device='cpu').manual_seed(seed)
                with torch.no_grad():
                    base.theta[:, 4:].copy_(.35*torch.randn(base.theta[:, 4:].shape,
                                                          generator=generator).to(device))
            trial = {'seed': seed, 'trained': False,
                     'parameters': base.theta.detach().cpu().tolist(), 'variants': {}}
            perm = torch.arange(n-1, -1, -1, device=device)
            with torch.no_grad():
                original = production.QuantumGraph(loader.normalization['quantum_scale']).to(device)
                original.load_state_dict(base.state_dict())
                baseline = original(x, mask, return_details=True)
                parity = (base(x,mask)-baseline['readout']).abs().max().item()
                if parity > 1e-12:
                    raise AssertionError('Probe baseline differs from production')
                trial['production_parity_max_abs'] = parity
                for variant in ('PV_ZX', 'VP_ZX', 'VV_ZX_serial', 'VV_XX'):
                    base.variant = variant
                    details = base(x, mask, return_details=True)
                    out = details['readout']
                    reordered = base(x[perm], mask[perm])
                    off = base(x,mask,interaction_scale=0.)
                    off_ref = original(x,mask,interaction_scale=0.)
                    entry = {'permutation_max_abs': (reordered-out[perm]).abs().max().item(),
                             'change_from_PV_max_abs': (out-baseline['readout']).abs().max().item(),
                             'zero_interaction_parity_max_abs': (off-off_ref).abs().max().item(),
                             'shallow_parity_max_abs': (details['shallow']['singles']-
                                                       baseline['shallow']['singles']).abs().max().item(),
                             'state_norm_error': abs(details['deep_state'].abs().square().sum().item()-1.)}
                    # Change one neighbor velocity, holding the target state fixed.
                    delta = torch.zeros_like(x)
                    delta[1, 2] = 1e-3
                    qp = base(x+delta, mask)[0]
                    qm = base(x-delta, mask)[0]
                    entry['target_readout_neighbor_vx_derivative_norm'] = ((qp-qm)/.002).norm().item()
                    if variant != 'VV_ZX_serial' and entry['permutation_max_abs'] > 1e-10:
                        raise AssertionError(f'{variant} unexpectedly failed permutation')
                    if entry['zero_interaction_parity_max_abs'] > 1e-10 or entry['shallow_parity_max_abs'] > 1e-10:
                        raise AssertionError('Candidate changed unaffected paths')
                    trial['variants'][variant] = entry
                base.variant = 'PV_ZX'
                if seed:
                    j, e, directions = jacobians(base, x, mask, 1e-4)
                    j2, e2, _ = jacobians(base, x, mask, 5e-5)
                    _, singular, vh = torch.linalg.svd(j, full_matrices=True)
                    direction = vh[-1]
                    perturbation = torch.einsum('d,dnc->nc',direction,directions)
                    changes = {}
                    for h in (1e-3, 5e-4):
                        pq, pe = features(base(x+h*perturbation,mask,return_details=True))
                        mq, me = features(base(x-h*perturbation,mask,return_details=True))
                        changes[str(h)] = {'q24_central_slope_norm': ((pq-mq)/(2*h)).norm().item(),
                                           'expanded_central_slope_norm': ((pe-me)/(2*h)).norm().item()}
                    trial['readout_diagnostic'] = {
                        'neighbor_input_dimensions': j.shape[1], 'q_dimensions': j.shape[0],
                        'expanded_dimensions': e.shape[0], 'q24': spectrum(j), 'expanded': spectrum(e),
                        'finite_difference_repeat_q_abs': (j-j2).abs().max().item(),
                        'finite_difference_repeat_expanded_abs': (e-e2).abs().max().item(),
                        'weak_direction_q_slope_norm': (j@direction).norm().item(),
                        'weak_direction_expanded_slope_norm': (e@direction).norm().item(),
                        'physical_direction_by_vehicle': perturbation.cpu().tolist(),
                        'central_step_recheck': changes}
            item['parameter_trials'].append(trial)
            result['elapsed_seconds'] = time.perf_counter()-start
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
            print(json.dumps({'n':n,'seed':seed,'elapsed_seconds':result['elapsed_seconds'],
                              'variants':trial['variants'],
                              'readout':trial.get('readout_diagnostic',{}).get('q24',{})}),flush=True)
    result['completed'] = True
    result['elapsed_seconds'] = time.perf_counter()-start
    result['peak_gpu_memory_bytes'] = torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device',default='cuda:1')
    parser.add_argument('--output',type=Path,default=ROOT/'reports/qgnn_connection_readout_probe/results.json')
    run(parser.parse_args())
