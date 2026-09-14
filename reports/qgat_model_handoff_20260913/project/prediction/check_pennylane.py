"""Archived PyTorch-reference versus independent PennyLane crosscheck of A02 Q24; no fitting or optimizer steps.

Run: python -m prediction.check_pennylane --device cuda:1
Both simulators use CUDA complex128. PennyLane gets an explicit CUDA |0>
StatePrep because its default initial state is CPU in the installed version.
"""
import argparse
import gc
import json
import math
import time
from pathlib import Path

import numpy as np
import pennylane as qml
import torch
from .quantum_torch_reference import QuantumGraph

ROOT = Path(__file__).resolve().parents[1]


def observables(n):
    singles = [getattr(qml, 'Pauli'+p)(2*i+r)
               for i in range(n) for r in range(2) for p in 'XYZ']
    pairs = [getattr(qml, 'Pauli'+p)(2*j) @ getattr(qml, 'Pauli'+q)(2*i+1)
             for j in range(n) for i in range(n) if i != j
             for p, q in [('Z','X'), ('Z','Y'), ('Z','Z'), ('X','Z')]]
    local = [getattr(qml, 'Pauli'+p)(2*i) @ getattr(qml, 'Pauli'+p)(2*i+1)
             for i in range(n) for p in 'XY']
    return singles + pairs + local


def make_circuit(n, depth, shallow=False, native=False):
    dev = qml.device('default.qubit', wires=2*n, shots=None)
    obs = observables(n)

    @qml.qnode(dev, interface='torch', diff_method='backprop')
    def circuit(x, mask, scales, theta):
        initial = torch.zeros(4**n, dtype=torch.complex128, device=x.device)
        initial[0] = 1
        qml.StatePrep(initial, wires=range(2*n))
        clean = torch.where(mask[:, None], x, 0.)
        for l in range(depth):
            gain = .25 + 3.75 * torch.sigmoid(theta[l, :4])
            for i in range(n):
                angles = (.5*torch.atan(clean[i]*gain/scales)
                          + (math.pi/2 if l == 0 else 0.)) * mask[i]
                for r in range(2):
                    qml.RY(angles[2*r], wires=2*i+r)
                    qml.RZ(angles[2*r+1], wires=2*i+r)
            for j in range(n):
                for i in range(n):
                    if i == j:
                        continue
                    delta = clean[j]-clean[i]
                    distance2 = delta[:2].square().sum()
                    approach = torch.tanh(-torch.dot(delta[:2],delta[2:])
                                          /torch.sqrt(distance2+1)/15)
                    edge = torch.cat((delta[:2]/45, delta[2:]/15, approach[None]))
                    angle = (torch.tanh(torch.dot(edge,theta[l,4:9])+theta[l,9])
                             * (math.pi/2)*torch.sigmoid(theta[l,10])
                             * torch.exp(-distance2/(2*45**2))/7 * mask[j]*mask[i])
                    qml.PauliRot(angle, 'ZX', wires=[2*j,2*i+1])
            if shallow:
                break
            for i in range(n):
                qml.IsingXX(theta[l,15]*mask[i], wires=[2*i,2*i+1])
                qml.RY(theta[l,12]*mask[i], wires=2*i)
                qml.RZ(theta[l,11]*mask[i], wires=2*i)
                qml.RY(theta[l,14]*mask[i], wires=2*i+1)
                qml.RZ(theta[l,13]*mask[i], wires=2*i+1)
        if native:
            return tuple(qml.expval(o) for o in obs)
        return qml.state()
    return circuit


def matrix_moments(state, n):
    """Independent dense 2x2/4x4 PennyLane matrices, no bit-index production code."""
    values = []
    for op in observables(n):
        wires = list(op.wires)
        order = wires + [w for w in range(2*n) if w not in wires]
        vector = state.reshape([2]*(2*n)).permute(order).reshape(2**len(wires),-1)
        matrix = torch.as_tensor(qml.matrix(op, wire_order=wires), device=state.device, dtype=state.dtype)
        values.append((vector.conj()*(matrix@vector)).sum().real)
    return torch.stack(values)


def unpack(raw, mask):
    n = len(mask)
    single = raw[:6*n].reshape(n,6)
    cursor = 6*n
    rows = []
    for j in range(n):
        cols = []
        for i in range(n):
            cols.append(raw[cursor:cursor+4] if i != j else raw[:4]*0)
            if i != j:
                cursor += 4
        rows.append(torch.stack(cols))
    pair = torch.stack(rows)
    connected = torch.stack([
        torch.stack([(pair[j,i]-single[j,[2,2,2,0]]*single[i,[3,4,5,5]])
                     *mask[j]*mask[i] if i != j else pair[j,i]*0
                     for i in range(n)]) for j in range(n)])
    local = raw[cursor:].reshape(n,2)
    return dict(singles=single,pair_raw=pair,pair_connected=connected,
                local_raw=local,local_connected=(local-single[:,:2]*single[:,3:5])*mask[:,None])


def readout(shallow, deep, x, mask):
    clean = torch.where(mask[:,None],x,0.)
    rows = []
    for i in range(len(mask)):
        weighted = []
        for j in range(len(mask)):
            delta = clean[j]-clean[i]
            approach = torch.tanh(-torch.dot(delta[:2],delta[2:])
                                  /torch.sqrt(delta[:2].square().sum()+1)/15)
            weighted.append(shallow['pair_connected'][j,i,1:]*approach)
        rows.append(torch.cat((shallow['singles'][i],
            shallow['pair_connected'][:,i,1:].sum(0)/7,deep['singles'][i],
            deep['local_connected'][i],deep['pair_connected'][:,i].sum(0)/7,
            torch.stack(weighted).sum(0)/7))*mask[i])
    return torch.stack(rows)


def run(device, selected_case=None):
    start = time.monotonic()
    torch.manual_seed(20260912)
    checks = []
    report = dict(pennylane_version=qml.__version__, torch_version=torch.__version__,
        device=device, gpu=torch.cuda.get_device_name(device), backend='default.qubit',
        interface='torch', diff_method='backprop', precision='float64/complex128',
        initial_state='explicit CUDA |0> via StatePrep; default CPU init fails on cuda:1',
        formal_training=False, optimizer_steps=0, checks=checks, passed=False,
        thresholds=dict(output=1e-10, gradient=1e-8), cases=[],
        measurement_method='All cases: PennyLane state + PennyLane local matrices; N3 mixed L3 additionally native qml.expval',
        scope='Selected finite cases; not proof for all parameters. No phase alignment applied.')

    def check(case, name, a, b, tolerance=1e-10):
        err = float((a.detach()-b.detach()).abs().max())
        evidence = dict(case=case,name=name,passed=math.isfinite(err) and err<=tolerance,
                        max_abs_error=err,threshold=tolerance,
                        production_max_abs=float(a.detach().abs().max()),
                        reference_max_abs=float(b.detach().abs().max()),elements=a.numel())
        if 'gradient' in name:
            evidence['reference_near_zero'] = evidence['reference_max_abs'] <= 1e-12
        checks.append(evidence)
        print(json.dumps(evidence),flush=True)
        if not evidence['passed']:
            raise AssertionError(f'{case}: {name}')

    cases = [(n,l,mode,None) for n in (2,3,4) for l in (1,2,3)
             for mode in ('initial','mixed')]
    cases += [(3,3,'masked',None),(8,1,'mixed',None),(8,3,'masked',None)]
    from frontend.symbol_dataset import SharedPredictionInputs
    loader = SharedPredictionInputs(ROOT/'data/f01d/inputs/train_snr_20.npz')
    counts = loader.arrays['track_exists'].sum(-1)
    candidates = np.argwhere(counts >= 3)
    sample_index, frame_index = map(int,candidates[0])
    sample = loader[sample_index]
    real = (sample['state_hat'][frame_index], sample['track_exists'][frame_index])
    cases.append((8,3,'real_train',real))
    # Append to preserve the original 22-case random-number schedule.
    cases.append((8,3,'mixed',None))
    scales = torch.as_tensor(loader.normalization['quantum_scale'],device=device,dtype=torch.float64)
    report['real_cache'] = dict(path='data/f01d/inputs/train_snr_20.npz',split='train',
        sample_index=sample_index, frame_index=frame_index, label_access=False,
        active_count=int(np.sum(real[1])))
    previous = None
    if selected_case:
        cases = [case for case in cases if f'N{case[0]}_L{case[1]}_{case[2]}' == selected_case]
        if not cases:
            raise ValueError(f'Unknown case: {selected_case}')
        previous = json.loads((ROOT/'reports/f02/pennylane_crosscheck.json').read_text())
        if not previous['passed'] or previous['device'] != device:
            raise ValueError('Single-case merge requires an existing passed report on the same device')
    try:
        for n,l,mode,real_values in cases:
            case = f'N{n}_L{l}_{mode}'
            if case == 'N8_L3_mixed':
                # Identical supplemental data/parameters for full and single-case runs.
                torch.manual_seed(20260912)
            mask = torch.ones(n,device=device,dtype=torch.bool)
            x = torch.randn(n,4,device=device,dtype=torch.float64)*15
            if mode == 'masked':
                mask[:] = False
                mask[[0, n//2, n-1] if n==8 else [0,2]] = True
                x[~mask] = float('nan')
            if real_values is not None:
                x = torch.as_tensor(real_values[0],device=device,dtype=torch.float64)
                mask = torch.as_tensor(real_values[1],device=device,dtype=torch.bool)
            model = QuantumGraph(scales,depth=l).to(device)
            if mode != 'initial':
                with torch.no_grad():
                    model.theta.add_(torch.randn_like(model.theta)*.45)
            xp = x.detach().clone().requires_grad_()
            xr = x.detach().clone().requires_grad_()
            tr = model.theta.detach().clone().requires_grad_()
            details = model(xp,mask,return_details=True)
            ps = make_circuit(n,l,shallow=True)(xr,mask,scales,tr)
            pd = make_circuit(n,l)(xr,mask,scales,tr)
            assert ps.device == xp.device and pd.device == xp.device
            assert ps.dtype == torch.complex128 and pd.dtype == torch.complex128
            check(case,'shallow_state_raw_no_phase_alignment',details['shallow_state'],ps)
            check(case,'deep_state_raw_no_phase_alignment',details['deep_state'],pd)
            rs,rd = matrix_moments(ps,n),matrix_moments(pd,n)
            ss,dd = unpack(rs,mask),unpack(rd,mask)
            for level,moments in [('shallow',ss),('deep',dd)]:
                for key,val in moments.items():
                    check(case,level+'_'+key,details[level][key],val)
            qr = readout(ss,dd,xr,mask)
            check(case,'Q24_readout',details['readout'],qr)
            native = n==3 and l==3 and mode=='mixed'
            if native:
                ns = torch.stack(make_circuit(n,l,shallow=True,native=True)(xr,mask,scales,tr))
                nd = torch.stack(make_circuit(n,l,native=True)(xr,mask,scales,tr))
                check(case,'native_expval_all_shallow_moments',rs,ns)
                check(case,'native_expval_all_deep_moments',rd,nd)
                qnative = readout(unpack(ns,mask),unpack(nd,mask),xr,mask)
                check(case,'native_expval_Q24',qr,qnative)
            for projection in range(2):
                w = torch.randn_like(qr)
                gp = torch.autograd.grad((details['readout']*w).sum(),(model.theta,xp),retain_graph=True)
                gr = torch.autograd.grad((qr*w).sum(),(tr,xr),retain_graph=True)
                for label,a,b in zip(('theta','input'),gp,gr):
                    check(case,f'projection{projection}_{label}_gradient',a,b,1e-8)
                if native:
                    gn = torch.autograd.grad((qnative*w).sum(),(tr,xr),retain_graph=True)
                    for label,a,b in zip(('theta','input'),gp,gn):
                        check(case,f'native_projection{projection}_{label}_gradient',a,b,1e-8)
                if (~mask).any():
                    check(case,f'projection{projection}_masked_input_gradient',gr[1][~mask],torch.zeros_like(gr[1][~mask]),1e-8)
            report['cases'].append(dict(name=case,n=n,depth=l,active=int(mask.sum()),
                 parameter_mode=mode,output_device=str(pd.device),native_expval=native,
                 random_seed=20260912 if case == 'N8_L3_mixed' else None,
                 rng_scope='independent_case' if case == 'N8_L3_mixed' else 'original_sequential_run'))
            del details,ps,pd,rs,rd,ss,dd,qr,model,xp,xr,tr,gp,gr
            if native:
                del ns,nd,qnative,gn
            gc.collect()
            torch.cuda.empty_cache()
        report['passed'] = all(c['passed'] for c in checks)
    finally:
        elapsed = time.monotonic()-start
        report['runs'] = [dict(case_selection=selected_case or 'all',
                              elapsed_seconds=elapsed, executed_cases=[c['name'] for c in report['cases']])]
        if previous is not None:
            checks[:0] = [c for c in previous['checks'] if c['case'] != selected_case]
            report['cases'][:0] = [c for c in previous['cases'] if c['name'] != selected_case]
            report['runs'][:0] = previous.get('runs', [dict(case_selection='original_22_cases',
                elapsed_seconds=previous['elapsed_seconds'], executed_cases=[c['name'] for c in previous['cases']])])
        report['check_count'] = len(checks)
        report['case_count'] = len(report['cases'])
        report['elapsed_seconds'] = sum(r['elapsed_seconds'] for r in report['runs'])
        report['max_output_error'] = max((c['max_abs_error'] for c in checks if 'gradient' not in c['name']),default=0.)
        report['max_gradient_error'] = max((c['max_abs_error'] for c in checks if 'gradient' in c['name']),default=0.)
        report['near_zero_gradient_checks'] = [c['case']+':'+c['name'] for c in checks if c.get('reference_near_zero')]
        path=ROOT/'reports/f02/pennylane_crosscheck.json'
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('checks','cases','near_zero_gradient_checks')}),flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device',default='cuda:1')
    parser.add_argument('--case',help='Run only this case and merge into an existing passed report')
    args = parser.parse_args()
    run(args.device,args.case)

