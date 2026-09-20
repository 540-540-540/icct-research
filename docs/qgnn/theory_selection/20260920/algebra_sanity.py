"""Small algebra checks, not an ICCT model, benchmark, or training run.

Uses a two-feature-qubit surrogate to check configuration-space identities.
The proposed Primary uses four/six feature qubits. No ICCT dataset is read.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
from scipy.linalg import expm

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)

def config_h(a: np.ndarray, risk: np.ndarray, joint: np.ndarray):
    n = len(a)
    tuples = [(j, k) for j in range(n) for k in range(n) if j != k]
    h = np.zeros((len(tuples), len(tuples)))
    for s, (j, k) in enumerate(tuples):
        for t, (l, m) in enumerate(tuples):
            h[s, t] = (float(k == m) * a[j, l] + float(j == l) * a[k, m])
        h[s, s] += .3 * (risk[j] + risk[k] + joint[j, k])
    h = h / max(1., np.abs(h).sum(1).max())
    return tuples, h

def conditional(features: np.ndarray, tuples):
    fs = []
    for j, k in tuples:
        # Fixed role slots, different vehicles, no trainable cross-agent preprocessing.
        rj = expm(-.5j * features[j, 0] * Y) @ expm(-.5j * features[j, 1] * Z)
        rk = expm(-.5j * features[k, 0] * X) @ expm(-.5j * features[k, 1] * Y)
        f = expm(-.5j * (features[j, 2] - features[k, 2]) * np.kron(Z, X)) @ np.kron(rj, rk)
        fs.append(f)
    return np.asarray(fs)

def run(blocks, dephase=False):
    m = len(blocks[0][0])
    psi = np.zeros((m, 4), dtype=complex); psi[:, 0] = 1/math.sqrt(m)
    rho = np.zeros((m, 4, 4), dtype=complex); rho[:, 0, 0] = 1/m
    for h, fs in blocks:
        u = expm(-1j * h)
        if not dephase:
            psi = np.einsum('sfg,sg->sf', fs, u @ psi)
        else:
            rho = np.einsum('st,tfg->sfg', np.abs(u)**2, rho)
            rho = np.einsum('sab,sbc,sdc->sad', fs, rho, fs.conj())
    if dephase:
        return rho.sum(0)
    return psi.T @ psi.conj(), psi

rng = np.random.default_rng(20260920)
n = 4
raw = []
blocks = []
for _ in range(4):
    a = rng.uniform(.1, .7, (n, n)); a=(a+a.T)/2; np.fill_diagonal(a,0)
    risk = rng.uniform(.1,1,n)
    joint = rng.uniform(0,1,(n,n)); joint=(joint+joint.T)/2
    features=rng.normal(size=(n,3))
    tuples,h=config_h(a,risk,joint)
    fs=conditional(features,tuples)
    raw.append((a,risk,joint,features)); blocks.append((h,fs))

rho,psi=run(blocks)
obs=[np.kron(p,q) for p in (I,X,Y,Z) for q in (I,X,Y,Z)][1:]
read=lambda r: np.asarray([np.trace(r@o).real for o in obs])
original=read(rho)
perm=np.array([2,0,3,1])
pblocks=[]
for a,risk,joint,features in raw:
    t,h=config_h(a[np.ix_(perm,perm)], risk[perm], joint[np.ix_(perm,perm)])
    pblocks.append((h,conditional(features[perm],t)))
prho,_=run(pblocks)
reverse,_=run(list(reversed(blocks)))
incoherent=run(blocks,dephase=True)
phase=np.exp(.731j)*psi
phase_rho=phase.T@phase.conj()
terminal=expm(-.8j*blocks[0][0])@psi
terminal_rho=terminal.T@terminal.conj()
check={
 'scope':'Toy configuration algebra only; feature qubits=2; not a full Primary implementation or ICCT experiment',
 'execution_location':'assistant artifact container, CPU',
 'seed':20260920,'neighbors':n,'ordered_configurations':len(tuples),
 'norm_error':float(abs(np.linalg.norm(psi)**2-1)),
 'density_trace_error':float(abs(np.trace(rho)-1)),
 'conditional_unitary_error':float(max(np.max(np.abs(f.conj().T@f-np.eye(4))) for _,fs in blocks for f in fs)),
 'permutation_readout_max_error':float(np.max(np.abs(original-read(prho)))),
 'global_phase_readout_max_error':float(np.max(np.abs(original-read(phase_rho)))),
 'terminal_index_only_hop_readout_max_error':float(np.max(np.abs(original-read(terminal_rho)))),
 'temporal_reversal_readout_max_difference':float(np.max(np.abs(original-read(reverse)))),
 'dephased_configuration_readout_max_difference':float(np.max(np.abs(original-read(incoherent)))),
 'no_new_training':True,
}
assert max(check[k] for k in ('norm_error','density_trace_error','conditional_unitary_error','permutation_readout_max_error','global_phase_readout_max_error','terminal_index_only_hop_readout_max_error'))<1e-10
assert check['temporal_reversal_readout_max_difference']>1e-5
assert check['dephased_configuration_readout_max_difference']>1e-5
check['status']='PASS_IDENTITY_CHECKS_NOT_PREDICTION_VALIDATION'
Path(__file__).with_name('algebra_sanity.json').write_text(json.dumps(check,indent=2))
print(json.dumps(check,indent=2))

# Resource arithmetic, independent of server framework installations.
rows=[]
for K in [8,12,16]:
    nn=K-1; b=math.ceil(math.log2(nn)); m1=nn; m2=nn*(nn-1)
    rows.append({'K':K,'M1':m1,'M2':m2,'Q1':b+4,'Q2':2*b+6,
      'full_state_dimensions':[2**(b+4),2**(2*b+6)],
      'reduced_state_dimensions':[m1*16,m2*64],
      'full_complex64_KiB_per_target':8*(2**(b+4)+2**(2*b+6))/1024,
      'reduced_complex64_KiB_per_target':8*(m1*16+m2*64)/1024,
      'conditional_pauli_rotations_per_target':4*(80*m1+156*m2),
      'backup_Q':2*K,'backup_complex64_MiB_single_state':8*2**(2*K)/2**20})
params={'temporal':224+6336+64,'quantum_encoding_and_dynamics':396+24+4+4,
 'quantum_softtoken_adapter':70+35*128+128+128*768+768,
 'type_time_embedding':6*768,'own_token_projection':64+32*768+768,
 'context_head_projection':320+160*64+64,
 'motion_embedding':1681*768,'continuous_motion':4*768,
 'history_adapter':2*768+2*(768*768+768),
 'future_queries':20*768,'token_head':2*768+768*1681+1681,
 'coordinate_head':2*834+834*256+256+256*2+2,'lora':4*(768*8+8*2304)}
resources={'primary_by_context':rows,'primary_trainable_parameter_breakdown':params,
 'primary_trainable_total':sum(params.values()),'parameter_count_status':'exact arithmetic for proposed layer definitions; not instantiated model count',
 'steps_20_epochs_15802_B32':20*math.ceil(15802/32),
 'single_gpu_training_minutes_excluding_eval_profile_scenarios':{str(t):20*math.ceil(15802/32)*t/60 for t in [.25,.5,.7,1.]}}
Path(__file__).with_name('complexity_tables.json').write_text(json.dumps(resources,indent=2))
print(json.dumps(resources,indent=2))
