"""Autograd cross-check of one local finite-difference diagnostic, no training."""
import json
from pathlib import Path
import torch
from probe_connections_readout import ROOT, GateProbe, SharedPredictionInputs, features, spectrum

torch.set_num_threads(2)
folder = ROOT/'reports/qgnn_connection_readout_probe'
previous = json.loads((folder/'results.json').read_text())
case = next(s for s in previous['samples'] if s['n'] == 8)
trial = next(t for t in case['parameter_trials'] if t['seed'] == 2023)
loader = SharedPredictionInputs(ROOT/'data/f01d/inputs/train_snr_20.npz')
x = torch.tensor(loader.arrays['state_hat'][case['sample_index'], case['frame'], case['active_slots']],
                 dtype=torch.float64,device='cuda:1',requires_grad=True)
model = GateProbe(loader.normalization['quantum_scale']).to(x.device)
model.variant = 'PV_ZX'
with torch.no_grad():
    model.theta.copy_(torch.tensor(trial['parameters'],dtype=torch.float64,device=x.device))
mask = torch.ones(8,dtype=torch.bool,device=x.device)
q,e = features(model(x,mask,return_details=True))
rows = []
for component in q:
    grad = torch.autograd.grad(component,x,retain_graph=True)[0]
    rows.append((grad[1:]*model.scales).flatten())
jac = torch.stack(rows)
physical = torch.tensor(trial['readout_diagnostic']['physical_direction_by_vehicle'],
                        dtype=x.dtype,device=x.device)
direction = (physical[1:]/model.scales).flatten()
slopes=[]
for component in e:
    grad=torch.autograd.grad(component,x,retain_graph=True)[0]
    slopes.append((grad*physical).sum())
result={'method':'reverse-mode automatic differentiation', 'seed':2023,'n':8,
        'labels_opened':False,'optimizer_steps':0,'q24_spectrum':spectrum(jac),
        'q24_weak_direction_slope_norm':(jac@direction).norm().item(),
        'expanded_weak_direction_slope_norm':torch.stack(slopes).norm().item()}
assert result['q24_weak_direction_slope_norm'] < 1e-6
assert abs(result['expanded_weak_direction_slope_norm']-
           trial['readout_diagnostic']['weak_direction_expanded_slope_norm']) < 1e-6
result['passed']=True
(folder/'autograd_check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
