"""Exploratory frozen-circuit probes; no main-model/LLM training, no test access."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from probe_connections_readout import ROOT, GateProbe, SharedPredictionInputs


def select_inputs(split, maximum):
    loader = SharedPredictionInputs(ROOT/f'data/f01d/inputs/{split}_snr_20.npz')
    metadata = json.loads((ROOT/f'data/f01d/metadata/{split}.json').read_text())['samples']
    groups = {}
    mask = loader.arrays['track_exists']
    eligible = (mask[:, -1] & (mask.sum(1) >= 3)).any(-1)
    for row in metadata:
        index = row['sample_index']
        if eligible[index]:
            groups.setdefault(row['episode_id'], []).append(index)
    keys = list(groups)
    chosen = np.linspace(0,len(keys)-1,min(maximum,len(keys))).round().astype(int)
    indices = [groups[keys[k]][len(groups[keys[k]])//2] for k in chosen]
    selected = [loader[i] for i in indices]
    pack = {k:torch.as_tensor(np.stack([s[k] for s in selected])) for k in selected[0]}
    manifest = {'split':split,'snr_db':20,'indices':indices,
                'rows':[metadata[i] for i in indices],
                'input_hash':loader.input_hash,
                'selection':'Evenly spaced episodes in cache order; middle input-eligible origin per episode; labels not used'}
    return pack, manifest, loader.normalization


def relation_readout(details):
    # Four bounded relative coordinates weight six existing pair moments.
    # Sum over source vehicles gives a fixed-width, permutation-equivariant
    # extension. This changes classical aggregation as well as readout width.
    c = torch.cat((details['shallow']['pair_connected'][...,1:],
                   details['deep']['pair_connected'][...,1:]),-1)
    weights = details['edge_features'][...,:4].tanh()
    extra = torch.einsum('bjic,bjik->bick',c,weights).flatten(-2)/7
    return torch.cat((details['readout'],extra),-1)


@torch.no_grad()
def encode(pack, normalization, seed, device, output, split):
    started=time.perf_counter()
    base=GateProbe(normalization['quantum_scale']).to(device)
    generator=torch.Generator().manual_seed(seed)
    base.theta[:,4:].copy_(.35*torch.randn(base.theta[:,4:].shape,generator=generator).to(device))
    flat=pack['state_hat'].flatten(0,1)
    masks=pack['track_exists'].flatten(0,1)
    nframes=len(flat)
    pv=torch.zeros(nframes,8,48)
    vp=torch.zeros_like(pv)
    # One warm/timing batch; abort costly unexpected behavior rather than
    # silently switching precision or dropping crowded scenes.
    for n in range(1,9):
        locations=(masks.sum(-1)==n).nonzero(as_tuple=True)[0]
        for offset in range(0,len(locations),16):
            loc=locations[offset:offset+16]
            slots=torch.stack([masks[i].nonzero(as_tuple=True)[0] for i in loc])
            x=torch.stack([flat[i,inds] for i,inds in zip(loc,slots)]).to(device=device,dtype=torch.float64)
            valid=torch.ones(x.shape[:-1],dtype=torch.bool,device=device)
            base.variant='PV_ZX'
            details=base(x,valid,return_details=True)
            pfeatures=relation_readout(details)
            del details
            if offset==0:
                perm=torch.arange(n-1,-1,-1,device=device)
                permuted=relation_readout(base(x[:,perm],valid[:,perm],return_details=True))
                if (permuted-pfeatures[:,perm]).abs().max().item()>1e-10:
                    raise AssertionError('Expanded readout failed permutation check')
            base.variant='VP_ZX'
            vfeatures=base(x,valid)
            for row,(idx,inds) in enumerate(zip(loc,slots)):
                pv[idx,inds]=pfeatures[row].cpu().float()
                vp[idx,inds,:24]=vfeatures[row].cpu().float()
            if time.perf_counter()-started>600:
                raise RuntimeError('Feature generation exceeded 10 minutes per seed/split; budget review required')
        if len(locations):
            print(json.dumps({'stage':'features','seed':seed,'split':split,'active_cars':n,
                              'frames':len(locations),'elapsed':time.perf_counter()-started}),flush=True)
    q=pv[:,:,:24]
    shape=(*pack['state_hat'].shape[:-1],48)
    pack['features']={'own_only':torch.zeros_like(pv).reshape(shape),
                      'PV24':torch.cat((q,torch.zeros_like(q)),-1).reshape(shape),
                      'VP24':vp.reshape(shape),
                      'PV48_square':torch.cat((q,q.square()),-1).reshape(shape),
                      'PV48_relation':pv.reshape(shape)}
    report={'seconds':time.perf_counter()-started,'frames':nframes,
            'theta':base.theta.cpu().tolist(),'trained':False,
            'peak_allocated_gpu_bytes':torch.cuda.max_memory_allocated(device)}
    (output/f'cost_seed{seed}_{split}.json').write_text(json.dumps(report,indent=2)+'\n')
    return pack


def main(args):
    torch.set_num_threads(2)
    output=ROOT/'reports/qgnn_frozen_prediction_pilot'
    output.mkdir(parents=True,exist_ok=True)
    train, tm, norm=select_inputs('train',64)
    val, vm, _=select_inputs('V_select',25)
    protocol={'status':'predeclared exploratory pilot','seeds':[2023,2024],
              'quantum_weights':'fixed untrained perturbations, shared among arms per seed',
              'train':tm,'development':vm,'test_or_V_confirm_access':False,
              'arms':['own_only','PV24','VP24','PV48_square','PV48_relation'],
              'head':'same 2-layer causal Transformer 128/4heads/FF256/dropout0, input graph48',
              'head_training':'12 epochs maximum, patience3, AdamW lr1e-3 wd1e-2, paired minibatches8, select V_select J including epoch0',
              'claims':'feature usefulness under fixed untrained circuits and this small head only; no architecture superiority or final model change',
              'cost_limit':'600 seconds feature extraction per seed/split; abort rather than change samples/precision',
              'readout_extension':'four tanh(relative position/velocity scaled) weights times six shallow/deep pair correlations, sum_source/7, append24',
              'width_control':'append squared original24; adds nonlinear basis but no new observed information',
              'selection_rule':'no candidate search or retraining in response to results'}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    # Labels are opened only AFTER input-only selection has been persisted.
    for split,pack,manifest in [('train',train,tm),('V_select',val,vm)]:
        with np.load(ROOT/f'data/f01d/labels/{split}.npz',allow_pickle=False) as labels:
            for key in ('future_position','label_valid'):
                pack[key]=torch.as_tensor(labels[key][manifest['indices']].copy())
    from pilot_heads import run_heads, PilotHead
    # A targeted head check: left padding, exact initial CV and real gradients.
    head=PilotHead().to(args.device)
    sx=train['state_hat'][:1,:,0].to(args.device).float()
    sz=train['standardized_state'][:1,:,0].to(args.device).float()
    sm=torch.ones((1,20),dtype=torch.bool,device=args.device)
    sm[:,:3]=False
    prediction=head(sx,sz,sm,sm,torch.zeros(1,20,48,device=args.device))
    expected=sx[:,-1:,:2]+sx[:,-1:,2:]*torch.arange(1,21,device=args.device)[None,:,None]*.1
    if not torch.isfinite(prediction).all() or not torch.allclose(prediction,expected,atol=1e-5,rtol=1e-6):
        raise AssertionError('Initial head or padding failed')
    prediction.sum().backward()
    if head.output[-1].weight.grad.abs().max().item()==0:
        raise AssertionError('Zero head gradient')
    del head,prediction
    all_results={}
    for seed in (2023,2024):
        for split,pack in [('train',train),('V_select',val)]:
            cached=output/f'features_{split}_seed{seed}.pt'
            if args.resume and cached.exists():
                pack['features']=torch.load(cached,map_location='cpu',weights_only=True)
            else:
                encode(pack,norm,seed,args.device,output,split)
                torch.save(pack['features'],cached)
        all_results[str(seed)]=run_heads(train,val,seed,output,args.device)
        (output/'summary.json').write_text(json.dumps(all_results,indent=2)+'\n')
    (output/'completed.json').write_text(json.dumps({'completed':True,'formal_training':False,
                                                   'quantum_trained':False,'LLM_trained':False})+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',default='cuda:1')
    parser.add_argument('--resume',action='store_true')
    main(parser.parse_args())
