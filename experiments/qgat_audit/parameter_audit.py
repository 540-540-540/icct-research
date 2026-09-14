"""Read-only deterministic initialization/checkpoint/optimizer audit. No training."""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.qgat_candidate.run_formal import (Predictor,read_config,source_contract,
    temporal_state,tensor_digest,candidate_optimizer,digest)
from prediction.training import mutable_state,balanced_schedule,write_json


def category(name):
    if name=='graph.theta':return 'quantum_theta'
    if name.startswith('graph.encoder.'):return 'graph_encoder'
    if name.startswith('graph.readout.'):return 'graph_readout'
    if name.startswith('graph.'):return 'graph_other'
    if 'graph_projection' in name:return 'graph_projection'
    if 'state_projection' in name or 'marker_projection' in name:return 'own_projection'
    if 'lora_A' in name:return 'lora_A'
    if 'lora_B' in name:return 'lora_B'
    if name.startswith('temporal.head.'):return 'head'
    return 'other'


def scalar(value):
    return float(value.item()) if torch.is_tensor(value) else float(value)


def inspect_checkpoint(path, initial, model, optimizer):
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    state=checkpoint['model'];saved=checkpoint['optimizer']
    name_by_id={id(p):n for n,p in model.named_parameters()}
    assert set(state)==set(initial)
    assert len(saved['param_groups'])==len(optimizer.param_groups)
    mapping={};actual_groups=[]
    for group,restored in zip(optimizer.param_groups,saved['param_groups']):
        assert len(group['params'])==len(restored['params'])
        assert group['lr']==restored['lr'] and group['weight_decay']==restored['weight_decay']
        names=[name_by_id[id(p)] for p in group['params']]
        actual_groups.append(dict(lr=restored['lr'],weight_decay=restored['weight_decay'],names=names))
        for parameter_id,name in zip(restored['params'],names):mapping[name]=saved['state'].get(parameter_id)
    assert not any('.llm.' in n and 'lora_' not in n for n in mapping)
    rows={};groups=defaultdict(list)
    for name,value in state.items():
        if name not in mapping:continue
        before=initial[name].double();after=value.double();delta=after-before
        moment=mapping[name]
        row=dict(shape=list(value.shape),elements=value.numel(),changed_elements=int(torch.count_nonzero(delta)),
                 maxabsdelta=float(delta.abs().max()),delta_l2=float(delta.norm()),init_l2=float(before.norm()),
                 relative_delta_l2=float(delta.norm()/before.norm()) if before.norm()>0 else None,
                 optimizer_state_present=moment is not None)
        if moment is not None:
            row.update(optimizer_step=scalar(moment['step']),exp_avg_l2=float(moment['exp_avg'].double().norm()),
                       exp_avg_maxabs=float(moment['exp_avg'].abs().max()),
                       exp_avg_nonzero=int(torch.count_nonzero(moment['exp_avg'])),
                       exp_avg_sq_l2=float(moment['exp_avg_sq'].double().norm()),
                       exp_avg_sq_nonzero=int(torch.count_nonzero(moment['exp_avg_sq'])))
        rows[name]=row;groups[category(name)].append(row)
    aggregates={}
    for group,parts in groups.items():
        norm=math.sqrt(sum(r['delta_l2']**2 for r in parts));base=math.sqrt(sum(r['init_l2']**2 for r in parts))
        aggregates[group]=dict(tensors=len(parts),elements=sum(r['elements'] for r in parts),
            changed_elements=sum(r['changed_elements'] for r in parts),maxabsdelta=max(r['maxabsdelta'] for r in parts),
            delta_l2=norm,relative_delta_l2=norm/base if base else None,
            optimizer_steps=sorted(set(r['optimizer_step'] for r in parts if r['optimizer_state_present'])),
            all_optimizer_states_present=all(r['optimizer_state_present'] for r in parts),
            exp_avg_l2=math.sqrt(sum(r.get('exp_avg_l2',0)**2 for r in parts)),
            exp_avg_nonzero=sum(r.get('exp_avg_nonzero',0) for r in parts),
            exp_avg_sq_nonzero=sum(r.get('exp_avg_sq_nonzero',0) for r in parts))
    theta=None
    if 'graph.theta' in state:
        name='graph.theta';theta=dict(initial=initial[name].tolist(),value=state[name].tolist(),
            delta=(state[name]-initial[name]).tolist(),exp_avg=mapping[name]['exp_avg'].tolist(),
            exp_avg_sq=mapping[name]['exp_avg_sq'].tolist(),step=scalar(mapping[name]['step']))
    progress=checkpoint['progress']
    return dict(epoch=progress['epoch'],cursor=progress['cursor'],optimizer_steps=progress['optimizer_steps'],
                stale=progress['stale'],best_J=progress['best_J'],history=progress['history'],
                groups=aggregates,parameters=rows,optimizer_groups=actual_groups,theta=theta,
                saved_mutable_keys=len(state),frozen_base_in_saved_state=any('.llm.' in k and 'lora_' not in k for k in state)),checkpoint['config']


def main():
    torch.set_num_threads(1)
    report=dict(mode='read-only CPU audit',optimizer_steps_executed=0,test_opened=False,blocking_bugs=[],jobs={},
        evidence_limits=['Checkpoint optimizer moments are endpoint state, not a recorded per-step theta-gradient history.',
        'Epoch graph_gradient_norm is the final batch aggregate over the graph, not a quantum-theta-specific gradient.',
        'Frozen GPT-2 base tensors are omitted from mutable checkpoints; direct initial/final base-weight subtraction is unavailable.',
        'Per-origin SNR exposure is reconstructed from sealed deterministic schedule and full manifest; per-step SNR IDs were not separately logged.',
        'No dedicated resume-event ledger exists; epoch and optimizer accounting can be checked, but restart counts are not inferred.',
        'Best-checkpoint group update magnitudes across seeds can cover different epoch counts; they are not controlled learning-rate or gradient comparisons.'])
    config_path=ROOT/'configs/qgat_formal.json';cfg=read_config(config_path)
    provenance=json.loads((ROOT/'reports/qgat_formal/provenance.json').read_text())
    current=source_contract(config_path)
    report['source_contract_matches']=current==provenance['source']
    if not report['source_contract_matches']:
        report['blocking_bugs'].append('Current source/data/environment contract differs from formal provenance')
        write_json(ROOT/'reports/qgat_audit/parameters.json',report);raise RuntimeError(report['blocking_bugs'][-1])
    sealed=json.loads((ROOT/'reports/qgat_formal/selected_checkpoints.json').read_text())
    fingerprints={};input_contracts={}
    for seed in cfg['seeds']:
        for name in cfg['models']:
            job=f'{name}-{seed}';directory=ROOT/'results/qgat_formal'/job
            status=json.loads((ROOT/'reports/qgat_formal/jobs'/f'{job}.json').read_text())
            saved_config=json.loads((directory/'config.json').read_text())
            history=json.loads((directory/'history.json').read_text())
            saved_parameters=json.loads((directory/'parameters.json').read_text())
            input_contracts[job]=(saved_config['input_hashes'],saved_config['validation_hashes'])
            model=Predictor(name,seed,cfg)
            fingerprints[job]=tensor_digest(temporal_state(model))
            optimizer=candidate_optimizer(model,'joint',cfg['graph_lr'])
            initial=mutable_state(model)
            best,best_config=inspect_checkpoint(directory/'best.pt',initial,model,optimizer)
            latest,latest_config=inspect_checkpoint(directory/'latest.pt',initial,model,optimizer)
            full=saved_config['indices']==list(range(5549))
            orders=[];exposure=np.zeros((5549,4),int)
            for epoch in range(latest['epoch']):
                order,snrs=balanced_schedule(5549,seed,epoch)
                assert len(np.unique(order))==5549
                for snr_index,snr in enumerate((5,10,15,20)):exposure[order[snrs==snr],snr_index]+=1
                orders.append({str(s):int((snrs==s).sum()) for s in (5,10,15,20)})
            minimum=min(history,key=lambda r:r['J'])
            numerical_expected_steps=math.ceil(5549/16)*latest['epoch']
            frozen=[n for n,p in model.named_parameters() if '.llm.' in n and 'lora_' not in n]
            checks=dict(full_5549_origin_manifest=full,
                explicit_training_contract_matches=saved_config['seed']==seed and saved_config['epochs']==20 and saved_config['patience']==4 and saved_config['phase']=='joint' and saved_config['batch_size']==16 and saved_config['micro_batch']==1 and saved_config['graph_lr']==3e-4,
                actual_saved_parameter_flags_match=[dict(name=n,shape=list(p.shape),trainable=p.requires_grad) for n,p in model.named_parameters()]==saved_parameters,
                epoch_history_contiguous=[r['epoch'] for r in history]==list(range(1,latest['epoch']+1)),balanced_snr_mode=saved_config['snr_mode']=='balanced',
                same_config_in_best_latest=best_config==latest_config==saved_config,
                checkpoint_contract_matches=best_config['contract']==provenance,
                rebuilt_temporal_matches_record=fingerprints[job]==status['temporal_initialization_sha256'],
                best_hash_matches_worker_and_seal=digest(directory/'best.pt')==status['best_checkpoint_sha256']==sealed['checkpoints'][job]['sha256'],
                best_epoch_is_minimum_history_J=best['epoch']==minimum['epoch'] and best['best_J']==minimum['J'],
                latest_history_matches_log=latest['history']==history,
                optimizer_steps_match_all_batches=latest['optimizer_steps']==numerical_expected_steps,
                stopping_matches_patience_or_cap=latest['epoch']==cfg['epochs'] or latest['stale']==cfg['patience'],
                completed_epoch_boundary=latest['cursor']==0,
                pretrained_base_all_frozen=all(not dict(model.named_parameters())[n].requires_grad for n in frozen),
                lora_all_trainable=all(p.requires_grad for n,p in model.named_parameters() if 'lora_' in n),
                frozen_base_excluded_from_optimizer_and_checkpoint=not best['frozen_base_in_saved_state'] and not latest['frozen_base_in_saved_state'])
            issues=[k for k,v in checks.items() if not v]
            report['blocking_bugs'].extend(f'{job}: {k}' for k in issues)
            report['jobs'][job]=dict(checks=checks,trained_epochs=latest['epoch'],best_epoch=best['epoch'],
                train_origin_count=5549,snr_epoch_counts=orders,per_origin_snr_exposure_min=exposure.min(0).tolist(),
                per_origin_snr_exposure_max=exposure.max(0).tolist(),expected_optimizer_steps=numerical_expected_steps,
                frozen_base_parameter_tensors=len(frozen),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                best=best,latest=latest)
            print(json.dumps(dict(job=job,checks_all_passed=not issues,epochs=latest['epoch'],best_epoch=best['epoch'],
                                 theta_best=best['groups'].get('quantum_theta'))),flush=True)
            del model,optimizer,initial
    report['all_jobs_same_input_and_validation_hashes']=all(v==next(iter(input_contracts.values())) for v in input_contracts.values())
    if not report['all_jobs_same_input_and_validation_hashes']:report['blocking_bugs'].append('Input digest mismatch across jobs')
    report['paired_temporal_init_matches']={str(seed):fingerprints[f'qgat-{seed}']==fingerprints[f'gnn-{seed}'] for seed in cfg['seeds']}
    if not all(report['paired_temporal_init_matches'].values()):report['blocking_bugs'].append('Paired temporal init mismatch')
    report['verified']=not report['blocking_bugs']
    write_json(ROOT/'reports/qgat_audit/parameters.json',report)
    print(json.dumps(dict(verified=report['verified'],blocking_bugs=report['blocking_bugs'],output='reports/qgat_audit/parameters.json')),flush=True)


if __name__=='__main__':main()
