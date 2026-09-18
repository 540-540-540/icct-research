"""Validation counterfactuals and train-only readout-scale diagnostics."""
import sys,json,argparse,time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from prediction.qgnn_final.model import build_model
from prediction.qgnn_final.quantum import evolve,moments
from frontend.sind_prediction_dataset import SinDPredictionDataset
from scripts.train_qgnn_final import evaluate,atomic_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args()
    torch.set_num_threads(4);directory=ROOT/a.run_dir
    cp=torch.load(directory/'best.pt',map_location='cpu',weights_only=False);c=cp['config']
    model=build_model(c['kind'],c['seed'],c['depth']).cuda().eval()
    missing,extra=model.load_state_dict(cp['model_state'],strict=False)
    if extra or any(not k.startswith('llm.gpt2.') for k in missing):raise ValueError('checkpoint mismatch')
    q=model.graph
    val=SinDPredictionDataset('val',c['snr'],ROOT,True)
    loader=DataLoader(val,batch_size=32,shuffle=False)
    report={'kind':c['kind'],'epoch':cp['epoch'],'test_set_used':False,'warning':'Post-training removal is an OOD dependency check, not a retrained causal ablation.','counterfactuals':{}}
    for mode in ['normal','no_entanglers','no_ZZZ','constant_graph_context']:
        hook=None
        q.interaction_scale=0. if mode=='no_entanglers' else 1.
        q.triple_scale=0. if mode=='no_ZZZ' else 1.
        if mode=='constant_graph_context':hook=q.register_forward_hook(lambda module,args,out:torch.zeros_like(out))
        result,_=evaluate(model,loader,torch.device('cuda:0'))
        if hook is not None:hook.remove()
        report['counterfactuals'][mode]=result
        atomic_json(directory/'mechanism_diagnostic.json',report);print(mode,json.dumps(result),flush=True)
    q.interaction_scale=q.triple_scale=1.
    train=SinDPredictionDataset('train',c['snr'],ROOT,True)
    data=next(iter(DataLoader(train,batch_size=64,shuffle=False)))
    h=data['history_state'].cuda();mask=data['vehicle_mask'].cuda()
    with torch.no_grad():
        own,risk,tw,ry,rz,pa,ta=q.circuit_inputs(h,mask)
        states=evolve(ry,rz,pa,ta,q.rx,mask)
        features=torch.cat([moments(s,mask,risk,tw) for s in states],-1)
        gate=torch.sigmoid(q.gate(torch.cat((own,features),-1)))
        z=q.local(own);delta=gate*q.readout(features)
        per=features[mask]
        last=moments(states[-1],mask,risk,tw)
        purity=(1+last[...,:3].square().sum(-1))/2
        report['train_only_readout']={'feature_std':per.std(0).tolist(),'feature_abs_mean':per.abs().mean(0).tolist(),'gate_mean':float(gate[mask].mean()),'local_rms':float(z[mask].square().mean().sqrt()),'interaction_readout_rms':float(delta[mask].square().mean().sqrt()),'single_qubit_purity_min':float(purity[mask].min()),'single_qubit_purity_mean':float(purity[mask].mean()),'pair_phase_rms':float(pa.square().mean().sqrt()),'triple_phase_rms':float(ta.square().mean().sqrt())}
    atomic_json(directory/'mechanism_diagnostic.json',report);print('TRAIN_READOUT',json.dumps(report['train_only_readout']),flush=True)
if __name__=='__main__':main()
