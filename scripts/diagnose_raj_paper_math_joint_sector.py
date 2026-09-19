from __future__ import annotations
import json, sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from frontend.sind_prediction_dataset import SinDPredictionDataset
from prediction.qgnn_paper_native.model import build_paper_model
from prediction.qgnn_final.common import physical_graph
from prediction.qgnn_paper_native.raj_joint import conditional_embedding_states

def load_graph(checkpoint=None):
    model=build_paper_model('raj_paper_math_quantum',2026,3,j=3,correction_cap_m=16.)
    if checkpoint:
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
        miss,unexp=model.load_state_dict(ck['model_state'],strict=False)
        bad=[k for k in miss if not (k.startswith('llm.gpt2.') and 'lora_' not in k)]
        if bad or unexp: raise RuntimeError((bad,unexp))
    return model.graph

@torch.no_grad()
def batch_stats(core,h,m):
    own,feat,valid,subs=core.builder(h,m,core.j)
    edge,risk=physical_graph(h,m)
    x0,_=core.loaders[0](feat,valid)
    ctype=torch.complex64 if x0.dtype==torch.float32 else torch.complex128
    x=x0.to(ctype)
    for l in range(core.rounds):
        x=core.adj[l](x,edge,risk,m)
        x=core.evolution[l](x)
        _,cond=core.loaders[l+1](feat,valid)
        x=core.loaders[l+1].reupload(x,cond,valid)
    pre=x.abs().square().sum((-1,-2))
    joint=core.joint(x,risk,m)
    full=joint.abs().square().sum(-1)
    factor=core.joint.project_factor_sector(joint)
    sector=factor.abs().square().sum((-1,-2))
    cond,prob=conditional_embedding_states(core.joint,joint)
    psum=prob.sum(-1)
    pn=prob/psum[:,None].clamp_min(1e-12)
    entropy=-(pn.clamp_min(1e-12)*pn.clamp_min(1e-12).log()).sum(-1)
    support=(prob>1e-8).sum(-1)
    return dict(pre=pre,full=full,sector=sector,psum=psum,entropy=entropy,support=support)

def collect(core,loader,device):
    core=core.to(device).eval()
    vals={k:[] for k in ['pre','full','sector','psum','entropy','support']}
    for batch in loader:
        h=batch['history_state'].to(device);m=batch['vehicle_mask'].to(device)
        out=batch_stats(core,h,m)
        for k,v in out.items(): vals[k].append(v.detach().cpu())
    vals={k:torch.cat(v) for k,v in vals.items()}
    ratio=vals['sector']/vals['full'].clamp_min(1e-12)
    return {
      'scenes':int(len(ratio)),
      'pre_norm2_mean':float(vals['pre'].mean()),
      'joint_norm2_mean':float(vals['full'].mean()),
      'factor_sector_mass_mean':float(vals['sector'].mean()),
      'factor_sector_fraction_mean':float(ratio.mean()),
      'factor_sector_fraction_min':float(ratio.min()),
      'factor_sector_fraction_max':float(ratio.max()),
      'projection_prob_sum_mean':float(vals['psum'].mean()),
      'projection_entropy_mean':float(vals['entropy'].mean()),
      'projection_support_mean':float(vals['support'].float().mean()),
    }

def main():
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    val=SinDPredictionDataset('val',0.,ROOT,True)
    sub=Subset(val,list(range(min(64,len(val)))))
    loader=DataLoader(sub,batch_size=16,shuffle=False,num_workers=0)
    ck=ROOT/'reports/qgnn/round4_raj_paper_math_quantum_4096_seed2026/best.pt'
    torch.manual_seed(2026+100003)
    init=load_graph(None)
    torch.manual_seed(2026+100003)
    best=load_graph(ck)
    payload={'scope':{'split':'val','scenes':len(sub),'snr_db':0,'test_used':False},
             'initial':collect(init,loader,device),
             'best_epoch_checkpoint':collect(best,loader,device)}
    for label,core in [('initial',init),('best',best)]:
        payload[label+'_joint_parameter_norms']={
          'node':float(core.joint.theta_node.detach().norm()),
          'embedding':float(core.joint.theta_emb.detach().norm()),
          'cross':float(core.joint.theta_cross.detach().norm())}
    out=ROOT/'reports/qgnn/round4_raj_paper_math_joint_sector_diagnostic_20260919.json'
    out.write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(payload,indent=2))
if __name__=='__main__':main()

