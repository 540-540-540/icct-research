from pathlib import Path
import sys, json, torch
ROOT=Path('/home/dell/YrM/ICCT'); sys.path.insert(0,str(ROOT))
from prediction.qgnn_paper_native.raj_mechanism import RowLocalFeatureLoader
from prediction.qgnn_paper_native.model import build_paper_model
from frontend.sind_prediction_dataset import SinDPredictionDataset
from prediction.q0.metrics import trajectory_metrics
from scripts.train_q0_motion_llm import token_loss
from torch.utils.data import DataLoader,Subset

torch.manual_seed(123)
loader=RowLocalFeatureLoader(10,20,32).double()
feat=torch.randn(2,7,10,dtype=torch.float64)
valid=torch.tensor([[1,1,1,1,1,0,0],[1,1,1,0,0,0,0]],dtype=torch.bool)
x,cond=loader(feat,valid)
global_norm=x.flatten(1).norm(dim=1)
row_norm=x.norm(dim=-1)
target=torch.tensor([1/5**0.5,1/3**0.5],dtype=torch.float64)
row_err=max((row_norm[0,:5]-target[0]).abs().max().item(),(row_norm[1,:3]-target[1]).abs().max().item())
zero_err=max(row_norm[0,5:].abs().max().item(),row_norm[1,3:].abs().max().item())
ctype=torch.complex128
z=torch.complex(torch.randn_like(x),torch.randn_like(x))
# normalize valid rows to same global convention before reupload
z=z/(z.norm(dim=-1,keepdim=True).clamp_min(1e-12))
scale=(valid.sum(-1,keepdim=True).double().rsqrt())[...,None]
z=z*scale*valid[...,None]
zr=loader.reupload(z,cond,valid)
reupload_global=zr.flatten(1).norm(dim=1)

assert (global_norm-1).abs().max()<1e-10
assert row_err<1e-10 and zero_err<1e-12
assert (reupload_global-1).abs().max()<1e-10

model=build_paper_model('raj_paper_math_rowlocal_quantum',2026,3,j=3,correction_cap_m=16.).cuda().train()
ds=SinDPredictionDataset('train',0.,ROOT,True)
b=next(iter(DataLoader(Subset(ds,[0,1,2,3]),batch_size=4)))
h=b['history_state'].cuda();m=b['vehicle_mask'].cuda();y=b['future_state'].cuda();ts=b['history_timestamp'].cuda()
out=model(h,m,ts)
metric=trajectory_metrics(out['prediction'],y,m)
tok=token_loss(out['token_logits'],model.llm.future_token_ids(h,y),m)
loss=metric['loss']+.035*tok
loss.backward()
grads=[p.grad.detach().float().square().sum() for p in model.graph.parameters() if p.grad is not None]
gn=float(torch.sqrt(torch.stack(grads).sum()))
payload={'status':'PASS','loader_global_norm_max_error':float((global_norm-1).abs().max()),'loader_equal_row_norm_max_error':row_err,'loader_masked_zero_max':zero_err,'reupload_global_norm_max_error':float((reupload_global-1).abs().max()),'model_prediction_shape':list(out['prediction'].shape),'model_graph_shape':list(out['graph_features'].shape),'finite_loss':bool(torch.isfinite(loss)),'graph_gradient_norm':gn,'graph_params':model.parameter_summary()['graph'],'test_used':False}
assert payload['finite_loss'] and gn>1e-8 and out['graph_features'].shape==(4,8,64)
(ROOT/'reports/qgnn/round4_raj_rowlocal_unit_smoke_20260919.json').write_text(json.dumps(payload,indent=2)+'\n')
print(json.dumps(payload,indent=2))
