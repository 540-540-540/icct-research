#!/usr/bin/env python3
"""Train-only LLM mechanism audit; forwards/gradients only, no optimizer updates."""
import json,time,collections,numpy as np,torch
from pathlib import Path
from safetensors import safe_open
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from prediction.qgnn_raj_pennylane.residual import RajResidualLLM
from frontend.sind_target_dataset import SinDTargetPredictionDataset
start=time.perf_counter();torch.set_num_threads(4);torch.manual_seed(302029)
m=RajResidualLLM(self_frame='ego_v1');m.set_training_phase('self');m.eval();r={}
r['scope']={'dataset_split':'train','test_used':False,'optimizer_updates':0,'sample_count':96,'sampling':'numpy default_rng choice without replacement over target views','sampling_seed':20260921,'model_initialization_seed':302029,'gradient_batches':3,'gradient_batch_size':32,'gradient_mode':'eval: deterministic; dropout disabled','weighting':'mean per-target error times existing target_weight; random subset weights are not renormalized and this is not a full origin-macro metric','checkpoint':'results/qgnn/self_repair_v1/target_llm_ego_v1_0db_seed2026/best.pt','source_head_audited':'175c16a','limits':'checkpoint interventions and local gradients do not identify retraining effects or the cause of the full validation gap'}
with safe_open(str(ROOT/'models/gpt2/model.safetensors'),framework='pt',device='cpu') as f:
 keys=set(f.keys());cnt=0;bad=[];absent=[]
 for n,p in m.gpt2.named_parameters():
  if 'lora_' in n:continue
  k=n.replace('.c_attn.base.','.c_attn.');k=k if k in keys else 'transformer.'+k
  if k not in keys:absent.append(n);continue
  cnt+=p.numel()
  if not torch.equal(p.detach(),f.get_tensor(k)):bad.append(n)
 r['pretrained_file_equality']={'parameters':cnt,'mismatched':bad,'absent':absent,'layers':len(m.gpt2.h),'ln_f_mean':float(m.gpt2.ln_f.weight.mean()),'ln_f_std':float(m.gpt2.ln_f.weight.std())}
init={n:p.detach().clone() for n,p in m.named_parameters() if p.requires_grad}
ck=torch.load(ROOT/r['scope']['checkpoint'],map_location='cpu',weights_only=False)
missing,extra=m.load_state_dict({k[4:]:v for k,v in ck['model_state'].items() if k.startswith('llm.')},strict=False)
assert not extra and all(n.startswith('base.gpt2.') and 'lora_' not in n for n in missing)
def group(n):
 if 'lora_' in n:return 'lora'
 for p in ['base.motion_embedding','base.history_adapter','own_history_adapter','base.continuous_motion','base.future_queries','base.token_head','self_head']:
  if n.startswith(p):return p
 return 'other'
gs=collections.defaultdict(lambda:[0,0,0.,0.])
for n,p in m.named_parameters():
 g=gs[group(n)];g[0 if p.requires_grad else 1]+=p.numel()
 if p.requires_grad:g[2]+=float((p.detach()-init[n]).square().sum());g[3]+=float(init[n].square().sum())
r['parameter_groups']={k:{'trainable':v[0],'frozen':v[1],'relative_update_l2':(v[2]/max(v[3],1e-30))**.5} for k,v in gs.items()}
ds=SinDTargetPredictionDataset('train',root_dir=ROOT);ids=np.random.default_rng(20260921).choice(len(ds),96,replace=False);rows=[ds[int(i)] for i in ids]
r['scope']['target_indices']=ids.tolist()
h=torch.stack([x['history_state'][:,0] for x in rows]);y=torch.stack([x['future_state'][:,0] for x in rows]);w=torch.stack([x['target_weight'] for x in rows]);mask=torch.ones(96,1,dtype=torch.bool);z=torch.zeros(96,1,64);im=torch.zeros_like(mask)
def run():return m(h[:,:,None],mask,z,im)
def score(o):
 d=(o['prediction'][:,:,0]-y[...,:2]).norm(dim=-1)
 return {'ADE':float((d.mean(1)*w).mean()),'FDE':float((d[:,-1]*w).mean())}
with torch.no_grad():
 o=run();r['train96_weighted_baseline']=score(o)
 frame,_=m._ego_frame(h);own=torch.cat(((h[...,:2]-h[:,-1:,:2])@frame/10,h[...,2:]@frame/10),-1)[:,1:]
 emb,_=m.base._history_motion_embeddings(h)
 r['embedding_rms']={k:float(v.square().mean().sqrt()) for k,v in [('motion',emb),('history_adapter',m.base.history_adapter(emb)),('own_adapter',m.own_history_adapter(own)),('gpt_wte',m.gpt2.wte.weight),('gpt_wpe',m.gpt2.wpe.weight)]}
 hook=m.own_history_adapter.register_forward_hook(lambda mod,i,o:torch.zeros_like(o));alt=run();hook.remove();r['without_own_adapter']={**score(alt),'prediction_delta_mean_m':float((o['prediction']-alt['prediction']).norm(dim=-1).mean())}
 orig=m.tokenizer.expected_motion;m.tokenizer.expected_motion=lambda p:(torch.zeros(p.shape[:-1]),torch.zeros(p.shape[:-1]));alt=run();m.tokenizer.expected_motion=orig;r['without_expected_motion']={**score(alt),'prediction_delta_mean_m':float((o['prediction']-alt['prediction']).norm(dim=-1).mean())}
 s=torch.cat((h[:,-1:],y),1);f,l=m.tokenizer.local_deltas(s[...,:2],s[...,2:]);tar=m.tokenizer.token_ids(s[...,:2],s[...,2:]);ft,lt=m.tokenizer._tables('cpu',h.dtype);err=torch.stack((f-ft[tar],l-lt[tar]),-1).norm(dim=-1)
 r['token_targets']={'vocab':m.base.vocab_size,'unique':int(tar.unique().numel()),'top5_fraction':float(torch.bincount(tar.flatten(),minlength=1681).topk(5).values.sum()/tar.numel()),'lateral_zero_fraction':float((lt[tar]==0).float().mean()),'forward_has_zero':bool((ft==0).any()),'quant_error_mean_m':float(err.mean()),'quant_error_p95_m':float(torch.quantile(err,.95)),'ce':float(torch.nn.functional.cross_entropy(o['token_logits'][:,:,0].reshape(-1,1681),tar.flatten())),'accuracy':float((o['token_logits'][:,:,0].argmax(-1)==tar).float().mean())}
r['gradient_batches']=[];params=[(n,p) for n,p in m.named_parameters() if p.requires_grad]
for lo in [0,32,64]:
 o=m(h[lo:lo+32,:,None],mask[:32],z[:32],im[:32]);d=(o['prediction'][:,:,0]-y[lo:lo+32,:,:2]).norm(dim=-1);co=((d.mean(1)+.5*d[:,-1])*w[lo:lo+32]).mean();ce=torch.nn.functional.cross_entropy(o['token_logits'][:,:,0].reshape(-1,1681),tar[lo:lo+32].reshape(-1),reduction='none').reshape(32,20).mean(1);aux=(ce*w[lo:lo+32]).mean()*.035
 a=torch.autograd.grad(co,[p for n,p in params],retain_graph=True,allow_unused=True);b=torch.autograd.grad(aux,[p for n,p in params],allow_unused=True);sums=collections.defaultdict(lambda:[0.,0.,0.])
 for (n,p),ga,gb in zip(params,a,b):
  v=sums[group(n)]
  if ga is not None:v[0]+=float(ga.square().sum())
  if gb is not None:v[1]+=float(gb.square().sum())
  if ga is not None and gb is not None:v[2]+=float((ga*gb).sum())
 r['gradient_batches'].append({'coordinate':float(co.detach()),'weighted_token':float(aux.detach()),'groups':{k:{'coordinate_norm':v[0]**.5,'weighted_token_norm':v[1]**.5,'cosine':v[2]/max((v[0]*v[1])**.5,1e-30)} for k,v in sums.items()}})
m.train();r['dropout_training']=[{'name':n,'p':v.p,'training':v.training} for n,v in m.named_modules() if isinstance(v,torch.nn.Dropout)];r['elapsed_seconds']=time.perf_counter()-start
folder=ROOT/'reports/qgnn/sind_target_self_repair';folder.mkdir(parents=True,exist_ok=True);dest=folder/'llm_mechanism_diagnostics.json';dest.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
