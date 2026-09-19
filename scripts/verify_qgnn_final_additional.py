"""CPU-only independent readout, symmetry and provenance checks; no test split."""
import sys,json,hashlib,itertools
from pathlib import Path
import torch
import pennylane as qml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from prediction.qgnn_final.quantum import moments
from prediction.qgnn_final.relational import RelationCarryingQuantumCore,cumulants
from prediction.qgnn_final.classical import AdaptiveClassicalCore
torch.set_num_threads(2)
r={'test_set_used':False,'checks':{}}
torch.manual_seed(777);q=RelationCarryingQuantumCore().double().eval()
torch.manual_seed(777);c=AdaptiveClassicalCore().double().eval()
for name in ['encoder','local']:
    a=getattr(q,name).state_dict();b=getattr(c,name).state_dict()
    assert all(torch.equal(a[k],b[k]) for k in a)
r['checks']['identical_own_encoder_and_local_initialization']=True
x=torch.randn(2,20,4,4,dtype=torch.float64);m=torch.ones(2,4,dtype=torch.bool)
with torch.no_grad():
    a=q(x,m);padded=torch.cat((x,torch.full((2,20,16,4),float('nan'),dtype=x.dtype)),2)
    mask=torch.cat((m,torch.zeros(2,16,dtype=torch.bool)),1)
    error=float((q(padded,mask)[:,:4]-a).abs().max());assert error<1e-8
    tied=x.repeat(1,1,3,1);mt=torch.ones(2,12,dtype=torch.bool);perm=torch.randperm(12)
    tie=float((q(tied[:,:,perm],mt)-q(tied,mt)[:,perm]).abs().max());assert tie<1e-8
    assert (q(x,torch.zeros_like(m))==0).all()
r['checks'].update(padding_across_patch_threshold_error=error,duplicate_history_permutation_error=tie,all_masked_zero=True)
with torch.no_grad():
    classic_error=float((c(padded,mask)[:,:4]-c(x,m)).abs().max());assert classic_error<1e-8
r['checks']['classical_padding_across_patch_threshold_error']=classic_error
n=4;pairs=list(itertools.combinations(range(n),2));triples=list(itertools.combinations(range(n),3))
psi=torch.randn(2**n,dtype=torch.complex128);psi=psi/psi.norm()
dev=qml.device('default.qubit',wires=n,shots=None)
@qml.qnode(dev,interface='torch')
def obs(state):
    qml.StatePrep(state,wires=range(n))
    operators=[qml.PauliX(i) for i in range(n)]+[qml.PauliY(i) for i in range(n)]+[qml.PauliZ(i) for i in range(n)]
    operators += [qml.prod(*[qml.PauliZ(i) for i in ids]) for ids in pairs+triples]
    return tuple(qml.expval(o) for o in operators)
y=torch.stack(obs(psi));z=y[8:12];zz=y[12:18];zzz=y[18:]
lookup={ij:k for k,ij in enumerate(pairs)}
c2=torch.zeros(n,n,dtype=torch.float64)
for k,(i,j) in enumerate(pairs):c2[i,j]=c2[j,i]=zz[k]-z[i]*z[j]
c3=torch.stack([zzz[k]-z[i]*zz[lookup[(j,l)]]-z[j]*zz[lookup[(i,l)]]-z[l]*zz[lookup[(i,j)]]+2*z[i]*z[j]*z[l] for k,(i,j,l) in enumerate(triples)])
a,b=cumulants(psi[None],n)
assert torch.allclose(a[0],c2,atol=1e-12) and torch.allclose(b[0],c3,atol=1e-12)
risk=torch.ones(1,n,n,dtype=torch.float64)-torch.eye(n)[None]
actual=moments(psi[None],torch.ones(1,n,dtype=torch.bool),risk,torch.ones(1,len(triples),dtype=torch.float64))[0]
expected=[]
for i in range(n):
    pair=c2[i,torch.arange(n)!=i];tri=c3[torch.tensor([i in t for t in triples])]
    expected.append(torch.stack((y[i],y[n+i],z[i],pair.mean(),torch.sqrt(pair.square().mean()+1e-12),tri.mean(),torch.sqrt(tri.square().mean()+1e-12))))
err=float((actual-torch.stack(expected)).abs().max());assert err<1e-10
r['checks']['independent_PennyLane_all_readout_max_error']=err
r['checks']['quantum_core_parameters']=sum(p.numel() for p in q.parameters())
r['checks']['cost_backup_C2_parameters']=sum(p.numel() for p in RelationCarryingQuantumCore(channels=2).parameters())
out=ROOT/'reports/qgnn/additional_checks_and_provenance.json'
r['status']='RUNNING';r['sha256']={}
def save():
    tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(r,indent=2)+'\n');tmp.replace(out)
save()
paths=[ROOT/f'data/sind/splits/{s}/samples.npz' for s in ['train','val']]
paths += [ROOT/f'data/sind/isac/{s}/sensing_cache.npz' for s in ['train','val']]
paths += [ROOT/'models/gpt2'/name for name in ['config.json','model.safetensors','pytorch_model.bin'] if (ROOT/'models/gpt2'/name).is_file()]
paths += [ROOT/'configs/qgnn_final_tokens.json',ROOT/'configs/qgnn_final_architecture.json']
for p in paths:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    r['sha256'][str(p.relative_to(ROOT))]={'sha256':h.hexdigest(),'bytes':p.stat().st_size}
    save();print('HASH',p.name,p.stat().st_size,flush=True)
r['status']='PASS';save();print(json.dumps(r),flush=True)
