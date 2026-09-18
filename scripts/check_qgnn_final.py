"""Independent circuit equivalence, symmetry, gradients, GPU and GPT-2 smoke checks."""
import sys,json,time,statistics,itertools,hashlib
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from prediction.qgnn_final.quantum import HypergraphQuantumCore,evolve,moments
from prediction.qgnn_final.classical import AdaptiveClassicalCore
from prediction.qgnn_final.model import build_model

OUT=ROOT/'reports/qgnn/engineering_checks.json'
report={'test_set_used':False,'checks':{},'profiling':{}}
def save():
    tmp=OUT.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(OUT)
def record(key,value):
    report['checks'][key]=value;save();print(key,json.dumps(value),flush=True)

def reference_check():
    import pennylane as qml
    torch.manual_seed(410)
    n=4;d=3;pairs=list(itertools.combinations(range(n),2));tri=list(itertools.combinations(range(n),3))
    shapes=[(1,d,n),(1,d,n),(1,d,len(pairs)),(1,d,len(tri)),(d,)]
    args=[(.3*torch.randn(s,dtype=torch.float64)).requires_grad_() for s in shapes]
    mask=torch.ones(1,n,dtype=torch.bool)
    state=evolve(*args,mask)[-1][0]
    device=qml.device('default.qubit',wires=n,shots=None)
    @qml.qnode(device,interface='torch',diff_method='backprop')
    def circuit(ry,rz,pa,ta,rx):
        for l in range(d):
            for i in range(n): qml.RY(ry[0,l,i],wires=i)
            for i in range(n): qml.RZ(rz[0,l,i],wires=i)
            for k,w in enumerate(pairs): qml.PauliRot(pa[0,l,k],'ZZ',wires=w)
            for k,w in enumerate(tri): qml.PauliRot(ta[0,l,k],'ZZZ',wires=w)
            for i in range(n): qml.RX(rx[l],wires=i)
        return qml.state()
    independent=[a.detach().clone().requires_grad_() for a in args]
    ref=circuit(*independent)
    probe=torch.randn(2**n,dtype=torch.complex128)
    grads=torch.autograd.grad((state.conj()*probe).sum().real,args)
    refs=torch.autograd.grad((ref.conj()*probe).sum().real,independent)
    error=float((state-ref).abs().max()); graderr=max(float((a-b).abs().max()) for a,b in zip(grads,refs))
    assert error<1e-10 and graderr<1e-10
    record('pennylane_state_and_gradient',{'max_state_error':error,'max_gradient_error':graderr,'norm_error':float((state.abs().square().sum()-1).abs())})

def structural_checks():
    torch.manual_seed(411)
    for kind,cls in [('quantum',HypergraphQuantumCore),('classical',AdaptiveClassicalCore)]:
        core=cls().double().eval()
        errors={}
        for n in [1,2,4,8,12,16,20]:
            x=torch.randn(2,20,n,4,dtype=torch.float64);x[...,:2]*=15
            mask=torch.ones(2,n,dtype=torch.bool);mask[0,1::3]=False
            perm=torch.randperm(n)
            with torch.no_grad():
                y=core(x,mask); yp=core(x[:,:,perm],mask[:,perm])
            error=float((yp-y[:,perm]).abs().max());assert error<1e-8
            assert y.shape==(2,n,64) and torch.isfinite(y).all() and (y[~mask]==0).all()
            errors[str(n)]=error
        record(kind+'_permutation_and_shapes',errors)
        x=torch.randn(1,20,4,4,dtype=torch.float64);mask=torch.ones(1,4,dtype=torch.bool)
        extended=torch.cat((x,torch.full_like(x,float('nan'))),2)
        me=torch.cat((mask,~mask),1)
        with torch.no_grad(): error=float((core(x,mask)-core(extended,me)[:,:4]).abs().max())
        assert error<1e-8;record(kind+'_padding_invariance',error)
    q=HypergraphQuantumCore().double()
    x=torch.randn(1,20,4,4,dtype=torch.float64,requires_grad=True);mask=torch.ones(1,4,dtype=torch.bool)
    probe=torch.randn(64,dtype=torch.float64)
    def cross():
        return float(torch.autograd.grad((q(x,mask)[0,0]*probe).sum(),x)[0][:,:,1:].norm())
    on=cross();q.interaction_scale=0.;off=cross();q.interaction_scale=1.
    assert on>1e-7 and off<1e-9
    record('quantum_cross_vehicle_path',{'entanglers_on_gradient':on,'entanglers_off_gradient':off})
    q.zero_grad();(q(x,mask)*probe).sum().backward()
    norms={name:float(p.grad.norm()) for name,p in q.named_parameters() if p.grad is not None}
    assert all(torch.isfinite(p.grad).all() for p in q.parameters() if p.grad is not None)
    assert norms['rx']>0 and norms['triple_angle.2.weight']>0
    record('quantum_parameter_gradients',norms)

def profile():
    from frontend.sind_prediction_dataset import SinDPredictionDataset
    from torch.utils.data import DataLoader
    from prediction.q0.metrics import trajectory_metrics
    from scripts.train_q0_motion_llm import token_loss
    import importlib.metadata as metadata
    report['environment']={p:metadata.version(p) for p in ['torch','pennylane','pennylane-lightning-gpu','transformers','custatevec-cu12']}
    report['environment']['gpu']=torch.cuda.get_device_name(0)
    ds=SinDPredictionDataset('train',0.,ROOT,True)
    batch=next(iter(DataLoader(ds,batch_size=32)))
    x=batch['history_state'].cuda();m=batch['vehicle_mask'].cuda();y=batch['future_state'].cuda();ts=batch['history_timestamp'].cuda()
    hashes={}
    for kind in ['quantum','classical','legacy_classical']:
        torch.manual_seed(2026)
        model=build_model(kind).cuda()
        hasher=hashlib.sha256()
        for name,p in model.llm.named_parameters():
            if p.requires_grad: hasher.update(name.encode());hasher.update(p.detach().cpu().numpy().tobytes())
        hashes[kind]=hasher.hexdigest()
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-4)
        measurements={}
        for bs in [16,32]:
            times=[];losses=[];torch.cuda.reset_peak_memory_stats()
            for step in range(7):
                model.train();opt.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.perf_counter()
                out=model(x[:bs],m[:bs],ts[:bs]);metric=trajectory_metrics(out['prediction'],y[:bs],m[:bs])
                loss=metric['loss']+.035*token_loss(out['token_logits'],model.llm.future_token_ids(x[:bs],y[:bs]),m[:bs])
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),3.);opt.step();torch.cuda.synchronize()
                if step>=2: times.append(time.perf_counter()-start);losses.append(float(loss))
            step_s=statistics.median(times)
            measurements[str(bs)]={'median_train_step_s':step_s,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,'sampled_losses':losses,'train_epoch_estimate_s':step_s*((15802+bs-1)//bs),'twenty_train_epochs_estimate_h':20*step_s*((15802+bs-1)//bs)/3600}
        model.eval()
        perm=torch.randperm(8,device=x.device)
        with torch.no_grad():
            a=model(x[:2],m[:2],ts[:2]);b=model(x[:2,:,perm],m[:2,perm],ts[:2])
        error=float((a['prediction'][:,:,perm]-b['prediction']).abs().max());assert error<2e-4
        record(kind+'_end_to_end_permutation',error)
        report['profiling'][kind]={'parameters':model.parameter_summary(),'batch_measurements':measurements}
        save();print('PROFILE',kind,json.dumps(report['profiling'][kind]),flush=True)
        del model,opt,out,loss,a,b;torch.cuda.empty_cache()
    assert len(set(hashes.values()))==1
    record('identical_shared_GPT2_token_initialization',hashes)
    q=HypergraphQuantumCore().cuda()
    dynamic={}
    for n in [8,12,16,20]:
        x=torch.randn(16,20,n,4,device='cuda');mask=torch.ones(16,n,dtype=torch.bool,device='cuda')
        times=[];torch.cuda.reset_peak_memory_stats()
        for s in range(5):
            q.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.perf_counter()
            out=q(x,mask);out.square().mean().backward();torch.cuda.synchronize()
            if s>0:times.append(time.perf_counter()-start)
        dynamic[str(n)]={'core_forward_backward_s':statistics.median(times),'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'circuit_instances_per_scene':1 if n<=8 else n}
    report['profiling']['dynamic_N_quantum_core']=dynamic;save();print('DYNAMIC',json.dumps(dynamic),flush=True)

if __name__=='__main__':
    torch.set_num_threads(4)
    reference_check();structural_checks();profile()
    report['status']='PASS';save();print('ALL_CHECKS_PASS',flush=True)
