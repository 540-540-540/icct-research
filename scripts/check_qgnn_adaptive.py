"""Round3 architecture acceptance. Only train samples are loaded."""
import sys,json,time,statistics,itertools,hashlib,traceback
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from prediction.qgnn_adaptive.quantum import SceneAdaptiveQuantumCore
from prediction.qgnn_adaptive.classical import SceneAdaptiveClassicalCore
from prediction.qgnn_adaptive.model import build_model
from prediction.qgnn_final.common import physical_graph
from prediction.qgnn_final.quantum import tables
OUT=ROOT/"reports/qgnn/round3_engineering_v1.json"
r={"status":"RUNNING","test_set_used":False,"checks":{},"profiling":{}}
def save():
    tmp=OUT.with_suffix(".tmp");tmp.write_text(json.dumps(r,indent=2));tmp.replace(OUT)
def rec(k,v):r["checks"][k]=v;save();print(k,json.dumps(v),flush=True)

def structural():
    torch.manual_seed(612);q=SceneAdaptiveQuantumCore().double().eval();c=SceneAdaptiveClassicalCore().double().eval()
    for name,model in [("quantum",q),("classical",c)]:
        errors={}
        for n in [1,2,4,8,12,16,20]:
            h=torch.randn(2,20,n,4,dtype=torch.float64);h[...,:2]*=10
            m=torch.ones(2,n,dtype=torch.bool);m[0,1::3]=False;p=torch.randperm(n)
            with torch.no_grad():a=model(h,m);b=model(h[:,:,p],m[:,p])
            e=float((a[:,p]-b).abs().max());assert e<1e-8
            assert a.shape==(2,n,64) and torch.isfinite(a).all() and (a[~m]==0).all()
            errors[n]=e
        rec(name+"_permutation_shapes",errors)
        h=torch.randn(2,20,4,4,dtype=torch.float64);m=torch.ones(2,4,dtype=torch.bool)
        hp=torch.cat((h,torch.full((2,20,16,4),float("nan"),dtype=h.dtype)),2);mp=torch.cat((m,torch.zeros(2,16,dtype=torch.bool)),1)
        with torch.no_grad():e=float((model(h,m)-model(hp,mp)[:,:4]).abs().max())
        assert e<1e-8;rec(name+"_padding_to20",e)
        assert (model(h,torch.zeros_like(m))==0).all()
    assert all(torch.equal(v,c.controller.state_dict()[k]) for k,v in q.controller.state_dict().items())
    rec("identical_adaptive_controller_initialization",True)
    h=torch.randn(1,20,4,4,dtype=torch.float64,requires_grad=True);m=torch.ones(1,4,dtype=torch.bool);probe=torch.randn(64,dtype=h.dtype)
    def cross():return float(torch.autograd.grad((q(h,m)[0,0]*probe).sum(),h)[0][:,:,1:].norm())
    on=cross();q.interaction_scale=0.;off=cross();q.interaction_scale=1.
    assert on>1e-7 and off<1e-8;rec("cross_vehicle_dependency",dict(on=on,entanglers_off=off))
    q.zero_grad();(q(h,m)*probe).sum().backward()
    norms={k:float(p.grad.norm()) for k,p in q.named_parameters() if p.grad is not None}
    assert all(torch.isfinite(p.grad).all() for p in q.parameters() if p.grad is not None)
    assert norms["controller.feedback_local"]>1e-8 and norms["controller.scene_head.weight"]>1e-8
    rec("parameter_gradient_norms",norms)
    own,st,_,_,_,trace=q.quantum_states(h,m,True)
    err=max(float((s.abs().square().sum(-1)-1).abs().max()) for s in st);assert err<1e-10
    assert all(((v["lambda2"]>=.2)&(v["lambda2"]<=1.8)).all() for v in trace)
    rec("state_norm",err)

def reference():
    import pennylane as qml
    torch.manual_seed(613);q=SceneAdaptiveQuantumCore().double().eval()
    h=torch.randn(1,20,3,4,dtype=torch.float64,requires_grad=True);m=torch.ones(1,3,dtype=torch.bool)
    own,risk,tw,ry,rz,pa,ta,rx,em=q.circuit_inputs(h,m);edge,w=physical_graph(h,m);d=q.controller.prepare(edge,w,m)
    n=3;ch=4;dev=qml.device("default.qubit",wires=n)
    pairs=list(itertools.combinations(range(n),2));triples=list(itertools.combinations(range(n),3))
    @qml.qnode(dev,interface="torch",diff_method="backprop")
    def circuit(iy,iz,ip,it,ix):
        for l in range(iy.shape[0]):
            for j in range(n):qml.RY(iy[l,j],j);qml.RZ(iz[l,j],j)
            for j,ij in enumerate(pairs):qml.PauliRot(ip[l,j],"ZZ",wires=ij)
            for j,ijk in enumerate(triples):qml.PauliRot(it[l,j],"ZZZ",wires=ijk)
            for j in range(n):qml.RX(ix[l],j)
        return qml.state()
    _,z,pp,tt,zz,zzz=tables(n,h.device,h.dtype)
    feedback=None;ps=[];ts=[]
    for l in range(3):
        l2,l3=q.controller(l,d,feedback);ps.append(pa[:,l]*l2.reshape(ch,-1));ts.append(ta[:,l]*l3.reshape(ch,-1))
        ip=torch.stack(ps,1);it=torch.stack(ts,1)
        refs=torch.stack([circuit(ry[c,:l+1],rz[c,:l+1],ip[c],it[c],rx[c,:l+1]) for c in range(ch)])
        prob=refs.abs().square();mz=prob@z;raw2=prob@zz
        k2=raw2-mz[:,pp[:,0]]*mz[:,pp[:,1]]
        k3=prob@zzz-mz[:,0:1]*raw2[:,2:3]-mz[:,1:2]*raw2[:,1:2]-mz[:,2:3]*raw2[:,0:1]+2*mz.prod(-1,keepdim=True)
        feedback=(torch.tanh(k2/.05)[None],torch.tanh(k3/.02)[None])
    actual=q.quantum_states(h,m)[1][-1];error=float((actual-refs).abs().max());assert error<1e-10
    probe=torch.randn_like(actual);a=torch.autograd.grad((actual.conj()*probe).sum().real,[h,q.controller.feedback_local],retain_graph=True)
    b=torch.autograd.grad((refs.conj()*probe).sum().real,[h,q.controller.feedback_local]);ge=max(float((x-y).abs().max()) for x,y in zip(a,b));assert ge<1e-9
    rec("independent_PennyLane_prefix_feedback",dict(state_max_error=error,gradient_max_error=ge))

def profile():
    from frontend.sind_prediction_dataset import SinDPredictionDataset
    from torch.utils.data import DataLoader
    from prediction.q0.metrics import trajectory_metrics
    from scripts.train_q0_motion_llm import token_loss
    import importlib.metadata as md
    r["environment"]={k:md.version(k) for k in ["torch","pennylane","pennylane-lightning-gpu","custatevec-cu12","transformers"]}
    r["environment"]["gpu"]=torch.cuda.get_device_name(0)
    ds=SinDPredictionDataset("train",0.,ROOT,True);batch=next(iter(DataLoader(ds,batch_size=32)))
    x=batch["history_state"].cuda();m=batch["vehicle_mask"].cuda();y=batch["future_state"].cuda();ts=batch["history_timestamp"].cuda();hashes={}
    for kind in ["quantum","classical"]:
        model=build_model(kind,adaptive_mode="feedback").cuda();digest=hashlib.sha256()
        for name,p in model.llm.named_parameters():
            if p.requires_grad:digest.update(name.encode());digest.update(p.detach().cpu().numpy().tobytes())
        hashes[kind]=digest.hexdigest();opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-4)
        times=[];losses=[];torch.cuda.reset_peak_memory_stats()
        for step in range(9):
            model.train();opt.zero_grad(set_to_none=True);torch.cuda.synchronize();tick=time.perf_counter();out=model(x,m,ts)
            loss=trajectory_metrics(out["prediction"],y,m)["loss"]+.035*token_loss(out["token_logits"],model.llm.future_token_ids(x,y),m)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),3.);opt.step();torch.cuda.synchronize()
            if step>=3:times.append(time.perf_counter()-tick);losses.append(float(loss))
        sec=statistics.median(times);r["profiling"][kind]={"parameters":model.parameter_summary(),"controller_params":sum(p.numel() for p in model.graph.controller.parameters()),"B32_step_median_s":sec,"peak_allocated_GiB":torch.cuda.max_memory_allocated()/2**30,"peak_reserved_GiB":torch.cuda.max_memory_reserved()/2**30,"losses":losses,"full20_train_only_estimate_s":9880*sec};save()
        model.eval();perm=torch.randperm(8,device=x.device)
        with torch.no_grad():a=model(x[:2],m[:2],ts[:2])["prediction"];bb=model(x[:2,:,perm],m[:2,perm],ts[:2])["prediction"]
        e=float((a[:,:,perm]-bb).abs().max());assert e<2e-4;rec(kind+"_end_to_end_permutation",e)
        del model,opt,out,loss;torch.cuda.empty_cache()
    assert len(set(hashes.values()))==1;rec("shared_initialization_sha256",hashes)
    q=SceneAdaptiveQuantumCore().cuda();dynamic={}
    for n in [8,12,16,20]:
        h=torch.randn(16,20,n,4,device="cuda");m=torch.ones(16,n,dtype=torch.bool,device="cuda");times=[];torch.cuda.reset_peak_memory_stats()
        for k in range(5):
            q.zero_grad();torch.cuda.synchronize();t=time.perf_counter();q(h,m).square().mean().backward();torch.cuda.synchronize()
            if k:times.append(time.perf_counter()-t)
        dynamic[n]={"B16_core_forward_backward_s":statistics.median(times),"peak_allocated_GiB":torch.cuda.max_memory_allocated()/2**30,"trajectories_per_scene":4*(1 if n<=8 else n)}
    r["profiling"]["dynamic_N"]=dynamic;save()

if __name__=="__main__":
    torch.set_num_threads(4)
    try:structural();reference();profile();r["status"]="PASS";save();print("ALL_PASS",json.dumps(r["profiling"]),flush=True)
    except Exception as e:r.update(status="FAILED",error=repr(e),traceback=traceback.format_exc());save();raise
