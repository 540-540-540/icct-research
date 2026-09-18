"""Factorial graph numerical contracts; no fitting and no dataset access."""
import sys,json,time,math
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.qgat_candidate.graph import QGATGraph
from experiments.qgat_factorial.graph import FactorialGraph


def gradients(output,weight,x,graph):
    return torch.autograd.grad((output*weight).sum(),(x,*graph.parameters()),allow_unused=True)


def max_gradient_error(a,b):
    error=0.
    for x,y in zip(a,b):
        assert (x is None)==(y is None)
        if x is not None:
            assert torch.isfinite(x).all() and torch.isfinite(y).all()
            error=max(error,float((x-y).abs().max()))
    return error


def main():
    torch.set_num_threads(2);torch.manual_seed(20260913)
    report={'checks':[]}
    old=QGATGraph().double();new=FactorialGraph(1,None).double()
    new.load_state_dict(old.state_dict())
    x=torch.randn(3,8,4,dtype=torch.float64,requires_grad=True)
    mask=torch.zeros(3,8,dtype=torch.bool);mask[1,[0,3,7]]=True;mask[2]=True
    detected=mask&(torch.rand_like(mask.float())>.3)
    a=old(x,x,mask,detected);b=new(x,x,mask,detected);weight=torch.randn_like(a)
    error=float((a-b).abs().max());ge=max_gradient_error(gradients(a,weight,x,old),gradients(b,weight,x,new))
    assert error<1e-9 and ge<1e-7
    report['checks'].append(dict(case='A equals original QGAT',output_error=error,all_parameter_and_input_gradient_error=ge))
    for qubits in (1,2):
        for radius in (None,45):
            graph=FactorialGraph(qubits,radius).double()
            for n in (1,3,8):
                z=torch.randn(1,n,4,dtype=torch.float64,requires_grad=True)
                m=torch.ones(1,n,dtype=torch.bool);d=m.clone()
                physical=z*20
                a=graph.features(physical,z,m,d);b=graph.features(physical,z,m,d,reference='full')
                assert a[...,-1].min()>=-1e-10 and a[...,-1].max()<=1+1e-10
                if qubits==2:
                    assert a[...,:15].square().sum(-1).max()<=3+1e-9
                    torch.testing.assert_close(a[...,16:31].square().sum(-1),3*a[...,15].square(),atol=1e-9,rtol=1e-8)
                weight=torch.randn_like(a)
                ga=gradients(a,weight,z,graph);gb=gradients(b,weight,z,graph)
                err=float((a-b).abs().max());ger=max_gradient_error(ga,gb)
                assert err<1e-9 and ger<1e-7
                report['checks'].append(dict(case='local contraction vs complete controlled circuit',
                    qubits=qubits,logical_wires=3+2*qubits,radius=radius,n=n,output_error=err,gradient_error=ger))
            z=torch.randn(3,8,4,dtype=torch.float64)
            m=torch.zeros(3,8,dtype=torch.bool);m[1,[0,3,7]]=True;m[2]=True;d=m.clone()
            p=torch.tensor([7,3,1,6,4,0,2,5]);physical=z*30
            original=graph(physical,z,m,d)
            shuffled=graph(physical[:,p],z[:,p],m[:,p],d[:,p])
            perm=float((shuffled-original[:,p]).abs().max());assert perm<1e-9
            dirty=z.clone();dirty[~m]=float('nan');badphysical=physical.clone();badphysical[~m]=float('nan')
            assert torch.equal(original,graph(badphysical,dirty,m,d))
            assert not original[~m].any()
            # Isolated valid query retains its self-loop; exactly 45m is inside.
            if radius==45:
                z=torch.randn(1,3,4,dtype=torch.float64)
                physical=torch.zeros_like(z);physical[0,:,0]=torch.tensor([0.,45.,90.00001])
                m=torch.ones(1,3,dtype=torch.bool)
                inside=graph.features(physical,z,m,m)
                direct=graph.features(physical[:,:2],z[:,:2],m[:,:2],m[:,:2])
                torch.testing.assert_close(inside[:,0],direct[:,0],atol=1e-10,rtol=1e-9)
                physical[0,1,0]=45.00001
                outside=graph.features(physical,z,m,m)
                self_only=graph.features(physical[:,:1],z[:,:1],m[:,:1],m[:,:1])
                torch.testing.assert_close(outside[:,0],self_only[:,0],atol=1e-10,rtol=1e-9)
            report['checks'].append(dict(case='permutation invalid NaN empty frames radius boundary self loop',qubits=qubits,radius=radius,permutation_error=perm))
    # A deliberately orthogonal key yields a zero raw readout with finite zero gradients.
    for qubits in (1,2):
        graph=FactorialGraph(qubits,None).double()
        with torch.no_grad():
            graph.encoder.weight.zero_();graph.encoder.bias.zero_();graph.theta.zero_()
            if qubits==1:graph.theta[1,0]=math.pi
            else:graph.theta[1,0,0]=math.pi
        z=torch.randn(1,3,4,dtype=torch.float64,requires_grad=True);m=torch.ones(1,3,dtype=torch.bool)
        for reference in (False,'full'):
            out=graph.features(z,z,m,m,reference=reference)
            assert not out.any()
            grads=torch.autograd.grad(out.sum(),(z,graph.theta))
            assert all(torch.isfinite(g).all() and not g.any() for g in grads)
        with torch.no_grad():
            if qubits==1:graph.theta[1,0]=math.pi-4e-6
            else:graph.theta[1,0,0]=math.pi-4e-6
        recovered=graph.features(z,z,m,m)
        assert recovered[...,-1].min()>1e-12 and recovered.abs().max()>.9
        report['checks'].append(dict(case='nonempty nearzero fallback finite zero gradients and recovery',qubits=qubits,
                                     threshold=1e-12,recovered_min_success=float(recovered[...,-1].min().detach())))
    graph=FactorialGraph(2,None).double()
    z=torch.tensor([.23,-.41,.37,.62],dtype=torch.float64,requires_grad=True)
    m=torch.ones(1,1,dtype=torch.bool)
    jac=torch.autograd.functional.jacobian(lambda v:graph.features(v.reshape(1,1,4),v.reshape(1,1,4),m,m).flatten(),z)
    sv=torch.linalg.svdvals(jac);rank=int((sv>1e-7).sum());assert rank>=4
    report['jacobian']=dict(input='four continuous state channels',output='actual 32 raw readouts with one self neighbor',singular_values=sv.tolist(),rank=rank,
                            interpretation='Four encoded pure-state degrees of freedom; not a prediction-performance guarantee.')
    costs=[]
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        for qubits in (1,2):
            for radius in (None,45):
                graph=FactorialGraph(qubits,radius).cuda()
                z=torch.randn(1,20,8,4,device='cuda');m=torch.ones(1,20,8,dtype=torch.bool,device='cuda')
                times=[]
                for repeat in range(3):
                    graph.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
                    tick=time.perf_counter();out=graph(z*20,z,m,m);(out*torch.randn_like(out)).sum().backward();torch.cuda.synchronize()
                    times.append(time.perf_counter()-tick)
                costs.append(dict(qubits=qubits,radius=radius,shape=[1,20,8,4],seconds=times[1:],peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20))
    report.update(passed=True,backend='PennyLane default.qubit backprop complex128 shots=None',costs=costs,
                  mixed_pauli_squared_norm_le3=True,coherent_nonidentity_squared_norm_eq3_identity_squared=True)
    folder=ROOT/'reports/qgat_factorial';folder.mkdir(parents=True,exist_ok=True)
    (folder/'graph_checks.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
