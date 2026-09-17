"""CUDA float64/complex128 mathematical acceptance, without training."""
import argparse
import json
import math
from pathlib import Path
import torch
from prediction.quantum import QuantumGraph, expectation, pauli_rotation
from prediction.reference import exact_moment, first_layer_moments, dense_circuit

torch.set_default_dtype(torch.float64)
DEVICE = 'cuda:0'
RESULT = {}


def record(name, error, tol=1e-10, **extra):
    error = float(error)
    RESULT[name] = dict(passed=error <= tol, max_abs_error=error, tolerance=tol, **extra)
    print(name, RESULT[name], flush=True)


def err(a,b):
    return (a-b).abs().max().item()


def model(n, depth=3, random=True):
    m = QuantumGraph([50.,50.,15.,15.],depth=depth).to(DEVICE).double()
    if random:
        with torch.no_grad():
            m.theta.add_(.4*torch.randn_like(m.theta))
    x = torch.randn(n,4,device=DEVICE)*torch.tensor([20.,20.,8.,8.],device=DEVICE)
    return m,x,torch.ones(n,device=DEVICE,dtype=torch.bool)


def flat_moments(d):
    return torch.cat([d[k].reshape(-1) for k in ('singles','pair_raw','pair_connected','local_connected')])


def checks_a():
    for n in (2,3,4,8):
        m,x,mask=model(n,1); x.requires_grad_()
        d=m(x,mask,return_details=True)
        a,g=d['encoding_angles'][0],d['angles'][0]
        ref=first_layer_moments(a,g)
        offdiag=(~torch.eye(n,device=DEVICE,dtype=torch.bool))[...,None]
        ref['raw']=ref['raw']*offdiag
        ref['connected']=ref['connected']*offdiag
        mapped=dict(singles=ref['singles'],pair_raw=ref['raw'],pair_connected=ref['connected'],local_connected=ref['local'])
        # Compare raw moments including self pairs; masks apply only at aggregation.
        f=flat_moments(d['shallow']); r=flat_moments(mapped)
        record(f'A_N{n}_all_raw_and_connected',err(f,r),count=f.numel())
        weights=torch.randn_like(f)
        ga=torch.autograd.grad((f*weights).sum(),(m.theta,x),retain_graph=True)
        gb=torch.autograd.grad((r*weights).sum(),(m.theta,x))
        for label,p,q in zip(('shared_theta','input'),ga,gb):
            near=p.abs()<1e-9
            record(f'A_N{n}_{label}',err(p,q),1e-8,near_zero_count=int(near.sum()),
                   near_zero_abs_error=float((p-q)[near].abs().max()) if near.any() else 0.,
                   near_zero_relative_error=float(((p-q).abs()/(p.abs()+1e-30))[near].max()) if near.any() else 0.)
        if n==2:
            # Central finite differences on a nonzero shared parameter, same scalar objective.
            analytic=ga[0][0,4].item(); values=[]
            for h in (1e-2,3e-3,1e-3,3e-4):
                with torch.no_grad():
                    m.theta[0,4]+=h; plus=(flat_moments(m(x,mask,return_details=True)['shallow'])*weights).sum()
                    m.theta[0,4]-=2*h; minus=(flat_moments(m(x,mask,return_details=True)['shallow'])*weights).sum()
                    m.theta[0,4]+=h
                values.append(abs(float((plus-minus)/(2*h))-analytic))
            record('A_shared_finite_difference_convergence',0 if all(b<a for a,b in zip(values,values[1:])) else 1,0,steps=[1e-2,3e-3,1e-3,3e-4],errors=values)


def cumulant(state,a,b,c):
    e=lambda ops: expectation(state,ops)
    return e(a|b|c)-e(a|b)*e(c)-e(a|c)*e(b)-e(b|c)*e(a)+2*e(a)*e(b)*e(c)


def checks_b_d():
    m,x,mask=model(3,1)
    d=m(x,mask,return_details=True); a=d['encoding_angles'][0]
    zero=torch.zeros_like(d['angles'])
    dz=m(x,mask,return_details=True,angles_override=zero)
    singles=dz['shallow']['singles'].reshape(3,2,3)
    recovered=torch.stack((torch.acos(singles[...,2]),torch.atan2(singles[...,1],singles[...,0])),-1).reshape(3,4)
    omega=.25+3.75*torch.sigmoid(m.theta[0,:4])
    xr=torch.tensor([50.,50.,15.,15.],device=DEVICE)/omega*torch.tan(2*(recovered-math.pi/2))
    record('B_encoding_inverse',err(x,xr))
    sh=d['shallow']; chi=d['shallow_state']
    old=[sh['local_connected'].flatten(),sh['pair_connected'][...,0].flatten()]
    old.append(torch.stack([cumulant(chi,{0:'Z'},{2:'Z'},{5:'X'})]))
    record('B_four_old_zero_channels',torch.cat(old).abs().max())
    g=zero.clone().requires_grad_(); chi0=m(x,mask,return_details=True,angles_override=g)['shallow_state']
    k=cumulant(chi0,{0:'Z'},{2:'Z'},{5:'Z'})
    grad=torch.autograd.grad(k,g)[0]
    record('B_three_vehicle_zero_cumulant_and_derivative',max(abs(float(k)),float(grad.abs().max())))
    # T03 local derivative and the paired opposite-source construction.
    gz=zero.clone().requires_grad_(); dd=m(x,mask,angles_override=gz,return_details=True)
    c=dd['shallow']['pair_connected'][0,1,2]
    actual=torch.autograd.grad(c,gz)[0][0,0,1]
    expected=(1-singles[0,0,2]**2)*singles[1,1,1]
    record('B_single_edge_derivative',err(actual,expected),lower_bound=float(expected))
    aa=.3; bb=1.; angle=.1
    xp=torch.zeros(3,4,device=DEVICE)
    xp[1,0]=50/omega[0]*torch.tan(2*torch.asin(torch.tensor(aa,device=DEVICE)))
    xp[2,0]=-xp[1,0]
    gp=zero.clone(); gp[0,1,0]=-angle; gp[0,2,0]=angle
    za=m(xp,mask,angles_override=gp,return_details=True)['shallow']['singles'][0,5]
    zb=m(xp,mask,angles_override=-gp,return_details=True)['shallow']['singles'][0,5]
    record('B_pair_difference',abs(float(za-zb)-2*aa*bb*math.sin(2*angle)))
    I=torch.eye(2,device=DEVICE,dtype=torch.complex128)
    Y=torch.tensor([[0,-1j],[1j,0]],device=DEVICE); Z=torch.diag(torch.tensor([1.,-1.],device=DEVICE)).to(torch.complex128)
    ry=torch.matrix_exp(-1j*math.pi/4*Y); rz=torch.matrix_exp(-1j*math.pi/4*Z)
    record('D_old_E0_cube_minus_identity',err(torch.linalg.matrix_power(rz@ry,3),-I))
    mm,xx,ma=model(2,3,False); zz=mm(torch.zeros_like(xx),ma,return_details=True)
    record('D_new_zero_later_encoding',zz['encoding_angles'][1:].abs().max())
    # Test four-term operator identity on an arbitrary complex pure state.
    s=torch.randn(64,device=DEVICE,dtype=torch.complex128);s=s/s.norm()
    gj,gk=.17,-.23
    out=pauli_rotation(pauli_rotation(s,{0:'Z',5:'X'},torch.tensor(gj,device=DEVICE)),{2:'Z',5:'X'},torch.tensor(gk,device=DEVICE))
    lhs=expectation(out,{5:'Z'})
    rhs=(math.cos(gj)*math.cos(gk)*expectation(s,{5:'Z'})+math.sin(gj)*math.cos(gk)*expectation(s,{0:'Z',5:'Y'})+math.cos(gj)*math.sin(gk)*expectation(s,{2:'Z',5:'Y'})-math.sin(gj)*math.sin(gk)*expectation(s,{0:'Z',2:'Z',5:'Z'}))
    record('D_two_source_four_term_expansion',err(lhs,rhs))
    # Independent Kronecker Pauli matrices + spectral exponentials for all gates.
    mm,xx,ma=model(2,3)
    dd=mm(xx,ma,return_details=True)
    dense=dense_circuit(dd['encoding_angles'],dd['angles'],mm.theta[:,11:])
    record('D_independent_dense_three_layer_state',err(dense,dd['deep_state']))
    probe=torch.randn_like(dense)
    grad1=torch.autograd.grad((dense.conj()*probe).sum().real,mm.theta,retain_graph=True)[0]
    grad2=torch.autograd.grad((dd['deep_state'].conj()*probe).sum().real,mm.theta)[0]
    record('D_independent_dense_shared_gradient',err(grad1,grad2),1e-8)


def checks_c_e_h():
    m,x,mask=model(3);x.requires_grad_()
    d=m(x,mask,return_details=True);q=d['readout']
    p=torch.tensor([2,0,1],device=DEVICE);qp=m(x[p],mask[p])
    record('C_permutation_output',err(q[p],qp))
    weights=torch.randn_like(q)
    g1=torch.autograd.grad((q*weights).sum(),(x,m.theta),retain_graph=True)
    g2=torch.autograd.grad((qp*weights[p]).sum(),(x,m.theta))
    record('C_permutation_gradients',max(err(a,b) for a,b in zip(g1,g2)),1e-8)
    padded=torch.cat((x,torch.randn(5,4,device=DEVICE)*100)); pm=torch.tensor([1,1,1,0,0,0,0,0],device=DEVICE,dtype=torch.bool)
    pq=m(padded,pm)
    record('C_fixed_16_qubit_vs_N3',err(pq[:3],q))
    pg=torch.autograd.grad((pq[:3]*weights).sum(),(x,m.theta))
    record('C_padding_gradients',max(err(a,b) for a,b in zip(g1,pg)),1e-8)
    record('C_masked_output_zero',pq[3:].abs().max())
    # Explicitly disconnect added ACTIVE node; all old angles retained.
    xa=torch.cat((x,torch.randn(1,4,device=DEVICE)));ga=torch.zeros(3,4,4,device=DEVICE);ga[:,:3,:3]=d['angles']
    qa=m(xa,torch.ones(4,device=DEVICE,dtype=torch.bool),angles_override=ga)
    record('C_disconnected_active_node',err(qa[:3],q))
    ag=torch.autograd.grad((qa[:3]*weights).sum(),(x,m.theta),retain_graph=True)
    record('C_disconnected_active_gradients',max(err(a,b) for a,b in zip(g1,ag)),1e-8)
    # Multilayer derivative insertion must equal the isolated two-vehicle result.
    zeros=torch.zeros(3,3,3,device=DEVICE,requires_grad=True)
    full=m(x,mask,angles_override=zeros,return_details=True)
    one=full['deep']['singles'][1,5]
    jac=torch.autograd.grad(one,zeros)[0]
    small=torch.zeros(3,2,2,device=DEVICE,requires_grad=True)
    sub=m(x[:2],mask[:2],angles_override=small,return_details=True)['deep']['singles'][1,5]
    sj=torch.autograd.grad(sub,small)[0]
    record('E_multilayer_two_vehicle_insertion',err(jac[:,0,1],sj[:,0,1]),1e-8,depth=3)
    # Analytic JVP at zero strength, then shrinking nonlinear residual.
    func=lambda strength:m(x,mask,interaction_scale=strength)
    q0,dq=torch.autograd.functional.jvp(func,torch.tensor(0.,device=DEVICE),torch.tensor(1.,device=DEVICE))
    hs=[.1,.05,.025,.0125]
    residual=[float((func(torch.tensor(h,device=DEVICE))-q0-h*dq).norm()) for h in hs]
    slopes=[math.log(a/b,2) for a,b in zip(residual,residual[1:])]
    record('E_strength_tangent_second_order',max(0.,1.8-min(slopes)),0,steps=hs,residual_norms=residual,observed_orders=slopes)
    # Physical approach feature independently computed in SI units.
    r=x[:,None,:2]-x[None,:,:2];u=x[:,None,2:]-x[None,:,2:]
    app=torch.tanh(-(r*u).sum(-1)/torch.sqrt((r*r).sum(-1)+1)/15)
    record('H_approach_formula_and_swap',max(err(app,d['approach']),err(app,app.T)))
    weighted=(d['shallow']['pair_connected'][...,1:]*app[...,None]).sum(0)/7
    # self connected could be nonzero for XZ; self edges must be excluded.
    weighted=weighted-d['shallow']['pair_connected'][torch.arange(3),torch.arange(3),1:]*app.diag()[:,None]/7
    record('H_appended_readout_indices',err(weighted,q[:,21:]))
    record('H_readout_bounds',max(0.,float(q.abs().max())-1),0)
    # Explicit product rule: input gradient through BOTH a and C.
    cc=d['shallow']['pair_connected'][...,1:]; loss=(app[...,None]*cc).sum()/7
    both=torch.autograd.grad(loss,x,retain_graph=True)[0]
    onlya=torch.autograd.grad((app[...,None]*cc.detach()).sum()/7,x,retain_graph=True)[0]
    onlyc=torch.autograd.grad((app.detach()[...,None]*cc).sum()/7,x)[0]
    record('H_weight_and_connection_gradient_product_rule',err(both,onlya+onlyc),1e-8,weight_path_norm=float(onlya.norm()),connection_path_norm=float(onlyc.norm()))
    oneq=m(x[:1],mask[:1]);record('H_no_neighbor_zero',oneq[:,[6,7,8,17,18,19,20,21,22,23]].abs().max())
    cases=torch.tensor([[0.,0.,1.,0.],[10.,0.,-1.,0.]],device=DEVICE)
    ad=m(cases,mask[:2],return_details=True)['approach'];record('H_approaching_positive',0 if ad[0,1]>0 else 1,0)
    cases[1,2]=3;ad=m(cases,mask[:2],return_details=True)['approach'];record('H_separating_negative',0 if ad[0,1]<0 else 1,0)
    cases[1,:2]=cases[0,:2];ad=m(cases,mask[:2],return_details=True)['approach'];record('H_coincident_finite_zero',ad.abs().max())
    cases[1,:2]+=10;cases[1,2:]=cases[0,2:];ad=m(cases,mask[:2],return_details=True)['approach'];record('H_equal_velocity_zero',ad.abs().max())
    # No temporal access: changing later frames cannot alter earlier frame output.
    history=torch.stack((x.detach(),x.detach()+2,x.detach()+3))
    histmask=mask.expand(3,-1)
    hq=m(history,histmask);changed=history.clone();changed[1:]+=100
    record('H_future_frame_mutation',err(hq[0],m(changed,histmask)[0]))
    old=QuantumGraph([50.,50.,15.,15.],depth=3,legacy=True).to(DEVICE)
    with torch.no_grad():old.theta.add_(.3*torch.randn_like(old.theta))
    new=QuantumGraph([50.,50.,15.,15.],depth=3).to(DEVICE).load_legacy(old)
    oq=old(x.detach(),mask);nq=new(x.detach(),mask)
    proj=torch.randn(21,9,device=DEVICE);padded_proj=torch.cat((proj,torch.zeros(3,9,device=DEVICE)))
    record('H_old_Q_embedding_and_zero_projection',max(err(oq,nq[:,:21]),err(oq@proj,nq@padded_proj)))
    loss=(nq*torch.randn_like(nq)).sum()
    newgrad=torch.autograd.grad(loss,new.theta)[0][:,8]
    record('H_activated_added_edge_gradient',0 if newgrad.abs().max()>1e-10 else 1,0,per_layer_gradient=newgrad.tolist())


def checks_g():
    m,x,mask=model(2)
    fn=lambda z:m(z.reshape(2,4),mask).reshape(-1)
    J=torch.autograd.functional.jacobian(fn,x.flatten())
    S=torch.diag(torch.linspace(.5,2.,8,device=DEVICE));delta=torch.randn(8,device=DEVICE)
    def distance(j):
        diff=j@delta;cov=j@S@j.T
        return diff@torch.linalg.pinv(cov,hermitian=True,rtol=1e-10)@diff
    d1=distance(J);d2=distance(7*J)
    sv=torch.linalg.svdvals(J@torch.linalg.cholesky(S));rank=int((sv>sv[0]*1e-5).sum())
    bound=delta@torch.linalg.solve(S,delta)
    delta_rep=J@delta;noise=J@S@J.T
    record('G_representation_and_noise_scaling',max(err((7*J)@delta,7*delta_rep),err((7*J)@S@(7*J).T,49*noise)))
    A=J@torch.linalg.cholesky(S);_,singular,vh=torch.linalg.svd(A,full_matrices=False)
    retained=singular>singular[0]*1e-5
    whitened=torch.linalg.solve_triangular(torch.linalg.cholesky(S),delta[:,None],upper=False)[:,0]
    projected=(vh[retained]@whitened).square().sum()
    record('G_pseudoinverse_vs_row_projection',err(d1,projected))
    record('G_scaling_distance_invariance',abs(float(d1-d2)),rank=rank,input_dimension=8,distance=float(d1),input_upper_bound=float(bound),representation_difference_scale=7,noise_covariance_scale=49)
    record('G_local_information_upper_bound',max(0.,float(d1-bound)))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='reports/f02/quantum_checks.json');parser.add_argument('--section',default='all');args=parser.parse_args()
    torch.manual_seed(20260912)
    for label,func in [('A',checks_a),('BD',checks_b_d),('CEH',checks_c_e_h),('G',checks_g)]:
        if args.section not in ('all',label):continue
        try:func()
        except Exception as exc:
            RESULT[label+'_execution_error']=dict(passed=False,error=repr(exc));print(label,repr(exc),flush=True)
    output=dict(dtype='float64/complex128',device=DEVICE,seed=20260912,passed=all(v['passed'] for v in RESULT.values()),checks=RESULT)
    Path(args.output).parent.mkdir(parents=True,exist_ok=True);Path(args.output).write_text(json.dumps(output,indent=2)+'\n')
    if not output['passed']:raise SystemExit(1)


if __name__=='__main__':main()
