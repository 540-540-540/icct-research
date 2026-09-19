"""Scene and quantum-cumulant feedback inside RC-HQGNN evolution.
Exact simulator model; hardware must estimate and replay circuit prefixes.
"""
from __future__ import annotations
import torch
from prediction.qgnn_final.quantum import tables,rotate
from prediction.qgnn_final.relational import RelationCarryingQuantumCore,cumulants
from prediction.qgnn_final.common import physical_graph
from .controller import CouplingController

def quantum_round(state,ry,rz,pa,ta,rx,mask):
    n=mask.shape[1];_,z,_,_,zz,zzz=tables(n,state.device,state.real.dtype)
    for i in range(n):state=rotate(state,ry[:,i],i,"Y")
    phase=rz@z.T+pa@zz.T+ta@zzz.T
    state=state*torch.exp(-.5j*phase)
    for i in range(n):state=rotate(state,rx*mask[:,i],i,"X")
    return state

class SceneAdaptiveQuantumCore(RelationCarryingQuantumCore):
    def __init__(self,depth=3,channels=4,controller_seed=802029,feedback=True,phase_mode=False):
        super().__init__(depth,channels)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(controller_seed);self.controller=CouplingController(depth,channels)
        self.feedback_enabled=feedback;self.phase_mode=phase_mode

    def modulate_angles(self,pa,ta,l2,l3,d,mask):
        if not self.phase_mode:
            return pa*l2.reshape_as(pa),ta*l3.reshape_as(ta)
        active=mask.sum(1).clamp_min(1).to(pa.dtype)
        n2=(d["pw"].square().sum(1)/active).clamp_min(1.).sqrt()
        n3=(d["tw"].square().sum(1)/active).clamp_min(1.).sqrt()
        p=1.5*(l2-1)*d["pw"][:,None]/n2[:,None,None]*self.interaction_scale
        t=1.5*(l3-1)*d["tw"][:,None]/n3[:,None,None]*self.interaction_scale*self.triple_scale
        return pa+p.reshape_as(pa),ta+t.reshape_as(ta)

    def quantum_states(self,history,mask,trace=False):
        own,risk,tw,ry,rz,pa,ta,rx,em=self.circuit_inputs(history,mask)
        edge,weights=physical_graph(history,mask);d=self.controller.prepare(edge,weights,mask)
        b,n,_=own.shape;ch=self.channels
        state=torch.zeros(b*ch,2**n,device=own.device,dtype=torch.complex128 if own.dtype==torch.float64 else torch.complex64)
        state[:,0]=1.;states=[];feedback=None;logs=[]
        for layer in range(self.depth):
            l2,l3=self.controller(layer,d,feedback if self.feedback_enabled else None)
            p,t=self.modulate_angles(pa[:,layer],ta[:,layer],l2,l3,d,mask)
            mix=rx[:,layer] if rx.ndim==2 else rx[layer].expand(b*ch)
            state=quantum_round(state,ry[:,layer],rz[:,layer],p,t,mix,em)
            states.append(state)
            if layer<self.depth-1 or trace:
                k2,k3=cumulants(state,n);pairs=d["pairs"]
                k2=k2[:,pairs[:,0],pairs[:,1]].reshape(b,ch,-1);k3=k3.reshape(b,ch,-1)
                feedback=(torch.tanh(k2/.05),torch.tanh(k3/.02))
                if trace:logs.append(dict(lambda2=l2,lambda3=l3,k2=k2,k3=k3,pair_phase=p,triple_phase=t))
        return own,states,risk,tw,em,logs

    def features_and_own(self,history,mask):
        own,states,risk,tw,em,_=self.quantum_states(history,mask)
        base=self.readout_features(states,em,risk,tw,own.shape[0])
        relation=self.relational_features(states,own,history,mask)
        return own,torch.cat((base,relation),-1)
