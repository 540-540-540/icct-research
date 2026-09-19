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
    def __init__(self,depth=3,channels=4,controller_seed=802029,feedback=True):
        super().__init__(depth,channels)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(controller_seed);self.controller=CouplingController(depth,channels)
        self.feedback_enabled=feedback

    def quantum_states(self,history,mask,trace=False):
        own,risk,tw,ry,rz,pa,ta,rx,em=self.circuit_inputs(history,mask)
        edge,weights=physical_graph(history,mask);d=self.controller.prepare(edge,weights,mask)
        b,n,_=own.shape;ch=self.channels
        state=torch.zeros(b*ch,2**n,device=own.device,dtype=torch.complex128 if own.dtype==torch.float64 else torch.complex64)
        state[:,0]=1.;states=[];feedback=None;logs=[]
        for layer in range(self.depth):
            l2,l3=self.controller(layer,d,feedback if self.feedback_enabled else None)
            p=pa[:,layer]*l2.reshape(b*ch,-1);t=ta[:,layer]*l3.reshape(b*ch,-1)
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
