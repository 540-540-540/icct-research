"""Four-cell QGAT experiment: one/two state qubits and all/45m neighbors."""
import math
import torch
from torch import nn
import pennylane as qml
from experiments.qgat_candidate.graph import QGATGraph, encode, unitary, observables


def two_unitary(angles, theta, wires):
    for i, wire in enumerate(wires):
        encode(angles[...,3*i:3*i+3], wire)
    qml.CNOT(wires=wires)
    for i, wire in enumerate(wires):
        qml.U3(theta[i,0],theta[i,1],theta[i,2],wires=wire)


def pauli_observables(amplitude, mask, probability):
    """15 mixed + 16 coherent Pauli expectations + postselection probability.

    Pauli order is lexicographic I,X,Y,Z tensor I,X,Y,Z; mixed excludes II.
    Coherent state is the uniform-index projection, not a classical mixture.
    """
    one=amplitude.new_tensor([[[1,0],[0,1]],[[0,1],[1,0]],
                              [[0,-1j],[1j,0]],[[1,0],[0,-1]]])
    paulis=torch.stack([torch.kron(p,q) for p in one for q in one])
    mixed=torch.einsum('...jd,pde,...je->...p',amplitude.conj(),paulis,amplitude).real
    coherent=amplitude.sum(-2)/mask.sum(-1).to(amplitude.real.dtype).clamp_min(1).sqrt()[...,None]
    coh=torch.einsum('...d,pde,...e->...p',coherent.conj(),paulis,coherent).real
    return torch.cat((mixed[...,1:],coh,probability[...,None]),-1)


class FactorialGraph(QGATGraph):
    def __init__(self, qubits=1, radius=None):
        if qubits not in (1,2) or radius not in (None,45):
            raise ValueError('Factorial graph requires qubits=1/2 and radius=None/45')
        if qubits==1:
            super().__init__()
        else:
            nn.Module.__init__(self)
            self.encoder=nn.Linear(6,6)
            self.theta=nn.Parameter(torch.empty(3,2,3).uniform_(-.3,.3))
            self.readout=nn.Sequential(nn.Linear(32,128),nn.LayerNorm(128))
            @qml.qnode(qml.device('default.qubit',wires=2,shots=None),interface='torch',diff_method='backprop')
            def local_state(angles,theta):
                initial=torch.zeros((*angles.shape[:-1],4),dtype=torch.complex128,device=angles.device)
                initial[...,0]=1
                qml.StatePrep(initial,wires=[0,1])
                two_unitary(angles,theta,[0,1])
                return qml.state()
            self.local_state=local_state
        self.qubits,self.radius=qubits,radius
        self.raw_dim=8 if qubits==1 else 32
        self.output_dim=128
        att=list(range(3,3+qubits));val=list(range(3+qubits,3+2*qubits))
        def apply(a,t,wires):
            if qubits==1: unitary(a,t,wires[0])
            else: two_unitary(a,t,wires)
        @qml.qnode(qml.device('default.qubit',wires=3+2*qubits,shots=None),interface='torch',diff_method='backprop')
        def full_circuit(angles,query,index_state,theta):
            qml.StatePrep(index_state,wires=[0,1,2])
            apply(query,theta[0],att)
            for i in range(8):
                controls=[(i>>j)&1 for j in (2,1,0)]
                qml.ctrl(qml.adjoint(apply),control=[0,1,2],control_values=controls)(angles[...,i,:],theta[1],att)
                qml.ctrl(apply,control=[0,1,2],control_values=controls)(angles[...,i,:],theta[2],val)
            return qml.state()
        self.full_circuit=full_circuit

    def features(self,state_hat,standardized_state,track_exists,detected,reference=False):
        shape=standardized_state.shape
        if shape[-1]!=4 or not 1<=shape[-2]<=8 or state_hat.shape!=shape:
            raise ValueError('Expected matching physical/standardized [...,1..8,4] states')
        if track_exists.shape!=shape[:-1] or detected.shape!=track_exists.shape:
            raise ValueError('Mask shape mismatch')
        m=track_exists.bool();z=torch.where(m[...,None],standardized_state,0)
        if not torch.isfinite(z).all() or (detected.bool()&~m).any():
            raise ValueError('Nonfinite valid state or detected without track')
        inp=torch.cat((z,m[...,None].to(z),detected[...,None].to(z)),-1)
        a=math.pi*torch.tanh(self.encoder(inp.to(self.encoder.weight.dtype)))
        n=shape[-2];a=a.reshape(-1,n,3*self.qubits).to(torch.float64);m=m.reshape(-1,n)
        pairs=m[:,:,None]&m[:,None,:]
        if self.radius is not None:
            xy=torch.where(track_exists[...,None].bool(),state_hat,0)[...,:2].reshape(-1,n,2)
            if not torch.isfinite(xy).all():
                raise ValueError('Nonfinite active physical position')
            pairs=pairs&((xy[:,:,None]-xy[:,None,:]).square().sum(-1)<=self.radius**2)
        theta=self.theta.to(torch.float64);batch=len(a);dim=2**self.qubits
        if reference is False:
            # Independent local states are evaluated once per node, not per query.
            q,k,v=[self.local_state(a.reshape(-1,3*self.qubits),t).reshape(batch,n,dim) for t in theta]
            gamma=torch.einsum('bjd,bkd->bkj',k.conj(),q)
            unnormalized=gamma[...,None]*v[:,None]*pairs[...,None]
            unnormalized=unnormalized/pairs.sum(-1).to(a.dtype).clamp_min(1).sqrt()[...,None,None]
        elif reference=='full':
            neighbors=torch.nn.functional.pad(a,(0,0,0,8-n))[:,None].expand(batch,n,8,3*self.qubits)
            masks=torch.nn.functional.pad(pairs,(0,8-n))
            prep=masks.to(a.dtype)
            prep=prep+(~masks.any(-1))[...,None]*torch.nn.functional.one_hot(torch.tensor(0,device=a.device),8)
            prep=prep/prep.sum(-1,keepdim=True).sqrt()
            state=self.full_circuit(neighbors.reshape(-1,8,3*self.qubits),a.reshape(-1,3*self.qubits),prep.reshape(-1,8),theta)
            unnormalized=state.reshape(batch,n,8,dim,dim)[:,:,:n,0,:]*pairs[...,None]
        else:
            raise ValueError('reference must be False or full')
        probability=unnormalized.abs().square().sum((-1,-2))
        usable=probability>1e-12
        amplitude=unnormalized/torch.where(usable,probability,1).sqrt()[...,None,None]
        result=(observables if self.qubits==1 else pauli_observables)(amplitude,pairs,probability)
        result=torch.where(usable[...,None],result,0)*m[...,None]
        return result.reshape(*shape[:-1],self.raw_dim)

    def forward(self,state_hat,standardized_state,track_exists,detected):
        raw=self.features(state_hat,standardized_state,track_exists,detected)
        return self.readout(raw.to(self.readout[0].weight.dtype))*track_exists[...,None]
