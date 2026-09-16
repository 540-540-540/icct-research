"""Four-qubit real-statevector message map inside the second bipartite layer."""
import sys
from pathlib import Path
import torch
from torch import nn
OLD=Path('/home/js_cn/sensing/双图研究框架/02_量测航迹关联图_方案/code_snapshot')
sys.path.insert(0,str(OLD))
from measurement_track_association import MeasurementTrackAssociationNet, AssociationGraphConfig

class Circuit(nn.Module):
    def __init__(self):
        super().__init__()
        self.theta=nn.Parameter(torch.randn(2,4)*.05)
        bits=((torch.arange(16)[:,None] >> torch.arange(4)) & 1)
        self.register_buffer('z',1.-2.*bits.float())
        self.register_buffer('cz',torch.prod(1.-2.*(bits*bits.roll(-1,1)).float(),1))
        for q in range(4):
            i=torch.arange(16); self.register_buffer('flip%d'%q,i^(1<<q))
    def state(self,x):
        s=torch.zeros(*x.shape[:-1],16,device=x.device,dtype=x.dtype); s[...,0]=1.
        for layer in range(2):
            for q in range(4):
                a=x[...,q]+self.theta[layer,q]
                c=torch.cos(a/2)[...,None]; t=torch.sin(a/2)[...,None]
                s=c*s-t*self.z[:,q]*s[...,getattr(self,'flip%d'%q)]
            s=s*self.cz
        return s
    def forward(self,x): return self.state(x).square()@self.z

class MessageMap(nn.Module):
    def __init__(self, original, kind):
        super().__init__(); self.original=original; self.kind=kind
        self.encoder=nn.Sequential(nn.LayerNorm(288),nn.Linear(288,4))
        self.mapping=Circuit() if kind=='quantum' else nn.Sequential(nn.Linear(4,4),nn.Tanh(),nn.Linear(4,4),nn.Tanh())
        self.readout=nn.Linear(4,96,bias=False); nn.init.zeros_(self.readout.weight)
    def forward(self,x):
        angles=torch.pi*torch.tanh(self.encoder(x))
        return self.original(x)+self.readout(self.mapping(angles))

def build(kind, initial):
    torch.manual_seed(2026)
    model=MeasurementTrackAssociationNet(AssociationGraphConfig(dropout=0.0))
    model.load_state_dict(initial,strict=True)
    if kind!='plain':
        torch.manual_seed(22026)
        model.layers[1].edge_message=MessageMap(model.layers[1].edge_message,kind)
    return model.cuda()
