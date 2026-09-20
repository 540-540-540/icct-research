from __future__ import annotations
import math,itertools
from functools import lru_cache
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .common import PrefixHistoryEncoder,block_statistics,pair_physics,trc_edge_features,apply_pauli_rotation,expect_pauli

@lru_cache(maxsize=8)
def _ordered_pairs(n:int):
    return tuple((a,b) for a in range(n) for b in range(n) if a!=b)

def _root_neighbors(n:int,root:int,device):
    return torch.tensor([j for j in range(n) if j!=root],device=device,dtype=torch.long)

def _atanh(x): return .5*math.log((1+x)/(1-x))

class TRCFeatureBase(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=PrefixHistoryEncoder('first',32)
    def base_features(self,history,mask,timestamps):
        h=self.encoder(history,mask)
        p,v,a,_=block_statistics(history,timestamps,mask)
        phys=pair_physics(p,v,a,mask);edge,risk=trc_edge_features(phys)
        return h,p,v,a,phys,edge,risk
    @staticmethod
    def role_desc(h,p,v,a,mask,root):
        rp=p[:,:,root:root+1];rv=v[:,:,root:root+1]
        relp=(p-rp)/30.;relv=(v-rv)/10.
        isroot=torch.zeros_like(mask,dtype=h.dtype);isroot[:,root]=1
        chi=torch.cat((relp,relv,v/10.,a/10.,isroot[:,None,:,None].expand(-1,4,-1,-1),mask[:,None,:,None].to(h.dtype).expand(-1,4,-1,-1)),-1)
        return torch.cat((h,chi),-1)
    @staticmethod
    def graph_parts(risk,phys,mask,root,neigh,branch):
        # risk/phys [B,4,N,N], neighbor-space A uses only sensed-history physics.
        wri=risk[:,:,root,neigh]
        wnn=risk[:,:,neigh][:,:,:,neigh]
        valid=mask[:,neigh]
        A=torch.sqrt(wri[:,:,:,None].clamp_min(0)*wri[:,:,None,:].clamp_min(0))*wnn
        eye=torch.eye(len(neigh),device=mask.device,dtype=torch.bool)[None,None]
        A=A.masked_fill(eye,0.)*valid[:,None,:,None]*valid[:,None,None,:]
        if branch==1:
            cfg_valid=valid
            pot=wri
            return A,pot,cfg_valid,None
        pairs=torch.tensor(_ordered_pairs(len(neigh)),device=mask.device,dtype=torch.long)
        x,y=pairs[:,0],pairs[:,1];M=len(pairs)
        same_y=(y[:,None]==y[None,:]).to(A.dtype);same_x=(x[:,None]==x[None,:]).to(A.dtype)
        # dest/source symmetric because A is symmetric.
        Ax=A[:,:,x[:,None],x[None,:]]*same_y
        Ay=A[:,:,y[:,None],y[None,:]]*same_x
        Acfg=Ax+Ay
        cfg_valid=valid[:,x]&valid[:,y]
        Acfg=Acfg*cfg_valid[:,None,:,None]*cfg_valid[:,None,None,:]
        wrx=wri[:,:,x];wry=wri[:,:,y];wjk=wnn[:,:,x,y]
        taux=phys['tau'][:,:,root,neigh[x]];tauy=phys['tau'][:,:,root,neigh[y]]
        eta=wrx*wry*wjk*torch.exp(-(taux-tauy).abs())
        pot=(wrx+wry+eta)/3.
        return Acfg,pot,cfg_valid,pairs

class TRCQuantumCore(TRCFeatureBase):
    """Temporal Rooted-Configuration QGNN with all target roots batched together."""
    def __init__(self,use_checkpoint:bool=True):
        super().__init__();self.use_checkpoint=bool(use_checkpoint)
        self.history_proj=nn.Linear(32,12)
        self.edge_scale=nn.Parameter(torch.full((12,),.25));self.edge_bias=nn.Parameter(torch.zeros(12))
        self.intra=nn.Parameter(torch.full((4,),.1))
        self.hop=nn.Parameter(torch.tensor([_atanh(.4),_atanh(.4)]))
        self.potential=nn.Parameter(torch.tensor([_atanh(.2),_atanh(.2)]))
    def _node_angles(self,roles):
        hist=math.pi*torch.tanh(self.history_proj(roles[...,:32]));direct=math.pi*torch.tanh(roles[...,32:42])
        return torch.cat((hist,direct,torch.zeros_like(direct[...,:2])),-1)
    def _encode(self,state,roles,edges):
        B,M,F=state.shape;R=roles.shape[2]
        x=state.reshape(B*M,F);ang=self._node_angles(roles).reshape(B*M,R,24)
        phi=(math.pi*torch.tanh(edges*self.edge_scale+self.edge_bias)).reshape(B*M,R,R,12)
        axis=('X','Y','Z');pairs=(('Z','X'),('X','Z'),('Y','Y'))
        for rr in range(4):
            for role in range(R):
                vals=ang[:,role,rr*6:(rr+1)*6]
                for k,ax in enumerate(axis):x=apply_pauli_rotation(x,vals[:,k],[(2*role,ax)])
                for k,ax in enumerate(axis):x=apply_pauli_rotation(x,vals[:,3+k],[(2*role+1,ax)])
                x=apply_pauli_rotation(x,self.intra[rr].expand(B*M),[(2*role,'Z'),(2*role+1,'Z')])
            for mu,(pa,pb) in enumerate(pairs):
                d=rr*3+mu
                for aa in range(R):
                    for bb in range(R):
                        if aa!=bb:x=apply_pauli_rotation(x,phi[:,aa,bb,d],[(2*aa,pa),(2*bb+1,pb)])
        return x.reshape(B,M,F)
    def _step(self,state,H,roles,edges):
        U=torch.matrix_exp((-1j*H).to(state.dtype));state=torch.bmm(U,state)
        return self._encode(state,roles,edges)
    def _readout(self,state,R):
        B=state.shape[0];vals=[];pq=('I','X','Y','Z')
        for p in pq:
            for qx in pq:
                if p=='I' and qx=='I':continue
                spec=[]
                if p!='I':spec.append((0,p))
                if qx!='I':spec.append((1,qx))
                vals.append(expect_pauli(state,spec).sum(1))
        for qa in (0,1):
            for qc in (0,1):
                for ax in ('X','Y','Z'):
                    tmp=0.
                    for nr in range(1,R):tmp=tmp+expect_pauli(state,[(qa,ax),(2*nr+qc,ax)]).sum(1)
                    vals.append(tmp/max(R-1,1))
        if R==3:
            for qa in (0,1):
                for qc in (0,1):
                    for qd in (0,1):
                        u=expect_pauli(state,[(qa,'Z'),(2+qc,'Z'),(4+qd,'Z')]).sum(1)
                        v=expect_pauli(state,[(qa,'Z'),(2+qd,'Z'),(4+qc,'Z')]).sum(1)
                        vals.append(.5*(u+v))
        else: vals += [state.real.new_zeros(B) for _ in range(8)]
        return torch.stack(vals,-1)
    @staticmethod
    def _desc_all(h,p,v,a,mask):
        B,S,N,_=h.shape
        relp=(p[:,:,None,:,:]-p[:,:,:,None,:])/30.;relv=(v[:,:,None,:,:]-v[:,:,:,None,:])/10.
        ha=h[:,:,None,:,:].expand(-1,-1,N,-1,-1);va=v[:,:,None,:,:].expand(-1,-1,N,-1,-1)/10.;aa=a[:,:,None,:,:].expand(-1,-1,N,-1,-1)/10.
        eye=torch.eye(N,device=h.device,dtype=h.dtype)[None,None,:,:,None].expand(B,S,-1,-1,-1)
        valid=mask[:,None,None,:,None].to(h.dtype).expand(-1,S,N,-1,-1)
        return torch.cat((ha,relp,relv,va,aa,eye,valid),-1)
    @staticmethod
    def _neighbors(N,device):
        return torch.stack([torch.tensor([j for j in range(N) if j!=r],device=device,dtype=torch.long) for r in range(N)],0)
    def _branch_static(self,risk,phys,mask,branch):
        B,S,N,_=risk.shape;neigh=self._neighbors(N,risk.device);K=N-1
        valid=torch.stack([mask[:,neigh[r]] for r in range(N)],1)
        wri=torch.stack([risk[:,:,r,neigh[r]] for r in range(N)],2)
        wnn=torch.stack([risk[:,:,neigh[r],:][...,neigh[r]] for r in range(N)],2)
        tau=torch.stack([phys['tau'][:,:,r,neigh[r]] for r in range(N)],2)
        A=torch.sqrt(wri[..., :,None].clamp_min(0)*wri[...,None,:].clamp_min(0))*wnn
        eye=torch.eye(K,device=risk.device,dtype=torch.bool)[None,None,None]
        A=A.masked_fill(eye,0.)*valid[:,None,:, :,None]*valid[:,None,:,None,:]
        if branch==1:
            roots=torch.arange(N,device=risk.device)[:,None].expand(N,K)
            ridx=torch.stack((roots,neigh),-1)
            return A,wri,valid,ridx
        pairs=torch.tensor(_ordered_pairs(K),device=risk.device,dtype=torch.long);x,y=pairs[:,0],pairs[:,1];M=len(pairs)
        same_y=(y[:,None]==y[None,:]).to(A.dtype);same_x=(x[:,None]==x[None,:]).to(A.dtype)
        Ax=A.index_select(-2,x).index_select(-1,x)*same_y;Ay=A.index_select(-2,y).index_select(-1,y)*same_x;Acfg=Ax+Ay
        cv=valid.index_select(-1,x)&valid.index_select(-1,y);Acfg=Acfg*cv[:,None,:,:,None]*cv[:,None,:,None,:]
        wrx=wri.index_select(-1,x);wry=wri.index_select(-1,y)
        temp=wnn.index_select(-2,x);yi=y.view(1,1,1,M,1).expand(B,S,N,M,1);wjk=torch.gather(temp,-1,yi).squeeze(-1)
        tx=tau.index_select(-1,x);ty=tau.index_select(-1,y);eta=wrx*wry*wjk*torch.exp(-(tx-ty).abs());pot=(wrx+wry+eta)/3.
        roots=torch.arange(N,device=risk.device)[:,None].expand(N,M);ridx=torch.stack((roots,neigh[:,x],neigh[:,y]),-1)
        return Acfg,pot,cv,ridx
    @staticmethod
    def _config_inputs(desc,edge,ridx):
        # desc [B,S,Rroot,Nagent,42], ridx [Rroot,M,Rrole]
        B,S,Nroot,_,_=desc.shape;M,R=ridx.shape[1],ridx.shape[2];roles=[];eds=[]
        for root in range(Nroot):
            ids=ridx[root];roles.append(desc[:,:,root][:,:,ids])
            rr=torch.zeros(B,S,M,R,R,12,device=edge.device,dtype=edge.dtype)
            for x in range(R):
                for y in range(R):
                    if x!=y:rr[:,:,:,x,y]=edge[:,:,ids[:,x],ids[:,y]]
            eds.append(rr)
        return torch.stack(roles,2),torch.stack(eds,2)
    def _branch_all(self,h,p,v,a,phys,edge,risk,mask,branch):
        B,S,N,_=h.shape;Ag,pot,cfg_valid,ridx=self._branch_static(risk,phys,mask,branch);M=cfg_valid.shape[-1];R=2 if branch==1 else 3;F=2**(2*R);BT=B*N
        desc=self._desc_all(h,p,v,a,mask);roles,reds=self._config_inputs(desc,edge,ridx)
        cv=cfg_valid.reshape(BT,M);count=cv.sum(-1).clamp_min(1).to(h.dtype)
        state=torch.zeros(BT,M,F,device=h.device,dtype=torch.complex64 if h.dtype==torch.float32 else torch.complex128);state[:,:,0]=cv.to(h.dtype)/count.sqrt()[:,None]
        outs=[]
        for s in range(4):
            A=Ag[:,s].reshape(BT,M,M);pp=pot[:,s].reshape(BT,M);rows=A.abs().sum(-1).amax(-1).clamp_min(1.)
            H=(.5*torch.tanh(self.hop[branch-1]))*(A/rows[:,None,None])+(.5*torch.tanh(self.potential[branch-1]))*torch.diag_embed(pp*cv)
            ro=roles[:,s].reshape(BT,M,R,42);re=reds[:,s].reshape(BT,M,R,R,12)
            state=checkpoint(self._step,state,H,ro,re,use_reentrant=False) if self.training and self.use_checkpoint else self._step(state,H,ro,re)
            outs.append(self._readout(state,R).reshape(B,N,35)*cfg_valid.any(-1)[...,None])
        return torch.stack(outs,2),cfg_valid.any(-1)
    def forward(self,history,mask,timestamps=None):
        history=torch.where(mask[:,None,:,None],history,0.);h,p,v,a,phys,edge,risk=self.base_features(history,mask,timestamps)
        z1,v1=self._branch_all(h,p,v,a,phys,edge,risk,mask,1);z2,v2=self._branch_all(h,p,v,a,phys,edge,risk,mask,2)
        z=torch.stack((z1,z2),2)*mask[:,:,None,None,None];vm=torch.stack((v1,v2),2)[:,:,:,None].expand(-1,-1,-1,4)&mask[:,:,None,None]
        return {'readout':z,'interaction_mask':vm,'own':h[:,-1]}

class RTCNCore(TRCFeatureBase):
    """Matched Rooted Temporal Configuration Network."""
    def __init__(self,width:int=64):
        super().__init__();self.width=int(width)
        self.in1=nn.Sequential(nn.Linear(108,width),nn.SiLU())
        self.in2=nn.Sequential(nn.Linear(198,width),nn.SiLU())
        self.q=nn.Linear(width,width,bias=False);self.k=nn.Linear(width,width,bias=False);self.vv=nn.Linear(width,width,bias=False)
        self.mlp=nn.Sequential(nn.Linear(2*width,2*width),nn.SiLU(),nn.Linear(2*width,width))
        self.norm=nn.LayerNorm(width);self.gamma=nn.Parameter(torch.tensor(1.))
        self.readout=nn.Linear(2*width,35)
    def _branch(self,h,p,v,a,phys,edge,risk,mask,root,branch):
        B,_,N,_=h.shape;neigh=_root_neighbors(N,root,h.device);nn=len(neigh)
        Ag,pot,cfg_valid,pairs=self.graph_parts(risk,phys,mask,root,neigh,branch)
        M=nn if branch==1 else nn*(nn-1);R=2 if branch==1 else 3
        hidden=h.new_zeros(B,M,self.width);outs=[]
        for s in range(4):
            desc=self.role_desc(h,p,v,a,mask,root)[:,s];rootd=desc[:,root]
            if branch==1:
                ridx=torch.stack((torch.full((M,),root,device=h.device,dtype=torch.long),neigh),1)
                roles=torch.stack((rootd[:,None].expand(-1,M,-1),desc[:,neigh]),2)
            else:
                ia,ib=pairs[:,0],pairs[:,1]
                ridx=torch.stack((torch.full((M,),root,device=h.device,dtype=torch.long),neigh[ia],neigh[ib]),1)
                roles=torch.stack((rootd[:,None].expand(-1,M,-1),desc[:,neigh[ia]],desc[:,neigh[ib]]),2)
            ef=[];es=edge[:,s]
            for x in range(R):
                for y in range(R):
                    if x!=y: ef.append(es[:,ridx[:,x],ridx[:,y]])
            x=torch.cat((roles.flatten(-2),torch.cat(ef,-1)),-1)
            inp=(self.in1(x) if branch==1 else self.in2(x))
            t=hidden+inp
            A=Ag[:,s];bias=A.clone();diag=1.+pot[:,s]
            ii=torch.arange(M,device=h.device);bias[:,ii,ii]=diag
            allowed=(bias>0)&cfg_valid[:,None,:]&cfg_valid[:,:,None]
            logits=torch.einsum('bmd,bnd->bmn',self.q(t),self.k(t))/math.sqrt(self.width)+self.gamma*torch.log(bias.clamp_min(1e-6))
            logits=logits.masked_fill(~allowed,-1e4)
            att=torch.softmax(logits,-1)*cfg_valid[:,:,None]
            msg=torch.einsum('bmn,bnd->bmd',att,self.vv(t))
            hidden=self.norm(t+self.mlp(torch.cat((t,msg),-1)))*cfg_valid[...,None]
            den=cfg_valid.sum(-1,keepdim=True).clamp_min(1).to(hidden.dtype)
            mean=hidden.sum(1)/den
            mx=hidden.masked_fill(~cfg_valid[...,None],-1e4).amax(1);mx=torch.where(cfg_valid.any(-1)[:,None],mx,torch.zeros_like(mx))
            outs.append(self.readout(torch.cat((mean,mx),-1)))
        return torch.stack(outs,1),cfg_valid.any(-1)
    def forward(self,history,mask,timestamps=None):
        history=torch.where(mask[:,None,:,None],history,0.)
        h,p,v,a,phys,edge,risk=self.base_features(history,mask,timestamps)
        zs=[];vms=[]
        for root in range(history.shape[2]):
            z1,v1=self._branch(h,p,v,a,phys,edge,risk,mask,root,1);z2,v2=self._branch(h,p,v,a,phys,edge,risk,mask,root,2)
            zs.append(torch.stack((z1,z2),1));vms.append(torch.stack((v1,v2),1)[:,:,None].expand(-1,-1,4))
        return {'readout':torch.stack(zs,1)*mask[:,:,None,None,None],'interaction_mask':torch.stack(vms,1)&mask[:,:,None,None],'own':h[:,-1]}
