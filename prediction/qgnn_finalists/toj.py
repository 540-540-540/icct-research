from __future__ import annotations
import math
from functools import lru_cache
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .common import PrefixHistoryEncoder,block_statistics,pair_physics,to_edge_features,to_patch_indices,local_twoqubit_unitary,apply_adjacent_twoqubit_unitary,apply_hadamards,expect_pauli,init_product_plus

def _atanh(x): return .5*math.log((1+x)/(1-x))

@lru_cache(maxsize=16)
def _ztable(m:int,device_str:str,dtype_str:str):
    device=torch.device(device_str);dtype=torch.float64 if dtype_str=='torch.float64' else torch.float32
    q=2*m;idx=torch.arange(2**q,device=device)
    za=[];zb=[]
    for j in range(m):
        za.append((1-2*((idx>>(q-1-2*j))&1)).to(dtype))
        zb.append((1-2*((idx>>(q-2-2*j))&1)).to(dtype))
    return torch.stack(za,-1),torch.stack(zb,-1)

class TOFeatureBase(nn.Module):
    def __init__(self,cap:int=6):
        super().__init__();self.cap=int(cap);self.encoder=PrefixHistoryEncoder('last',32)
    def base_features(self,history,mask,timestamps):
        h=self.encoder(history,mask);p,v,a,_=block_statistics(history,timestamps,mask)
        phys=pair_physics(p,v,a,mask);even,odd,risk=to_edge_features(phys)
        slots,valid=to_patch_indices(risk,phys,mask,self.cap)
        return h,p,v,a,phys,even,odd,risk,slots,valid
    def patch(self,h,p,v,a,even,odd,risk,slots,valid):
        B,S,N,D=h.shape;M=slots.shape[-1];T=N
        batch=torch.arange(B,device=h.device)[:,None,None]
        # [B,T,S,M,*]
        hp=h.permute(0,2,1,3)[batch,slots].permute(0,1,3,2,4)
        pp=p.permute(0,2,1,3)[batch,slots].permute(0,1,3,2,4)
        vp=v.permute(0,2,1,3)[batch,slots].permute(0,1,3,2,4)
        rootp=pp[:,:,:,0:1];rootv=vp[:,:,:,0:1]
        speed=vp.square().sum(-1,keepdim=True).sqrt()/10.
        isroot=torch.zeros(B,T,S,M,1,device=h.device,dtype=h.dtype);isroot[:,:,:,0]=1
        node=torch.cat((hp,(pp-rootp)/30.,(vp-rootv)/10.,speed,isroot),-1)
        # pair gather via explicit M<=6 loops keeps indexing clear and differentiable wrt features.
        ep=torch.zeros(B,T,S,M,M,6,device=h.device,dtype=h.dtype)
        op=torch.zeros_like(ep);rp=torch.zeros(B,T,S,M,M,device=h.device,dtype=h.dtype)
        for x in range(M):
            ix=slots[:,:,x]
            for y in range(M):
                iy=slots[:,:,y]
                # Explicit B loop is small (formal B<=32) and avoids ambiguous advanced indexing.
                for b in range(B):
                    ep[b,:,:,x,y]=even[b,:,slots[b,:,x],slots[b,:,y]].permute(1,0,2)
                    op[b,:,:,x,y]=odd[b,:,slots[b,:,x],slots[b,:,y]].permute(1,0,2)
                    rp[b,:,:,x,y]=risk[b,:,slots[b,:,x],slots[b,:,y]].permute(1,0)
        pairvalid=valid[:,:,:,None]&valid[:,:,None,:]
        eye=torch.eye(M,device=h.device,dtype=torch.bool)[None,None]
        pairvalid=pairvalid&~eye
        ep=ep*pairvalid[:,:,None,:,:,None];op=op*pairvalid[:,:,None,:,:,None];rp=rp*pairvalid[:,:,None]
        return node,ep,op,rp

class TOJQGNNCore(TOFeatureBase):
    """Time-Ordered Joint-register QGNN with fused commuting Hamiltonian families."""
    def __init__(self,cap:int=6,use_checkpoint:bool=True):
        super().__init__(cap);self.use_checkpoint=bool(use_checkpoint)
        self.node_gain=nn.Parameter(torch.full((38,),_atanh(.25)));self.node_bias=nn.Parameter(torch.zeros(38))
        self.even_gain=nn.Parameter(torch.full((6,),_atanh(.25)));self.even_bias=nn.Parameter(torch.zeros(6))
        self.odd_gain=nn.Parameter(torch.full((6,),_atanh(.25)));
        self.eta=nn.Parameter(torch.full((4,),_atanh(.25)))
        self.register_buffer('idx_plus',torch.tensor([0,1,2,3,4,5,1,3],dtype=torch.long))
        self.register_buffer('idx_minus',torch.tensor([0,1,2,3,4,5,2,3],dtype=torch.long))
    def _local_bank(self,state,node,valid,bank):
        BT,M,_=node.shape
        padded=torch.cat((node,torch.zeros(BT,M,2,device=node.device,dtype=node.dtype)),-1)
        gain=torch.cat((torch.tanh(self.node_gain),torch.zeros(2,device=node.device,dtype=node.dtype)))
        bias=torch.cat((self.node_bias,torch.zeros(2,device=node.device,dtype=node.dtype)))
        rawg=gain[bank*10:(bank+1)*10];bb=bias[bank*10:(bank+1)*10]
        ang=(math.pi/2)*torch.tanh(padded[...,bank*10:(bank+1)*10]*rawg+bb)
        if bank==3: ang=ang.clone();ang[...,-2:]=0
        ang=ang*valid[...,None]
        eta=.2*torch.tanh(self.eta[bank]);U=local_twoqubit_unitary(ang,eta.expand(BT,M)*valid)
        for j in range(M): state=apply_adjacent_twoqubit_unitary(state,U[:,j],j)
        return state
    def _phase_family(self,state,J,za,zb,family):
        M=J.shape[-1]
        if family=='ZZ':
            phase=.5*torch.einsum('sm,bmn,sn->bs',za,J,za)
            return state*torch.exp(-.5j*phase)
        if family=='XX':
            qs=[2*j for j in range(M)];state=apply_hadamards(state,qs)
            phase=.5*torch.einsum('sm,bmn,sn->bs',za,J,za);state=state*torch.exp(-.5j*phase)
            return apply_hadamards(state,qs)
        if family=='ZX':
            qs=[2*j+1 for j in range(M)];state=apply_hadamards(state,qs)
            phase=torch.einsum('sm,bmn,sn->bs',za,J,zb);state=state*torch.exp(-.5j*phase)
            return apply_hadamards(state,qs)
        if family=='XZ':
            qs=[2*j for j in range(M)];state=apply_hadamards(state,qs)
            phase=torch.einsum('sm,bmn,sn->bs',za,J,zb);state=state*torch.exp(-.5j*phase)
            return apply_hadamards(state,qs)
        raise ValueError(family)
    def _bank(self,state,node,even,odd,risk,valid,bank):
        BT,M,_=node.shape;za,zb=_ztable(M,str(state.device),str(node.dtype));za=za.to(node.dtype);zb=zb.to(node.dtype)
        state=self._local_bank(state,node,valid,bank)
        gp=torch.tanh(self.even_gain);bp=self.even_bias;gm=torch.tanh(self.odd_gain)
        i0=int(self.idx_plus[2*bank]);i1=int(self.idx_plus[2*bank+1]);j0=int(self.idx_minus[2*bank]);j1=int(self.idx_minus[2*bank+1])
        beta0=(math.pi/2)*torch.tanh(gp[i0]*even[...,i0]+bp[i0]);beta1=(math.pi/2)*torch.tanh(gp[i1]*even[...,i1]+bp[i1])
        gamma0=(math.pi/2)*torch.tanh(gm[j0]*odd[...,j0]);gamma1=(math.pi/2)*torch.tanh(gm[j1]*odd[...,j1])
        pv=valid[:,:,None]&valid[:,None,:];eye=torch.eye(M,device=node.device,dtype=torch.bool)[None];pv=pv&~eye
        kappa=(risk/5.)*pv
        Jz=kappa*beta0;Jx=kappa*beta1;Jd1=kappa*gamma0;Jd2=kappa*gamma1
        state=self._phase_family(state,Jz,za,zb,'ZZ');state=self._phase_family(state,Jx,za,zb,'XX')
        state=self._phase_family(state,Jd1,za,zb,'ZX');state=self._phase_family(state,Jd2,za,zb,'XZ')
        return state
    def _stage(self,state,node,even,odd,risk,valid):
        for bank in range(4): state=self._bank(state,node,even,odd,risk,valid,bank)
        return state
    def _readout(self,state,risk,valid):
        BT,M=valid.shape;vals=[];pq=('I','X','Y','Z')
        for p in pq:
            for q in pq:
                if p=='I' and q=='I':continue
                spec=[]
                if p!='I':spec.append((0,p))
                if q!='I':spec.append((1,q))
                vals.append(expect_pauli(state,spec))
        root_single={}
        for qu,ax in [(0,'X'),(0,'Y'),(0,'Z'),(1,'X'),(1,'Y'),(1,'Z')]:root_single[(qu,ax)]=expect_pauli(state,[(qu,ax)])
        corr=[];weights=[]
        for j in range(1,M):
            nv=valid[:,j].to(state.real.dtype);w=risk[:,0,j]*nv;weights.append(w)
            row=[]
            for qu,ax in [(0,'X'),(0,'Y'),(0,'Z'),(1,'X'),(1,'Y'),(1,'Z')]:
                ns=expect_pauli(state,[(2*j+qu,ax)]);pair=expect_pauli(state,[(qu,ax),(2*j+qu,ax)])
                row.append((pair-root_single[(qu,ax)]*ns)*nv)
            ns=expect_pauli(state,[(2*j+1,'X')]);pair=expect_pauli(state,[(0,'Z'),(2*j+1,'X')]);row.append((pair-root_single[(0,'Z')]*ns)*nv)
            ns=expect_pauli(state,[(2*j+1,'Z')]);pair=expect_pauli(state,[(0,'X'),(2*j+1,'Z')]);row.append((pair-root_single[(0,'X')]*ns)*nv)
            corr.append(torch.stack(row,-1))
        if corr:
            C=torch.stack(corr,1);W=torch.stack(weights,1);den=valid[:,1:].sum(-1).clamp_min(1).to(C.dtype)
            mean=C.sum(1)/den[:,None];wmean=(C*W[...,None]).sum(1)/W.sum(1).clamp_min(1e-6)[:,None]
            has=valid[:,1:].any(-1)[:,None];mean=mean*has;wmean=wmean*has
        else: mean=state.real.new_zeros(BT,8);wmean=mean
        return torch.cat((torch.stack(vals,-1),mean,wmean),-1)
    def forward(self,history,mask,timestamps=None):
        history=torch.where(mask[:,None,:,None],history,0.)
        h,p,v,a,phys,even,odd,risk,slots,valid=self.base_features(history,mask,timestamps)
        node,ep,op,rp=self.patch(h,p,v,a,even,odd,risk,slots,valid)
        B,N,S,M,_=node.shape;BT=B*N
        node=node.reshape(BT,S,M,38);ep=ep.reshape(BT,S,M,M,6);op=op.reshape(BT,S,M,M,6);rp=rp.reshape(BT,S,M,M);va=valid.reshape(BT,M)
        state=init_product_plus(va).to(history.dtype if False else torch.complex64)
        outs=[]
        for s in range(4):
            if self.training and self.use_checkpoint:
                state=checkpoint(self._stage,state,node[:,s],ep[:,s],op[:,s],rp[:,s],va,use_reentrant=False)
            else: state=self._stage(state,node[:,s],ep[:,s],op[:,s],rp[:,s],va)
            outs.append(self._readout(state,rp[:,s],va))
        z=torch.stack(outs,1).reshape(B,N,4,31)*mask[:,:,None,None]
        return {'readout':z,'interaction_mask':mask[:,:,None].expand(-1,-1,4),'own':h[:,-1]}

class TRTGNCore(TOFeatureBase):
    """Matched Temporal Rooted Tensor GNN on the same M<=6 sensed-history context."""
    def __init__(self,cap:int=6,width:int=32,rank:int=16):
        super().__init__(cap);self.width=width;self.rank=rank
        self.node_in=nn.Linear(38,width);self.pair_in=nn.Linear(88,width)
        self.node_norm=nn.LayerNorm(width);self.pair_norm=nn.LayerNorm(width)
        self.A=nn.Linear(width,rank);self.B=nn.Linear(width,rank);self.tensor_bias_a=nn.Parameter(torch.zeros(rank));self.tensor_bias_b=nn.Parameter(torch.zeros(rank))
        self.pair_update=nn.Sequential(nn.Linear(124,64),nn.SiLU(),nn.Linear(64,width))
        self.C=nn.Linear(width,width);self.gru=nn.GRUCell(width,width)
        self.root_out=nn.Linear(width,15);self.pair_out=nn.Linear(width,8)
    def _round(self,u,pair,e,risk,valid):
        BT,M,_,W=pair.shape;L=self.A(pair)+self.tensor_bias_a;R=self.B(pair)+self.tensor_bias_b
        tens=pair.new_zeros(BT,M,M,self.rank);den=pair.new_zeros(BT,M,M)
        for l in range(M):
            w=(risk[:,:,l][:,:,None]*risk[:,l,:][:,None,:])
            lv=L[:,:,l][:,:,None,:];rv=R[:,l,:][:,None,:,:]
            ok=(valid[:,:,None]&valid[:,None,:]&valid[:,l,None,None]).to(pair.dtype)
            # exclude mediator equal to either endpoint
            jj=torch.arange(M,device=pair.device);ok[:,jj,l]=0;ok[:,l,jj]=0
            ww=w*ok;tens=tens+ww[...,None]*(lv*rv);den=den+ww
        tens=tens/den.clamp_min(1e-6)[...,None]
        uj=u[:,:,None,:].expand(-1,-1,M,-1);uk=u[:,None,:,:].expand(-1,M,-1,-1)
        inp=torch.cat((pair,uj,uk,e,tens),-1)
        pv=valid[:,:,None]&valid[:,None,:]&~torch.eye(M,device=pair.device,dtype=torch.bool)[None]
        pair=self.pair_norm(pair+self.pair_update(inp))*pv[...,None]
        msg=(self.C(pair)*risk[...,None]).sum(2)/risk.sum(2).clamp_min(1e-6)[...,None]
        u=self.gru(msg.reshape(BT*M,W),u.reshape(BT*M,W)).reshape(BT,M,W)*valid[...,None]
        return u,pair
    def _readout(self,u,pair,risk,valid):
        root=self.root_out(u[:,0]);pc=self.pair_out(pair[:,0,1:]);vm=valid[:,1:]
        den=vm.sum(-1).clamp_min(1).to(pc.dtype);mean=(pc*vm[...,None]).sum(1)/den[:,None]
        w=risk[:,0,1:]*vm;wmean=(pc*w[...,None]).sum(1)/w.sum(-1).clamp_min(1e-6)[:,None]
        has=vm.any(-1)[:,None];return torch.cat((root,mean*has,wmean*has),-1)
    def forward(self,history,mask,timestamps=None):
        history=torch.where(mask[:,None,:,None],history,0.)
        h,p,v,a,phys,even,odd,risk,slots,valid=self.base_features(history,mask,timestamps)
        node,ep,op,rp=self.patch(h,p,v,a,even,odd,risk,slots,valid)
        e=torch.cat((ep,op),-1);B,N,S,M,_=node.shape;BT=B*N;va=valid.reshape(BT,M)
        node=node.reshape(BT,S,M,38);e=e.reshape(BT,S,M,M,12);rp=rp.reshape(BT,S,M,M)
        u=node.new_zeros(BT,M,self.width);pair=node.new_zeros(BT,M,M,self.width);outs=[]
        for s in range(4):
            ni=self.node_in(node[:,s]);ej=node[:,s,:,None,:].expand(-1,-1,M,-1);ek=node[:,s,None,:,:].expand(-1,M,-1,-1)
            pin=self.pair_in(torch.cat((ej,ek,e[:,s]),-1))
            u=self.node_norm(u+ni)*va[...,None]
            pv=va[:,:,None]&va[:,None,:]&~torch.eye(M,device=node.device,dtype=torch.bool)[None]
            pair=self.pair_norm(pair+pin)*pv[...,None]
            for _ in range(4):u,pair=self._round(u,pair,e[:,s],rp[:,s],va)
            outs.append(self._readout(u,pair,rp[:,s],va))
        z=torch.stack(outs,1).reshape(B,N,4,31)*mask[:,:,None,None]
        return {'readout':z,'interaction_mask':mask[:,:,None].expand(-1,-1,4),'own':h[:,-1]}
