"""A02 directed Q24 graph, exact differentiable pure-state simulation (§6).

Qubits are ordered P0,V0,P1,V1,..., most significant first. Edge arrays
use [source, target]. No density matrices, shot noise, or learned slot IDs.
"""
import math
from functools import lru_cache

import torch
from torch import nn


@lru_cache(maxsize=64)
def _basis(size, device):
    return torch.arange(size, device=device)


def pauli_apply(state, operators):
    """Apply a Pauli string {qubit: 'X'/'Y'/'Z'} to [..., 2**n]."""
    size = state.shape[-1]
    n = size.bit_length() - 1
    if size != 1 << n:
        raise ValueError('State dimension must be a power of two')
    indices = _basis(size, state.device)
    flip = 0
    phase = torch.ones(size, dtype=state.dtype, device=state.device)
    for bit, op in operators.items():
        if not 0 <= bit < n or op not in ('X', 'Y', 'Z'):
            raise ValueError('Invalid Pauli operator')
        bitmask = 1 << (n - 1 - bit)
        sign = 1 - 2 * ((indices & bitmask) != 0).to(state.real.dtype)
        if op in ('X', 'Y'):
            flip ^= bitmask
        if op == 'Z':
            phase = phase * sign
        elif op == 'Y':
            # Output-index form: Y|psi>[0] = -i psi[1].
            phase = phase * (-1j * sign)
    return state[..., indices ^ flip] * phase


class _Expectation(torch.autograd.Function):
    """Recompute P|psi> in backward; do not retain one state per readout."""
    @staticmethod
    def forward(ctx, state, operators):
        ctx.save_for_backward(state)
        ctx.operators = operators
        return (state.conj() * pauli_apply(state, operators)).sum(-1).real

    @staticmethod
    def backward(ctx, grad):
        state, = ctx.saved_tensors
        return 2 * grad.unsqueeze(-1) * pauli_apply(state, ctx.operators), None


def expectation(state, operators):
    return _Expectation.apply(state, operators)


def pauli_rotation(state, operators, angle):
    angle = torch.as_tensor(angle, dtype=state.real.dtype, device=state.device)
    return (torch.cos(angle / 2).unsqueeze(-1) * state
            - 1j * torch.sin(angle / 2).unsqueeze(-1) * pauli_apply(state, operators))


def edge_features(state_hat):
    """Physical relative state, kernel and smooth approach; source,target axes."""
    relative = state_hat.unsqueeze(-2) - state_hat.unsqueeze(-3)
    r, u = relative[..., :2], relative[..., 2:]
    distance2 = r.square().sum(-1)
    approach = torch.tanh(-(r * u).sum(-1) / torch.sqrt(distance2 + 1) / 15)
    edge = torch.cat((r / 45, u / 15, approach.unsqueeze(-1)), -1)
    return edge, torch.exp(-distance2 / (2 * 45**2)), approach


class QuantumGraph(nn.Module):
    """48-parameter L3 Q24; legacy=True gives the nested 45-parameter Q21.

    State/mask accept arbitrary leading batch dimensions; the final dimensions
    are [N,4]/[N], 1<=N<=8. Masked state entries are ignored, including NaNs.
    The model uses float64/complex128; move with .to(device), not .float().
    """
    def __init__(self, scales, depth=3, legacy=False):
        super().__init__()
        if depth not in (1, 2, 3):
            raise ValueError('Depth 1 is validation-only; supported depths: 1,2,3')
        scales = torch.as_tensor(scales, dtype=torch.float64)
        if scales.shape != (4,) or not torch.isfinite(scales).all() or (scales < 1).any():
            raise ValueError('scales must be four finite training Q95 scales >= 1')
        self.register_buffer('scales', scales.clone())
        self.depth, self.legacy = depth, legacy
        self.output_dim = 21 if legacy else 24
        theta = torch.zeros(depth, 15 if legacy else 16, dtype=torch.float64)
        theta[:, :4] = math.log(0.2 / 0.8)
        self.theta = nn.Parameter(theta)

    def physical_parameters(self):
        k = 4 if self.legacy else 5
        return dict(gains=.25 + 3.75 * self.theta[:, :4].sigmoid(),
                    weights=self.theta[:, 4:4+k], bias=self.theta[:, 4+k],
                    gamma=math.pi / 2 * self.theta[:, 5+k].sigmoid(),
                    mixing=self.theta[:, 6+k:11+k])

    def load_legacy(self, old):
        """Exact T13 embedding of an old Q21 model, including nonzero parameters."""
        if self.legacy or not old.legacy or old.depth != self.depth:
            raise ValueError('Expected new Q24 and same-depth old Q21')
        with torch.no_grad():
            self.scales.copy_(old.scales)
            self.theta[:, :8].copy_(old.theta[:, :8])
            self.theta[:, 8].zero_()
            self.theta[:, 9:].copy_(old.theta[:, 8:])
        return self

    @staticmethod
    def _moments(state, mask):
        n = mask.shape[-1]
        single = torch.stack([expectation(state, {2*i+reg: p})
                              for i in range(n) for reg in range(2)
                              for p in ('X', 'Y', 'Z')], -1).reshape(*mask.shape, 6)
        zero = single[..., 0, 0] * 0
        pairs, connected = [], []
        for j in range(n):
            for i in range(n):
                raw, conn = [], []
                for p, q, pi, qi in [('Z','X',2,3), ('Z','Y',2,4),
                                      ('Z','Z',2,5), ('X','Z',0,5)]:
                    value = expectation(state, {2*j:p, 2*i+1:q}) if i != j else zero
                    raw.append(value)
                    conn.append((value-single[...,j,pi]*single[...,i,qi])
                                * (mask[...,j] & mask[...,i]) if i != j else zero)
                pairs.append(torch.stack(raw, -1))
                connected.append(torch.stack(conn, -1))
        shape = (*mask.shape[:-1], n, n, 4)
        local = torch.stack([expectation(state, {2*i:p,2*i+1:p})
                             for i in range(n) for p in ('X','Y')], -1)
        local = local.reshape(*mask.shape, 2)
        local_c = local - single[..., :2] * single[..., 3:5]
        return dict(singles=single, pair_raw=torch.stack(pairs,-2).reshape(shape),
                    pair_connected=torch.stack(connected,-2).reshape(shape),
                    local_raw=local, local_connected=local_c * mask.unsqueeze(-1))

    def forward(self, state_hat, track_exists, interaction_scale=1.,
                return_details=False, angles_override=None):
        if self.theta.dtype != torch.float64:
            raise ValueError('Q reference implementation requires float64 parameters')
        x = torch.as_tensor(state_hat, dtype=torch.float64, device=self.theta.device)
        mask = torch.as_tensor(track_exists, dtype=torch.bool, device=x.device)
        if x.ndim < 2 or x.shape[-1] != 4 or mask.shape != x.shape[:-1]:
            raise ValueError('Expected [...,N,4] states and [...,N] mask')
        n = x.shape[-2]
        if not 1 <= n <= 8:
            raise ValueError('N must be between 1 and 8')
        x = torch.where(mask.unsqueeze(-1), x, 0.)
        if not torch.isfinite(x).all():
            raise ValueError('Active states must be finite')
        edge, kernel, approach = edge_features(x)
        pair_mask = (mask.unsqueeze(-1) & mask.unsqueeze(-2)
                     & ~torch.eye(n,dtype=torch.bool,device=x.device))
        par = self.physical_parameters()
        rotation = .5 * torch.atan(x.unsqueeze(-3) * par['gains'][:,None,:] / self.scales)
        center = torch.zeros(self.depth,dtype=x.dtype,device=x.device)
        center[0] = math.pi / 2
        rotation = (rotation + center[:,None,None]) * mask[...,None,:,None]
        if angles_override is None:
            ef = edge[...,:4] if self.legacy else edge
            logits = torch.einsum('...jik,lk->...lji', ef, par['weights'])
            angles = (logits + par['bias'][:,None,None]).tanh()
            angles = angles * par['gamma'][:,None,None] * kernel.unsqueeze(-3) / 7
        else:
            angles = torch.as_tensor(angles_override,dtype=x.dtype,device=x.device)
            angles = torch.broadcast_to(angles, (*x.shape[:-2],self.depth,n,n))
        angles = angles * pair_mask.unsqueeze(-3) * interaction_scale
        state = torch.zeros((*x.shape[:-2], 1 << (2*n)),dtype=torch.complex128,device=x.device)
        state[...,0] = 1
        encoded, layers = [], []
        for l in range(self.depth):
            for i in range(n):
                for reg in range(2):
                    state = pauli_rotation(state,{2*i+reg:'Y'},rotation[...,l,i,2*reg])
                    state = pauli_rotation(state,{2*i+reg:'Z'},rotation[...,l,i,2*reg+1])
            if return_details:
                encoded.append(state)
            for j in range(n):
                for i in range(n):
                    if i != j:
                        state = pauli_rotation(state,{2*j:'Z',2*i+1:'X'},angles[...,l,j,i])
            if l == 0:
                shallow_state = state
            alpha_p,beta_p,alpha_v,beta_v,eta = par['mixing'][l]
            for i in range(n):
                state = pauli_rotation(state,{2*i:'X',2*i+1:'X'},eta*mask[...,i])
                for reg,alpha,beta in [(0,alpha_p,beta_p),(1,alpha_v,beta_v)]:
                    state = pauli_rotation(state,{2*i+reg:'Y'},beta*mask[...,i])
                    state = pauli_rotation(state,{2*i+reg:'Z'},alpha*mask[...,i])
            if return_details:
                layers.append(state)
        shallow, deep = self._moments(shallow_state,mask), self._moments(state,mask)
        sc = shallow['pair_connected'][...,1:]
        q = torch.cat((shallow['singles'],sc.sum(-3)/7,deep['singles'],
                       deep['local_connected'],deep['pair_connected'].sum(-3)/7), -1)
        if not self.legacy:
            q = torch.cat((q,(sc*approach.unsqueeze(-1)).sum(-3)/7),-1)
        q = q * mask.unsqueeze(-1)
        if not return_details:
            return q
        return dict(readout=q,shallow_state=shallow_state,deep_state=state,
                    layer_states=layers,encoded_states=encoded,angles=angles,
                    approach=approach,edge_features=edge,kernel=kernel,
                    encoding_angles=rotation,shallow=shallow,deep=deep)

    def forward_history(self, state_hat, track_exists, interaction_scale=1.,
                        checkpoint=True):
        """One compact register per frame; checkpoint avoids retaining T circuits."""
        from torch.utils.checkpoint import checkpoint as recompute
        x = torch.as_tensor(state_hat,dtype=torch.float64,device=self.theta.device)
        mask = torch.as_tensor(track_exists,dtype=torch.bool,device=x.device)
        if x.ndim != 4 or mask.shape != x.shape[:-1] or x.shape[-1] != 4:
            raise ValueError('History requires [B,T,N,4] and [B,T,N]')
        outputs = []
        for frame, valid in zip(x.flatten(0,1),mask.flatten(0,1)):
            indices = valid.nonzero(as_tuple=True)[0]
            # where removes inactive NaNs while maintaining a zero input gradient.
            blank = torch.where(valid[:,None],frame,0.).sum(-1,keepdim=True)*0
            blank = blank.expand(-1,self.output_dim)
            if indices.numel():
                compact = frame.index_select(0,indices)
                # Bind mask in the closure; backward recomputation occurs later.
                def evaluate(value):
                    return self(value,torch.ones(value.shape[-2],dtype=torch.bool,
                                                 device=value.device),interaction_scale)
                q = (recompute(evaluate,compact,use_reentrant=False)
                     if checkpoint and torch.is_grad_enabled() else evaluate(compact))
                blank = blank.index_copy(0,indices,q)
            outputs.append(blank)
        return torch.stack(outputs).reshape(*x.shape[:-1],self.output_dim)
