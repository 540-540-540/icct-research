"""Isolated Raj paper-math faithful SinD adaptation.

This prototype restores the structural blocks V-A-W-M-readout while keeping the
ICCT task interface. It is intentionally separate from the accepted main model.
It is not an official-code reproduction and not a gate-for-gate transcription
of the paper's hierarchical loader.
"""
from __future__ import annotations

import itertools
import math
from functools import lru_cache

import torch
from torch import nn

from prediction.qgnn_final.common import BoundedCore, physical_graph
from .raj_subset import SubsetFeatureBuilder
from .raj_paper import CompoundEvolution, rdm_operators_cpu
from .raj_joint import JointRegisterMixer, givens_shell_pairs, oriented_pair_tables, conditional_embedding_states


class ControlledFeatureLoader(nn.Module):
    """Reduced-basis functional analogue of subset-controlled V(x).

    Initial load: one globally normalised factored state [subset, embedding].
    Re-upload: one orthogonal Householder map per subset, acting on embedding only.
    """

    def __init__(self, feat_dim: int, embed_dim: int, hidden: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.net = nn.Sequential(nn.Linear(feat_dim, hidden), nn.SiLU(), nn.Linear(hidden, embed_dim))

    def amplitudes(self, feat: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = torch.nn.functional.softplus(self.net(feat)) * valid[..., None]
        flat_norm = raw.flatten(1).norm(dim=1, keepdim=True).clamp_min(1e-12)
        state = raw / flat_norm[:, None]
        row_norm = raw.norm(dim=-1, keepdim=True)
        cond = raw / row_norm.clamp_min(1e-12)
        cond = torch.where(valid[..., None], cond, torch.zeros_like(cond))
        return state, cond

    def reference(self, cond: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        ref = torch.zeros_like(cond)
        ref[..., 0] = valid.to(cond.dtype)
        return ref

    def reupload(self, x: torch.Tensor, cond: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Apply H_s(cond_s) independently to every subset row."""
        ref = self.reference(cond, valid).to(x.dtype)
        target = cond.to(x.dtype)
        u = ref - target
        den = (u.conj() * u).real.sum(-1, keepdim=True)
        inner = (u.conj() * x).sum(-1, keepdim=True)
        reflected = x - 2.0 * u * inner / den.clamp_min(1e-12)
        same = den <= 1e-12
        out = torch.where(same, x, reflected)
        return out * valid[..., None]

    def forward(self, feat: torch.Tensor, valid: torch.Tensor):
        state, cond = self.amplitudes(feat, valid)
        return state, cond


class EquivariantSubsetAdjacency(nn.Module):
    """Graph-conditioned node-register Givens message passing at fixed j."""

    def __init__(self, n: int, j: int, edge_dim: int = 16, hidden: int = 32):
        super().__init__()
        self.n, self.j = n, j
        self.angle = nn.Sequential(nn.Linear(edge_dim, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        li, ri = oriented_pair_tables(n, j)
        self.register_buffer("_pair_left", li)
        self.register_buffer("_pair_right", ri)

    @staticmethod
    def _node_key(weights: torch.Tensor, node: int) -> tuple[float, ...]:
        row = weights[node]
        vals = sorted((float(v) for i, v in enumerate(row) if i != node), reverse=True)
        return (float(row.abs().sum()), float((row * row).sum()), *vals)

    def _ordered_pairs(self, weights: torch.Tensor, mask: torch.Tensor) -> list[tuple[int, int]]:
        nodes = [i for i in range(self.n) if bool(mask[i])]
        keys = {i: self._node_key(weights, i) for i in nodes}
        pairs = []
        for a, b in itertools.combinations(nodes, 2):
            if keys[a] == keys[b]:
                continue
            u, v = (a, b) if keys[a] > keys[b] else (b, a)
            pairs.append((float(weights[a, b]), keys[u], keys[v], u, v))
        pairs.sort(key=lambda z: (-z[0], tuple(-x for x in z[1]), tuple(-x for x in z[2])))
        return [(u, v) for _, _, _, u, v in pairs]

    def _orders_cpu(self, risk: torch.Tensor, mask: torch.Tensor):
        rc = risk.detach().cpu(); mc = mask.detach().cpu()
        return [self._ordered_pairs(rc[b], mc[b]) for b in range(len(rc))]

    def _apply_pair(self, x: torch.Tensor, a: int, b: int, theta: torch.Tensor) -> torch.Tensor:
        ia, ib = givens_shell_pairs(self.n, self.j, a, b)
        ia, ib = ia.to(x.device), ib.to(x.device)
        xa, xb = x[ia], x[ib]
        c = torch.cos(theta / 2).to(x.real.dtype)
        s = torch.sin(theta / 2).to(x.real.dtype)
        y = x.clone(); y[ia] = c * xa - s * xb; y[ib] = s * xa + c * xb
        return y

    def _apply_pair_batch(self, x, a, b, theta, enabled):
        safe_a = torch.where(enabled, a, torch.zeros_like(a))
        safe_b = torch.where(enabled, b, torch.ones_like(b))
        ia = self._pair_left[safe_a, safe_b]
        ib = self._pair_right[safe_a, safe_b]
        E = x.shape[-1]
        iae = ia[..., None].expand(-1, -1, E)
        ibe = ib[..., None].expand(-1, -1, E)
        xa = torch.gather(x, 1, iae); xb = torch.gather(x, 1, ibe)
        th = torch.where(enabled, theta, torch.zeros_like(theta))
        c = torch.cos(th / 2).to(x.real.dtype)[:, None, None]
        s = torch.sin(th / 2).to(x.real.dtype)[:, None, None]
        y = x.scatter(1, iae, c * xa - s * xb)
        y = y.scatter(1, ibe, s * xa + c * xb)
        return y

    def _forward_reference(self, x, edge, risk, mask):
        rc, mc = risk.detach().cpu(), mask.detach().cpu()
        rows = []
        for bi in range(x.shape[0]):
            z = x[bi]
            for a, b in self._ordered_pairs(rc[bi], mc[bi]):
                theta = torch.tanh(self.angle(edge[bi, a, b]).squeeze(-1)) * (math.pi / 2)
                z = self._apply_pair(z, a, b, theta)
            rows.append(z)
        return torch.stack(rows, 0)

    def forward(self, x: torch.Tensor, edge: torch.Tensor, risk: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        orders = self._orders_cpu(risk, mask)
        B = x.shape[0]; device = x.device
        z = x
        max_pairs = max((len(o) for o in orders), default=0)
        batch = torch.arange(B, device=device)
        for rank in range(max_pairs):
            enabled = torch.tensor([rank < len(o) for o in orders], device=device, dtype=torch.bool)
            a = torch.tensor([o[rank][0] if rank < len(o) else 0 for o in orders], device=device)
            b = torch.tensor([o[rank][1] if rank < len(o) else 1 for o in orders], device=device)
            ef = edge[batch, a, b]
            theta = torch.tanh(self.angle(ef).squeeze(-1)) * (math.pi / 2)
            z = self._apply_pair_batch(z, a, b, theta, enabled)
        return z


class ConditionalRDMReadout(nn.Module):
    def __init__(self, n: int, j: int, D: int = 6, k: int = 3, out: int = 64):
        super().__init__()
        self.n, self.j, self.D, self.k = n, j, D, k
        self.net = nn.Sequential(nn.Linear(2 * D * D, 128), nn.SiLU(), nn.Linear(128, out), nn.LayerNorm(out))
        subs = list(itertools.combinations(range(n), j))
        inc = torch.zeros(len(subs), n)
        for si, s in enumerate(subs):
            inc[si, list(s)] = 1.0
        self.register_buffer("incidence", inc)

    def subset_rdm(self, conditional: torch.Tensor) -> torch.Tensor:
        op = rdm_operators_cpu(self.D, self.k).to(conditional.device, conditional.dtype)
        return torch.einsum("bse,pqef,bsf->bspq", conditional.conj(), op, conditional)

    def forward(self, conditional: torch.Tensor, prob: torch.Tensor, valid: torch.Tensor, mask: torch.Tensor):
        gamma = self.subset_rdm(conditional)
        raw = torch.cat((gamma.real, gamma.imag), dim=-1).flatten(-2)
        subset_z = self.net(raw) * valid[..., None]
        inc = self.incidence.to(subset_z.device, subset_z.dtype)
        w = prob.to(subset_z.dtype) * valid.to(subset_z.dtype)
        den = torch.einsum("bs,sn->bn", w, inc).clamp_min(1e-12)
        pooled = torch.einsum("bsh,bs,sn->bnh", subset_z, w, inc) / den[..., None]
        has = (torch.einsum("bs,sn->bn", valid.to(subset_z.dtype), inc) > 0).to(subset_z.dtype)
        return pooled * has[..., None] * mask[..., None], gamma


class RajPaperMathQGNNCore(BoundedCore):
    """Single-j=3 paper-math faithful SinD adaptation for diagnosis only."""

    def __init__(self, j: int = 3, rounds: int = 3, D: int = 6, k: int = 3, n: int = 8):
        super().__init__()
        if n != 8:
            raise ValueError("BoundedCore patch size is fixed at N=8 for this prototype")
        self.n, self.j, self.rounds, self.D, self.k = n, j, rounds, D, k
        self.builder = SubsetFeatureBuilder(32)
        self.embed_dim = math.comb(D, k)
        self.loaders = nn.ModuleList(ControlledFeatureLoader(self.builder.out_dim, self.embed_dim) for _ in range(rounds + 1))
        self.adj = nn.ModuleList(EquivariantSubsetAdjacency(n, j) for _ in range(rounds))
        self.evolution = nn.ModuleList(CompoundEvolution(D, k) for _ in range(rounds))
        self.joint = JointRegisterMixer(n=n, D=D, j=j, k=k)
        self.readout = ConditionalRDMReadout(n, j, D, k, 64)

    def forward_patch(self, history: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        own, feat, valid, subs = self.builder(history, mask, self.j)
        if not bool(valid.any()):
            return history.new_zeros(history.shape[0], history.shape[2], 64)
        edge, risk = physical_graph(history, mask)
        x0, _ = self.loaders[0](feat, valid)
        ctype = torch.complex64 if x0.dtype == torch.float32 else torch.complex128
        x = x0.to(ctype)
        for l in range(self.rounds):
            x = self.adj[l](x, edge, risk, mask)
            x = self.evolution[l](x)
            _, cond = self.loaders[l + 1](feat, valid)
            x = self.loaders[l + 1].reupload(x, cond, valid)
        joint = self.joint(x, risk, mask)
        conditional, prob = conditional_embedding_states(self.joint, joint)
        z, _ = self.readout(conditional, prob, valid, mask)
        return z
