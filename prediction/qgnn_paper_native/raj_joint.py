"""Paper-math joint-register primitives for Raj-style QGNN fidelity experiments.

This module is intentionally isolated from the accepted ICCT main model.
It implements the main-text/Supplementary-Note-1 interpretation of M(theta_M):
node-pair, embedding-pair, and cross-register Givens blocks acting in the
fixed-total-particle-number shell. It does NOT implement the hierarchical
controlled-Givens loader V(x); callers must not label a model using this module
as a verbatim full-paper reproduction unless that loader is also implemented.
"""
from __future__ import annotations

import itertools
from functools import lru_cache
from math import comb

import torch
from torch import nn


@lru_cache(maxsize=64)
def fixed_weight_subsets(m: int, weight: int) -> tuple[tuple[int, ...], ...]:
    return tuple(itertools.combinations(range(m), weight))


@lru_cache(maxsize=64)
def fixed_weight_index(m: int, weight: int) -> dict[tuple[int, ...], int]:
    return {s: i for i, s in enumerate(fixed_weight_subsets(m, weight))}


@lru_cache(maxsize=64)
def factored_to_joint_indices(n: int, D: int, j: int, k: int) -> torch.Tensor:
    """Index map [C(n,j), C(D,k)] -> C(n+D,j+k) joint basis."""
    node_subs = fixed_weight_subsets(n, j)
    emb_subs = fixed_weight_subsets(D, k)
    lookup = fixed_weight_index(n + D, j + k)
    out = torch.empty(len(node_subs), len(emb_subs), dtype=torch.long)
    for si, s in enumerate(node_subs):
        for ei, e in enumerate(emb_subs):
            joint = tuple(sorted((*s, *(n + q for q in e))))
            out[si, ei] = lookup[joint]
    return out


@lru_cache(maxsize=512)
def givens_shell_pairs(m: int, weight: int, a: int, b: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Paired basis indices mixed by a Givens rotation on modes (a,b)."""
    if not (0 <= a < m and 0 <= b < m and a != b):
        raise ValueError((m, weight, a, b))
    # Keep the caller-provided orientation. RBS/Givens rotations are oriented;
    # replacing it by numeric qubit order would make the map label-dependent.
    basis = fixed_weight_subsets(m, weight)
    lookup = fixed_weight_index(m, weight)
    left, right = [], []
    for idx, s in enumerate(basis):
        st = set(s)
        if a in st and b not in st:
            partner = tuple(sorted((st - {a}) | {b}))
            left.append(idx)
            right.append(lookup[partner])
    return torch.tensor(left, dtype=torch.long), torch.tensor(right, dtype=torch.long)



class JointRegisterMixer(nn.Module):
    """Raj Supplementary-Note-1 joint mixer on the total-weight j+k shell.

    Interpretation frozen for this prototype:
    - one final M after L V-A-W rounds (main-text composition);
    - all three blocks from Supplementary Note 1;
    - trainable angles are indexed by canonical rank, not node identity.
    """

    def __init__(self, n: int = 8, D: int = 6, j: int = 3, k: int = 3, init_std: float = 0.03):
        super().__init__()
        self.n, self.D, self.j, self.k = n, D, j, k
        self.m = n + D
        self.total_weight = j + k
        self.joint_dim = comb(self.m, self.total_weight)
        self.node_pairs = tuple(itertools.combinations(range(n), 2))
        self.emb_pairs = tuple(itertools.combinations(range(D), 2))
        self.theta_node = nn.Parameter(torch.randn(len(self.node_pairs)) * init_std)
        self.theta_emb = nn.Parameter(torch.randn(len(self.emb_pairs)) * init_std)
        self.theta_cross = nn.Parameter(torch.randn(n, D) * init_std)
        self.register_buffer("_factor_joint", factored_to_joint_indices(n, D, j, k))

    def factor_to_joint(self, factor_state: torch.Tensor) -> torch.Tensor:
        expected = (len(fixed_weight_subsets(self.n, self.j)), len(fixed_weight_subsets(self.D, self.k)))
        if factor_state.shape[-2:] != expected:
            raise ValueError(f"expected factor state [...,{expected[0]},{expected[1]}], got {tuple(factor_state.shape)}")
        flat = factor_state.reshape(-1, expected[0], expected[1])
        out = flat.new_zeros(flat.shape[0], self.joint_dim)
        idx = self._factor_joint.to(flat.device).reshape(-1)
        out[:, idx] = flat.reshape(flat.shape[0], -1)
        return out.reshape(*factor_state.shape[:-2], self.joint_dim)

    def project_factor_sector(self, joint_state: torch.Tensor) -> torch.Tensor:
        idx = self._factor_joint.to(joint_state.device)
        return joint_state[..., idx]

    def _apply_pair(self, x: torch.Tensor, a: int, b: int, theta: torch.Tensor) -> torch.Tensor:
        ia, ib = givens_shell_pairs(self.m, self.total_weight, a, b)
        ia, ib = ia.to(x.device), ib.to(x.device)
        va, vb = x[..., ia], x[..., ib]
        c = torch.cos(theta / 2).to(x.real.dtype)
        s = torch.sin(theta / 2).to(x.real.dtype)
        y = x.clone()
        y[..., ia] = c * va - s * vb
        y[..., ib] = s * va + c * vb
        return y

    def _canonical_pair_order(self, weights: torch.Tensor, active: torch.Tensor) -> list[tuple[int, int]]:
        # A Givens gate is oriented, so endpoint orientation must also be
        # label-invariant. Use the same graph-derived node rank as the cross block.
        node_order = self._canonical_node_order(weights, active)
        rank = {node: r for r, node in enumerate(node_order)}
        rows = []
        for a, b in self.node_pairs:
            if bool(active[a]) and bool(active[b]):
                u, v = (a, b) if rank[a] < rank[b] else (b, a)
                rows.append((float(weights[a, b].detach().cpu()), rank[u], rank[v], u, v))
        rows.sort(key=lambda z: (-z[0], z[1], z[2]))
        return [(u, v) for _, _, _, u, v in rows]

    def _canonical_node_order(self, weights: torch.Tensor, active: torch.Tensor) -> list[int]:
        degree = weights.abs().sum(-1)
        nodes = [i for i in range(self.n) if bool(active[i])]
        return sorted(nodes, key=lambda i: (-float(degree[i].detach().cpu()), i))

    def forward(self, factor_state: torch.Tensor, weights: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """factor_state [B,C(n,j),C(D,k)], weights [B,n,n], mask [B,n].

        Returns the full joint-shell state [B,C(n+D,j+k)].
        """
        if factor_state.ndim != 3:
            raise ValueError("factor_state must be [B,S,E]")
        if weights.shape != (factor_state.shape[0], self.n, self.n):
            raise ValueError("weights must be [B,n,n]")
        if mask.shape != (factor_state.shape[0], self.n) or mask.dtype != torch.bool:
            raise ValueError("mask must be bool [B,n]")

        joint = self.factor_to_joint(factor_state)
        rows = []
        for bi in range(joint.shape[0]):
            x = joint[bi]
            active = mask[bi]
            pair_order = self._canonical_pair_order(weights[bi], active)
            for rank, (a, b) in enumerate(pair_order):
                x = self._apply_pair(x, a, b, self.theta_node[rank])
            for rank, (ea, eb) in enumerate(self.emb_pairs):
                x = self._apply_pair(x, self.n + ea, self.n + eb, self.theta_emb[rank])
            node_order = self._canonical_node_order(weights[bi], active)
            for rank, node in enumerate(node_order):
                for eb in range(self.D):
                    x = self._apply_pair(x, node, self.n + eb, self.theta_cross[rank, eb])
            rows.append(x)
        return torch.stack(rows, 0)


def conditional_embedding_states(
    mixer: JointRegisterMixer,
    joint_state: torch.Tensor,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project onto every j-subset and normalise the conditional k-particle state.

    Returns:
      conditional [B,C(n,j),C(D,k)]
      projection_prob [B,C(n,j)]
    """
    factor = mixer.project_factor_sector(joint_state)
    prob = factor.abs().square().sum(-1)
    conditional = factor / prob.clamp_min(eps).sqrt()[..., None]
    conditional = torch.where((prob > eps)[..., None], conditional, torch.zeros_like(conditional))
    return conditional, prob
