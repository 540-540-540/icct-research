from __future__ import annotations

import itertools
from functools import lru_cache
from math import comb

import torch

NODE_QUBITS = 8
EMBED_QUBITS = 6
EMBED_WEIGHT = 3
TOTAL_QUBITS = NODE_QUBITS + EMBED_QUBITS
NODE_PAIRS = tuple(itertools.combinations(range(NODE_QUBITS), 2))
EMBED_PAIRS = tuple(itertools.combinations(range(EMBED_QUBITS), 2))
EMBED_RING = tuple((i, (i + 1) % EMBED_QUBITS) for i in range(EMBED_QUBITS))
OBS_PER_NODE = 1 + EMBED_QUBITS + 4 * len(EMBED_PAIRS)

@lru_cache(maxsize=8)
def node_subsets(j: int):
    return tuple(itertools.combinations(range(NODE_QUBITS), j))

@lru_cache(maxsize=4)
def embed_subsets():
    return tuple(itertools.combinations(range(EMBED_QUBITS), EMBED_WEIGHT))

def _basis_index(occupied):
    idx = 0
    for wire in occupied:
        idx |= 1 << (TOTAL_QUBITS - 1 - wire)
    return idx

@lru_cache(maxsize=8)
def joint_basis_indices(j: int):
    rows = []
    for s in node_subsets(j):
        for e in embed_subsets():
            rows.append((s, e, _basis_index((*s, *(NODE_QUBITS + q for q in e)))))
    return tuple(rows)

@lru_cache(maxsize=8)
def subset_incidence(j: int):
    subs = node_subsets(j)
    out = torch.zeros(len(subs), NODE_QUBITS, dtype=torch.float64)
    for si, s in enumerate(subs):
        out[si, list(s)] = 1.0
    return out

@lru_cache(maxsize=1)
def pair_lookup():
    table = torch.full((NODE_QUBITS, NODE_QUBITS), -1, dtype=torch.long)
    for k, (a, b) in enumerate(NODE_PAIRS):
        table[a, b] = table[b, a] = k
    return table

def uniform_joint_state(mask: torch.Tensor, j: int) -> torch.Tensor:
    if mask.ndim != 1 or mask.numel() != NODE_QUBITS:
        raise ValueError(f"uniform_joint_state expects [{NODE_QUBITS}] mask")
    active = tuple(bool(v) for v in mask.detach().cpu().tolist())
    valid = [(s, e, idx) for s, e, idx in joint_basis_indices(j) if all(active[i] for i in s)]
    state = torch.zeros(1 << TOTAL_QUBITS, dtype=torch.complex128, device=mask.device)
    if not valid:
        state[0] = 1.0
        return state
    amp = (len(valid) ** -0.5)
    state[torch.tensor([row[2] for row in valid], device=mask.device)] = amp
    return state
def pad_vehicle_register(history: torch.Tensor, mask: torch.Tensor):
    """Pad a <=8 vehicle scene to the fixed 8-qubit register."""
    if history.ndim != 4 or mask.ndim != 2:
        raise ValueError("Expected history [B,T,N,4] and mask [B,N]")
    n = history.shape[2]
    if n > NODE_QUBITS:
        raise ValueError(f"Vehicle register accepts at most {NODE_QUBITS} slots")
    if n == NODE_QUBITS:
        return history, mask
    extra_history = history.new_zeros(
        history.shape[0], history.shape[1], NODE_QUBITS - n, history.shape[3]
    )
    extra_mask = torch.zeros(
        mask.shape[0], NODE_QUBITS - n, dtype=torch.bool, device=mask.device
    )
    return torch.cat((history, extra_history), 2), torch.cat((mask, extra_mask), 1)

def pool_subset_features(feat: torch.Tensor, valid: torch.Tensor, j: int):
    inc = subset_incidence(j).to(feat.device, feat.dtype)
    weight = valid.to(feat.dtype)
    den_node = torch.einsum("bs,sn->bn", weight, inc).clamp_min(1.0)
    node = torch.einsum("bsf,bs,sn->bnf", feat, weight, inc) / den_node[..., None]
    den_scene = weight.sum(1, keepdim=True).clamp_min(1.0)
    scene = (feat * weight[..., None]).sum(1) / den_scene
    return node, scene

def directed_pair_features(edge: torch.Tensor):
    rows = []
    for a, b in NODE_PAIRS:
        rows.append(torch.cat((edge[:, a, b], edge[:, b, a]), -1))
    return torch.stack(rows, 1)

def pair_valid_mask(mask: torch.Tensor):
    return torch.stack([mask[:, a] & mask[:, b] for a, b in NODE_PAIRS], 1)

def branch_agent_mask(mask: torch.Tensor, j: int):
    enough = mask.sum(1, keepdim=True) >= j
    return mask & enough

def parameter_summary(module):
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in module.parameters() if not p.requires_grad)
    return {"trainable": trainable, "frozen": frozen, "total": trainable + frozen}
