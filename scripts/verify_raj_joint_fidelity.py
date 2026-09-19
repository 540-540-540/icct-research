"""Deterministic unit checks for the isolated Raj joint-register mixer."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from prediction.qgnn_paper_native.raj_joint import (
    JointRegisterMixer,
    conditional_embedding_states,
    fixed_weight_index,
    fixed_weight_subsets,
)


def normalized_factor(batch: int, n: int, D: int, j: int, k: int) -> torch.Tensor:
    torch.manual_seed(41)
    S = len(fixed_weight_subsets(n, j))
    E = len(fixed_weight_subsets(D, k))
    real = torch.randn(batch, S, E, dtype=torch.float64)
    imag = torch.randn(batch, S, E, dtype=torch.float64)
    x = torch.complex(real, imag)
    return x / x.flatten(1).norm(dim=1)[:, None, None]


def graph_weights() -> torch.Tensor:
    w = torch.zeros(1, 4, 4, dtype=torch.float64)
    vals = {(0, 1): .91, (0, 2): .72, (0, 3): .53,
            (1, 2): .84, (1, 3): .31, (2, 3): .65}
    for (a, b), v in vals.items():
        w[0, a, b] = w[0, b, a] = v
    return w


def permute_factor(x: torch.Tensor, pi: list[int], n: int, j: int) -> torch.Tensor:
    subs = fixed_weight_subsets(n, j)
    lookup = {s: i for i, s in enumerate(subs)}
    out = torch.zeros_like(x)
    for old_i, s in enumerate(subs):
        new_s = tuple(sorted(pi[q] for q in s))
        out[:, lookup[new_s]] = x[:, old_i]
    return out


def permute_graph(w: torch.Tensor, mask: torch.Tensor, pi: list[int]):
    wp = torch.zeros_like(w)
    mp = torch.zeros_like(mask)
    for old_i, new_i in enumerate(pi):
        mp[:, new_i] = mask[:, old_i]
        for old_j, new_j in enumerate(pi):
            wp[:, new_i, new_j] = w[:, old_i, old_j]
    return wp, mp


def permute_joint(x: torch.Tensor, pi: list[int], n: int, D: int, weight: int) -> torch.Tensor:
    basis = fixed_weight_subsets(n + D, weight)
    lookup = fixed_weight_index(n + D, weight)
    out = torch.zeros_like(x)
    for old_i, s in enumerate(basis):
        mapped = tuple(sorted(pi[q] if q < n else q for q in s))
        out[:, lookup[mapped]] = x[:, old_i]
    return out


def main():
    n, D, j, k = 4, 3, 2, 1
    mask = torch.ones(1, n, dtype=torch.bool)
    w = graph_weights()
    x = normalized_factor(1, n, D, j, k)

    mixer = JointRegisterMixer(n=n, D=D, j=j, k=k, init_std=0.0).double()
    assert mixer.joint_dim == 35
    assert mixer._factor_joint.numel() == 18
    assert mixer._factor_joint.unique().numel() == 18

    # Zero-angle M is the identity embedding of H_n^j x H_D^k into H_{n+D}^{j+k}.
    with torch.no_grad():
        mixer.theta_node.zero_()
        mixer.theta_emb.zero_()
        mixer.theta_cross.zero_()
    identity = mixer(x, w, mask)
    embedded = mixer.factor_to_joint(x)
    identity_err = float((identity - embedded).abs().max())
    assert identity_err < 1e-12

    cond, prob = conditional_embedding_states(mixer, identity)
    expected_prob = x.abs().square().sum(-1)
    probability_err = float((prob - expected_prob).abs().max())
    assert probability_err < 1e-12
    live = expected_prob > 1e-12
    phase_free_cond_err = float(
        (cond[live] - x[live] / expected_prob[live, None].sqrt()).abs().max()
    )
    assert phase_free_cond_err < 1e-12

    # Non-zero mixer is norm preserving on the full fixed-weight shell.
    torch.manual_seed(7)
    with torch.no_grad():
        mixer.theta_node.copy_(torch.randn_like(mixer.theta_node) * .17)
        mixer.theta_emb.copy_(torch.randn_like(mixer.theta_emb) * .13)
        mixer.theta_cross.copy_(torch.randn_like(mixer.theta_cross) * .11)
    mixed = mixer(x, w, mask)
    norm_err = float((mixed.norm(dim=-1) - x.flatten(1).norm(dim=-1)).abs().max())
    assert norm_err < 1e-10

    # Cross-register gates must actually move amplitude outside the factor sector.
    projected = mixer.project_factor_sector(mixed)
    sector_mass = float(projected.abs().square().sum())
    full_mass = float(mixed.abs().square().sum())
    cross_leak_mass = full_mass - sector_mass
    assert cross_leak_mass > 1e-8

    # The mixer must be differentiable with respect to its trainable angles.
    mixer.zero_grad(set_to_none=True)
    probe = torch.linspace(-1.0, 1.0, mixer.joint_dim, dtype=torch.float64)
    loss = (mixed.abs().square() * probe).sum()
    loss.backward()
    grad_norms = {
        "node": float(mixer.theta_node.grad.norm()),
        "embedding": float(mixer.theta_emb.grad.norm()),
        "cross": float(mixer.theta_cross.grad.norm()),
    }
    assert all(torch.isfinite(torch.tensor(v)) and v > 1e-10 for v in grad_norms.values())

    # Exact node-permutation equivariance for unique canonical graph keys.
    pi = [2, 0, 3, 1]  # old node -> relabelled node
    xp = permute_factor(x, pi, n, j)
    wp, mp = permute_graph(w, mask, pi)
    mixed_p = mixer(xp, wp, mp)
    expected_p = permute_joint(mixed, pi, n, D, j + k)
    equivariance_err = float((mixed_p - expected_p).abs().max())
    assert equivariance_err < 1e-10

    # Production-size dimensional sanity check (N=8,D=6,j=3,k=3).
    production = JointRegisterMixer(n=8, D=6, j=3, k=3)
    assert production.joint_dim == 3003
    assert production._factor_joint.shape == (56, 20)

    payload = {
        "status": "PASS",
        "interpretation": "single final M; all three Supplementary-Note-1 Givens blocks",
        "small_case": {"n": n, "D": D, "j": j, "k": k, "joint_dim": mixer.joint_dim},
        "identity_max_abs_error": identity_err,
        "conditional_probability_max_abs_error": probability_err,
        "conditional_state_max_abs_error": phase_free_cond_err,
        "norm_max_abs_error": norm_err,
        "cross_register_leak_mass": cross_leak_mass,
        "gradient_norms": grad_norms,
        "permutation_equivariance_max_abs_error": equivariance_err,
        "production_case": {
            "n": 8, "D": 6, "j": 3, "k": 3,
            "factor_shape": [56, 20],
            "factor_dim": 1120,
            "joint_dim": production.joint_dim,
        },
        "test_set_used": False,
    }
    out = Path("reports/qgnn/round4_raj_joint_mixer_unit_20260919.json")
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
