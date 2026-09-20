"""Torch-only primitive reference for the PennyLane Raj core.

This module is deliberately not the formal interaction model. It is used only
for gate-convention, ordering, and primitive equivalence checks.
"""
from __future__ import annotations

from collections.abc import Iterable

import torch


def _indices_for_weight_transfer(n_qubits: int, a: int, b: int, device):
    if n_qubits < 1 or not (0 <= a < n_qubits and 0 <= b < n_qubits) or a == b:
        raise ValueError((n_qubits, a, b))
    indices = torch.arange(1 << n_qubits, device=device, dtype=torch.long)
    bit_a = 1 << (n_qubits - 1 - a)
    bit_b = 1 << (n_qubits - 1 - b)
    left = indices[((indices & bit_a) != 0) & ((indices & bit_b) == 0)]
    right = left ^ bit_a ^ bit_b
    return left, right


def single_excitation(state: torch.Tensor, a: int, b: int, angle: torch.Tensor):
    """Apply the PennyLane SingleExcitation convention to a full statevector."""
    if state.shape[-1] <= 0 or state.shape[-1] & (state.shape[-1] - 1):
        raise ValueError("state must end in a power-of-two dimension")
    n_qubits = state.shape[-1].bit_length() - 1
    left, right = _indices_for_weight_transfer(
        n_qubits, a, b, state.device
    )
    shape = (1,) * (state.ndim - 1) + (left.numel(),)
    left_index = left.reshape(shape).expand(*state.shape[:-1], -1)
    right_index = right.reshape(shape).expand(*state.shape[:-1], -1)
    va = torch.gather(state, -1, left_index)
    vb = torch.gather(state, -1, right_index)
    angle = torch.as_tensor(angle, dtype=state.real.dtype, device=state.device)
    c = torch.cos(angle / 2).unsqueeze(-1)
    s = torch.sin(angle / 2).unsqueeze(-1)
    updated = state.scatter(-1, left_index, c * va + s * vb)
    return updated.scatter(-1, right_index, -s * va + c * vb)


def ising_zz(state: torch.Tensor, a: int, b: int, angle: torch.Tensor):
    """Apply PennyLane IsingZZ(theta) = exp(-i theta Z_a Z_b / 2)."""
    if state.shape[-1] <= 0 or state.shape[-1] & (state.shape[-1] - 1):
        raise ValueError("state must end in a power-of-two dimension")
    n_qubits = state.shape[-1].bit_length() - 1
    if not (0 <= a < n_qubits and 0 <= b < n_qubits) or a == b:
        raise ValueError((n_qubits, a, b))
    indices = torch.arange(1 << n_qubits, device=state.device, dtype=torch.long)
    bit_a = 1 << (n_qubits - 1 - a)
    bit_b = 1 << (n_qubits - 1 - b)
    z_a = 1 - 2 * ((indices & bit_a) != 0).to(state.real.dtype)
    z_b = 1 - 2 * ((indices & bit_b) != 0).to(state.real.dtype)
    angle = torch.as_tensor(angle, dtype=state.real.dtype, device=state.device)
    phase = torch.exp(-0.5j * angle.unsqueeze(-1) * z_a * z_b)
    return state * phase


def apply_sequence(
    state: torch.Tensor,
    single_ops: Iterable[tuple[int, int, torch.Tensor]],
    zz_ops: Iterable[tuple[int, int, torch.Tensor]],
):
    """Apply an explicit reference sequence; no matrix exponential is used."""
    for a, b, angle in single_ops:
        state = single_excitation(state, a, b, angle)
    for a, b, angle in zz_ops:
        state = ising_zz(state, a, b, angle)
    return state
