from __future__ import annotations

from functools import lru_cache

import pennylane as qml
import torch

from .common import EMBED_PAIRS, EMBED_QUBITS, NODE_QUBITS

RAW_OBS_PER_NODE = 1 + EMBED_QUBITS + 2 * len(EMBED_PAIRS)


@lru_cache(maxsize=1)
def _basis(size: int, device: str):
    return torch.arange(size, device=torch.device(device))


def _pauli_apply(state: torch.Tensor, operators: dict[int, str]) -> torch.Tensor:
    """Apply a Pauli string to a batched statevector without materializing a matrix."""
    size = state.shape[-1]
    n_qubits = size.bit_length() - 1
    if size != 1 << n_qubits:
        raise ValueError("State dimension must be a power of two")
    indices = torch.arange(size, device=state.device)
    flip = 0
    phase = torch.ones(size, dtype=state.dtype, device=state.device)
    for wire, op in operators.items():
        bitmask = 1 << (n_qubits - 1 - int(wire))
        sign = 1 - 2 * ((indices & bitmask) != 0).to(state.real.dtype)
        if op in ("X", "Y"):
            flip ^= bitmask
        if op == "Z":
            phase = phase * sign
        elif op == "Y":
            phase = phase * (-1j * sign)
        elif op != "X":
            raise ValueError(f"Unsupported Pauli {op}")
    return state[..., indices ^ flip] * phase


class _Expectation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state: torch.Tensor, operators: dict[int, str]):
        ctx.save_for_backward(state)
        ctx.operators = operators
        return (state.conj() * _pauli_apply(state, operators)).sum(-1).real

    @staticmethod
    def backward(ctx, grad):
        (state,) = ctx.saved_tensors
        applied = _pauli_apply(state, ctx.operators)
        return 2 * grad.unsqueeze(-1) * applied, None


def pauli_expectation(state: torch.Tensor, operators: dict[int, str]) -> torch.Tensor:
    return _Expectation.apply(state, operators)


def conditional_embedding_rdm_features(
    state: torch.Tensor,
    active_mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """37 measurable features/vehicle: occupancy + conditional embedding 1-RDM.

    For vehicle i, P_i=(I-Z_i)/2.  The returned terms are:
      <P_i>,
      <P_i n_e>/<P_i> for 6 embedding modes,
      <P_i (XX+YY)/2>/<P_i> and
      <P_i (YX-XY)/2>/<P_i> for all 15 embedding-mode pairs.

    The statevector is only an exact-simulator optimization.  These features are
    linear combinations of Pauli expectations and therefore have an equivalent
    qml.expval definition (see formal_observables).
    """
    if state.ndim != 2:
        raise ValueError("Expected batched state [B, 2**Q]")
    if active_mask.shape != (state.shape[0], NODE_QUBITS):
        raise ValueError("Expected active mask [B,8]")

    one = state.real.new_ones(state.shape[0])
    z_node = torch.stack(
        [pauli_expectation(state, {i: "Z"}) for i in range(NODE_QUBITS)], -1
    )
    z_embed = torch.stack(
        [
            pauli_expectation(state, {NODE_QUBITS + e: "Z"})
            for e in range(EMBED_QUBITS)
        ],
        -1,
    )
    occupancy = (one[:, None] - z_node) * 0.5
    denom = occupancy.clamp_min(eps)

    rows = []
    for i in range(NODE_QUBITS):
        values = [occupancy[:, i]]
        for e in range(EMBED_QUBITS):
            ze = z_embed[:, e]
            zz = pauli_expectation(state, {i: "Z", NODE_QUBITS + e: "Z"})
            numerator = (one - z_node[:, i] - ze + zz) * 0.25
            values.append(numerator / denom[:, i])

        for p, q in EMBED_PAIRS:
            wp, wq = NODE_QUBITS + p, NODE_QUBITS + q
            xx = pauli_expectation(state, {wp: "X", wq: "X"})
            yy = pauli_expectation(state, {wp: "Y", wq: "Y"})
            yx = pauli_expectation(state, {wp: "Y", wq: "X"})
            xy = pauli_expectation(state, {wp: "X", wq: "Y"})
            zxx = pauli_expectation(state, {i: "Z", wp: "X", wq: "X"})
            zyy = pauli_expectation(state, {i: "Z", wp: "Y", wq: "Y"})
            zyx = pauli_expectation(state, {i: "Z", wp: "Y", wq: "X"})
            zxy = pauli_expectation(state, {i: "Z", wp: "X", wq: "Y"})

            real_num = (xx + yy - zxx - zyy) * 0.25
            imag_num = (yx - xy - zyx + zxy) * 0.25
            values.extend((real_num / denom[:, i], imag_num / denom[:, i]))

        rows.append(torch.stack(values, -1))

    result = torch.stack(rows, 1)
    return result * active_mask[..., None].to(result.dtype)


def _hamiltonian(coeffs, ops):
    return qml.Hamiltonian(coeffs, ops)


def formal_observables():
    """Hermitian observables whose expectations produce the unnormalized 37D raw set.

    Classical normalization by <P_i> is applied after measurement.  This path is
    used for numerical validation; efficient training evaluates the identical
    expectations from qml.state() without exposing amplitudes as model features.
    """
    observables = []
    identity = qml.Identity(0)
    for i in range(NODE_QUBITS):
        zi = qml.PauliZ(i)
        p_i = _hamiltonian([0.5, -0.5], [identity, zi])
        observables.append(p_i)

        for e in range(EMBED_QUBITS):
            ze = qml.PauliZ(NODE_QUBITS + e)
            observables.append(
                _hamiltonian(
                    [0.25, -0.25, -0.25, 0.25],
                    [identity, zi, ze, zi @ ze],
                )
            )

        for p, q in EMBED_PAIRS:
            wp, wq = NODE_QUBITS + p, NODE_QUBITS + q
            xx = qml.PauliX(wp) @ qml.PauliX(wq)
            yy = qml.PauliY(wp) @ qml.PauliY(wq)
            yx = qml.PauliY(wp) @ qml.PauliX(wq)
            xy = qml.PauliX(wp) @ qml.PauliY(wq)
            observables.append(
                _hamiltonian(
                    [0.25, 0.25, -0.25, -0.25],
                    [xx, yy, zi @ xx, zi @ yy],
                )
            )
            observables.append(
                _hamiltonian(
                    [0.25, -0.25, -0.25, 0.25],
                    [yx, xy, zi @ yx, zi @ xy],
                )
            )
    return tuple(observables)


def normalize_formal_raw(raw: torch.Tensor, active_mask: torch.Tensor, eps: float = 1e-6):
    """Normalize qml.expval raw observables into the same 37D feature convention."""
    if raw.shape[-2:] != (NODE_QUBITS, RAW_OBS_PER_NODE):
        raise ValueError("Unexpected formal raw shape")
    occupancy = raw[..., :1].clamp_min(eps)
    normalized = torch.cat((raw[..., :1], raw[..., 1:] / occupancy), -1)
    return normalized * active_mask[..., None].to(device=normalized.device, dtype=normalized.dtype)
