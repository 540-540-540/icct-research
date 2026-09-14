"""Five-qubit QCGAT adaptation for permutation-equivariant vehicle prediction.

Source: Li et al., MLST 7 (2026) 015007, doi:10.1088/2632-2153/ae32dd.
Shared 6->3 angle encoder, independent Q/K/V U3 gates, attention postselection.
Eight invariant observable features replace the paper's index-sensitive density
flattening. This is an adaptation, not a reproduction of its classification head.
"""
import math
import torch
from torch import nn
import pennylane as qml


def encode(a, wire):
    qml.RY(a[..., 0], wires=wire)
    qml.RX(a[..., 1], wires=wire)
    qml.RY(a[..., 2], wires=wire)


def unitary(a, theta, wire):
    encode(a, wire)
    qml.U3(theta[0], theta[1], theta[2], wires=wire)


def one_qubit_state(a, theta):
    """Independent differentiable analytic reference to the gate sequence."""
    dtype = torch.complex128 if a.dtype == torch.float64 else torch.complex64
    x = torch.stack((torch.ones_like(a[..., 0]), torch.zeros_like(a[..., 0])), -1).to(dtype)
    for angle, axis in zip(a.unbind(-1), ('y', 'x', 'y')):
        c, s = torch.cos(angle/2), torch.sin(angle/2)
        v0, v1 = x.unbind(-1)
        x = (torch.stack((c*v0-s*v1, s*v0+c*v1), -1) if axis == 'y'
             else torch.stack((c*v0-1j*s*v1, -1j*s*v0+c*v1), -1))
    t, p, l = theta.unbind(-1)
    c, s = torch.cos(t/2), torch.sin(t/2)
    x0, x1 = x.unbind(-1)
    return torch.stack((c*x0-torch.exp(1j*l)*s*x1,
                        torch.exp(1j*p)*s*x0+torch.exp(1j*(p+l))*c*x1), -1)


def observables(amplitude, mask, probability):
    """Normalized index+value amplitudes -> 3 mixed + 4 coherent + success."""
    def bloch(v):
        a, b = v.unbind(-1)
        cross = a.conj()*b
        return torch.stack((2*cross.real, 2*cross.imag, a.abs().square()-b.abs().square()), -1)
    mixed = bloch(amplitude).sum(-2)
    coherent = amplitude.sum(-2)/mask.sum(-1).to(amplitude.real.dtype).clamp_min(1).sqrt()[..., None]
    intensity = coherent.abs().square().sum(-1, keepdim=True)
    return torch.cat((mixed, intensity, bloch(coherent), probability[..., None]), -1)


class QGATGraph(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(6, 3)
        self.theta = nn.Parameter(torch.empty(3, 3).uniform_(-.3, .3))
        self.readout = nn.Sequential(nn.Linear(8, 128), nn.LayerNorm(128))
        device = qml.device('default.qubit', wires=5, shots=None)

        @qml.qnode(device, interface='torch', diff_method='backprop')
        def circuit(angles, query, index_state, theta):
            qml.StatePrep(index_state, wires=[0, 1, 2])
            unitary(query, theta[0], 3)
            for i in range(8):
                controls = [(i >> j) & 1 for j in (2, 1, 0)]
                qml.ctrl(qml.adjoint(unitary), control=[0, 1, 2], control_values=controls)(angles[..., i, :], theta[1], 3)
                qml.ctrl(unitary, control=[0, 1, 2], control_values=controls)(angles[..., i, :], theta[2], 4)
            return qml.state()
        self.circuit = circuit

        @qml.qnode(qml.device('default.qubit', wires=1, shots=None), interface='torch', diff_method='backprop')
        def local_state(angles, theta):
            qml.StatePrep(torch.stack((torch.ones_like(angles[..., 0]), torch.zeros_like(angles[..., 0])), -1), wires=0)
            unitary(angles, theta, 0)
            return qml.state()
        self.local_state = local_state

    def features(self, standardized_state, track_exists, detected, reference=False):
        shape = standardized_state.shape
        if shape[-1] != 4 or not 1 <= shape[-2] <= 8:
            raise ValueError('Expected [...,1..8,4] states')
        if track_exists.shape != shape[:-1] or detected.shape != track_exists.shape:
            raise ValueError('Mask shape mismatch')
        m = track_exists.bool()
        z = torch.where(m[..., None], standardized_state, 0)
        if not torch.isfinite(z).all() or (detected.bool() & ~m).any():
            raise ValueError('Nonfinite valid state or detected without track')
        inp = torch.cat((z, m[..., None].to(z), detected[..., None].to(z)), -1)
        a = math.pi * torch.tanh(self.encoder(inp.to(self.encoder.weight.dtype)))
        n = shape[-2]
        a = a.reshape(-1, n, 3).to(torch.float64)
        m = m.reshape(-1, n)
        if n < 8:
            a = torch.nn.functional.pad(a, (0, 0, 0, 8-n))
            m = torch.nn.functional.pad(m, (0, 8-n))
        theta = self.theta.to(torch.float64)
        batch = len(a)
        # All-empty frames have a harmless index-0 state, then zero output.
        prep = m.to(a.dtype)
        prep = prep + (m.sum(-1) == 0)[..., None] * torch.nn.functional.one_hot(torch.tensor(0, device=a.device), 8)
        prep = prep/prep.sum(-1, keepdim=True).sqrt()
        query = a[:, :n].reshape(-1, 3)
        neighbors = a[:, None].expand(batch, n, 8, 3).reshape(-1, 8, 3)
        masks = m[:, None].expand(batch, n, 8).reshape(-1, 8)
        if reference != 'full':
            # Exact contraction of the controlled circuit; no approximation or
            # loss of index coherence. Training still uses differentiable PL gates.
            state_fn = one_qubit_state if reference is True else self.local_state
            q = state_fn(query, theta[0])
            k = state_fn(neighbors.reshape(-1, 3), theta[1]).reshape(-1, 8, 2)
            v = state_fn(neighbors.reshape(-1, 3), theta[2]).reshape(-1, 8, 2)
            gamma = (k.conj()*q[:, None]).sum(-1)
            unnormalized = gamma[..., None]*v*masks[..., None]
            unnormalized = unnormalized/masks.sum(-1).to(a.dtype).clamp_min(1).sqrt()[:, None, None]
        else:
            idx = prep[:, None].expand(batch, n, 8).reshape(-1, 8)
            state = self.circuit(neighbors, query, idx, theta)
            unnormalized = state.reshape(-1, 8, 2, 2)[:, :, 0, :] * masks[..., None]
        probability = unnormalized.abs().square().sum((-1, -2))
        usable = probability > 1e-12
        # Explicit zero fallback for impossible/numerically vanishing postselection.
        normalized = unnormalized / torch.where(usable, probability, 1).sqrt()[:, None, None]
        result = observables(normalized, masks, probability)
        result = torch.where(usable[:, None], result, 0).reshape(batch, n, 8)
        result = result * m[:, :n, None]
        return result.reshape(*shape[:-1], 8)

    def forward(self, state_hat, standardized_state, track_exists, detected):
        if state_hat.shape != standardized_state.shape:
            raise ValueError('Physical and standardized state shapes must match')
        f = self.features(standardized_state, track_exists, detected)
        return self.readout(f.to(self.readout[0].weight.dtype)) * track_exists[..., None]


class CandidatePredictor(nn.Module):
    def __init__(self, normalization=None):
        super().__init__()
        from prediction.temporal import TrajectoryPredictor
        self.graph = QGATGraph()
        self.temporal = TrajectoryPredictor(128)

    def forward(self, state_hat, standardized_state, track_exists, detected):
        values = (state_hat, standardized_state, track_exists, detected)
        return self.temporal(*values, self.graph(*values))
