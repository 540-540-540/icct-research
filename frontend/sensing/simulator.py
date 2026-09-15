"""Shared multi-target echo simulator (A domain only).

Implements the frozen V1 model in reports/sensing_design/sens_design_02/SYSTEM_MODEL.md
sections 2-3: one shared observation per BS and frame, all target echoes summed before a
single receiver noise realization. Truth states never leave this module.
"""
from __future__ import annotations

import hashlib
import math
from typing import Sequence

import torch

from .waveform import ArrayConfig, PaperWaveform


def derive_seed(text: str) -> int:
    digest = hashlib.sha256(text.encode()).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def _generator(seed: int, device) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def _phases(episode: int, frame: int, bs: int, source_keys: Sequence[int], device, dtype) -> torch.Tensor:
    values = []
    for key in source_keys:
        seed = derive_seed(f"2026:{episode}:{frame}:{bs}:{int(key)}:phase")
        generator = _generator(seed, device)
        values.append((torch.rand(1, device=device, dtype=dtype, generator=generator) * 2 - 1) * math.pi)
    if not values:
        return torch.zeros(0, device=device, dtype=dtype)
    return torch.stack(values).reshape(-1)


@torch.no_grad()
def synthesize_shared(
    positions,
    velocities,
    source_keys: Sequence[int],
    stations,
    boresights,
    waveform: PaperWaveform,
    array: ArrayConfig,
    rcs_m2,
    snr_ref_db: float,
    episode: int,
    frame: int,
    height_m: float | None = None,
    noise: bool = True,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.float64,
) -> dict:
    if dtype not in (torch.float64, torch.float32):
        raise ValueError("dtype must be float64 or float32")
    if height_m is None or not math.isfinite(float(height_m)) or float(height_m) < 0:
        raise ValueError("height_m must be the finite A01/config height difference (5 m)")
    p = torch.as_tensor(positions, device=device, dtype=dtype)
    v = torch.as_tensor(velocities, device=device, dtype=dtype)
    if p.numel() == 0:
        p = torch.zeros((0, 2), device=device, dtype=dtype)
    if v.numel() == 0:
        v = torch.zeros((0, 2), device=device, dtype=dtype)
    s = torch.as_tensor(stations, device=device, dtype=dtype)
    b = torch.as_tensor(boresights, device=device, dtype=dtype)
    sigma = torch.as_tensor(rcs_m2, device=device, dtype=dtype).reshape(-1, 1)
    if p.ndim != 2 or p.shape[1] != 2 or v.shape != p.shape or s.shape != (3, 2) or b.shape != (3,):
        raise ValueError("Expected positions/velocities [Nt,2], stations [3,2], boresights [3]")
    keys = [int(k) for k in source_keys]
    if len(keys) != len(p):
        raise ValueError("source_keys must provide one stable key per target")
    if sigma.numel() != len(p):
        raise ValueError("rcs_m2 must provide one value per target")
    if not math.isfinite(float(snr_ref_db)):
        raise ValueError("snr_ref_db must be finite")

    delta = s[None] - p[:, None]
    to_target = p[:, None] - s[None]
    rho = torch.linalg.vector_norm(delta, dim=-1)
    if len(p) and torch.any(rho <= 0):
        raise ValueError("Target coincides with station")
    height = rho.new_tensor(float(height_m))
    radii = torch.sqrt(rho ** 2 + height ** 2)
    radial = (delta * v[:, None]).sum(-1) / radii
    bearing = (torch.remainder(torch.atan2(to_target[..., 1], to_target[..., 0]) - b[None] + math.pi,
                               2 * math.pi) - math.pi)
    visibility = ((radii >= 10.0) & (radii <= 300.0) & (bearing.abs() <= math.radians(70.0))) if len(p) else \
        torch.zeros((0, 3), device=device, dtype=torch.bool)
    u = torch.sin(bearing)
    amplitude = 10 ** (float(snr_ref_db) / 20) * (100.0 / radii.clamp_min(1e-9)) ** 2 * torch.sqrt(sigma / 10.0)
    amplitude = torch.where(visibility, amplitude, torch.zeros_like(amplitude))

    xs, ys, noise_cubes = [], [], []
    k = torch.arange(waveform.K, device=device, dtype=dtype)
    n = torch.arange(waveform.N, device=device, dtype=dtype)
    a = torch.arange(array.elements, device=device, dtype=dtype)
    for bs in range(3):
        seed_waveform = derive_seed(f"2026:{episode}:{frame}:{bs}:waveform")
        generator = _generator(seed_waveform, device)
        q = torch.randint(0, 4, (waveform.K, waveform.N), generator=generator, device=device)
        x = torch.polar(torch.ones_like(q, dtype=dtype), math.pi / 4 + math.pi / 2 * q.to(dtype))
        phases = _phases(episode, frame, bs, keys, device, dtype)
        if len(p):
            delay = torch.exp(-1j * 4 * math.pi * waveform.df / waveform.c * radii[:, bs, None] * k)
            doppler = torch.exp(1j * 4 * math.pi * waveform.fc * waveform.T / waveform.c * radial[:, bs, None] * n)
            element = torch.exp(1j * math.pi * a[None] * u[:, bs, None])
            scalar = amplitude[:, bs] * torch.polar(torch.ones_like(phases), phases)
            contribution = (scalar[:, None, None, None] * element[:, :, None, None]
                            * delay[:, None, :, None] * doppler[:, None, None, :]).sum(0)
        else:
            contribution = torch.zeros((array.elements, waveform.K, waveform.N), device=device, dtype=torch.complex128)
            contribution = contribution.to(torch.complex64 if dtype == torch.float32 else torch.complex128)
        clean = contribution * x[None, :, :]
        seed_noise = derive_seed(f"2026:{episode}:{frame}:{bs}:noise")
        noise_generator = _generator(seed_noise, device)
        real = torch.randn((array.elements, waveform.K, waveform.N), device=device, dtype=dtype,
                           generator=noise_generator)
        imag = torch.randn((array.elements, waveform.K, waveform.N), device=device, dtype=dtype,
                           generator=noise_generator)
        noise_cube = torch.complex(real, imag) / math.sqrt(2)
        xs.append(x)
        ys.append(clean + noise_cube if noise else clean)
        noise_cubes.append(noise_cube)

    output_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
    return {
        "Y": torch.stack(ys).to(output_dtype),
        "X": torch.stack(xs).to(output_dtype),
        "W": torch.stack(noise_cubes).to(output_dtype),
        "alpha": amplitude.T.contiguous(),
        "visible": visibility.T.contiguous(),
    }


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise SystemExit("synthesize_shared requires CUDA per frozen design")
    w = PaperWaveform()
    array = ArrayConfig()
    stations = [[-64.23372566, -106.90678644], [114.23372566, 0.0], [-64.23372566, 106.90678644]]
    boresights = [0.0, math.pi, 0.0]
    out = synthesize_shared([[0.0, 0.0]], [[5.0, -3.0]], [7], stations, boresights, w, array, [10.0], 20.0,
                            episode=0, frame=0, height_m=5.0, device=device)
    assert out["Y"].shape == (3, array.elements, w.K, w.N)
    assert out["X"].shape == (3, w.K, w.N)
    assert out["W"].shape == out["Y"].shape
    assert out["Y"].dtype == torch.complex128 and out["alpha"].shape == (3, 1)
    print({"Y": list(out["Y"].shape), "alpha": out["alpha"].tolist(), "visible": out["visible"].tolist()})