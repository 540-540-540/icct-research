"""Frozen V1 waveform, array and geometry constants (SENS-DESIGN-02).

Design references: SYSTEM_MODEL.md sections 1-4, DECISIONS.md D03/D04, INTERFACE_SPEC.md 3.1.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PaperWaveform:
    fc: float = 24e9
    B: float = 93.1e6
    K: int = 256
    N: int = 256
    T: float = 12.375e-6
    c: float = 299792458.0

    def __post_init__(self):
        if self.K < 2 or self.N < 2 or int(self.K) != self.K or int(self.N) != self.N:
            raise ValueError("K and N must be integers >= 2")
        if any(not math.isfinite(v) or v <= 0 for v in (self.fc, self.B, self.T, self.c)):
            raise ValueError("Waveform constants must be finite and positive")

    @property
    def df(self) -> float:
        return self.B / self.K

    @property
    def unambiguous_range(self) -> float:
        return self.c / (2 * self.df)

    @property
    def unambiguous_velocity(self) -> float:
        return self.c / (4 * self.fc * self.T)

    @property
    def range_resolution(self) -> float:
        return self.c / (2 * self.B)

    @property
    def velocity_resolution(self) -> float:
        return self.c / (2 * self.fc * self.N * self.T)

    @property
    def wavelength(self) -> float:
        return self.c / self.fc

    @property
    def cpi(self) -> float:
        return self.N * self.T


@dataclass(frozen=True)
class ArrayConfig:
    elements: int = 16
    spacing_wavelengths: float = 0.5
    fft_size: int = 64
    window: str = "rect"
    aoa_max_peaks: int = 1
    aoa_nms_radius: int = 4

    def __post_init__(self):
        if self.elements < 2 or int(self.elements) != self.elements:
            raise ValueError("elements must be an integer >= 2")
        if self.fft_size < self.elements or int(self.fft_size) != self.fft_size:
            raise ValueError("fft_size must be an integer >= elements")
        if self.window != "rect":
            raise ValueError("V1 only implements the rectangular array window")
        if self.aoa_max_peaks != 1 or int(self.aoa_max_peaks) != self.aoa_max_peaks:
            raise ValueError("V1 freezes aoa_max_peaks=1")


def load_geometry(root: Path | str = ROOT, path: str = "reports/f01a/geometry.json") -> dict:
    payload = json.loads((Path(root) / path).read_text())
    stations = torch.tensor(payload["stations_xy_m"], dtype=torch.float64)
    boresights = torch.tensor(payload["boresight_rad"], dtype=torch.float64)
    if stations.shape != (3, 2) or boresights.shape != (3,):
        raise ValueError("A01 geometry must provide three planar stations and boresights")
    return {
        "stations": stations,
        "boresights": boresights,
        "height_difference_m": float(payload["station_height_m"] - payload["vehicle_height_m"]),
        "fov_half_angle_deg": float(payload["fov_half_angle_deg"]),
        "range_m": [float(v) for v in payload["range_m"]],
    }


def periodic_hann(n: int, device, dtype) -> torch.Tensor:
    index = torch.arange(n, device=device, dtype=dtype)
    return 0.5 - 0.5 * torch.cos(2 * torch.pi * index / n)


if __name__ == "__main__":
    w = PaperWaveform()
    assert abs(w.range_resolution - 1.610056165413534) < 1e-12
    assert abs(w.unambiguous_range - 412.1743783458647) < 1e-9
    assert abs(w.unambiguous_velocity - 252.35055387205387) < 1e-9
    array = ArrayConfig()
    assert array.elements == 16 and array.fft_size == 64
    print({"waveform": {"df": w.df, "range_resolution": w.range_resolution,
                        "unambiguous_velocity": w.unambiguous_velocity, "cpi": w.cpi},
           "array": array.__dict__})