"""T1 shared-echo structural test (VALIDATION_PLAN.md T1, P0-2 seed contract)."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import _common  # noqa: E402

OUT = ROOT / "reports/f01e/p1_shared_echo"
EPISODE = 9000
FRAME = 0
SNR_REF = 20.0
TARGETS = [
    ([-20.0, -5.0], [8.0, -4.0], 101),
    ([30.0, 20.0], [-6.0, 3.0], 202),
    ([0.0, 60.0], [4.0, 1.0], 303),
]


def relative_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-300))


def source_hashes() -> dict:
    names = ["frontend/sensing/waveform.py", "frontend/sensing/simulator.py", "configs/shared_frontend.json"]
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("T1 requires CUDA per frozen design")
    device = "cuda:0"
    config = _common.load_frontend_config()
    waveform, array = _common.build_objects(config)
    geometry = _common.load_geometry_config()
    stations, boresights = geometry["stations"], geometry["boresights"]

    positions = [target[0] for target in TARGETS]
    velocities = [target[1] for target in TARGETS]
    keys = [target[2] for target in TARGETS]
    checks = {}

    shared_noisy = _common.synthesize(positions, velocities, keys, stations, boresights, waveform, array,
                                      config, SNR_REF, EPISODE, FRAME, device, noise=True)
    shared_clean = _common.synthesize(positions, velocities, keys, stations, boresights, waveform, array,
                                      config, SNR_REF, EPISODE, FRAME, device, noise=False)
    clean = shared_clean["Y"]
    noise = shared_noisy["Y"] - clean

    summed = torch.stack([
        _common.synthesize([position], [velocity], [key], stations, boresights, waveform, array,
                           config, SNR_REF, EPISODE, FRAME, device, noise=False)["Y"]
        for position, velocity, key in TARGETS]).sum(0)
    linearity = relative_error(clean, summed)
    checks["clean_shared_equals_sum_of_single_clean"] = {"passed": linearity < 1e-12,
                                                         "max_relative_error": linearity}

    noise_power = [float(noise[b].abs().square().mean()) for b in range(3)]
    checks["single_receiver_noise_with_unit_variance"] = {
        "passed": all(abs(value - 1.0) < 0.05 for value in noise_power),
        "mean_power_per_bs": noise_power}

    permuted = _common.synthesize(positions[::-1], velocities[::-1], keys[::-1], stations, boresights,
                                  waveform, array, config, SNR_REF, EPISODE, FRAME, device, noise=True)
    permuted_clean = _common.synthesize(positions[::-1], velocities[::-1], keys[::-1], stations, boresights,
                                        waveform, array, config, SNR_REF, EPISODE, FRAME, device, noise=False)
    permuted_noise = permuted["Y"] - permuted_clean["Y"]
    order_error = relative_error(permuted_clean["Y"], clean)
    x_identical = bool(torch.equal(shared_noisy["X"], permuted["X"]))
    noise_difference = float((noise - permuted_noise).abs().max())
    checks["waveform_and_noise_independent_of_target_order"] = {
        "passed": x_identical and noise_difference <= 1e-12 and order_error < 1e-12,
        "X_bitwise_identical": x_identical, "max_noise_difference": noise_difference,
        "max_relative_clean_error": order_error,
        "note": "noise realization is target-order independent; residual equals clean summation-order rounding"}

    repeat = _common.synthesize([positions[0]], [velocities[0]], [101], stations, boresights, waveform, array,
                                config, SNR_REF, EPISODE, FRAME, device, noise=False)
    same = _common.synthesize([positions[0]], [velocities[0]], [101], stations, boresights, waveform, array,
                              config, SNR_REF, EPISODE, FRAME, device, noise=False)
    other = _common.synthesize([positions[0]], [velocities[0]], [999], stations, boresights, waveform, array,
                               config, SNR_REF, EPISODE, FRAME, device, noise=False)
    reproducible = bool(torch.equal(repeat["Y"], same["Y"]))
    phase_differs = relative_error(other["Y"], repeat["Y"]) > 1e-3
    checks["phase_seeded_by_source_key"] = {"passed": reproducible and phase_differs,
                                            "same_key_bitwise": reproducible,
                                            "different_key_changes_echo": phase_differs}

    one_target = _common.synthesize([positions[0]], [velocities[0]], [101], stations, boresights,
                                    waveform, array, config, SNR_REF, EPISODE, FRAME, device, noise=True)
    shape_ok = shared_noisy["Y"].shape == one_target["Y"].shape == (3, array.elements, waveform.K, waveform.N)
    no_target_dim = shape_ok and shared_noisy["Y"].ndim == 4
    checks["no_target_dimension_in_observation"] = {
        "passed": bool(no_target_dim), "shared_shape": list(shared_noisy["Y"].shape),
        "single_shape": list(one_target["Y"].shape)}

    repeat_noisy = _common.synthesize(positions, velocities, keys, stations, boresights, waveform, array,
                                  config, SNR_REF, EPISODE, FRAME, device, noise=True)
    checks["bitwise_determinism_same_call"] = {
        "passed": bool(torch.equal(shared_noisy["Y"], repeat_noisy["Y"])
                       and torch.equal(shared_noisy["X"], repeat_noisy["X"]))}

    OUT.mkdir(parents=True, exist_ok=True)
    crop = 32
    np.savez_compressed(
        OUT / "shared_echo_frame.npz",
        clean_crop=clean[:, :, :crop, :crop].cpu().numpy(),
        noise_crop=noise[:, :, :crop, :crop].cpu().numpy(),
        X_crop=shared_noisy["X"][:, :crop, :crop].cpu().numpy(),
        alpha=shared_noisy["alpha"].cpu().numpy(),
        visible=shared_noisy["visible"].cpu().numpy(),
        full_shape=np.asarray(shared_noisy["Y"].shape, dtype=np.int64))
    checks["diagnostic_frame_saved"] = {"passed": True, "path": "reports/f01e/p1_shared_echo/shared_echo_frame.npz",
                                        "note": "32x32 crop only; full Y is never persisted"}

    summary = {"test": "T1", "episode": EPISODE, "frame": FRAME, "snr_ref_db": SNR_REF,
               "linearity_max_relative_error": linearity, "noise_power_per_bs": noise_power,
               "source_hashes": source_hashes(), "checks": checks}
    passed = all(entry["passed"] for entry in checks.values())
    _common.write_json(OUT / "summary.json", summary)
    _common.write_json(OUT / "checks.json", {"test": "T1", "passed": passed, "checks": checks})
    print(json.dumps({"T1": "PASS" if passed else "FAIL", "linearity": linearity,
                      "noise_power": noise_power}, ensure_ascii=False))


if __name__ == "__main__":
    main()