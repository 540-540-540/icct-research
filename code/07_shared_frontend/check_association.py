"""P3 association/fusion unit tests A-H (SENS-REBUILD-03B section 13)."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import _common  # noqa: E402

OUT = ROOT / "reports/f01e/p3_association"


def make_detection(bs: int, x: float, y: float, sigma: float = 0.2, power: float = 100.0,
                   quality: float = 25.0, grid=(0, 0, 0), time_ns: int = 0) -> dict:
    return {"station_id": bs, "time_ns": time_ns, "x_m": float(x), "y_m": float(y),
            "C_xy": [[sigma ** 2, 0.0], [0.0, sigma ** 2]], "bs_mask": 1 << bs, "n_bs": 1,
            "peak_to_noise_db": float(quality), "peak_power": float(power), "grid_index": list(grid)}


def canonical(observations: list[dict]) -> list[tuple]:
    return sorted((observation["n_bs"], observation["bs_mask"], round(observation["x_m"], 6),
                   round(observation["y_m"], 6)) for observation in observations)


def fused_covariance_finite(observations: list[dict]) -> bool:
    for observation in observations:
        covariance = np.asarray(observation["C_xy"], dtype=float)
        if covariance.shape != (2, 2) or not np.isfinite(covariance).all():
            return False
        if np.linalg.eigvalsh(covariance).min() <= 0:
            return False
        if not np.isfinite([observation["x_m"], observation["y_m"], observation["quality_db"]]).all():
            return False
    return True


def main() -> None:
    from frontend.fusion.association import fuse_frame

    config = _common.load_frontend_config()
    checks = {}

    # A. standard 2-BS fusion
    observations, diagnostics = fuse_frame({0: [make_detection(0, 100.0, 0.0)],
                                            1: [make_detection(1, 100.3, 0.0)], 2: []}, config)
    checks["A_two_bs_fusion"] = {"passed": len(observations) == 1 and observations[0]["n_bs"] == 2
                                 and observations[0]["bs_mask"] == 0b011 and fused_covariance_finite(observations),
                                 "detail": {"observations": observations, "diagnostics": diagnostics}}

    # B. standard 3-BS fusion
    observations, diagnostics = fuse_frame({0: [make_detection(0, 100.0, -50.0)],
                                            1: [make_detection(1, 100.3, -50.1)],
                                            2: [make_detection(2, 99.8, -49.9)]}, config)
    checks["B_three_bs_fusion"] = {"passed": len(observations) == 1 and observations[0]["n_bs"] == 3
                                   and observations[0]["bs_mask"] == 0b111 and fused_covariance_finite(observations),
                                   "detail": {"observations": observations, "diagnostics": diagnostics}}

    # C. out-of-gate pair must be revoked by post-filter
    observations, diagnostics = fuse_frame({0: [make_detection(0, 0.0, 0.0)],
                                            1: [make_detection(1, 50.0, 0.0)], 2: []}, config)
    checks["C_gate_rejection"] = {"passed": all(observation["n_bs"] == 1 for observation in observations)
                                  and len(observations) == 2 and diagnostics["postfilter_rejections"] >= 1,
                                  "detail": {"observations": observations, "diagnostics": diagnostics}}

    # D. BS0 missing
    observations, diagnostics = fuse_frame({0: [], 1: [make_detection(1, 10.0, 10.0)],
                                            2: [make_detection(2, 10.2, 10.0)]}, config)
    checks["D_bs0_missing"] = {"passed": len(observations) == 1 and observations[0]["n_bs"] == 2
                               and observations[0]["bs_mask"] == 0b110,
                               "detail": {"observations": observations, "diagnostics": diagnostics}}

    # E. BS1 missing, BS0 singleton joins BS2 at stage 2
    observations, diagnostics = fuse_frame({0: [make_detection(0, 20.0, 20.0)], 1: [],
                                            2: [make_detection(2, 20.3, 20.0)]}, config)
    checks["E_bs1_missing"] = {"passed": len(observations) == 1 and observations[0]["n_bs"] == 2
                               and observations[0]["bs_mask"] == 0b101,
                               "detail": {"observations": observations, "diagnostics": diagnostics}}

    # F. all singleton
    observations, diagnostics = fuse_frame({0: [make_detection(0, 0.0, 0.0)],
                                            1: [make_detection(1, 100.0, 0.0)],
                                            2: [make_detection(2, 0.0, 100.0)]}, config)
    checks["F_all_singleton"] = {"passed": len(observations) == 3
                                 and all(observation["n_bs"] == 1 for observation in observations),
                                 "detail": {"observations": observations, "diagnostics": diagnostics}}

    # G. near-neighbor conflict: one-to-one usage and deterministic grouping
    base = {0: [make_detection(0, 0.0, 0.0, grid=(5, 5, 5)), make_detection(0, 2.0, 0.1, grid=(6, 5, 5))],
            1: [make_detection(1, 0.2, 0.1, grid=(5, 5, 5)), make_detection(1, 1.9, -0.1, grid=(6, 5, 5))],
            2: [make_detection(2, 0.1, -0.2, grid=(5, 5, 5)), make_detection(2, 2.1, 0.2, grid=(6, 5, 5))]}
    observations, diagnostics = fuse_frame(copy.deepcopy(base), config)
    per_bs_usage_ok = True
    for observation in observations:
        mask = observation["bs_mask"]
        if mask.bit_count() != observation["n_bs"]:
            per_bs_usage_ok = False
    total_members = sum(observation["n_bs"] for observation in observations)
    checks["G_near_neighbor_conflict"] = {"passed": per_bs_usage_ok and total_members == 6
                                          and len(observations) == 2 and fused_covariance_finite(observations),
                                          "detail": {"observations": observations, "diagnostics": diagnostics}}

    # H. permutation invariance
    import random

    reference = canonical(fuse_frame(copy.deepcopy(base), config)[0])
    permutation_ok = True
    for seed in range(10):
        shuffled = {}
        rng = random.Random(seed)
        for bs, entries in base.items():
            local = copy.deepcopy(entries)
            rng.shuffle(local)
            shuffled[bs] = local
        permutation_ok &= canonical(fuse_frame(shuffled, config)[0]) == reference
    checks["H_permutation_invariance"] = {"passed": permutation_ok, "detail": {"reference": reference}}

    passed = all(entry["passed"] for entry in checks.values())
    summary = {"test": "P3 association/fusion", "passed": passed, "checks": checks,
               "source_hashes": {"frontend/fusion/association.py":
                                 hashlib.sha256((ROOT / "frontend/fusion/association.py").read_bytes()).hexdigest()}}
    OUT.mkdir(parents=True, exist_ok=True)
    _common.write_json(OUT / "summary.json", summary)
    _common.write_json(OUT / "checks.json", {"test": "P3 association/fusion", "passed": passed, "checks": checks})
    print(json.dumps({"P3": "PASS" if passed else "FAIL",
                      "checks": {key: entry["passed"] for key, entry in checks.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()