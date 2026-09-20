#!/usr/bin/env python3
"""Independent P1.1 correctness/performance gate for the repaired Raj-PennyLane core."""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pennylane as qml
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prediction.qgnn_raj_pennylane.common import (
    NODE_QUBITS,
    OBS_PER_NODE,
    TOTAL_QUBITS,
    canonical_joint_state,
    canonical_pack,
)
from prediction.qgnn_raj_pennylane.observables import formal_observables
from prediction.qgnn_raj_pennylane.quantum import PennyLaneRajMultiJCore


def synthetic_history(batch: int, n: int = 8, seed: int = 20260920):
    generator = torch.Generator(device="cpu").manual_seed(seed + batch + n)
    t = torch.arange(20, dtype=torch.float32)[None, :, None, None]
    centers = torch.arange(n, dtype=torch.float32)[None, None, :, None]
    velocity = torch.randn(batch, 1, n, 2, generator=generator) * 0.25
    position = torch.randn(batch, 1, n, 2, generator=generator) * 3.0
    position = position + centers[..., :1] * 7.0 + velocity * t
    speed = torch.randn(batch, 1, n, 2, generator=generator) * 0.1
    history = torch.cat((position, speed.expand(batch, 20, n, 2)), -1)
    return history, torch.ones(batch, n, dtype=torch.bool)


def sind_train_micro_samples(root: Path, snr_db: float = 0.0, limit: int = 2):
    sample_path = root / "data" / "sind" / "splits" / "train" / "samples.npz"
    cache_path = root / "data" / "sind" / "isac" / "train" / "sensing_cache.npz"
    with np.load(sample_path, allow_pickle=False) as samples:
        scene_ids = samples["scene_id"]
        start_frames = samples["start_frame"]
        vehicle_ids = samples["vehicle_ids"]
        vehicle_masks = samples["vehicle_mask"]
        candidates = np.flatnonzero(vehicle_masks.sum(axis=1) >= 3)[:limit]
        count_distribution = Counter(map(int, vehicle_masks.sum(axis=1)))
    with np.load(cache_path, allow_pickle=False) as cache:
        levels = cache["snr_levels_db"]
        hits = np.flatnonzero(np.isclose(levels, snr_db))
        if len(hits) != 1:
            raise ValueError(f"SinD train cache has no unique SNR {snr_db}")
        snr_index = int(hits[0])
        state_hat = cache["state_hat"][snr_index]
        lookup = {
            (int(scene), int(frame), int(vehicle)): row
            for row, (scene, frame, vehicle) in enumerate(
                zip(cache["scene_id"], cache["frame"], cache["vehicle_id"])
            )
        }
        histories, masks = [], []
        for sample_index in candidates:
            sample_index = int(sample_index)
            mask = vehicle_masks[sample_index].astype(np.bool_)
            history = np.zeros((20, 8, 4), dtype=np.float32)
            scene = int(scene_ids[sample_index])
            start = int(start_frames[sample_index])
            for t in range(20):
                for slot in range(8):
                    if not mask[slot]:
                        continue
                    key = (scene, start + t, int(vehicle_ids[sample_index, slot]))
                    if key not in lookup:
                        raise KeyError(f"missing SinD sensing key {key}")
                    history[t, slot] = state_hat[lookup[key]]
            histories.append(history)
            masks.append(mask)
    return (
        candidates,
        np.stack(histories),
        np.stack(masks),
        float(levels[snr_index]),
        dict(sorted(count_distribution.items())),
    )


def record(checks, name, passed, **evidence):
    row = {"passed": bool(passed), **evidence}
    checks[name] = row
    print(json.dumps({"check": name, **row}, ensure_ascii=False), flush=True)


def finite_nonzero_gradients(model):
    groups = {
        "feature_and_angle_generation": [],
        "quantum_specific_circuit": [],
        "post_measurement_readout": [],
        "multi_j_fusion": [],
    }
    quantum_suffixes = ("subset_bias", "embed_bias", "cross_bias", "cross_basis")
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("fusion."):
            group = "multi_j_fusion"
        elif ".readout." in name:
            group = "post_measurement_readout"
        elif any(name.endswith(suffix) for suffix in quantum_suffixes):
            group = "quantum_specific_circuit"
        else:
            group = "feature_and_angle_generation"
        grad = parameter.grad
        groups[group].append(
            {
                "name": name,
                "numel": parameter.numel(),
                "grad_none": grad is None,
                "finite": False if grad is None else bool(torch.isfinite(grad).all()),
                "max_abs": None if grad is None else float(grad.abs().max()),
                "nonzero": False if grad is None else float(grad.abs().sum()) > 1e-16,
            }
        )
    result = {}
    for group, rows in groups.items():
        result[group] = {
            "parameter_tensors": len(rows),
            "parameter_count": sum(row["numel"] for row in rows),
            "finite": all(row["finite"] for row in rows),
            "nonzero_tensors": sum(row["nonzero"] for row in rows),
            "all_finite_nonzero": bool(rows)
            and all(row["finite"] and row["nonzero"] for row in rows),
            "max_abs_min": min(
                (row["max_abs"] for row in rows if row["max_abs"] is not None),
                default=None,
            ),
            "max_abs_max": max(
                (row["max_abs"] for row in rows if row["max_abs"] is not None),
                default=None,
            ),
        }
    return result


def specs_for_branch(branch, history, mask):
    packed_history, packed_mask, _ = canonical_pack(history, mask)
    _, feat, valid, _ = branch.builder(packed_history, packed_mask, branch.j)
    subset, graph, embed, cross = branch._angles(
        packed_history, packed_mask, feat, valid
    )
    count = int(packed_mask[0].sum())
    state = canonical_joint_state(count, branch.j).to(
        device=history.device, dtype=torch.complex128
    )
    specs = qml.specs(branch._state_qnode)(
        state,
        subset[:1].to(torch.float64),
        graph[:1].to(torch.float64),
        embed[:1].to(torch.float64),
        cross[:1].to(torch.float64),
    )
    resources = specs["resources"]
    return {
        "num_wires": TOTAL_QUBITS,
        "num_gates": resources.num_gates,
        "depth": resources.depth,
        "gate_types": dict(resources.gate_types),
        "gate_sizes": {str(k): v for k, v in resources.gate_sizes.items()},
        "stateprep_is_logical_undecomposed": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="reports/qgnn/raj_pennylane_p1_1/repair_preflight_20260920.json",
    )
    args = parser.parse_args()

    torch.set_num_threads(min(8, torch.get_num_threads()))
    if not torch.cuda.is_available():
        raise RuntimeError("P1.1 performance gate requires CUDA")
    device = torch.device("cuda:0")
    checks = {}
    report = {
        "schema": "raj_pennylane_p1_1_repair_v1",
        "scope": {
            "formal_training": False,
            "optimizer_steps": 0,
            "validation_opened": False,
            "test_opened": False,
            "labels_opened": False,
        },
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "pennylane": qml.__version__,
            "cuda_device": torch.cuda.get_device_name(0),
        },
    }

    torch.manual_seed(20260920)
    core = PennyLaneRajMultiJCore(rounds=3).to(device)
    breakdown = core.parameter_breakdown()
    report["parameter_breakdown"] = breakdown
    record(
        checks,
        "parameter_budget",
        100000 <= breakdown["total_interaction_core"] <= 150000,
        **breakdown,
    )
    metadata = core.circuit_metadata()
    report["circuit"] = metadata
    record(
        checks,
        "fixed_circuit_topology",
        metadata["j2"]["fixed_topology"]
        and metadata["j3"]["fixed_topology"]
        and not metadata["j2"]["scene_specific_qnode_cache"]
        and not metadata["j3"]["scene_specific_qnode_cache"],
        metadata=metadata,
    )
    record(
        checks,
        "formal_observable_contract",
        OBS_PER_NODE == 37 and len(formal_observables()) == 8 * 37,
        observables_per_vehicle=OBS_PER_NODE,
        formal_observable_count=len(formal_observables()),
    )

    history, mask = synthetic_history(1, 8, seed=7777)
    history, mask = history.to(device), mask.to(device)

    # Full CUDA forward/backward and gradient groups.
    core.train()
    core.zero_grad(set_to_none=True)
    output = core(history, mask)
    loss = (output["readout"] * torch.randn_like(output["readout"])).mean()
    loss.backward()
    gradient_groups = finite_nonzero_gradients(core)
    report["gradient_checks"] = gradient_groups
    record(
        checks,
        "cuda_forward_backward",
        bool(torch.isfinite(output["readout"]).all())
        and bool(torch.isfinite(loss))
        and all(v["all_finite_nonzero"] for v in gradient_groups.values()),
        readout_shape=list(output["readout"].shape),
        branch_shape=list(output["branch_readout"].shape),
        interaction_tokens=core.interaction_tokens,
        gradient_groups=gradient_groups,
    )

    # Fast exact-statevector feature path must equal direct formal qml.expval.
    core.eval()
    packed_history, packed_mask, _ = canonical_pack(history, mask)
    equivalence = {}
    with torch.no_grad():
        for name, branch in (("j2", core.j2), ("j3", core.j3)):
            _, fast_raw = branch.forward_packed(
                packed_history, packed_mask, return_raw=True
            )
            formal_raw = branch.formal_expval_features(
                packed_history,
                packed_mask,
                device_name="default.qubit",
                diff_method="backprop",
            )
            equivalence[name] = {
                "max_abs": float((fast_raw - formal_raw).abs().max()),
                "mean_abs": float((fast_raw - formal_raw).abs().mean()),
            }
    report["statevector_expval_equivalence"] = equivalence
    record(
        checks,
        "statevector_expval_equivalence",
        all(row["max_abs"] <= 1e-8 for row in equivalence.values()),
        **equivalence,
    )

    backend_consistency = {}
    with torch.no_grad():
        for name, branch in (("j2", core.j2), ("j3", core.j3)):
            default_raw = branch.formal_expval_features(
                packed_history, packed_mask,
                device_name="default.qubit", diff_method="backprop",
            )
            lightning_raw = branch.formal_expval_features(
                packed_history, packed_mask,
                device_name="lightning.gpu", diff_method="adjoint",
            )
            backend_consistency[name] = {
                "max_abs": float((default_raw - lightning_raw).abs().max()),
                "mean_abs": float((default_raw - lightning_raw).abs().mean()),
            }
    report["formal_backend_consistency"] = backend_consistency
    record(
        checks,
        "formal_backend_consistency",
        all(row["max_abs"] <= 5e-6 for row in backend_consistency.values()),
        **backend_consistency,
    )

    gradient_equivalence = {}
    torch.manual_seed(77)
    from prediction.qgnn_raj_pennylane.observables import conditional_embedding_rdm_features, normalize_formal_raw
    for name, branch in (("j2", core.j2), ("j3", core.j3)):
        _, feat, valid, _ = branch.builder(packed_history, packed_mask, branch.j)
        base_angles = branch._angles(packed_history, packed_mask, feat, valid)
        fast_angles = [value.detach().clone().requires_grad_(True) for value in base_angles]
        state = canonical_joint_state(8, branch.j).to(device=device, dtype=torch.complex128)
        psi = branch._state_qnode(state, *[value.to(torch.float64) for value in fast_angles])
        fast_feature = conditional_embedding_rdm_features(psi, packed_mask)
        weight = torch.randn_like(fast_feature)
        fast_loss = (fast_feature * weight).sum()
        fast_grad = torch.autograd.grad(fast_loss, fast_angles)

        formal_angles = [value.detach().clone().requires_grad_(True) for value in base_angles]
        formal_qnode = branch._build_formal_qnode("default.qubit", "backprop")
        formal_values = formal_qnode(state, *[value.to(torch.float64) for value in formal_angles])
        formal_raw = torch.stack(formal_values, -1).reshape(1, NODE_QUBITS, OBS_PER_NODE)
        formal_feature = normalize_formal_raw(formal_raw, packed_mask)
        formal_loss = (formal_feature * weight).sum()
        formal_grad = torch.autograd.grad(formal_loss, formal_angles)
        errors = [float((a-b).abs().max()) for a,b in zip(fast_grad, formal_grad)]
        gradient_equivalence[name] = {
            "loss_abs": float((fast_loss.detach()-formal_loss.detach()).abs()),
            "gradient_max_abs": max(errors),
            "by_angle_group_abs": errors,
        }
    report["gradient_equivalence"] = gradient_equivalence
    record(
        checks,
        "statevector_expval_gradient_equivalence",
        all(row["gradient_max_abs"] <= 1e-8 for row in gradient_equivalence.values()),
        **gradient_equivalence,
    )

    # Structural ablations.
    with torch.no_grad():
        base = core(history, mask)["readout"]
        variants = {
            "j2_off": core(history, mask, branch_scales=(0.0, 1.0))["readout"],
            "j3_off": core(history, mask, branch_scales=(1.0, 0.0))["readout"],
            "interaction_off": core(history, mask, interaction_scale=0.0)["readout"],
            "johnson_off": core(history, mask, johnson_scale=0.0)["readout"],
            "entangling_off": core(history, mask, entangling_scale=0.0)["readout"],
            "subset_phase_off": core(history, mask, subset_scale=0.0)["readout"],
        }
    ablations = {
        name: float((base - value).abs().max()) for name, value in variants.items()
    }
    report["ablations"] = ablations
    record(
        checks,
        "mechanism_participation",
        all(value > 1e-7 for value in ablations.values()),
        **ablations,
    )

    # Stronger permutation/padding checks than the original Luna preflight.
    permutation_errors = []
    generator = torch.Generator().manual_seed(404)
    with torch.no_grad():
        original = core(history, mask)["readout"]
        for _ in range(8):
            permutation = torch.randperm(8, generator=generator).to(device)
            inverse = torch.argsort(permutation)
            permuted = core(
                history[:, :, permutation], mask[:, permutation]
            )["readout"][:, inverse]
            permutation_errors.append(float((permuted - original).abs().max()))
    padding = {}
    for n in (3, 5, 7):
        compact_h, compact_m = history[:, :, :n], mask[:, :n]
        noisy_padding = torch.randn(1, 20, 8 - n, 4, device=device) * 1e5
        padded_h = torch.cat((compact_h, noisy_padding), 2)
        padded_m = torch.cat(
            (compact_m, torch.zeros(1, 8 - n, dtype=torch.bool, device=device)), 1
        )
        with torch.no_grad():
            compact_out = core(compact_h, compact_m)["readout"]
            padded_out = core(padded_h, padded_m)["readout"]
        padding[str(n)] = {
            "active_max_abs": float(
                (compact_out - padded_out[:, :n]).abs().max()
            ),
            "inactive_max_abs": float(padded_out[:, n:].abs().max()),
        }
    report["invariance"] = {
        "permutation_errors": permutation_errors,
        "padding": padding,
    }
    record(
        checks,
        "permutation_padding",
        max(permutation_errors) <= 2e-5
        and all(
            row["active_max_abs"] <= 2e-5 and row["inactive_max_abs"] <= 2e-5
            for row in padding.values()
        ),
        permutation_max_abs=max(permutation_errors),
        padding=padding,
    )

    # Mixed active counts in one optimizer batch.
    mixed_h, mixed_m = synthetic_history(4, 8, seed=8888)
    mixed_h, mixed_m = mixed_h.to(device), mixed_m.to(device)
    for row, count in enumerate((8, 7, 5, 3)):
        mixed_m[row, count:] = False
        mixed_h[row, :, count:] = 0
    with torch.no_grad():
        mixed_out = core(mixed_h, mixed_m)
    report["mixed_active_count"] = {
        "counts": [8, 7, 5, 3],
        "shape": list(mixed_out["readout"].shape),
        "finite": bool(torch.isfinite(mixed_out["readout"]).all()),
        "j2_groups": core.j2.last_group_count,
        "j3_groups": core.j3.last_group_count,
    }
    record(
        checks,
        "mixed_active_count_batch",
        report["mixed_active_count"]["finite"]
        and core.j2.last_group_count == 4
        and core.j3.last_group_count == 4,
        **report["mixed_active_count"],
    )

    # Fixed-weight initialization for both higher-order branches.
    weight_rows = {}
    for j in (2, 3):
        state = canonical_joint_state(8, j)
        nonzero = torch.where(state.abs() > 1e-12)[0]
        node_weights, embed_weights = set(), set()
        for index in nonzero.tolist():
            bits = [
                int(bool(index & (1 << (TOTAL_QUBITS - 1 - wire))))
                for wire in range(TOTAL_QUBITS)
            ]
            node_weights.add(sum(bits[:8]))
            embed_weights.add(sum(bits[8:]))
        weight_rows[str(j)] = {
            "node_weights": sorted(node_weights),
            "embedding_weights": sorted(embed_weights),
            "nonzero_basis_states": len(nonzero),
        }
    report["fixed_weight"] = weight_rows
    record(
        checks,
        "fixed_hamming_weight",
        weight_rows["2"]["node_weights"] == [2]
        and weight_rows["3"]["node_weights"] == [3]
        and weight_rows["2"]["embedding_weights"] == [3]
        and weight_rows["3"]["embedding_weights"] == [3],
        **weight_rows,
    )

    # Logical resource accounting; StatePrep remains explicitly undecomposed.
    specs = {
        "j2": specs_for_branch(core.j2, packed_history, packed_mask),
        "j3": specs_for_branch(core.j3, packed_history, packed_mask),
    }
    report["logical_resources"] = specs
    record(
        checks,
        "logical_resource_accounting",
        specs["j2"]["num_wires"] == 14
        and specs["j3"]["num_wires"] == 14
        and specs["j2"]["stateprep_is_logical_undecomposed"]
        and specs["j3"]["stateprep_is_logical_undecomposed"],
        **specs,
    )

    # Actual SinD train-history-only micro smoke.
    indices, real_h_np, real_m_np, snr_db, count_distribution = sind_train_micro_samples(
        ROOT, limit=2
    )
    real_h = torch.as_tensor(real_h_np, dtype=torch.float32, device=device)
    real_m = torch.as_tensor(real_m_np, dtype=torch.bool, device=device)
    core.train()
    core.zero_grad(set_to_none=True)
    real_out = core(real_h, real_m)
    real_loss = (real_out["readout"] * torch.randn_like(real_out["readout"])).mean()
    real_loss.backward()
    real_grads = finite_nonzero_gradients(core)
    report["actual_data"] = {
        "dataset": "SinD",
        "split": "train",
        "snr_db": snr_db,
        "sample_indices": [int(v) for v in indices],
        "active_counts": [int(v) for v in real_m.sum(-1)],
        "labels_opened": False,
        "readout_shape": list(real_out["readout"].shape),
        "finite": bool(torch.isfinite(real_out["readout"]).all())
        and bool(torch.isfinite(real_loss)),
        "gradient_groups": real_grads,
        "train_active_count_distribution": {
            str(k): int(v) for k, v in count_distribution.items()
        },
    }
    record(
        checks,
        "actual_train_micro_smoke",
        report["actual_data"]["finite"]
        and all(v["all_finite_nonzero"] for v in real_grads.values()),
        **report["actual_data"],
    )

    # Full repaired-core speed gate.
    profile = []
    for batch in (1, 4, 32):
        h, m = synthetic_history(batch, 8, seed=12000 + batch)
        h, m = h.to(device), m.to(device)
        core.train()
        core.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        out = core(h, m)
        profile_loss = out["readout"].square().mean()
        profile_loss.backward()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        profile.append(
            {
                "batch": batch,
                "seconds_forward_backward": elapsed,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "readout_shape": list(out["readout"].shape),
                "finite": bool(torch.isfinite(profile_loss)),
            }
        )
    report["profile"] = profile
    t1, t4, t32 = (row["seconds_forward_backward"] for row in profile)
    record(
        checks,
        "batch32_training_speed",
        t32 <= 20.0 and t4 < 4.0 * t1 and t32 < 16.0 * t4,
        profile=profile,
        batch32_threshold_seconds=20.0,
        scaling_t4_over_4t1=t4 / (4 * t1),
        scaling_t32_over_32t1=t32 / (32 * t1),
    )

    report["checks"] = checks
    report["passed"] = all(row["passed"] for row in checks.values())
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "check_count": len(checks),
                "output": str(output_path),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
