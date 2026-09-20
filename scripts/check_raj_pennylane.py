#!/usr/bin/env python3
"""P1 correctness and short profile for the native PennyLane Raj core.

This script never opens labels, validation data, or prediction test data. It
does not run an optimizer step and does not perform trajectory training.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
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
    pool_subset_features,
    uniform_joint_state,
)
from prediction.qgnn_raj_pennylane.quantum import PennyLaneRajMultiJCore
from prediction.qgnn_raj_pennylane.torch_reference import (
    ising_zz,
    single_excitation,
)


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
    if len(candidates) == 0:
        raise RuntimeError("no SinD train sample with >=3 active vehicles")
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
    return candidates, np.stack(histories), np.stack(masks), float(levels[snr_index])

def finite_tensor(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value).all())


def nvidia_snapshot():
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    rows = []
    for line in raw.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3:
            rows.append(
                {
                    "index": int(parts[0]),
                    "used_mib": int(parts[1]),
                    "total_mib": int(parts[2]),
                }
            )
    return rows


def nvidia_peak_start():
    try:
        return subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total",
                "--format=csv,noheader,nounits",
                "--loop-ms=100",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


def nvidia_peak_stop(process):
    if process is None:
        return []
    try:
        process.terminate()
    except ProcessLookupError:
        pass
    try:
        raw, _ = process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        raw, _ = process.communicate()
    peaks = {}
    for line in raw.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        index, used, total = (int(part) for part in parts)
        row = peaks.setdefault(index, {"index": index, "peak_used_mib": used, "total_mib": total})
        row["peak_used_mib"] = max(row["peak_used_mib"], used)
    return [peaks[index] for index in sorted(peaks)]

def group_gradients(model: torch.nn.Module):
    groups = {
        "feature_and_angle_generation": [],
        "quantum_specific_circuit": [],
        "post_measurement_readout": [],
        "multi_j_fusion": [],
    }
    quantum_names = (
        ".node_bias",
        ".embed_bias",
        ".cross_bias",
        ".cross_basis",
    )
    feature_names = (
        ".builder.",
        ".node_phase.",
        ".graph_angle.",
        ".embed_angle.",
        ".cross_strength.",
    )
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("fusion."):
            group = "multi_j_fusion"
        elif any(token in name for token in quantum_names):
            group = "quantum_specific_circuit"
        elif ".readout." in name:
            group = "post_measurement_readout"
        elif any(token in name for token in feature_names):
            group = "feature_and_angle_generation"
        else:
            continue
        groups[group].append((name, parameter))
    result = {}
    for group, items in groups.items():
        gradients = [parameter.grad for _, parameter in items]
        finite = all(
            gradient is not None and torch.isfinite(gradient).all()
            for gradient in gradients
        )
        nonzero = [
            (name, float(gradient.abs().max()))
            for (name, _), gradient in zip(items, gradients)
            if gradient is not None and float(gradient.abs().sum()) > 1e-16
        ]
        result[group] = {
            "parameter_tensors": len(items),
            "finite": bool(finite),
            "nonzero_tensors": len(nonzero),
            "max_abs_by_tensor": dict(nonzero),
            "all_finite_nonzero": bool(finite and len(nonzero) == len(items)),
        }
    return result


def occupied_count(index: int, wires):
    return sum((index >> (TOTAL_QUBITS - 1 - wire)) & 1 for wire in wires)


def record(checks, name, passed, **evidence):
    row = {"passed": bool(passed), **evidence}
    checks[name] = row
    print(json.dumps({"check": name, **row}, ensure_ascii=False), flush=True)


def primitive_check(checks):
    try:
        torch.manual_seed(17)
        state = torch.randn(16, dtype=torch.complex128)
        state = state / state.norm()
        angle_a = torch.tensor(0.37, dtype=torch.float64)
        angle_b = torch.tensor(-0.23, dtype=torch.float64)
        reference = ising_zz(
            single_excitation(state, 0, 2, angle_a), 1, 3, angle_b
        )
        device = qml.device("default.qubit", wires=4, shots=None)

        @qml.qnode(device, interface="torch", diff_method="backprop")
        def circuit(value, first, second):
            qml.StatePrep(value, wires=range(4))
            qml.SingleExcitation(first, wires=[0, 2])
            qml.IsingZZ(second, wires=[1, 3])
            return qml.state()

        production = circuit(state, angle_a, angle_b)
        error = float((production - reference).abs().max())
        record(
            checks,
            "torch_reference_single_excitation_isingzz",
            error <= 1e-10,
            max_abs_error=error,
            role="reference-only primitive equivalence; not formal readout",
        )
    except Exception as exc:
        record(checks, "torch_reference_single_excitation_isingzz", False, error=str(exc))


def environment_report(report, checks):
    devices = {}
    for name in ("default.qubit", "lightning.qubit", "lightning.gpu"):
        try:
            device = qml.device(name, wires=TOTAL_QUBITS, shots=None)
            devices[name] = {"available": True, "device": str(device)}
        except Exception as exc:
            devices[name] = {"available": False, "error": str(exc)}
    report["environment"] = {
        "python": sys.version,
        "torch": torch.__version__,
        "pennylane": qml.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_devices": [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ],
        "devices": devices,
    }
    record(
        checks,
        "environment_required_devices",
        devices.get("default.qubit", {}).get("available", False)
        and devices.get("lightning.gpu", {}).get("available", False),
        pennylane_version=qml.__version__,
        required=("default.qubit", "lightning.gpu"),
        devices=devices,
    )


def qnode_metadata(core, history, mask):
    branch = core.j2
    with torch.no_grad():
        _, feature, valid, _ = branch.builder(history, mask, branch.j)
        node_feature, _ = pool_subset_features(feature, valid, branch.j)
        edge, _ = __import__(
            "prediction.qgnn_final.common", fromlist=["physical_graph"]
        ).physical_graph(history, mask)
        order = branch._canonical_pair_order(edge[0], mask[0], node_feature[0])
        angles = branch._angles(history, mask, feature, valid)
        qnode = branch.qnode_for_order(order)
        state = uniform_joint_state(mask[0], branch.j)
    spec = qml.specs(qnode)(
        state,
        angles[0][0].to(torch.float64),
        angles[1][0].to(torch.float64),
        angles[2][0].to(torch.float64),
        angles[3][0].to(torch.float64),
    )
    resources = spec["resources"]
    resource_data = {
        "num_wires": int(len(qnode.device.wires)),
        "num_gates": int(resources.num_gates),
        "depth": int(resources.depth),
        "gate_types": {str(key): int(value) for key, value in resources.gate_types.items()},
        "gate_sizes": {str(key): int(value) for key, value in resources.gate_sizes.items()},
    }
    return {
        "device": str(qnode.device),
        "interface": str(qnode.interface),
        "diff_method": str(qnode.diff_method),
        "shots": str(qnode.device.shots),
        "qubits": TOTAL_QUBITS,
        "pair_order_length": len(order),
        "observables_per_vehicle": OBS_PER_NODE,
        "observable_count": len(branch._observables),
        "qml_specs": resource_data,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--skip-profile", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(min(8, torch.get_num_threads()))
    checks = {}
    report = {
        "schema": "raj_pennylane_p1_preflight_v1",
        "scope": {
            "formal_training": False,
            "optimizer_steps": 0,
            "labels_opened": False,
            "prediction_test_opened": False,
            "rounds": args.rounds,
        },
    }
    environment_report(report, checks)
    primitive_check(checks)

    history, mask = synthetic_history(1)
    core = None
    try:
        core = PennyLaneRajMultiJCore(
            rounds=args.rounds,
            device_name="default.qubit",
            diff_method="backprop",
        )
        report["parameter_breakdown"] = core.parameter_breakdown()
        report["circuit"] = core.circuit_metadata()
        total = report["parameter_breakdown"]["total_interaction_core"]
        record(
            checks,
            "parameter_budget",
            100000 <= total <= 150000,
            total_interaction_core=total,
            target="100k-150k",
            breakdown=report["parameter_breakdown"],
        )
    except Exception as exc:
        record(checks, "core_construction", False, error=str(exc))

    if core is not None:
        try:
            core.train()
            core.zero_grad(set_to_none=True)
            output = core(history, mask)
            projection = torch.randn_like(output["readout"])
            loss = (output["readout"] * projection).mean()
            loss.backward()
            group_result = group_gradients(core)
            report["gradient_checks"] = group_result
            record(
                checks,
                "qnode_forward_output_contract",
                tuple(output["readout"].shape) == (1, 8, 64)
                and tuple(output["branch_readout"].shape) == (1, 8, 2, 64)
                and finite_tensor(output["readout"]),
                readout_shape=list(output["readout"].shape),
                branch_shape=list(output["branch_readout"].shape),
                interaction_mask_shape=list(output["interaction_mask"].shape),
            )
            record(
                checks,
                "finite_forward_backward",
                finite_tensor(output["readout"])
                and torch.isfinite(loss)
                and all(row["finite"] for row in group_result.values()),
                loss=float(loss.detach()),
                gradient_groups=group_result,
            )
            for group, row in group_result.items():
                record(
                    checks,
                    f"gradient_{group}",
                    row["all_finite_nonzero"],
                    **row,
                )
        except Exception as exc:
            record(checks, "synthetic_forward_backward", False, error=str(exc))

        try:
            core.eval()
            with torch.no_grad():
                base = core(history, mask)["readout"]
                no_j2 = core(history, mask, branch_scales=(0.0, 1.0))["readout"]
                no_j3 = core(history, mask, branch_scales=(1.0, 0.0))["readout"]
                no_interaction = core(history, mask, interaction_scale=0.0)["readout"]
                no_johnson = core(history, mask, johnson_scale=0.0)["readout"]
                no_entangling = core(history, mask, entangling_scale=0.0)["readout"]
            changes = {
                "j2_off_max_abs": float((base - no_j2).abs().max()),
                "j3_off_max_abs": float((base - no_j3).abs().max()),
                "interaction_off_max_abs": float((base - no_interaction).abs().max()),
                "johnson_off_max_abs": float((base - no_johnson).abs().max()),
                "entangling_off_max_abs": float((base - no_entangling).abs().max()),
            }
            report["ablations"] = changes
            for name, value in changes.items():
                record(checks, name, value > 1e-7, max_abs_change=value)
        except Exception as exc:
            record(checks, "ablation_checks", False, error=str(exc))

        try:
            h5, m5 = history[:, :, :5], mask[:, :5]
            padded = torch.cat(
                (h5, torch.randn(1, 20, 3, 4) * 9999.0), dim=2
            )
            padded_mask = torch.cat(
                (m5, torch.zeros(1, 3, dtype=torch.bool)), dim=1
            )
            permutation = torch.tensor([2, 4, 0, 3, 1, 7, 6, 5])
            inverse = torch.argsort(permutation)
            with torch.no_grad():
                compact = core(h5, m5)["readout"]
                padded_output = core(padded, padded_mask)["readout"]
                original = core(history, mask)["readout"]
                permuted = core(
                    history[:, :, permutation], mask[:, permutation]
                )["readout"]
            padding_error = float((compact - padded_output[:, :5]).abs().max())
            padded_output_error = float(padded_output[:, 5:].abs().max())
            permutation_error = float(
                (permuted[:, inverse] - original).abs().max()
            )
            report["mask_permutation"] = {
                "padding_active_max_abs": padding_error,
                "padding_output_max_abs": padded_output_error,
                "permutation_max_abs": permutation_error,
                "tolerance": 2e-5,
            }
            record(
                checks,
                "padding_invariance",
                padding_error <= 2e-5 and padded_output_error <= 2e-5,
                active_max_abs=padding_error,
                padded_output_max_abs=padded_output_error,
            )
            record(
                checks,
                "vehicle_permutation_consistency",
                permutation_error <= 2e-5,
                inverse_permute_max_abs=permutation_error,
            )
        except Exception as exc:
            record(checks, "mask_permutation_checks", False, error=str(exc))

        try:
            state = uniform_joint_state(mask[0], 2)
            nonzero = torch.where(state.abs() > 1e-12)[0].tolist()
            node_weights = {
                occupied_count(index, range(NODE_QUBITS)) for index in nonzero
            }
            embedding_weights = {
                occupied_count(index, range(NODE_QUBITS, TOTAL_QUBITS))
                for index in nonzero
            }
            record(
                checks,
                "fixed_hamming_weight_initialization",
                node_weights == {2} and embedding_weights == {3},
                node_weights=sorted(node_weights),
                embedding_weights=sorted(embedding_weights),
                nonzero_basis_states=len(nonzero),
            )
        except Exception as exc:
            record(checks, "fixed_hamming_weight_initialization", False, error=str(exc))

        try:
            report["qnode_identity"] = qnode_metadata(core, history, mask)
            record(
                checks,
                "qnode_identity",
                report["qnode_identity"]["qubits"] == 14
                and report["qnode_identity"]["shots"] == "Shots(total=None)",
                **report["qnode_identity"],
            )
        except Exception as exc:
            record(checks, "qnode_identity", False, error=str(exc))

        try:
            cpu_core = core
            gpu_core = PennyLaneRajMultiJCore(
                rounds=args.rounds,
                device_name="lightning.gpu",
                diff_method="adjoint",
            )
            gpu_core.load_state_dict(cpu_core.state_dict())
            cpu_core.eval()
            gpu_core.eval()
            with torch.no_grad():
                cpu_output = cpu_core(history, mask)
                gpu_output = gpu_core(history, mask)
            output_error = float(
                (cpu_output["readout"] - gpu_output["readout"]).abs().max()
            )
            branch_error = float(
                (cpu_output["branch_readout"] - gpu_output["branch_readout"])
                .abs()
                .max()
            )
            report["cpu_gpu_consistency"] = {
                "cpu_device": "default.qubit",
                "cpu_diff_method": "backprop",
                "gpu_device": "lightning.gpu",
                "gpu_diff_method": "adjoint",
                "readout_max_abs": output_error,
                "branch_readout_max_abs": branch_error,
                "tolerance": 5e-5,
            }
            record(
                checks,
                "cpu_gpu_numerical_consistency",
                output_error <= 5e-5 and branch_error <= 5e-5,
                **report["cpu_gpu_consistency"],
            )
        except Exception as exc:
            record(checks, "cpu_gpu_numerical_consistency", False, error=str(exc))

        try:
            report["circuit"]["representative_specs"] = qnode_metadata(
                core, history, mask
            )
            record(
                checks,
                "circuit_resource_profile",
                report["circuit"]["representative_specs"]["qml_specs"]["num_wires"]
                == 14,
                **report["circuit"]["representative_specs"],
            )
        except Exception as exc:
            record(checks, "circuit_resource_profile", False, error=str(exc))

        try:
            candidates, history_array, mask_array, actual_snr = (
                sind_train_micro_samples(ROOT)
            )
            real_history = torch.as_tensor(history_array, dtype=torch.float32)
            real_mask = torch.as_tensor(mask_array, dtype=torch.bool)
            core.train()
            core.zero_grad(set_to_none=True)
            real_output = core(real_history, real_mask)
            real_loss = (
                real_output["readout"] * torch.randn_like(real_output["readout"])
            ).mean()
            real_loss.backward()
            real_gradients = [
                parameter.grad
                for parameter in core.parameters()
                if parameter.requires_grad and parameter.grad is not None
            ]
            real_gradients_finite = bool(real_gradients) and all(
                torch.isfinite(gradient).all() for gradient in real_gradients
            )
            real_gradients_nonzero = sum(
                float(gradient.abs().sum()) > 1e-16 for gradient in real_gradients
            )
            report["actual_data"] = {
                "dataset": "SinD",
                "sample_paths": [
                    "data/sind/splits/train/samples.npz",
                    "data/sind/isac/train/sensing_cache.npz",
                ],
                "snr_db": actual_snr,
                "split": "train",
                "sample_indices": [int(index) for index in candidates],
                "active_counts": [int(row.sum()) for row in real_mask],
                "labels_opened": False,
                "readout_shape": list(real_output["readout"].shape),
                "finite_loss": bool(torch.isfinite(real_loss)),
                "gradient_tensors": len(real_gradients),
                "missing_gradient_tensors": sum(
                    parameter.requires_grad and parameter.grad is None
                    for parameter in core.parameters()
                ),
                "nonzero_gradient_tensors": int(real_gradients_nonzero),
            }
            record(
                checks,
                "actual_train_micro_smoke",
                tuple(real_output["readout"].shape) == (len(candidates), 8, 64)
                and finite_tensor(real_output["readout"])
                and torch.isfinite(real_loss)
                and real_gradients_finite
                and real_gradients_nonzero > 0,
                **report["actual_data"],
            )
        except Exception as exc:
            record(checks, "actual_train_micro_smoke", False, error=str(exc))

    if not args.skip_profile:
        try:
            profile_core = PennyLaneRajMultiJCore(
                rounds=args.rounds,
                device_name="lightning.gpu",
                diff_method="adjoint",
            )
            if core is not None:
                profile_core.load_state_dict(core.state_dict())
            profile_rows = []
            for batch in (1, 2, 4):
                profile_history, profile_mask = synthetic_history(
                    batch, seed=20261000 + batch
                )
                profile_core.train()
                profile_core.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                monitor = nvidia_peak_start()
                started = time.perf_counter()
                profile_output = profile_core(profile_history, profile_mask)
                profile_loss = profile_output["readout"].square().mean()
                profile_loss.backward()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                nvidia_peak = nvidia_peak_stop(monitor)
                profile_rows.append(
                    {
                        "batch": batch,
                        "readout_shape": list(profile_output["readout"].shape),
                        "step_seconds_forward_backward": elapsed,
                        "torch_peak_allocated_gib": (
                            torch.cuda.max_memory_allocated() / 2**30
                            if torch.cuda.is_available()
                            else 0.0
                        ),
                        "torch_peak_reserved_gib": (
                            torch.cuda.max_memory_reserved() / 2**30
                            if torch.cuda.is_available()
                            else 0.0
                        ),
                        "nvidia_smi_peak": nvidia_peak,
                        "nvidia_smi_after": nvidia_snapshot(),
                        "finite": bool(
                            finite_tensor(profile_output["readout"])
                            and torch.isfinite(profile_loss)
                        ),
                    }
                )
            report["short_profile"] = profile_rows
            record(
                checks,
                "short_profile_b1_b2_b4",
                all(row["finite"] for row in profile_rows)
                and [row["batch"] for row in profile_rows] == [1, 2, 4],
                profile=profile_rows,
            )
        except Exception as exc:
            record(checks, "short_profile_b1_b2_b4", False, error=str(exc))
    else:
        report["short_profile"] = {"skipped": True}
        record(checks, "short_profile_b1_b2_b4", True, skipped=True)

    report["checks"] = checks
    report["passed"] = all(row["passed"] for row in checks.values())
    output_path = (
        ROOT / "reports" / "qgnn" / "raj_pennylane_preflight"
        / "preflight_20260920.json"
    )
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
