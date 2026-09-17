"""Synthetic CPU/CUDA circuit and graph checks; no data, checkpoints, or training."""
import argparse
import json
from pathlib import Path
import platform

import pennylane as qml
import torch

from experiments.qgnn_hzy.graph import HZYQGNNGraph
from experiments.qgnn_inherited.graph import InheritedQGNNGraph


def check_tape(core, angles):
    tape = core._circuit.construct((angles, core.weights), {})
    expected = [('StatePrep', list(range(6)), None)]
    expected.extend(('Hadamard', [q], None) for q in range(6))
    for layer in range(3):
        for q in range(6):
            expected.extend([('RZ', [q], angles[..., q]), ('RY', [q], angles[..., q + 6])])
        for q in range(6):
            expected.extend((gate, [q], core.weights[layer, q, p])
                            for p, gate in enumerate(('RZ', 'RY', 'RZ')))
        expected.extend(('CNOT', [q, (q + 1) % 6], None) for q in range(6))
    assert len(tape.operations) == len(expected) == 115
    for operation, (name, wires, parameter) in zip(tape.operations, expected):
        assert operation.name == name and list(operation.wires) == wires
        if parameter is not None:
            torch.testing.assert_close(operation.parameters[0], parameter, atol=0, rtol=0)
    assert len(tape.measurements) == 12
    observables = [qml.PauliZ(q) for q in range(6)]
    observables.extend(qml.PauliZ(q) @ qml.PauliZ((q + 1) % 6) for q in range(6))
    for measurement, observable in zip(tape.measurements, observables):
        assert qml.equal(measurement.obs, observable)


def run_checks():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    model = HZYQGNNGraph(seed=2026).cpu().eval()
    torch.manual_seed(17)
    old = InheritedQGNNGraph(seed=2026).cpu().eval()
    assert model.state_dict().keys() == old.state_dict().keys()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, old.state_dict()[name], atol=0, rtol=0)
    parameters = sum(p.numel() for p in model.parameters())
    assert parameters == sum(p.numel() for p in old.parameters())
    assert [layer.core.weights.numel() for layer in model.graph_layers] == [54, 54]
    initial = {name: value.clone() for name, value in model.state_dict().items()}
    core_checks = []
    for layer in model.graph_layers:
        core = layer.core
        first, second = core._replicate_for_data_parallel(), core._replicate_for_data_parallel()
        assert len({id(core._circuit), id(first._circuit), id(second._circuit)}) == 3
        assert len({id(core._device), id(first._device), id(second._device)}) == 3
        angles = torch.randn(4, 12, requires_grad=True)
        check_tape(core, angles)
        output = core(angles)
        assert output.shape == (4, 12) and torch.isfinite(output).all()
        with torch.no_grad():
            serial = torch.cat([core(row[None]) for row in angles.detach()])
            inference = core(angles.detach())
        torch.testing.assert_close(output, serial)
        torch.testing.assert_close(output, inference)
        assert not inference.requires_grad
        probe = torch.randn_like(output)
        (output * probe).sum().backward()
        assert angles.grad is not None and torch.isfinite(angles.grad).all()
        sensitivity = angles.grad.abs().sum(0)
        assert (sensitivity > 1e-8).all(), sensitivity
        assert core.weights.grad is not None and torch.isfinite(core.weights.grad).all()
        assert core.weights.grad.abs().sum() > 0
        core_checks.append(dict(input_angle_gradient_l1_by_column=sensitivity.tolist(),
                                quantum_weight_gradient_norm=float(core.weights.grad.norm()),
                                quantum_weight_nonzero_probe_gradients=int((core.weights.grad.abs() > 1e-8).sum())))

    model.zero_grad(set_to_none=True)
    state = torch.randn(1, 2, 3, 4)
    state[..., :2] *= 4
    standardized = state / state.new_tensor([20., 20., 15., 15.])
    exists = torch.tensor([[[True, True, False], [False, False, False]]])
    detected = exists.clone()
    output = model(state, standardized, exists, detected)
    assert output.shape == (1, 2, 3, 128) and torch.isfinite(output).all()
    assert torch.count_nonzero(output[~exists]) == 0
    (output * torch.randn_like(output)).sum().backward()
    graph_gradients = []
    for layer in model.graph_layers:
        gradient = layer.core.weights.grad
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
        graph_gradients.append(float(gradient.norm()))
    with torch.no_grad():
        bad_state, bad_z, bad_detection = state.clone(), standardized.clone(), detected.float()
        bad_state[~exists] = bad_z[~exists] = bad_detection[~exists] = float('nan')
        torch.testing.assert_close(model(bad_state, bad_z, exists, bad_detection), output)
        perm = torch.tensor([2, 0, 1])
        permuted = model(state[..., perm, :], standardized[..., perm, :], exists[..., perm], detected[..., perm])
        torch.testing.assert_close(permuted, output[..., perm, :], atol=3e-6, rtol=3e-5)
        short = model(state[:, :1, :2], standardized[:, :1, :2], exists[:, :1, :2], detected[:, :1, :2])
        torch.testing.assert_close(short, output[:, :1, :2], atol=3e-6, rtol=3e-5)
        empty = model(state, standardized, torch.zeros_like(exists), detected)
        assert torch.isfinite(empty).all() and torch.count_nonzero(empty) == 0
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, initial[name], atol=0, rtol=0)

    cuda_checked = False
    if torch.cuda.is_available():
        gpu = HZYQGNNGraph(seed=2026).cuda().eval()
        gpu.load_state_dict(model.state_dict())
        for cpu_layer, gpu_layer in zip(model.graph_layers, gpu.graph_layers):
            cpu_layer.core.zero_grad(set_to_none=True)
            cpu_angles = torch.randn(3, 12, requires_grad=True)
            gpu_angles = cpu_angles.detach().cuda().requires_grad_(True)
            cpu_result, gpu_result = cpu_layer.core(cpu_angles), gpu_layer.core(gpu_angles)
            torch.testing.assert_close(gpu_result.cpu(), cpu_result, atol=3e-6, rtol=3e-5)
            with torch.no_grad():
                inference = gpu_layer.core(gpu_angles.detach())
            assert not inference.requires_grad
            torch.testing.assert_close(inference.cpu(), cpu_result.detach(), atol=3e-6, rtol=3e-5)
            probe = torch.randn_like(cpu_result)
            (cpu_result * probe).sum().backward()
            (gpu_result * probe.cuda()).sum().backward()
            torch.testing.assert_close(gpu_angles.grad.cpu(), cpu_angles.grad, atol=3e-6, rtol=3e-5)
            torch.testing.assert_close(gpu_layer.core.weights.grad.cpu(), cpu_layer.core.weights.grad,
                                       atol=3e-6, rtol=3e-5)
        with torch.no_grad():
            result = gpu(state.cuda(), standardized.cuda(), exists.cuda(), detected.cuda())
        torch.testing.assert_close(result.cpu(), output, atol=3e-6, rtol=3e-5)
        cuda_checked = True
    return dict(passed=True, circuit_revision=model.circuit_revision,
                environment=dict(host=platform.node(), python=platform.python_version(),
                                 torch=torch.__version__, pennylane=qml.__version__),
                graph_parameters=parameters, quantum_parameters_per_layer=54, quantum_parameters_total=108,
                gate_operations_per_core=115, hadamard_count_per_core=6, reupload_rounds=3,
                checks=['exact gate order, wires, and input/weight mapping', 'Z and adjacent ZZ readout order',
                        'independent replica simulators', 'batched/single/no_grad CPU equivalence',
                        'all 12 input columns and each quantum weight tensor have nonzero finite probe gradients',
                        'both quantum cores receive gradients through the graph', 'masks, empty frames, permutation, padding',
                        'same initial parameters and trainable parameter count as inherited graph', 'parameters unchanged'],
                core_checks=core_checks, graph_quantum_gradient_norms=graph_gradients,
                cuda_output_and_gradient_equivalence_checked=cuda_checked,
                optimizer_steps_executed=0, historical_checkpoint_loaded=False,
                dataset_opened=False, confirmation_opened=False, test_opened=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = run_checks()
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding='utf-8')
    print(rendered)
