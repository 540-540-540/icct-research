"""CPU and available-CUDA checks; no optimizer, training loop, or experiment launch."""
import copy

import torch

from experiments.qgnn_inherited.graph import InheritedQGNNGraph, _archive, _base


def run_checks():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    model = InheritedQGNNGraph(seed=2026).cpu().eval()
    for layer in model.graph_layers:
        first = layer.core._replicate_for_data_parallel()
        second = layer.core._replicate_for_data_parallel()
        assert first._circuit is not layer.core._circuit and first._circuit is not second._circuit
        assert first._device is not layer.core._device and first._device is not second._device
    state = torch.randn(1, 2, 3, 4)
    state[..., :2] *= 4
    standardized = state / state.new_tensor([20., 20., 15., 15.])
    exists = torch.tensor([[[True, True, False], [False, False, False]]])
    detected = exists.clone()
    output = model(state, standardized, exists, detected)
    assert output.shape == (1, 2, 3, 128)
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output[~exists]) == 0
    probe = torch.randn_like(output)
    (output * probe).sum().backward()
    for layer in model.graph_layers:
        grad = layer.core.weights.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0

    with torch.no_grad():
        bad_state, bad_z = state.clone(), standardized.clone()
        bad_state[~exists] = float('nan')
        bad_z[~exists] = float('nan')
        bad_detected = detected.float()
        bad_detected[~exists] = float('nan')
        torch.testing.assert_close(model(bad_state, bad_z, exists, bad_detected), output)
        perm = torch.tensor([2, 0, 1])
        permuted = model(state[..., perm, :], standardized[..., perm, :], exists[..., perm], detected[..., perm])
        torch.testing.assert_close(permuted, output[..., perm, :], atol=3e-6, rtol=3e-5)
        short = model(state[:, :1, :2], standardized[:, :1, :2], exists[:, :1, :2], detected[:, :1, :2])
        torch.testing.assert_close(short, output[:, :1, :2], atol=3e-6, rtol=3e-5)
        empty = model(state, standardized, torch.zeros_like(exists), detected)
        assert torch.isfinite(empty).all() and torch.count_nonzero(empty) == 0

    # Independently compose the original layer calls, with identical parameters,
    # and compare the adapter's valid-frame output and every parameter gradient.
    reference_input = copy.deepcopy(model.input)
    reference_layers = []
    for original, role, offset in zip(model.graph_layers, ('physical', 'contextual'), (103, 101)):
        reference = _archive.PhysicsAlignedMessageLayer(_base.ForecasterConfig(), 'quantum', role, 2026 + offset)
        reference.load_state_dict(original.state_dict())
        reference.eval()
        reference_layers.append(reference)
    x = state[:, 0, :2]
    z = standardized[:, 0, :2]
    flags = torch.ones(1, 2, 1)
    h = reference_input(torch.cat((z, flags, flags), -1))
    edges, distance = _base.build_edge_features(x[..., :2], x[..., 2:])
    for layer in reference_layers:
        h, _ = layer(h, edges, distance <= 45.)
    model.zero_grad(set_to_none=True)
    adapted = model(state[:, :1, :2], standardized[:, :1, :2], exists[:, :1, :2], detected[:, :1, :2])[:, 0]
    torch.testing.assert_close(adapted, h, atol=1e-6, rtol=1e-5)
    weight = probe[:, 0, :2]
    (adapted * weight).sum().backward()
    (h * weight).sum().backward()
    for actual_module, reference_module in zip([model.input, *model.graph_layers], [reference_input, *reference_layers]):
        for (name, actual), (other_name, expected) in zip(actual_module.named_parameters(), reference_module.named_parameters()):
            assert name == other_name
            assert actual.grad is not None and expected.grad is not None, name
            torch.testing.assert_close(actual.grad, expected.grad, atol=3e-6, rtol=3e-5, msg=name)
    checks = ['replicas have independent quantum nodes and devices', 'CPU forward/backward', 'both quantum-core gradients',
                'masks and empty frames', 'permutation and padding',
                'archived-layer output and gradient equivalence']
    if torch.cuda.is_available():
        gpu_model = InheritedQGNNGraph(seed=2026).cuda().eval()
        for layer in gpu_model.graph_layers:
            cpu_core = _archive.DualAxisPennyLanePQC(depth=3)
            cpu_core.load_state_dict({key: value.cpu() for key, value in layer.core.state_dict().items()})
            cpu_angles = torch.randn(3, 12, requires_grad=True)
            gpu_angles = cpu_angles.detach().cuda().requires_grad_(True)
            cpu_result = cpu_core(cpu_angles)
            gpu_result = layer.core(gpu_angles)
            torch.testing.assert_close(gpu_result.cpu(), cpu_result, atol=3e-6, rtol=3e-5)
            with torch.no_grad():
                inference_result = layer.core(gpu_angles.detach())
            assert not inference_result.requires_grad
            torch.testing.assert_close(inference_result.cpu(), cpu_result.detach(), atol=3e-6, rtol=3e-5)
            probe_core = torch.randn_like(cpu_result)
            (cpu_result * probe_core).sum().backward()
            (gpu_result * probe_core.cuda()).sum().backward()
            for gpu_grad, cpu_grad in ((layer.core.weights.grad, cpu_core.weights.grad),
                                       (gpu_angles.grad, cpu_angles.grad)):
                assert gpu_grad is not None and torch.isfinite(gpu_grad).all() and gpu_grad.abs().sum() > 0
                torch.testing.assert_close(gpu_grad.cpu(), cpu_grad, atol=3e-6, rtol=3e-5)
        checks.append('both CUDA quantum cores match archived CPU outputs and angle/weight gradients')
        checks.append('CUDA no_grad core outputs match gradient-enabled execution')
    return dict(passed=True, checks=checks, cuda_available=torch.cuda.is_available(), optimizer_steps_executed=0)


if __name__ == '__main__':
    print(run_checks())
