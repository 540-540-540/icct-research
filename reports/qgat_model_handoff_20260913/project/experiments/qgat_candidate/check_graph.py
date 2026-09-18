"""Numerical contract for the five-qubit candidate; CPU correctness and GPU cost."""
import sys, json, time, math
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.qgat_candidate.graph import QGATGraph


def main():
    torch.manual_seed(20260913)
    torch.set_num_threads(2)
    graph = QGATGraph().double()
    x = torch.randn(4, 8, 4, dtype=torch.float64, requires_grad=True)
    m = torch.zeros(4, 8, dtype=torch.bool)
    m[1, 4] = True
    m[2, [0, 3, 7]] = True
    m[3] = True
    d = m & (torch.rand(4, 8) > .3)
    pl = graph.features(x, m, d)
    ref = graph.features(x, m, d, reference=True)
    full = graph.features(x, m, d, reference='full')
    output_error = (pl-ref).abs().max().item()
    full_error = (pl-full).abs().max().item()
    weight = torch.randn_like(pl)
    params = (x, graph.theta, graph.encoder.weight, graph.encoder.bias)
    gp = torch.autograd.grad((pl*weight).sum(), params)
    gr = torch.autograd.grad((ref*weight).sum(), params)
    gf = torch.autograd.grad((full*weight).sum(), params)
    grad_error = max((a-b).abs().max().item() for a,b in zip(gp, gr))
    full_grad_error = max((a-b).abs().max().item() for a,b in zip(gp, gf))
    p = torch.tensor([7, 3, 1, 6, 4, 0, 2, 5])
    perm = graph.features(x[:, p], m[:, p], d[:, p])
    perm_error = (perm-pl[:, p]).abs().max().item()
    dirty = x.detach().clone()
    dirty[~m] = float('nan')
    pad_error = (graph.features(dirty, m, d)-pl).abs().max().item()
    z = pl[m]
    assert output_error < 1e-9 and grad_error < 1e-7
    assert full_error < 1e-9 and full_grad_error < 1e-7
    assert perm_error < 1e-9 and pad_error == 0 and not pl[~m].any()
    assert z[:, :3].norm(dim=-1).max() <= 1+1e-9
    assert torch.allclose(z[:, 4:7].norm(dim=-1), z[:, 3], atol=1e-9)
    assert z[:, [3, 7]].min() >= -1e-9 and z[:, [3, 7]].max() <= 1+1e-9
    assert gp[1].abs().sum() > 0 and gp[2].abs().sum() > 0
    for n in [1, 3]:
        small = graph.features(x[:, :n], m[:, :n], d[:, :n])
        assert small.shape == (4, n, 8) and torch.isfinite(small).all()
    edge = QGATGraph().double()
    with torch.no_grad():
        edge.encoder.weight.zero_(); edge.encoder.bias.zero_(); edge.theta.zero_()
        edge.theta[1, 0] = math.pi
    ex = torch.randn(1, 3, 4, dtype=torch.float64, requires_grad=True)
    em = torch.ones(1, 3, dtype=torch.bool)
    for backend in (False, True, 'full'):
        out = edge.features(ex, em, em, reference=backend)
        assert not out.any()
        eg = torch.autograd.grad(out.sum(), (ex, edge.theta))
        assert all(torch.isfinite(g).all() and not g.any() for g in eg)
    with torch.no_grad():
        edge.theta[1, 0] = math.pi - 4e-6
    above = edge.features(ex, em, em)
    assert above[..., 7].min() > 1e-12 and above.abs().max() > .9
    report = dict(passed=True, backend='PennyLane default.qubit backprop complex128', qubits=5,
                  output_error=output_error, gradient_error=grad_error, permutation_error=perm_error,
                  full_circuit_output_error=full_error, full_circuit_gradient_error=full_grad_error,
                  padding_error=pad_error, quantum_gradient_l1=float(gp[1].abs().sum()),
                  empty_and_variable_nodes=True, near_zero_threshold=1e-12,
                  nonempty_near_zero_fallback_and_recovery=True)
    if torch.cuda.is_available():
        g = QGATGraph().cuda()
        a = torch.randn(1, 20, 8, 4, device='cuda')
        mask = torch.ones(1, 20, 8, dtype=torch.bool, device='cuda')
        costs=[]
        for i in range(4):
            g.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start=time.perf_counter()
            y=g(a,a,mask,mask)
            (y*torch.randn_like(y)).sum().backward()
            torch.cuda.synchronize()
            costs.append(time.perf_counter()-start)
        report.update(graph_forward_backward_seconds=costs[1:], peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
    target=ROOT/'reports/qgat_candidate/graph_checks.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
