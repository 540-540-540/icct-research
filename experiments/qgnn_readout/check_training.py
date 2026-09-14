"""Check full-model pairing and optimizer coverage without optimization or data access."""
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.qgnn_readout.run import (
    CELLS, REPORT, TEMP, Predictor, configuration, guard_resume, optimizer, provenance, torch, training,
)


def main():
    torch.set_num_threads(1)
    cfg = configuration()
    original = Predictor('original', cfg)
    old_rng = torch.get_rng_state().clone()
    candidate = Predictor('x_readout', cfg)
    assert torch.equal(old_rng, torch.get_rng_state())
    old, new = original.state_dict(), candidate.state_dict()
    extra = set(new) - set(old)
    assert extra == {f'graph.graph_layers.{layer}.{head}' for layer in (0, 1) for head in ('x_score', 'x_gate')}
    assert set(old) <= set(new) and all(torch.equal(value, new[name]) for name, value in old.items())
    assert sum(new[name].numel() for name in extra) == 96
    assert all(not new[name].count_nonzero() for name in extra)
    counts = {}
    for cell, model in zip(CELLS, (original, candidate)):
        opt = optimizer(model, 'joint', cfg['graph_lr'])
        named = dict(model.named_parameters())
        expected = {id(p) for p in named.values() if p.requires_grad}
        params = [p for group in opt.param_groups for p in group['params']]
        assert len(params) == len({id(p) for p in params})
        assert {id(p) for p in params} == expected
        settings = {id(p):(g['lr'], g['weight_decay']) for g in opt.param_groups for p in g['params']}
        for layer in model.graph.graph_layers:
            assert settings[id(layer.core.weights)] == (cfg['graph_lr'], 0.)
        for name in extra:
            if name in named:
                assert settings[id(named[name])] == (cfg['graph_lr'], .01)
        assert all(not p.requires_grad for name,p in named.items() if '.llm.' in name and 'lora' not in name)
        counts[cell] = dict(total=sum(p.numel() for p in named.values()),
                            trainable=sum(p.numel() for p in named.values() if p.requires_grad),
                            graph=sum(p.numel() for p in model.graph.parameters()),
                            quantum=sum(layer.core.weights.numel() for layer in model.graph.graph_layers))
    assert counts['original']['trainable'] == 825014
    assert counts['x_readout']['trainable'] == 825110
    try:
        guard_resume(TEMP/'missing-checkpoint-probe', True)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError('Resume without checkpoint accepted')
    for split in ('V_confirm', 'test'):
        from experiments.qgnn_readout.run import shared
        try:
            shared.dataset(split)
        except ValueError:
            pass
        else:
            raise AssertionError('Forbidden dataset accepted')
    report = dict(passed=True, source=provenance(cfg), counts=counts,
                  full_shared_initialization_exact=True, subsequent_rng_equal=True,
                  optimizer_exact_coverage=True, quantum_weight_decay=0., x_head_weight_decay=.01,
                  missing_checkpoint_resume_refused=True, forbidden_splits_refused_before_loading=True,
                  optimizer_steps=0, dataset_access=False)
    training.write_json(REPORT/'training_checks.json', report)
    print(json.dumps({k:v for k,v in report.items() if k != 'source'}), flush=True)


if __name__ == '__main__':
    main()
