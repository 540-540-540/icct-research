"""Full-model initialization and optimizer checks without data or updates."""
import json
import os
from pathlib import Path
import random
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from experiments.qgnn_directed_context.run import (
    CELLS, REPORT, TEMP, Predictor, configuration, optimizer, paired, provenance,
    shared, torch, training,
)


def _rng_state():
    return torch.get_rng_state().clone(), np.random.get_state(), random.getstate()


def _assert_rng_equal(expected):
    actual = _rng_state()
    assert torch.equal(actual[0], expected[0])
    assert actual[1][0] == expected[1][0]
    assert np.array_equal(actual[1][1], expected[1][1])
    assert actual[1][2:] == expected[1][2:]
    assert actual[2] == expected[2]


def _optimizer_settings(opt, model):
    named = dict(model.named_parameters())
    by_id = {id(parameter): name for name, parameter in named.items()}
    parameters = [parameter for group in opt.param_groups for parameter in group['params']]
    identities = [id(parameter) for parameter in parameters]
    assert len(identities) == len(set(identities))
    assert set(identities) == {id(parameter) for parameter in named.values() if parameter.requires_grad}
    return {by_id[id(parameter)]:(group['lr'], group['weight_decay'])
            for group in opt.param_groups for parameter in group['params']}


def main():
    torch.set_num_threads(1)
    assert not torch.cuda.is_available(), 'CPU checks must not see a GPU'
    assert CELLS == ('original', 'directed_context')
    cfg = configuration()
    original = Predictor('original', cfg)
    expected_rng = _rng_state()
    candidate = Predictor('directed_context', cfg)
    _assert_rng_equal(expected_rng)

    old, new = original.state_dict(), candidate.state_dict()
    extra = sorted(set(new) - set(old))
    expected_extra = [f'graph.graph_layers.{layer}.{name}' for layer in range(2)
                      for name in ('receiver_context', 'sender_context')]
    assert extra == expected_extra
    assert set(old).issubset(new)
    assert all(torch.equal(value, new[name]) for name, value in old.items())
    assert sum(new[name].numel() for name in extra) == 2560
    assert all(torch.count_nonzero(new[name]) == 0 for name in extra)

    counts, groups = {}, {}
    for cell, model in ((CELLS[0], original), (CELLS[1], candidate)):
        opt = optimizer(model, 'joint', cfg['graph_lr'])
        assert isinstance(opt, torch.optim.AdamW)
        settings = _optimizer_settings(opt, model)
        named = dict(model.named_parameters())
        quantum = {id(layer.core.weights) for layer in model.graph.graph_layers}
        frozen = [parameter for name, parameter in named.items() if '.llm.' in name and 'lora_' not in name]
        frozen_ids = {id(parameter) for parameter in frozen}
        assert frozen and all(not parameter.requires_grad for parameter in frozen)
        assert all(name not in settings for name, parameter in named.items() if id(parameter) in frozen_ids)
        for name, parameter in named.items():
            if not parameter.requires_grad:
                continue
            expected_lr = cfg['graph_lr'] if name.startswith('graph.') else (3e-5 if 'lora_' in name else 1e-4)
            expected_wd = 0. if parameter.ndim < 2 or id(parameter) in quantum else .01
            assert settings[name] == (expected_lr, expected_wd), (cell, name, settings[name])
        for name in extra:
            if name in named:
                assert settings[name] == (cfg['graph_lr'], .01)
        assert len(opt.state) == 0
        counts[cell] = dict(total=sum(p.numel() for p in named.values()),
                            trainable=sum(p.numel() for p in named.values() if p.requires_grad),
                            graph=sum(p.numel() for p in model.graph.parameters()),
                            quantum=sum(layer.core.weights.numel() for layer in model.graph.graph_layers))
        groups[cell] = [dict(lr=group['lr'], weight_decay=group['weight_decay'],
                             tensors=len(group['params']),
                             parameters=sum(parameter.numel() for parameter in group['params']))
                        for group in opt.param_groups]
    assert counts['directed_context']['trainable'] - counts['original']['trainable'] == 2560
    assert counts['directed_context']['graph'] - counts['original']['graph'] == 2560
    assert counts['directed_context']['quantum'] == counts['original']['quantum'] == 108

    try:
        paired.guard_resume(TEMP/'missing-checkpoint-probe', True)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError('Resume without checkpoint accepted')

    dataset_constructor = training.Dataset
    calls = []
    def refuse_loader(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError('Forbidden split reached Dataset constructor')
    training.Dataset = refuse_loader
    try:
        for split in ('V_confirm', 'test'):
            try:
                shared.dataset(split)
            except ValueError:
                pass
            else:
                raise AssertionError(f'Forbidden split accepted: {split}')
    finally:
        training.Dataset = dataset_constructor
    assert not calls

    assert all(torch.equal(value, candidate.state_dict()[name]) for name, value in original.state_dict().items())
    report = dict(
        passed=True, source=provenance(cfg), counts=counts, optimizer_groups=groups,
        full_shared_initialization_exact=True, subsequent_rng_equal=True,
        extra_parameters=2560, extra_parameters_zero=True,
        optimizer_exact_coverage=True, quantum_weight_decay=0., directed_context_weight_decay=.01,
        frozen_gpt_excluded_from_optimizer=True,
        missing_checkpoint_resume_refused=True,
        forbidden_splits_refused_before_loading=True, dataset_constructor_calls=len(calls),
        optimizer_steps=0, dataset_access=False,
        limitation='Full-model construction and optimizer evidence only; real-task gradients are recorded by D0.',
    )
    training.write_json(REPORT/'training_checks.json', report)
    print(json.dumps({key:value for key, value in report.items() if key != 'source'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
