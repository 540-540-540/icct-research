"""CPU full-model/optimizer checks; no dataset loading or optimizer steps."""
import json
import os
from pathlib import Path
import random
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from experiments.qgnn_motionframe.run import (
    CELLS, REPORT, TEMP, Predictor, configuration, guard_resume, optimizer, prior,
    provenance, shared, torch, training,
)
from experiments.qgat_candidate.run_formal import Predictor as CanonicalGNN


def rng_state():
    return torch.get_rng_state().clone(), np.random.get_state(), random.getstate()


def check_rng(expected):
    current = rng_state()
    assert torch.equal(current[0], expected[0])
    assert current[1][0] == expected[1][0]
    assert np.array_equal(current[1][1], expected[1][1])
    assert current[1][2:] == expected[1][2:]
    assert current[2] == expected[2]


def check_state(actual, expected):
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        assert torch.equal(actual[name], value), name


def optimizer_settings(opt, model):
    named = dict(model.named_parameters())
    by_id = {id(parameter): name for name, parameter in named.items()}
    parameters = [parameter for group in opt.param_groups for parameter in group['params']]
    identities = [id(parameter) for parameter in parameters]
    assert len(identities) == len(set(identities)), 'Duplicated optimizer parameter'
    assert set(identities) == {id(parameter) for parameter in named.values() if parameter.requires_grad}
    return {by_id[id(parameter)]: (group['lr'], group['weight_decay'])
            for group in opt.param_groups for parameter in group['params']}


def main():
    torch.set_num_threads(1)
    assert not torch.cuda.is_available(), 'CPU checks must not see a GPU'
    assert tuple(CELLS) == ('motion_frame', 'gnn')
    cfg = configuration()
    original = prior.Predictor('original', prior.configuration())
    original_rng = rng_state()
    candidate = Predictor('motion_frame', cfg)
    check_rng(original_rng)
    check_state(candidate.state_dict(), original.state_dict())
    candidate_names, original_names = dict(candidate.named_parameters()), dict(original.named_parameters())
    assert candidate_names.keys() == original_names.keys()
    extra_parameters = sum(parameter.numel() for parameter in candidate_names.values()) - sum(
        parameter.numel() for parameter in original_names.values())
    assert extra_parameters == 0

    gnn = Predictor('gnn', cfg)
    gnn_rng = rng_state()
    check_state(gnn.temporal.state_dict(), original.temporal.state_dict())
    canonical = CanonicalGNN('gnn', 2026, cfg)
    check_rng(gnn_rng)
    check_state(gnn.state_dict(), canonical.state_dict())
    del canonical

    counts, group_records = {}, {}
    for cell, model in zip(CELLS, (candidate, gnn)):
        opt = optimizer(model, 'joint', cfg['graph_lr'])
        assert isinstance(opt, torch.optim.AdamW)
        settings = optimizer_settings(opt, model)
        named = dict(model.named_parameters())
        quantum = {id(layer.core.weights) for layer in model.graph.graph_layers} if cell == 'motion_frame' else set()
        frozen_gpt = [parameter for name, parameter in named.items() if '.llm.' in name and 'lora_' not in name]
        assert frozen_gpt and all(not parameter.requires_grad for parameter in frozen_gpt)
        assert all(name not in settings for name in named if '.llm.' in name and 'lora_' not in name)
        for name, (learning_rate, weight_decay) in settings.items():
            parameter = named[name]
            expected_lr = cfg['graph_lr'] if name.startswith('graph.') else (3e-5 if 'lora_' in name else 1e-4)
            expected_wd = 0. if parameter.ndim < 2 or id(parameter) in quantum else .01
            assert (learning_rate, weight_decay) == (expected_lr, expected_wd), (cell, name, learning_rate, weight_decay)
        if cell == 'gnn':
            standard = shared.ORIGINAL_OPTIMIZER(model, 'joint', cfg['graph_lr'])
            assert type(opt) is type(standard) and opt.defaults == standard.defaults
            assert settings == optimizer_settings(standard, model)
            del standard
        counts[cell] = dict(total=sum(parameter.numel() for parameter in named.values()),
                            trainable=sum(parameter.numel() for parameter in named.values() if parameter.requires_grad),
                            graph=sum(parameter.numel() for parameter in model.graph.parameters()),
                            quantum=sum(parameter.numel() for parameter in named.values() if id(parameter) in quantum),
                            frozen_gpt=sum(parameter.numel() for parameter in frozen_gpt))
        group_records[cell] = [dict(lr=group['lr'], weight_decay=group['weight_decay'],
                                    tensors=len(group['params']), parameters=sum(parameter.numel() for parameter in group['params']))
                               for group in opt.param_groups]
        assert len(opt.state) == 0, 'No optimizer state should exist before stepping'
        del opt
    assert counts['motion_frame']['graph'] == 219278
    assert counts['motion_frame']['trainable'] == 825014

    missing = TEMP / 'missing-checkpoint-probe'
    assert not (missing / 'latest.pt').exists()
    try:
        guard_resume(missing, True)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError('Resume without checkpoint accepted')

    dataset_constructor = shared.training.Dataset
    constructor_calls = []
    def refuse_loader(*args, **kwargs):
        constructor_calls.append((args, kwargs))
        raise AssertionError('Forbidden split reached the dataset constructor')
    shared.training.Dataset = refuse_loader
    try:
        for split in ('V_confirm', 'test'):
            try:
                shared.dataset(split)
            except ValueError:
                pass
            else:
                raise AssertionError(f'Forbidden split accepted: {split}')
    finally:
        shared.training.Dataset = dataset_constructor
    assert not constructor_calls

    # Recheck full states after optimizer construction, including frozen weights.
    check_state(candidate.state_dict(), original.state_dict())
    check_state(gnn.temporal.state_dict(), original.temporal.state_dict())
    report = dict(
        passed=True, source=provenance(cfg),
        currentcode={str(Path(__file__).relative_to(ROOT)): Path(__file__).read_text(encoding='utf-8')},
        counts=counts, optimizer_groups=group_records, extra_parameters=extra_parameters,
        full_qgnn_initialization_exact=True, full_state_includes_frozen_gpt=True,
        qgnn_subsequent_rng_equal=True, rng_backends=['torch_cpu', 'numpy', 'python'],
        gnn_temporal_matches_original_qgnn=True, gnn_full_state_matches_canonical=True,
        gnn_subsequent_rng_matches_canonical=True,
        optimizer_exact_coverage=True, all_learning_rates_and_weight_decay_verified=True,
        quantum_weight_decay=0., gnn_optimizer_matches_standard=True,
        frozen_gpt_excluded_from_optimizer=True, initialization_unchanged_after_optimizer_construction=True,
        missing_checkpoint_resume_refused=True, forbidden_splits_refused_before_loading=True,
        dataset_constructor_calls=len(constructor_calls), optimizer_steps=0,
        dataset_access=False, device='cpu', cuda_available=False,
    )
    training.write_json(REPORT / 'training_checks.json', report)
    print(json.dumps({key: value for key, value in report.items() if key not in ('source', 'currentcode')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
