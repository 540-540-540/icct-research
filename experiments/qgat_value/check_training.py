"""CPU-only integration checks for the fresh relation-score/value comparison."""
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.qgat_value import run

training, trial = run.training, run.trial
VALUE_NAME = 'graph.value_edge_encoder.weight'


def same_tensors(left, right):
    assert left.keys() == right.keys(), 'Shared tensor names differ'
    for name, value in left.items():
        assert value.dtype == right[name].dtype and torch.equal(value, right[name]), name


def check_initialization(models):
    states = {cell: training.mutable_state(model) for cell, model in models.items()}
    base, value = (states[cell] for cell in run.CELLS)
    assert set(value) - set(base) == {VALUE_NAME}
    assert not set(base) - set(value)
    same_tensors(base, {name: value[name] for name in base})
    assert value[VALUE_NAME].shape == (4, 7) and not value[VALUE_NAME].count_nonzero()
    graphs = {cell: model.graph.state_dict() for cell, model in models.items()}
    base_graph, value_graph = (graphs[cell] for cell in run.CELLS)
    assert 'edge_encoder.weight' in base_graph, 'The shared score encoder must be checked'
    assert set(value_graph) - set(base_graph) == {'value_edge_encoder.weight'}
    same_tensors(base_graph, {name: value_graph[name] for name in base_graph})
    temporal = [{name: tensor for name, tensor in state.items() if name.startswith('temporal.')}
                for state in (base, value)]
    same_tensors(*temporal)
    counts = {}
    for cell, model in models.items():
        assert all(p.device.type == 'cpu' for p in model.parameters())
        assert not hasattr(model.graph, 'forward_history')
        counts[cell] = dict(trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
                           graph=sum(p.numel() for p in model.graph.parameters() if p.requires_grad))
    assert counts == {'relation_D': {'trainable': 610304, 'graph': 4568},
                      'relation_value': {'trainable': 610332, 'graph': 4596}}
    return dict(shared_graph_tensors=len(base_graph), shared_temporal_tensors=len(temporal[0]),
                score_encoder_included=True, extra_parameter=VALUE_NAME,
                extra_shape=[4, 7], extra_initially_zero=True, parameter_counts=counts)


def check_optimizers(models, cfg):
    named_groups = {}
    for cell, model in models.items():
        optimizer = trial.make_optimizer(model, 'joint', cfg['graph_lr'])
        names = {id(p): name for name, p in model.named_parameters()}
        active = {id(p) for p in model.parameters() if p.requires_grad}
        grouped, settings = [], {}
        for group in optimizer.param_groups:
            for parameter in group['params']:
                identifier = id(parameter)
                assert identifier in names and identifier in active
                grouped.append(identifier)
                settings[names[identifier]] = dict(lr=group['lr'], weight_decay=group['weight_decay'])
        assert len(grouped) == len(set(grouped)), 'An optimizer parameter occurs more than once'
        assert set(grouped) == active, 'The optimizer omits an active parameter'
        assert settings['graph.theta'] == dict(lr=.0003, weight_decay=0.)
        assert settings['graph.edge_encoder.weight'] == dict(lr=.0003, weight_decay=.01)
        named_groups[cell] = settings
    base, value = (named_groups[cell] for cell in run.CELLS)
    assert set(value) - set(base) == {VALUE_NAME} and not set(base) - set(value)
    assert all(base[name] == value[name] for name in base)
    assert value[VALUE_NAME] == dict(lr=.0003, weight_decay=.01)
    return dict(shared_parameter_settings_equal=True, no_missing_or_duplicate_parameters=True,
                active_parameter_tensors={cell: len(groups) for cell, groups in named_groups.items()},
                value_settings=value[VALUE_NAME], theta_settings=base['graph.theta'])


def check_schedule(train, cfg):
    assert train.n == cfg['train_origins'] == 5549
    assert cfg['epochs'] == 100 and cfg['batch_size'] == 16 and cfg['micro_batch'] == 8
    assert tuple(cfg['snr_db']) == training.SNRS == (5, 10, 15, 20)
    ids = np.arange(train.n, dtype=np.int64)
    visits = np.zeros((train.n, 4), dtype=np.int16)
    updates = 0
    for epoch in range(cfg['epochs']):
        order, conditions = training.balanced_schedule(train.n, cfg['seed'], epoch, ids)
        assert np.array_equal(np.sort(order), ids)
        assert conditions.shape == order.shape and np.isin(conditions, training.SNRS).all()
        for column, snr in enumerate(training.SNRS):
            visits[order[conditions == snr], column] += 1
        sizes = [len(order[start:start+16]) for start in range(0, len(order), 16)]
        assert sizes == [16] * 346 + [13]
        updates += len(sizes)
    assert updates == 34700 and np.all(visits == 25)
    return dict(epochs=100, origins=5549, updates_per_epoch=347, tail_batch=13,
                expected_optimizer_steps=updates, visits_per_origin_per_snr=25,
                duplicate_or_omitted_origins=False)


def check_resume_guards():
    run.TEMP.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix='check_training_', dir=run.TEMP))
    assert folder.resolve().parent == run.TEMP.resolve()
    assert folder.name.startswith('check_training_')
    checked = []

    def case(name):
        path = folder/name
        path.mkdir()
        return path

    def refuse(label, path, resume):
        try:
            run.guard_resume(path, resume)
        except (ValueError, FileNotFoundError, FileExistsError, RuntimeError):
            checked.append(label)
        else:
            raise AssertionError(f'Resume guard accepted {label}')

    empty = dict(epoch=0, cursor=16, optimizer_steps=1, history=[], best_J=None)
    selected = dict(epoch=1, cursor=0, optimizer_steps=347,
                    history=[dict(epoch=1, J=1.)], best_J=1.)
    try:
        path = case('empty')
        assert run.guard_resume(path, False) is None
        refuse('missing_latest_rejected', path, True)
        path = case('first_epoch_before_validation')
        torch.save(dict(config={}, progress=empty), path/'latest.pt')
        assert run.guard_resume(path, True) == empty
        checked.append('epoch0_without_best_allowed')
        path = case('validated_without_best')
        torch.save(dict(config={}, progress=selected), path/'latest.pt')
        refuse('history_without_best_rejected', path, True)
        torch.save(dict(config={}, progress=selected), path/'best.pt')
        assert run.guard_resume(path, True) == selected
        checked.append('validated_with_best_allowed')
        torch.save(dict(config={'different': True}, progress=selected), path/'best.pt')
        refuse('best_latest_config_mismatch_rejected', path, True)
        path = case('selected_without_best')
        torch.save(dict(config={}, progress=dict(empty, best_J=1.)), path/'latest.pt')
        refuse('best_J_without_best_rejected', path, True)
        for filename in ('latest.pt', 'best.pt', 'history.json', 'config.json'):
            path = case('fresh_'+filename.replace('.', '_'))
            if filename.endswith('.pt'):
                torch.save(dict(config={}, progress=empty), path/filename)
            else:
                (path/filename).write_text('{}')
            refuse('fresh_rejects_'+filename, path, False)
    finally:
        # Windows project cleanup is delegated to the prescribed workspace cleaner.
        if os.name != 'nt':
            assert folder.resolve().parent == run.TEMP.resolve()
            shutil.rmtree(folder)
    return dict(passed=checked, fixtures_removed=not folder.exists(),
                retained_fixture_directory=str(folder.relative_to(ROOT)) if folder.exists() else None)


def check_value_batches(graph_type, seed):
    """Exercise the new value einsum in FP32 with full, micro8 and serial batches."""
    output_tolerance = dict(atol=3e-5, rtol=2e-5)
    gradient_tolerance = dict(atol=3e-5, rtol=3e-4)
    records = {}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        generator = torch.Generator(device='cpu').manual_seed(seed)
        graph = graph_type().float().eval()
        assert all(p.dtype == torch.float32 and p.device.type == 'cpu' for p in graph.parameters())
        with torch.no_grad():
            for encoder in (graph.edge_encoder, graph.value_edge_encoder):
                encoder.weight.copy_(.04 * torch.randn(encoder.weight.shape, generator=generator))
        physical = torch.randn((16, 2, 3, 4), generator=generator)
        physical[..., :2] *= 2
        physical[..., 0] += torch.tensor([0., 5., 11.])
        exists = torch.ones((16, 2, 3), dtype=torch.bool)
        exists[::4, 0, 2] = False
        detected = exists.clone()
        detected[1::3, 1, 1] = False
        inputs = dict(state_hat=physical,
                      standardized_state=physical/torch.tensor([30., 30., 15., 15.]),
                      track_exists=exists, detected=detected)
        weights = torch.randn((16, 2, 3, 128), generator=generator) * .1
        quantum = graph.local_state(torch.zeros((1, 6), dtype=torch.float64),
                                    graph.theta[0].to(torch.float64))
        assert quantum.dtype == torch.complex128
        for length in (16, 13):
            reference_output = reference_gradients = None
            for micro in (1, 8, length):
                graph.zero_grad(set_to_none=True)
                outputs = []
                for start in range(0, length, micro):
                    stop = min(start+micro, length)
                    output = graph(**{name: value[start:stop] for name, value in inputs.items()})
                    assert output.shape == (stop-start, 2, 3, 128) and output.dtype == torch.float32
                    assert torch.isfinite(output).all()
                    (output.mul(weights[start:stop]).sum()/length).backward()
                    outputs.append(output.detach())
                actual = torch.cat(outputs)
                gradients = {}
                for name, parameter in graph.named_parameters():
                    assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
                    gradients[name] = parameter.grad.detach().clone()
                if micro == 1:
                    reference_output, reference_gradients = actual, gradients
                    continue
                torch.testing.assert_close(actual, reference_output, **output_tolerance)
                assert gradients.keys() == reference_gradients.keys()
                differences = {}
                for name in gradients:
                    torch.testing.assert_close(gradients[name], reference_gradients[name], **gradient_tolerance)
                    differences[name] = float((gradients[name]-reference_gradients[name]).abs().max())
                largest = max(differences, key=differences.get)
                records[f'batch{length}_micro{micro}'] = dict(
                    output_max_abs=float((actual-reference_output).abs().max()),
                    gradient_max_abs=differences[largest], gradient_max_abs_parameter=largest)
    return dict(passed=True, input_shape=[16, 2, 3, 4], tested_global_batches=[16, 13],
                tested_micro_batches=[1, 8, 'full'], score_and_value_weights_nonzero=True,
                parameter_dtype='float32', quantum_state_dtype='complex128',
                output_tolerance=output_tolerance, gradient_tolerance=gradient_tolerance,
                comparisons=records)


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('CPU checks require an empty CUDA_VISIBLE_DEVICES before process launch')
    torch.set_num_threads(1)
    assert run.CELLS == ('relation_D', 'relation_value')
    cfg = run.configuration()
    graph_report = json.loads((run.REPORT/'graph_checks.json').read_text())
    assert graph_report.get('passed') and run.graph_check_current(), 'Graph checks are missing or stale'
    source = run.provenance(cfg)
    rejected = []
    for split in ('V_confirm', 'test'):
        try:
            trial.dataset(split)
        except ValueError:
            rejected.append(split)
        else:
            raise AssertionError(f'Forbidden dataset accepted: {split}')
    train, validation = trial.dataset('train'), trial.dataset('V_select')
    assert validation.n == cfg['validation_origins'] == 400
    schedule = check_schedule(train, cfg)
    guards = check_resume_guards()
    models = {cell: run.Predictor(cell, cfg).cpu() for cell in run.CELLS}
    initialization = check_initialization(models)
    optimizers = check_optimizers(models, cfg)
    inputs, truth = train.batch([0], [20], 'cpu')
    with torch.no_grad():
        outputs = {cell: model.eval()(**inputs) for cell, model in models.items()}
    base, value = (outputs[cell] for cell in run.CELLS)
    assert base.keys() == value.keys()
    assert torch.equal(base['origin_eligible'], value['origin_eligible'])
    assert base['origin_eligible'].any()
    for name in base:
        torch.testing.assert_close(base[name], value[name], atol=2e-6, rtol=2e-6)
    initial_error = float((base['prediction']-value['prediction']).abs().max())
    candidate = models['relation_value'].train()
    training.seed_all(cfg['seed'])
    candidate.zero_grad(set_to_none=True)
    loss = training.masked_trajectory_loss(**candidate(**inputs), **truth)
    assert not loss['skip_optimizer'] and torch.isfinite(loss['loss'])
    loss['loss'].backward()
    gradient = candidate.graph.value_edge_encoder.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().max() > 0
    for name, parameter in candidate.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name
    gradient_check = dict(finite=True, nonzero=True, max_abs=float(gradient.abs().max()),
                          norm=float(gradient.norm()), loss=float(loss['loss'].detach()),
                          origin=0, snr_db=20, model_training_mode=True)
    candidate.zero_grad(set_to_none=True)
    batching = check_value_batches(type(candidate.graph), cfg['seed'])
    assert run.graph_check_current(), 'Graph source changed during CPU checks'
    assert source == run.provenance(cfg), 'Run source changed during CPU checks'
    result = dict(passed=True, source=source, cpu_only=True, cuda_work_executed=False,
                  optimizer_steps_executed=0, formal_optimizer_steps_executed=0,
                  formal_training_started=False, graph_checks_current=True,
                  initialization=initialization, optimizer_groups=optimizers,
                  initial_real_prediction=dict(origin=0, snr_db=20, masks_equal=True,
                                               max_abs=initial_error, atol=2e-6, rtol=2e-6),
                  new_value_gradient=gradient_check, value_batching=batching,
                  schedule=schedule, forbidden_splits_rejected=rejected, resume_guards=guards)
    training.write_json(run.REPORT/'checks.json', result)
    print(json.dumps({key: value for key, value in result.items() if key != 'source'},
                     indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
