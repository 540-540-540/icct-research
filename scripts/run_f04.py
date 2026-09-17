#!/usr/bin/env python3
"""Execute the bounded F04 development gate; never starts F05 or reads test."""
import argparse
import hashlib
import json
import sys
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prediction.preflight import (CANDIDATES, SMALL_HEAD, SmallPredictor, balanced_indices,
                                  budget_gate, choose_candidate, confirmation_decision)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_hashes():
    return {str(p.relative_to(ROOT)): digest(p) for p in
            sorted((ROOT/'prediction').glob('*.py')) + [ROOT/'scripts/run_f04.py', ROOT/'scripts/run_f05.py',
            ROOT/'configs/prediction.json', ROOT/'frontend/symbol_dataset.py',
            ROOT/'code/00_remote_shared_dependencies/target_interaction_graph.py']}


def data_hashes(splits):
    paths = [ROOT/'data/f01d/normalization.json']
    for split in splits:
        paths.extend([ROOT/f'data/f01d/labels/{split}.npz', ROOT/f'data/f01d/metadata/{split}.json'])
        paths.extend(ROOT/f'data/f01d/inputs/{split}_snr_{s}.npz' for s in (5,10,15,20))
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def pretrained_hashes():
    paths = [ROOT/'models/gpt2/config.json']
    paths.extend(sorted((ROOT/'models/gpt2').glob('*.safetensors')))
    paths.extend(sorted((ROOT/'models/gpt2').glob('pytorch_model*.bin')))
    if len(paths) == 1:
        raise ValueError('Local GPT-2 model weights missing')
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def load_model_checkpoint(model, path):
    from prediction.training import load_weights
    state = torch.load(path, map_location=next(model.parameters()).device, weights_only=False)
    load_weights(model, state['model'])


def fit_small_trial(spec):
    """One isolated model process, with one allocated GPU and one output directory."""
    from prediction.training import Dataset, fit
    torch.set_num_threads(1)
    train, validation = Dataset('train'), Dataset('V_select')
    c = spec['candidate']
    model = SmallPredictor(c['model'], train.normalization, c['depth'], spec['seed']).to(spec['device'])
    result = fit(model, train, validation, Path(spec['run_dir']), seed=spec['seed'],
                 epochs=spec['epochs'], patience=spec['patience'], graph_lr=c['graph_lr'],
                 batch_size=16, micro_batch=spec['micro_batch'], indices=spec['indices'],
                 snr_mode='nominal' if spec['stage'] == 'selection' else 'balanced',
                 resume=spec['resume'], contract=spec['contract'])
    result.update(candidate=c, seed=spec['seed'], stage=spec['stage'])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--devices', help='Comma-separated allocated GPUs for independent small-head trials')
    p.add_argument('--budget', type=Path, default=ROOT/'configs/training_budget.json')
    p.add_argument('--run-dir', type=Path, default=ROOT/'results/development/f04')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--check-only', action='store_true', help='Validate sealed contract without starting computation')
    p.add_argument('--self-check', action='store_true', help='Run CPU gate and temporal contract checks; no data training')
    args = p.parse_args()
    if args.self_check:
        from prediction.preflight import self_check
        print(json.dumps(self_check(), indent=2)); return 0
    decision_path = ROOT/'reports/preflight_decision.json'
    devices = args.devices.split(',') if args.devices else [args.device]
    if len(set(devices)) != len(devices) or any(not d.startswith('cuda:') for d in devices):
        p.error('Use distinct explicit CUDA devices')
    args.device = devices[0]
    try:
        budget = budget_gate(json.loads(args.budget.read_text()))
    except (OSError, ValueError, KeyError) as exc:
        result = dict(status='INCOMPLETE', stage='budget', reason=str(exc))
        if not args.check_only:
            save(decision_path, result)
        print(json.dumps(result)); return 2
    if args.check_only:
        print(json.dumps(dict(status='READY', budget=str(args.budget), candidates=CANDIDATES,
                              selection=dict(seed=2025, max_origins=512, epochs=8, patience=3),
                              confirmation=dict(seeds=[2023, 2024], max_origins=2048, epochs=12, patience=3),
                              review=dict(max_rounds=1, warmup_epochs=2, joint_epochs=3))))
        return 0
    from prediction.training import Dataset, fit, evaluate, copy_own_weights
    from prediction.model import build_model
    args.run_dir.mkdir(parents=True, exist_ok=True)
    import fcntl
    run_lock = (args.run_dir/'.run.lock').open('w')
    try:
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('Another F04 coordinator already owns this run directory')
    contract_path = args.run_dir/'contract.json'
    metadata = json.loads((ROOT/'data/f01d/metadata/train.json').read_text())
    selection_ids = balanced_indices(metadata, 512)
    confirmation_ids = balanced_indices(metadata, 2048)
    current_contract = dict(revision='A04-F04', candidates=CANDIDATES, small_head=SMALL_HEAD,
                            selection_indices=selection_ids, confirmation_indices=confirmation_ids,
                            selection=dict(seed=2025, epochs=8, patience=3, snr='nominal20'),
                            confirmation=dict(seeds=[2023, 2024], epochs=12, patience=3, snr='balanced'),
                            review=dict(rounds=1, warmup_epochs=2, joint_epochs=3, seeds=[2023,2024]),
                            budget=budget, code_hashes=code_hashes(),
                            data_hashes=data_hashes(('train','V_select','V_confirm')),
                            pretrained_hashes=pretrained_hashes(),
                            metadata_hash=digest(ROOT/'data/f01d/metadata/train.json'))
    if contract_path.exists():
        if not args.resume:
            raise ValueError('F04 run already exists; use --resume to retain all trials')
        if json.loads(contract_path.read_text()) != current_contract:
            raise ValueError('Sealed F04 contract/code changed; do not silently resume a different experiment')
    else:
        save(contract_path, current_contract)
    train, validation = Dataset('train'), Dataset('V_select')
    micro = budget['micro_batch']
    completed_path = args.run_dir/'completed_trials.json'
    completed = json.loads(completed_path.read_text()) if completed_path.exists() else {}

    def audit(event, key, **fields):
        with (args.run_dir/'trial_audit.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(time=time.time(), event=event, trial_id=key, **fields), allow_nan=False)+'\n')

    def check_spent_budget():
        train_seconds = sum(v['elapsed_seconds'] for v in completed.values())
        interrupted_seconds = 0.
        for path in args.run_dir.glob('*/latest.pt'):
            if path.parent.name not in completed:
                saved = torch.load(path, map_location='cpu', weights_only=False)
                interrupted_seconds += saved['progress']['elapsed_seconds']
        eval_seconds = sum(json.loads(path.read_text()).get('evaluation_seconds', 0)
                           for path in args.run_dir.glob('*/confirm_metrics.json'))
        gpu_hours = (train_seconds + interrupted_seconds + eval_seconds) / 3600
        save(args.run_dir/'costs.json', dict(training_and_validation_seconds=train_seconds,
              interrupted_or_running_checkpoint_seconds=interrupted_seconds,
              final_confirmation_seconds=eval_seconds, accounted_gpu_hours=gpu_hours,
              limit_gpu_hours=budget['f04_reserved_gpu_hours'],
              note='Sum across model trials including checkpointed interruptions; startup, checkpoint writing and unsaved interrupted batches are not included'))
        if gpu_hours > budget['f04_reserved_gpu_hours']:
            raise RuntimeError('Measured F04 cost exceeded its sealed reservation; no further trials or PASS')

    def run_batch(jobs):
        pending = [job for job in jobs if job[0] not in completed]
        if pending:
            with ProcessPoolExecutor(max_workers=len(devices), max_tasks_per_child=1,
                                     mp_context=multiprocessing.get_context('spawn')) as pool:
                running = {}
                free = list(devices)
                while pending or running:
                    while pending and free:
                        check_spent_budget()
                        key, c, seed, stage, indices, epochs, patience = pending.pop(0)
                        device = free.pop(0)
                        directory = args.run_dir/key
                        contract = dict(f04_contract=str(contract_path), stage=stage, candidate=c,
                                        input_hashes=train.input_hashes, seed=seed)
                        spec = dict(candidate=c, seed=seed, stage=stage, indices=indices, epochs=epochs,
                                    patience=patience, device=device, micro_batch=micro, run_dir=str(directory),
                                    resume=args.resume and directory.exists(), contract=contract)
                        future = pool.submit(fit_small_trial, spec)
                        audit('started', key, device=device, candidate=c, seed=seed, stage=stage)
                        running[future] = (key, device)
                    finished, _ = wait(running, return_when=FIRST_COMPLETED)
                    for future in finished:
                        key, device = running.pop(future)
                        try:
                            result = future.result()
                        except Exception as exc:
                            audit('failed', key, error=f'{type(exc).__name__}: {exc}')
                            save(args.run_dir/key/'failure.json', dict(status='failed', trial_id=key,
                                  reason=f'{type(exc).__name__}: {exc}'))
                            raise
                        if result['status'] != 'complete' or result.get('numerical_gate') is not True:
                            raise RuntimeError(f'Trial {key} did not complete with a numerical pass')
                        completed[key] = result
                        save(completed_path, completed)
                        audit('completed', key, elapsed_seconds=result['elapsed_seconds'])
                        check_spent_budget()
                        free.append(device)
        return {job[0]: completed[job[0]] for job in jobs}

    def run_trial(key, candidate, seed, stage, indices, epochs, patience, model=None, phase='joint'):
        if key in completed:
            return completed[key]
        check_spent_budget()
        audit('started', key, candidate=candidate, seed=seed, stage=stage, phase=phase)
        model = model if model is not None else SmallPredictor(candidate['model'], train.normalization,
                                                               candidate['depth'], seed).to(args.device)
        directory = args.run_dir/key
        try:
            summary = fit(model, train, validation, directory, seed=seed, epochs=epochs,
                          patience=patience, graph_lr=candidate['graph_lr'], phase=phase,
                          batch_size=16, micro_batch=micro, indices=indices,
                          snr_mode='nominal' if stage == 'selection' else 'balanced',
                          resume=args.resume and directory.exists(),
                          contract=dict(f04_contract=str(contract_path), stage=stage, candidate=candidate,
                                        input_hashes=train.input_hashes, seed=seed))
        except Exception as exc:
            audit('failed', key, error=f'{type(exc).__name__}: {exc}')
            save(directory/'failure.json', dict(status='failed', trial_id=key,
                  reason=f'{type(exc).__name__}: {exc}'))
            raise
        summary.update(candidate=candidate, seed=seed, stage=stage)
        if summary['status'] not in ('completed', 'passed', 'complete'):
            raise RuntimeError(f'Trial {key} incomplete: {summary["status"]}')
        completed[key] = summary
        save(completed_path, completed)
        audit('completed', key, elapsed_seconds=summary['elapsed_seconds'])
        check_spent_budget()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return summary

    try:
        selected_path = args.run_dir/'selected_candidates.json'
        if selected_path.exists():
            selected = json.loads(selected_path.read_text())
        else:
            trials = list(run_batch([('select_'+c['id'], c, 2025, 'selection', selection_ids, 8, 3)
                                     for c in CANDIDATES]).values())
            selected = {name: choose_candidate([t for t in trials if t['candidate']['model'] == name])['candidate']
                        for name in ('qgnn', 'gnn')}
            save(selected_path, selected)
        # Both structures and learning rates are fixed before opening V_confirm.
        confirmed_trials = run_batch([(f'confirm_{seed}_{name}', selected[name], seed, 'confirmation',
                                       confirmation_ids, 12, 3)
                                      for seed in (2023, 2024) for name in ('qgnn', 'gnn')])
        confirm = Dataset('V_confirm')

        def score_pairs(stage, trials, llm=False):
            path = args.run_dir/(stage+'_metrics.json')
            if path.exists():
                return json.loads(path.read_text())
            pairs = {}
            for seed in (2023, 2024):
                pair = {}
                for name in ('qgnn', 'gnn'):
                    candidate = selected[name]
                    model = (build_model(name, train.normalization, args.device, candidate['depth']) if llm else
                             SmallPredictor(name, train.normalization, candidate['depth'], seed).to(args.device))
                    summary = trials[f'{stage}_{seed}_{name}']
                    load_model_checkpoint(model, summary['best_checkpoint'])
                    tick = time.perf_counter()
                    metrics = evaluate(model, confirm, args.device, micro_batch=micro)
                    metrics['evaluation_seconds'] = time.perf_counter()-tick
                    save(args.run_dir/f'{stage}_{seed}_{name}'/'confirm_metrics.json', metrics)
                    check_spent_budget()
                    pair[name] = metrics
                    del model
                    torch.cuda.empty_cache()
                # Full per-SNR denominators must match, not just the headline values.
                for snr in pair['qgnn']['by_snr']:
                    qm, gm = pair['qgnn']['by_snr'][snr], pair['gnn']['by_snr'][snr]
                    for field in qm:
                        if ('count' in field or 'denominator' in field or 'scenes' in field) and qm[field] != gm.get(field):
                            raise ValueError(f'Unequal confirmation denominator: {snr}/{field}')
                denominator_fields = ('origin', 'snr_db', 'ade_targets', 'fde_targets')
                for qr, gr in zip(pair['qgnn']['per_scene'], pair['gnn']['per_scene'], strict=True):
                    if any(qr[k] != gr[k] for k in denominator_fields):
                        raise ValueError('Unequal confirmation target denominators')
                pairs[str(seed)] = pair
            save(path, pairs)
            return pairs

        pairs = score_pairs('confirm', confirmed_trials)
        decision = confirmation_decision(pairs, budget['gates'])
        decision.update(stage='confirmation', selected=selected, contract=str(contract_path))
        save(args.run_dir/'initial_decision.json', decision)
        if decision['status'] == 'REVIEW':
            # Persist exposure before any review; crashes resume this same round.
            save(args.run_dir/'review_exposure.json', dict(round=1, confirmation_is_development=True,
                                                          fixed_candidates=selected))
            review_trials = {}
            for seed in (2023, 2024):
                torch.manual_seed(seed)
                own = build_model('gnn', train.normalization, args.device)
                warm = run_trial(f'review_own_{seed}', selected['gnn'], seed, 'review',
                                 confirmation_ids, 2, 3, model=own, phase='warmup')
                del own
                for name in ('qgnn', 'gnn'):
                    torch.manual_seed(seed)
                    model = build_model(name, train.normalization, args.device, selected[name]['depth'])
                    copy_own_weights(model, warm['best_checkpoint'])
                    key = f'review_{seed}_{name}'
                    review_trials[key] = run_trial(key, selected[name], seed, 'review',
                                                   confirmation_ids, 3, 3, model=model)
                    del model
            pairs = score_pairs('review', review_trials, llm=True)
            decision = confirmation_decision(pairs, budget['gates'])
            decision.update(stage='review', review_round=1, confirmation_is_development=True,
                            selected=selected, contract=str(contract_path))
            if decision['status'] == 'REVIEW':
                decision.update(status='STOP', reason='One allowed GPT-2 review exhausted without PASS')
        check_spent_budget()
        save(decision_path, decision)
        save(args.run_dir/'decision.json', decision)
        if decision['status'] == 'PASS':
            provenance = dict(f04_contract=str(contract_path),
                          f04_contract_sha256=digest(contract_path),
                          f04_decision_sha256=digest(decision_path),
                          code_hashes=code_hashes(), input_hashes=train.input_hashes,
                          normalization_hash=digest(ROOT/'data/f01d/normalization.json'),
                          data_hashes=data_hashes(('train','V_select')),
                          pretrained_hashes=pretrained_hashes(),
                          checkpoint_hashes={v['best_checkpoint']: digest(v['best_checkpoint']) for v in completed.values()})
            sealed = dict(revision='A04', status='PASS', models=selected, budget=budget,
                          budget_accepted=True, provenance=provenance, main_seeds=[2026,2027,2028],
                          training=dict(batch_size=16, micro_batch=micro, warmup_epochs=3,
                                        adapt_epochs=2, joint_epochs=15, patience=4))
            # JSON is a YAML 1.2 subset, retaining a single stdlib serializer.
            save(ROOT/'configs/selected_protocol.yaml', sealed)
        print(json.dumps(decision, indent=2))
        return 0 if decision['status'] == 'PASS' else 3
    except Exception as exc:
        save(decision_path, dict(status='INCOMPLETE', reason=f'{type(exc).__name__}: {exc}',
                                 run_dir=str(args.run_dir), completed_trials=list(completed)))
        raise


if __name__ == '__main__':
    raise SystemExit(main())
