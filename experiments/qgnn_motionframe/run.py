"""Motion-frame QGNN and fresh strong GNN under the completed original Q protocol."""
import argparse
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.qgnn_inherited.run import shared, training, optimizer as quantum_optimizer
from experiments.qgnn_readout import run as prior
from experiments.qgnn_motionframe.graph import MotionFrameQGNNGraph
from prediction.classical import GNNGraph

CELLS = ('motion_frame', 'gnn')
REPORT = ROOT/'reports/qgnn_motionframe'
RESULTS = ROOT/'results/qgnn_motionframe/seed2026'
TEMP = ROOT/'.codex-work/qgnn-motionframe-smoke-20260914'
BASE_EVALUATE = training.evaluate


def configuration():
    cfg = json.loads((ROOT/'configs/qgnn_motionframe.json').read_text(encoding='utf-8-sig'))
    fixed = dict(seed=2026, cells=list(CELLS), devices=dict(zip(CELLS, ('cuda:0', 'cuda:1'))),
                 graph_seed_offset=200000, temporal_seed_offset=100000, epochs=150, patience=20,
                 min_delta=0, train_origins=5549, validation_origins=400, batch_size=16,
                 micro_batch=16, validation_micro_batch=16, graph_lr=.0003,
                 snr_db=[5, 10, 15, 20], allowed_splits=['train', 'V_select'])
    for key, value in fixed.items():
        if cfg[key] != value:
            raise ValueError(f'Frozen motion-frame setting changed: {key}')
    if cfg['low_speed_fallback_mps'] != 1. or cfg['original_reference'] != 'results/qgnn_readout/seed2026/original':
        raise ValueError('Encoding or original reference changed')
    return cfg


class Predictor(nn.Module):
    def __init__(self, cell, cfg):
        super().__init__()
        if cell not in CELLS:
            raise ValueError('Unknown motion-frame trial cell')
        seed = cfg['seed'] + cfg['graph_seed_offset']
        training.seed_all(seed)
        self.graph = MotionFrameQGNNGraph(seed=seed) if cell == 'motion_frame' else GNNGraph()
        training.seed_all(cfg['seed'] + cfg['temporal_seed_offset'])
        self.temporal = shared.TrajectoryPredictor(128)

    def forward(self, state_hat, standardized_state, track_exists, detected):
        args = (state_hat, standardized_state, track_exists, detected)
        return self.temporal(*args, self.graph(*args))


def optimizer(model, phase, graph_lr):
    if isinstance(model.graph, MotionFrameQGNNGraph):
        return quantum_optimizer(model, phase, graph_lr)
    return shared.ORIGINAL_OPTIMIZER(model, phase, graph_lr)


def provenance(cfg):
    old_cfg = prior.configuration()
    source = prior.provenance(old_cfg)
    path = ROOT/cfg['original_reference']
    saved = json.loads((path/'config.json').read_text())
    summary = json.loads((path/'summary.json').read_text())
    if saved['contract']['source'] != source or summary['status'] != 'complete':
        raise ValueError('Completed original Q source or assets changed; reference reuse refused')
    for key in ('seed', 'graph_seed_offset', 'temporal_seed_offset', 'epochs', 'patience', 'min_delta',
                'train_origins', 'validation_origins', 'batch_size', 'micro_batch',
                'validation_micro_batch', 'graph_lr', 'snr_db', 'allowed_splits'):
        if cfg[key] != old_cfg[key]:
            raise ValueError(f'Original reference does not match new protocol: {key}')
    for key in ('seed', 'epochs', 'patience', 'graph_lr', 'batch_size', 'micro_batch'):
        if saved[key] != cfg[key]:
            raise ValueError(f'Original saved training configuration differs: {key}')
    if saved['indices'] != list(range(cfg['train_origins'])) or saved['phase'] != 'joint' or saved['snr_mode'] != 'balanced':
        raise ValueError('Original training coverage or schedule differs')
    if saved['validation_hashes']['selected_origins'] != list(range(cfg['validation_origins'])):
        raise ValueError('Original validation coverage differs')
    if summary['optimizer_steps'] != len(summary['history'])*347 or not summary['numerical_gate']:
        raise ValueError('Original training completion is not valid')
    code = ['experiments/qgnn_motionframe/run.py', 'experiments/qgnn_motionframe/graph.py',
            'experiments/qgnn_motionframe/check_graph.py', 'experiments/qgnn_motionframe/check_training.py',
            'prediction/classical.py', 'experiments/qgat_candidate/run_formal.py',
            'experiments/qgat_candidate/graph.py']
    source['code_text'].update({p:(ROOT/p).read_text() for p in code})
    source['configuration'] = cfg
    source['original_reuse'] = dict(path=cfg['original_reference'], source_matches_completed_run=True,
        best_metrics=summary['best_metrics'], epochs=len(summary['history']),
        optimizer_steps=summary['optimizer_steps'], stopped_early=summary['stopped_early'],
        input_hashes=saved['input_hashes'], validation_hashes=saved['validation_hashes'])
    for name in ('config.json', 'summary.json', 'best.pt', 'latest.pt',
                 f'validation_epoch_{summary["best_metrics"]["epoch"]:02d}.json'):
        info = (path/name).stat()
        source['asset_modification_records'][str((path/name).relative_to(ROOT))] = dict(bytes=info.st_size, mtime_ns=info.st_mtime_ns)
    return source


def guard_resume(run_dir, resume):
    run_dir = Path(run_dir)
    latest, best = run_dir/'latest.pt', run_dir/'best.pt'
    if not resume:
        if any((run_dir/n).exists() for n in ('latest.pt', 'best.pt', 'history.json', 'config.json')):
            raise FileExistsError('Existing run; use its explicit --resume')
        return None
    if not latest.is_file():
        raise FileNotFoundError('No latest checkpoint; fresh-start fallback refused')
    saved = torch.load(latest, map_location='cpu', weights_only=False)
    progress = saved['progress']
    if (progress['history'] or progress['best_J'] is not None or progress['epoch']) and not best.is_file():
        raise FileNotFoundError('Validated run is missing its best checkpoint')
    if best.is_file() and torch.load(best, map_location='cpu', weights_only=False)['config'] != saved['config']:
        raise ValueError('Best/latest configuration mismatch')
    return progress


class SmokeInterrupted(Exception):
    pass


def worker(args, cfg, source, run_root, report):
    cell = args.worker
    if os.environ.get('CUDA_VISIBLE_DEVICES') != cfg['devices'][cell].split(':')[1] or torch.cuda.device_count() != 1:
        raise ValueError('Worker must see only its assigned physical GPU')
    torch.cuda.set_device(0)
    contract = json.loads((report/'protocol.json').read_text())
    if contract['source'] != source:
        raise ValueError('Worker source changed')
    run_dir, path = run_root/cell, report/'jobs'/f'{cell}.json'
    status = dict(status='initializing', cell=cell, pid=os.getpid(), started_at=time.time(), device=cfg['devices'][cell])
    training.write_json(path, status)
    try:
        starting = guard_resume(run_dir, args.resume)
        initial_steps = starting['optimizer_steps'] if starting else 0
        train, valid = shared.dataset('train'), shared.dataset('V_select')
        assert (train.n, valid.n) == (cfg['train_origins'], cfg['validation_origins'])
        ids = shared.origins(train.n, 17 if args.smoke else train.n)
        valid = shared.Subset(valid, shared.origins(valid.n, 17 if args.smoke else valid.n))
        reference = source['original_reuse']
        if train.input_hashes != reference['input_hashes']:
            raise ValueError('Training inputs differ from reused original Q')
        for key, value in reference['validation_hashes'].items():
            if key != 'selected_origins' and valid.input_hashes[key] != value:
                raise ValueError('Validation inputs differ from reused original Q')
        batches = []
        if args.smoke:
            original_batch = valid.batch
            def recorded_batch(ids, *rest, **kwargs):
                batches.append(len(ids))
                return original_batch(ids, *rest, **kwargs)
            valid.batch = recorded_batch
        def evaluate(model, dataset, device=None, micro_batch=1, **kwargs):
            return BASE_EVALUATE(model, dataset, device, micro_batch=cfg['validation_micro_batch'], **kwargs)
        training.evaluate, training.configure_phase = evaluate, optimizer
        model = Predictor(cell, cfg).to('cuda:0')
        options = dict(seed=cfg['seed'], epochs=1 if args.smoke else cfg['epochs'], patience=cfg['patience'],
                       graph_lr=cfg['graph_lr'], phase='joint', batch_size=cfg['batch_size'],
                       micro_batch=cfg['micro_batch'], indices=ids, snr_mode='balanced', contract=contract)
        status.update(status='training', historical_trained_checkpoint_loaded=False, train_origins=len(ids),
                      V_select_origins=valid.n, wall_training_started_at=time.time())
        training.write_json(path, status)
        resume = args.resume
        if args.smoke and not resume:
            save = training.save_checkpoint
            def interrupt_after_first_update(target, payload):
                save(target, payload)
                if Path(target).name == 'latest.pt' and payload['progress']['optimizer_steps'] == 1:
                    raise SmokeInterrupted()
            training.save_checkpoint = interrupt_after_first_update
            try:
                training.fit(model, train, valid, run_dir, resume=False, **options)
                raise AssertionError('Smoke interruption failed')
            except SmokeInterrupted:
                progress = guard_resume(run_dir, True)
                assert progress['cursor'] == 16 and not (run_dir/'best.pt').exists()
                status['prevalidation_resume_checked'] = True
            finally:
                training.save_checkpoint = save
            del model
            model = Predictor(cell, cfg).to('cuda:0')
            resume = True
        result = training.fit(model, train, valid, run_dir, resume=resume, **options)
        latest = torch.load(run_dir/'latest.pt', map_location='cpu', weights_only=False)
        best = torch.load(run_dir/'best.pt', map_location='cpu', weights_only=False)
        assert result['numerical_gate']
        assert result['optimizer_steps'] == len(result['history'])*int(np.ceil(len(ids)/cfg['batch_size']))
        assert all(torch.equal(v.detach().cpu(), best['model'][k]) for k,v in training.mutable_state(model).items())
        assert all(torch.isfinite(v).all() for v in latest['model'].values())
        for state in latest['optimizer']['state'].values():
            assert int(state['step']) == result['optimizer_steps']
            assert all(torch.isfinite(v).all() for v in state.values() if isinstance(v, torch.Tensor))
        added = result['optimizer_steps'] - initial_steps
        if args.smoke:
            assert result['optimizer_steps'] == 2 and len(result['history']) == 1
            assert batches == ([16, 1]*4 if added else []), batches
        status.update(status='complete', finished_at=time.time(), training=result,
                      additional_optimizer_steps_this_process=added, best_weights_restored=True,
                      validation_batches=batches if args.smoke else None,
                      peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
        training.write_json(path, status)
    except BaseException as exc:
        status.update(status='failed', error=f'{type(exc).__name__}: {exc}', finished_at=time.time())
        training.write_json(path, status)
        raise


def summarize(states, cfg, run_root, smoke):
    best = {cell:json.loads((run_root/cell/f'validation_epoch_{state["training"]["best_metrics"]["epoch"]:02d}.json').read_text())
            for cell,state in states.items()}
    identities = [[(r['origin'], r['snr_db'], r['ade_targets'], r['fde_targets']) for r in best[c]['per_scene']] for c in CELLS]
    assert identities[0] == identities[1] and len(identities[0]) == (17 if smoke else 400)*4
    reference = json.loads((ROOT/cfg['original_reference']/'summary.json').read_text())
    comparisons = dict(motion_vs_fresh_gnn={k:100*(1-best['motion_frame'][k]/best['gnn'][k]) for k in ('ADE', 'FDE')})
    if not smoke:
        ref_best = json.loads((ROOT/cfg['original_reference']/f'validation_epoch_{reference["best_metrics"]["epoch"]:02d}.json').read_text())
        assert [(r['origin'], r['snr_db'], r['ade_targets'], r['fde_targets']) for r in ref_best['per_scene']] == identities[0]
        best['original_reused'] = ref_best
        comparisons['motion_vs_original'] = {k:100*(1-best['motion_frame'][k]/ref_best[k]) for k in ('ADE', 'FDE')}
        comparisons['original_vs_fresh_gnn'] = {k:100*(1-ref_best[k]/best['gnn'][k]) for k in ('ADE', 'FDE')}
    return dict(status='complete', smoke_only=smoke, seed=cfg['seed'], jobs=states, best_J_epoch_metrics=best,
                improvement_percent=comparisons, original_reused=reference['best_metrics'],
                common_denominators_verified=True,
                confirmation_opened=False, test_opened=False, limitation=cfg['limitation'])


def coordinate(args, cfg, source, run_root, report):
    report.mkdir(parents=True, exist_ok=True)
    with (report/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = dict(source=source, mode='isolated-smoke' if args.smoke else 'motion-and-fresh-gnn150-patience20',
                        confirmation_opened=False, test_opened=False)
        path = report/'protocol.json'
        if path.exists():
            if not args.resume:
                raise FileExistsError('Run exists; use --resume explicitly')
            if json.loads(path.read_text()) != protocol:
                raise ValueError('Resume source contract mismatch')
        elif args.resume:
            raise FileNotFoundError('No paired run to resume')
        else:
            training.write_json(path, protocol)
        active, logs, states = {}, [], {}
        status = dict(status='running', pid=os.getpid(), started_at=time.time(), devices=cfg['devices'])
        training.write_json(report/'coordinator.json', status)
        try:
            for cell in CELLS:
                logfile = report/'logs'/f'{cell}.log'
                logfile.parent.mkdir(parents=True, exist_ok=True)
                log = logfile.open('a', buffering=1)
                logs.append(log)
                cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--smoke' if args.smoke else '--run', '--worker', cell]
                if args.resume:
                    cmd.append('--resume')
                active[cell] = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                                env=dict(os.environ, CUDA_VISIBLE_DEVICES=cfg['devices'][cell].split(':')[1]))
            while active:
                time.sleep(1)
                for cell, child in list(active.items()):
                    if child.poll() is None:
                        continue
                    del active[cell]
                    if child.returncode:
                        raise RuntimeError(f'{cell} failed: '+ '\n'.join((report/'logs'/f'{cell}.log').read_text().splitlines()[-12:]))
                    states[cell] = json.loads((report/'jobs'/f'{cell}.json').read_text())
                    assert states[cell]['status'] == 'complete'
            if source != provenance(cfg):
                raise ValueError('Source or assets modified during trial')
            summary = summarize(states, cfg, run_root, args.smoke)
            training.write_json(report/'summary.json', summary)
            if not args.smoke:
                lines = ['# 行驶方向关系编码与强 GNN 对照结果', '',
                         'seed2026；仅 V_select；上限 150 轮，patience=20。按最低 J 的同一检查点分别报告 ADE、FDE。', '',
                         '| 模型 | 实际轮数 | 最佳轮次 | ADE | FDE | 早停 |',
                         '|---|---:|---:|---:|---:|---|']
                for cell in CELLS:
                    job = states[cell]['training']
                    best = job['best_metrics']
                    lines.append(f'| {cell} | {len(job["history"])} | {best["epoch"]} | {best["ADE"]:.6f} | {best["FDE"]:.6f} | {job["stopped_early"]} |')
                ref = source['original_reuse']
                b = ref['best_metrics']
                lines.append(f'| original_reused | {ref["epochs"]} | {b["epoch"]} | {b["ADE"]:.6f} | {b["FDE"]:.6f} | {ref["stopped_early"]} |')
                lines.append('')
                for name, gain in summary['improvement_percent'].items():
                    lines.append(f'{name}：ADE 改善 {gain["ADE"]:.3f}%，FDE 改善 {gain["FDE"]:.3f}%（负值表示退步）。')
                lines.extend(['',
                              '这是单种子开发结果。若达到 150 轮上限，须另行检查曲线判断训练是否充分；不能仅凭进程完成宣称收敛。',
                              '原 QGNN 复用已完成且来源、数据、训练规则完全匹配的 150/patience20 结果；强 GNN 本轮从头训练。',
                              '此强 GNN 对照用于预测性能比较，不单独隔离量子计算必要性。未训练 QGAT，未读取 V_confirm 或 test。'])
                (report/'RESULT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
            status['status'] = 'complete'
            return summary
        except BaseException as exc:
            for child in active.values():
                if child.poll() is None:
                    child.terminate()
            for child in active.values():
                child.wait()
            status.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            status['finished_at'] = time.time()
            training.write_json(report/'coordinator.json', status)
            for log in logs:
                log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--smoke', action='store_true')
    modes.add_argument('--run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--worker', choices=CELLS)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = configuration()
    source = provenance(cfg)
    checks = json.loads((REPORT/'training_checks.json').read_text())
    graph = json.loads((REPORT/'graph_checks.json').read_text())
    if not checks['passed'] or checks['source'] != source or not graph['passed']:
        raise ValueError('Current implementation checks required')
    if not graph['graph_sources'] or any((ROOT/p).read_text() != s for p,s in graph['graph_sources'].items()):
        raise ValueError('Graph checks are stale')
    if args.run:
        smoke = json.loads((REPORT/'smoke_checks.json').read_text())
        if not smoke['passed'] or smoke['source'] != source or smoke.get('completed_resume_additional_steps') != dict.fromkeys(CELLS, 0):
            raise ValueError('Current complete GPU smoke/resume required')
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    run_root, report = (TEMP/'results', TEMP/'reports') if args.smoke else (RESULTS, REPORT/'seed2026')
    if args.worker:
        worker(args, cfg, source, run_root, report)
        return
    if torch.cuda.device_count() < 2:
        raise RuntimeError('Two GPUs required')
    summary = coordinate(args, cfg, source, run_root, report)
    if args.smoke:
        checks = dict(passed=True, source=source, summary=summary, formal_optimizer_steps=0)
        if args.resume:
            checks['completed_resume_additional_steps'] = {c:s['additional_optimizer_steps_this_process'] for c,s in summary['jobs'].items()}
            assert checks['completed_resume_additional_steps'] == dict.fromkeys(CELLS, 0)
        training.write_json(REPORT/'smoke_checks.json', checks)
    print(json.dumps(dict(status='complete', smoke_only=args.smoke, improvement_percent=summary['improvement_percent'])), flush=True)


if __name__ == '__main__':
    main()
