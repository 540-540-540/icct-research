"""Paired fresh original/directed-context QGNN trial using the proven runner."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import sys
import threading

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.qgnn_directed_context.graph import DirectedContextQGNNGraph
from experiments.qgnn_inherited.graph import InheritedQGNNGraph
from experiments.qgnn_inherited.run import optimizer, shared, training
from experiments.qgnn_readout import run as paired

CELLS = ('original', 'directed_context')
REPORT = ROOT/'reports/qgnn_directed_context'
RESULTS = ROOT/'results/qgnn_directed_context/seed2026'
TEMP = ROOT/'.codex-work/qgnn-directed-context-smoke-20260914'


def configuration():
    cfg = json.loads((ROOT/'configs/qgnn_directed_context.json').read_text(encoding='utf-8-sig'))
    fixed = dict(seed=2026, cells=list(CELLS), devices=dict(zip(CELLS, ('cuda:0', 'cuda:1'))),
                 graph_seed_offset=200000, temporal_seed_offset=100000, epochs=150, patience=20,
                 min_delta=0, train_origins=5549, validation_origins=400, batch_size=16,
                 micro_batch=16, validation_micro_batch=16, graph_lr=.0003,
                 snr_db=[5, 10, 15, 20], allowed_splits=['train', 'V_select'])
    for key, value in fixed.items():
        if cfg[key] != value:
            raise ValueError(f'Frozen directed-context setting changed: {key}')
    return cfg


class Predictor(nn.Module):
    def __init__(self, cell, cfg):
        super().__init__()
        if cell not in CELLS:
            raise ValueError('Unknown directed-context trial cell')
        seed = cfg['seed'] + cfg['graph_seed_offset']
        training.seed_all(seed)
        graph = InheritedQGNNGraph if cell == 'original' else DirectedContextQGNNGraph
        self.graph = graph(seed=seed)
        training.seed_all(cfg['seed'] + cfg['temporal_seed_offset'])
        self.temporal = shared.TrajectoryPredictor(128)

    def forward(self, state_hat, standardized_state, track_exists, detected):
        args = (state_hat, standardized_state, track_exists, detected)
        return self.temporal(*args, self.graph(*args))


def provenance(cfg):
    code = ['experiments/qgnn_directed_context/run.py',
            'experiments/qgnn_directed_context/graph.py',
            'experiments/qgnn_directed_context/check_graph.py',
            'experiments/qgnn_directed_context/check_training.py',
            'experiments/qgnn_directed_context/diagnose.py',
            'configs/qgnn_directed_context.json',
            'experiments/qgnn_readout/run.py',
            'experiments/qgnn_inherited/run.py', 'experiments/qgnn_inherited/graph.py',
            'experiments/qgat_factorial/run_trial.py', 'prediction/training.py',
            'prediction/temporal.py', 'prediction/evaluation_cache.py',
            'frontend/symbol_dataset.py', 'frontend/echo_source.py',
            'code/00_remote_shared_dependencies/target_interaction_graph.py',
            'code/02_pennylane_qgnn_core/physics_aligned_dual_model.py']
    assets = ['models/gpt2/config.json', 'models/gpt2/model.safetensors', 'data/f01d/normalization.json']
    for split in ('train', 'V_select'):
        assets.append(f'data/f01d/labels/{split}.npz')
        assets.extend(f'data/f01d/inputs/{split}_snr_{snr}.npz' for snr in training.SNRS)
    records = {}
    for path in assets:
        info = (ROOT/path).stat()
        records[path] = dict(bytes=info.st_size, mtime_ns=info.st_mtime_ns)
    return dict(configuration=cfg, code_text={path:(ROOT/path).read_text() for path in code},
                asset_modification_records=records,
                versions={name:importlib.metadata.version(name)
                          for name in ('torch', 'numpy', 'pennylane', 'transformers')},
                asset_record_scope='Size/mtime modification detection; Dataset hashes and checkpoint contracts remain active.')


def summarize(states, cfg, run_root, smoke):
    best = {cell:json.loads((run_root/cell/f'validation_epoch_{state["training"]["best_metrics"]["epoch"]:02d}.json').read_text())
            for cell, state in states.items()}
    identities = [[(row['origin'], row['snr_db'], row['ade_targets'], row['fde_targets'])
                   for row in best[cell]['per_scene']] for cell in CELLS]
    assert identities[0] == identities[1] and len(identities[0]) == (17 if smoke else 400)*4
    delta = {key:best['directed_context'][key]-best['original'][key] for key in ('ADE', 'FDE', 'J')}
    gain = {key:100*(1-best['directed_context'][key]/best['original'][key]) for key in ('ADE', 'FDE')}
    return dict(status='complete', smoke_only=smoke, seed=cfg['seed'], jobs=states,
                best_J_epoch_metrics=best, directed_minus_original=delta,
                directed_improvement_percent=gain, development_gate_passed=gain['ADE'] > 0 and gain['FDE'] > 0,
                common_denominators_verified=True, confirmation_opened=False, test_opened=False,
                limitation=cfg['limitation'],
                x_minus_original=delta, x_improvement_percent=gain)


def _install_runner():
    paired.__file__ = __file__
    paired.CELLS = CELLS
    paired.REPORT = REPORT
    paired.RESULTS = RESULTS
    paired.TEMP = TEMP
    paired.Predictor = Predictor
    paired.configuration = configuration
    paired.provenance = provenance
    paired.summarize = summarize
    paired.optimizer = optimizer


def _stream_logs(report, stop, resume):
    positions = {cell:(report/'logs'/f'{cell}.log').stat().st_size
                 if resume and (report/'logs'/f'{cell}.log').is_file() else 0
                 for cell in CELLS}
    while not stop.wait(.5):
        for cell in CELLS:
            path = report/'logs'/f'{cell}.log'
            if not path.is_file():
                continue
            with path.open(encoding='utf-8', errors='replace') as log:
                log.seek(positions[cell])
                for line in log:
                    print(line if line.startswith(f'[{cell}] ') else f'[{cell}] {line}', end='', flush=True)
                positions[cell] = log.tell()
    for cell in CELLS:
        path = report/'logs'/f'{cell}.log'
        if path.is_file():
            with path.open(encoding='utf-8', errors='replace') as log:
                log.seek(positions[cell])
                for line in log:
                    print(line if line.startswith(f'[{cell}] ') else f'[{cell}] {line}', end='', flush=True)


def worker(args, cfg, source, run_root, report):
    paired.worker(args, cfg, source, run_root, report)
    if args.worker == 'directed_context':
        path = report/'jobs'/f'{args.worker}.json'
        status = json.loads(path.read_text())
        latest = torch.load(run_root/args.worker/'latest.pt', map_location='cpu', weights_only=False)
        changed = {name:int(value.count_nonzero()) for name, value in latest['model'].items()
                   if name.endswith(('receiver_context', 'sender_context'))}
        assert len(changed) == 4 and all(changed.values())
        status['directed_context_changed_elements'] = changed
        training.write_json(path, status)


def coordinate(args, cfg, source, run_root, report):
    stop = threading.Event()
    monitor = threading.Thread(target=_stream_logs, args=(report, stop, args.resume), daemon=True)
    monitor.start()
    try:
        summary = paired.coordinate(args, cfg, source, run_root, report)
    finally:
        stop.set()
        monitor.join()
    if not args.smoke:
        lines = ['# 可学习有向上下文残差 QGNN 配对结果', '',
                 'seed2026；仅 V_select；最多 150 轮，patience=20。ADE、FDE来自同一个最低J检查点。', '',
                 '| 模型 | 实际轮数 | 最佳轮次 | ADE | FDE | 早停 |',
                 '|---|---:|---:|---:|---:|---|']
        for cell in CELLS:
            job, best = summary['jobs'][cell]['training'], summary['best_J_epoch_metrics'][cell]
            lines.append(f'| {cell} | {len(job["history"])} | {best["epoch"]} | {best["ADE"]:.6f} | {best["FDE"]:.6f} | {job["stopped_early"]} |')
        gain = summary['directed_improvement_percent']
        verdict = '通过单种子开发门槛' if summary['development_gate_passed'] else '未通过双指标开发门槛'
        lines.extend(['', f'候选相对原Q：ADE改善 {gain["ADE"]:.3f}%，FDE改善 {gain["FDE"]:.3f}%；{verdict}。', '',
                      '这是单种子V_select开发证据，不是多种子确认、测试集结论或量子优势证据。'])
        (report/'RESULT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return summary


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
    _install_runner()
    cfg = configuration()
    source = provenance(cfg)
    checks = json.loads((REPORT/'training_checks.json').read_text())
    graph = json.loads((REPORT/'graph_checks.json').read_text())
    d0 = json.loads((REPORT/'d0.json').read_text())
    if (not checks['passed'] or checks['source'] != source or not graph['passed']
            or not d0['passed'] or d0['source'] != source):
        raise ValueError('Current CPU, D0, and graph checks required')
    if any((ROOT/path).read_text() != text for path, text in graph['graph_sources'].items()):
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
        result = dict(passed=True, source=source, summary=summary, formal_optimizer_steps=0)
        if args.resume:
            result['completed_resume_additional_steps'] = {
                cell:state['additional_optimizer_steps_this_process'] for cell, state in summary['jobs'].items()}
            assert result['completed_resume_additional_steps'] == dict.fromkeys(CELLS, 0)
        training.write_json(REPORT/'smoke_checks.json', result)
    print(json.dumps(dict(status='complete', smoke_only=args.smoke,
                          directed_minus_original=summary['directed_minus_original'])), flush=True)


if __name__ == '__main__':
    main()
