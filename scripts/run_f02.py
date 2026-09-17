"""Run or assemble F02 mathematical/interface validation; never train a model."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports/f02'
SUMMARY = ROOT / 'reports/theory_validation.json'


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def run_stage(name, module, arguments=()):
    started = time.time()
    print(f'F02 {name}: starting', flush=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    with (REPORTS / f'{name}.log').open('w') as log:
        process = subprocess.Popen([sys.executable, '-u', '-m', module, *arguments],
                                   cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line)
        code = process.wait()
    if code:
        raise RuntimeError(f'{name} failed with exit code {code}; see reports/f02/{name}.log')
    return time.time() - started


def assemble():
    frontend = json.loads((ROOT/'reports/f01d/run_status.json').read_text())
    if frontend.get('status') != 'complete' or not frontend.get('structural_validation_passed'):
        raise RuntimeError('F01D complete structural validation is required')
    quantum = json.loads((REPORTS/'quantum_checks.json').read_text())
    interface = json.loads((REPORTS/'interface.json').read_text())
    baseline = json.loads((REPORTS/'plain_baseline.json').read_text())
    primary = json.loads((REPORTS/'primary_models.json').read_text())
    interface_checks = dict(interface['checks'])
    interface_checks.update({'GNN_source_' + k: v for k, v in baseline['checks'].items()})
    interface_checks.update({'primary_' + k: v for k, v in primary['checks'].items()})
    records = quantum['checks']
    sections = {}
    for label in 'ABCDEFGH':
        selected = interface_checks if label == 'F' else {
            key: value for key, value in records.items() if key.startswith(label + '_')}
        sections[f'F02-{label}'] = dict(passed=bool(selected) and all(v['passed'] for v in selected.values()),
                                     check_count=len(selected),
                                     report=['reports/f02/interface.json', 'reports/f02/plain_baseline.json', 'reports/f02/primary_models.json'] if label == 'F' else 'reports/f02/quantum_checks.json',
                                     failed_checks=[key for key, value in selected.items() if not value['passed']])
    # Existing reports may be reused only if their code has not subsequently changed.
    source_groups = {
        'quantum_checks.json': ['prediction/quantum.py', 'prediction/reference.py', 'prediction/check_quantum.py'],
        'interface.json': ['prediction/quantum.py', 'prediction/classical.py', 'prediction/temporal.py',
                           'prediction/check_interfaces.py', 'frontend/symbol_dataset.py'],
        'plain_baseline.json': ['prediction/classical.py', 'prediction/check_plain_baseline.py',
                                'code/00_remote_shared_dependencies/target_interaction_graph.py'],
        'primary_models.json': ['prediction/model.py', 'prediction/check_primary_models.py',
                                'prediction/quantum.py', 'prediction/classical.py', 'prediction/temporal.py'],
    }
    for report, paths in source_groups.items():
        for path in paths:
            if (ROOT/path).stat().st_mtime_ns > (REPORTS/report).stat().st_mtime_ns:
                raise RuntimeError(f'{report} predates {path}; rerun the affected checks')
    sources = sorted(set(sum(source_groups.values(), [])))
    source_versions = {path: dict(bytes=(ROOT/path).stat().st_size,
                                 mtime_ns=(ROOT/path).stat().st_mtime_ns) for path in sources}
    passed = all(bool(r.get('passed')) for r in [quantum, interface, baseline, primary]) and all(v['passed'] for v in sections.values())
    import torch
    import transformers
    return dict(stage='F02', status='passed' if passed else 'failed', passed=passed,
                formula_version='研究方案书1.4 / A04 / A02 Q24 / A03-symbol-v4-cuda-stable-solver',
                primary_models={'QGNN': 'PennyLane quantum graph and common GPT-2', 'GNN': 'original Plain graph-attention core and common GPT-2'},
                main_entrypoint='prediction.model.build_model',
                retained_A02_GNN_checks='Auxiliary legacy checks only; not the primary GNN',
                sections=sections, total_checks=sum(s['check_count'] for s in sections.values()),
                reference_precision='float64/complex128', time_model_precision='float32 pretrained GPT-2',
                quantum_output_tolerance=1e-10, quantum_gradient_absolute_tolerance=1e-8,
                real_input=interface['real_cache'], frontend_status='complete',
                environment=dict(python=sys.version.split()[0], torch=torch.__version__, transformers=transformers.__version__),
                code_files=source_versions, assembled_unix=time.time(),
                formal_training_started=False, optimizer_steps=0,
                limitations=['Finite deterministic numerical examples; no prediction accuracy or quantum advantage claim.',
                             'complex64 circuit production has not been qualified.',
                             'F03 cost and F04 predictive preflight remain unexecuted.',
                             'New connection and readout-width isolation experiments cancelled by user.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assemble-existing', action='store_true', help='Assemble already completed, current reports without rerunning checks')
    args = parser.parse_args()
    REPORTS.mkdir(parents=True, exist_ok=True)
    started = time.time()
    status = dict(stage='F02', status='running', started_unix=started,
                  formal_training_started=False, optimizer_steps=0)
    write_json(REPORTS/'run_status.json', status)
    durations = {}
    try:
        if not args.assemble_existing:
            durations['quantum'] = run_stage('quantum_checks', 'prediction.check_quantum')
            durations['interface'] = run_stage('interface', 'prediction.check_interfaces', ['--device', 'cuda:1'])
            durations['gnn'] = run_stage('plain_baseline', 'prediction.check_plain_baseline')
            durations['primary_models'] = run_stage('primary_models', 'prediction.check_primary_models')
        result = assemble()
        result['rerun_stage_seconds'] = durations
        write_json(SUMMARY, result)
        status.update(status=result['status'], finished_unix=time.time(), passed=result['passed'])
        write_json(REPORTS/'run_status.json', status)
        print(json.dumps(dict(status=result['status'], sections=result['sections'], report=str(SUMMARY)), ensure_ascii=False), flush=True)
        if not result['passed']:
            raise SystemExit(1)
    except Exception as exc:
        status.update(status='failed', finished_unix=time.time(), error=str(exc))
        write_json(REPORTS/'run_status.json', status)
        write_json(SUMMARY, dict(stage='F02', status='incomplete', passed=False, error=str(exc), formal_training_started=False))
        raise


if __name__ == '__main__':
    main()
