"""Run the six F05 main trainings after a sealed F04 PASS. Never opens test data."""
import argparse
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def file_hash(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    resolved = path.resolve()
    if not resolved.is_relative_to(ROOT.resolve()) or any('test' in part.lower() for part in resolved.relative_to(ROOT).parts):
        raise RuntimeError(f'F05 refuses provenance access outside project or locked test: {path}')
    digest = hashlib.sha256()
    with resolved.open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_protocol(path):
    import yaml
    protocol = yaml.safe_load(Path(path).read_text())
    if protocol.get('status') != 'PASS' or not protocol.get('budget_accepted'):
        raise RuntimeError('F04 PASS and accepted training budget must be sealed first')
    decision = json.loads((ROOT/'reports/preflight_decision.json').read_text())
    if decision.get('status') != 'PASS':
        raise RuntimeError('F04 decision is not PASS')
    provenance = protocol.get('provenance', {})
    if not provenance.get('code_hashes') or not provenance.get('checkpoint_hashes') or not provenance.get('data_hashes'):
        raise RuntimeError('Missing sealed code/data/checkpoint provenance')
    required_code = {'prediction/training.py', 'prediction/model.py', 'prediction/quantum.py',
                     'prediction/temporal.py', 'prediction/classical.py', 'prediction/evaluation_cache.py', 'scripts/run_f05.py'}
    if not required_code <= set(provenance['code_hashes']):
        raise RuntimeError('F05 implementation is absent from the sealed code manifest')
    required_data = {f'data/f01d/inputs/{split}_snr_{snr}.npz' for split in ('train','V_select') for snr in (5,10,15,20)}
    required_data |= {'data/f01d/labels/train.npz', 'data/f01d/labels/V_select.npz', 'data/f01d/normalization.json'}
    if not required_data <= set(provenance['data_hashes']):
        raise RuntimeError('Training/validation/normalization data are absent from sealed manifest')
    pretrained = provenance.get('pretrained_hashes', {})
    if 'models/gpt2/config.json' not in pretrained or not any(k.endswith(('.safetensors', '.bin')) for k in pretrained):
        raise RuntimeError('Frozen GPT-2 dependency must be sealed with configuration and weights')
    for collection in ('code_hashes','data_hashes','checkpoint_hashes','pretrained_hashes'):
        for file, expected in provenance[collection].items():
            if file_hash(file) != expected:
                raise RuntimeError(f'Sealed {collection} changed: {file}')
    if not provenance.get('f04_contract') or file_hash(provenance['f04_contract']) != provenance.get('f04_contract_sha256'):
        raise RuntimeError('F04 execution contract differs from sealed provenance')
    if file_hash('reports/preflight_decision.json') != provenance.get('f04_decision_sha256'):
        raise RuntimeError('F04 decision differs from sealed provenance')
    cfg = protocol['training']
    for key, value in dict(batch_size=16, warmup_epochs=3, adapt_epochs=2, joint_epochs=15, patience=4).items():
        if cfg.get(key) != value:
            raise RuntimeError(f'Unrecognized F05 schedule: {key}; requires documented common amendment')
    if not isinstance(cfg.get('micro_batch'), int) or not 1 <= cfg['micro_batch'] <= 16:
        raise RuntimeError('F03 micro-batch must be sealed in [1,16]')
    models = protocol['models']
    if models['qgnn']['depth'] not in (2,3) or models['gnn']['depth'] != 2:
        raise RuntimeError('Invalid selected graph depth')
    if any(models[m]['graph_lr'] not in (1e-4,3e-4) for m in ('qgnn','gnn')):
        raise RuntimeError('Invalid selected graph learning rate')
    return protocol


def warmup(seed, train, validation, device, run_dir, cfg, contract, resume):
    from prediction.model import build_model
    from prediction.training import seed_all, fit
    path = run_dir / f'seed_{seed}' / 'own_only'
    seed_all(seed)
    model = build_model('gnn', train.normalization, device)
    result = fit(model, train, validation, path, seed=seed, epochs=cfg['warmup_epochs'], patience=4,
                 graph_lr=3e-4, phase='warmup', batch_size=cfg['batch_size'], micro_batch=cfg['micro_batch'],
                 resume=resume, contract=contract)
    del model
    import torch
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=('both','qgnn','gnn'), default='both')
    p.add_argument('--seeds', type=int, nargs='+', default=[2026,2027,2028])
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--protocol', type=Path, default=ROOT/'configs/selected_protocol.yaml')
    p.add_argument('--run-dir', type=Path, default=ROOT/'results/f05')
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if not set(args.seeds) <= {2026,2027,2028} or len(set(args.seeds)) != len(args.seeds):
        p.error('F05 seeds are distinct members of 2026/2027/2028')
    try:
        protocol = read_protocol(args.protocol)
    except (FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        print(json.dumps(dict(ready=False, stage='F05', blocker=str(exc), optimizer_steps_executed=0)), flush=True)
        raise SystemExit(2)
    import torch
    from prediction.training import Dataset, fit, seed_all, copy_own_weights, write_json, load_weights
    from prediction.model import build_model
    if not torch.cuda.is_available() or (args.model == 'both' and torch.cuda.device_count() < 2):
        raise RuntimeError('Requested F05 CUDA devices are unavailable')
    train, validation = Dataset('train'), Dataset('V_select')
    cfg = protocol['training']
    if protocol['provenance'].get('input_hashes') != train.input_hashes:
        raise RuntimeError('Training model-input digest differs from F04 seal')
    contract = dict(protocol=protocol, train_hashes=train.input_hashes, V_select_hashes=validation.input_hashes,
                    normalization=train.normalization)
    if args.check_only:
        print(json.dumps(dict(ready=True, stage='F05', seeds=args.seeds, model=args.model,
                              training_origins=train.n, validation_origins=validation.n,
                              schedule=cfg, optimizer_steps_executed=0, test_opened=False)), flush=True)
        return
    args.run_dir.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        own_dir = args.run_dir / f'seed_{seed}' / 'own_only'
        # The default two-GPU coordinator is the single writer of each shared warmup.
        existing = own_dir/'summary.json'
        if existing.exists():
            if not args.resume:
                raise FileExistsError('Existing F05 results require --resume')
            own = json.loads(existing.read_text())
            checkpoint = torch.load(own['best_checkpoint'], map_location='cpu', weights_only=False)
            if checkpoint['config']['contract'] != contract or checkpoint['config']['seed'] != seed:
                raise RuntimeError('Shared warmup contract differs')
        else:
            own = warmup(seed, train, validation, 'cuda:1' if args.model == 'both' else args.device,
                         args.run_dir, cfg, contract, args.resume)
        if args.model == 'both':
            children = []
            try:
                for name, device in [('qgnn','cuda:0'),('gnn','cuda:1')]:
                    command = [sys.executable, '-u', str(Path(__file__).resolve()), '--model', name,
                               '--device', device, '--seeds', str(seed), '--protocol', str(args.protocol),
                               '--run-dir', str(args.run_dir), '--resume']
                    children.append(subprocess.Popen(command, cwd=ROOT, env=dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')))
                codes = [c.wait() for c in children]
            except BaseException:
                for c in children:
                    if c.poll() is None:
                        c.terminate()
                for c in children:
                    c.wait()
                raise
            if any(codes):
                raise RuntimeError(f'F05 seed {seed} child failed: {codes}; resume preserves saved progress')
            continue
        seed_all(seed+10000)
        spec = protocol['models'][args.model]
        model = build_model(args.model, train.normalization, args.device, depth=spec['depth'])
        copy_own_weights(model, own['best_checkpoint'])
        model_dir = args.run_dir/f'seed_{seed}'/args.model
        adapt = fit(model, train, validation, model_dir/'adapt', seed=seed, epochs=cfg['adapt_epochs'],
                    patience=4, graph_lr=spec['graph_lr'], phase='adapt', batch_size=cfg['batch_size'],
                    micro_batch=cfg['micro_batch'], resume=args.resume, contract=contract)
        # Stage B has exactly two epochs; transfer its final state into joint training.
        adapt_last = torch.load(model_dir/'adapt/latest.pt', map_location='cpu', weights_only=False)
        load_weights(model, adapt_last['model'])
        joint = fit(model, train, validation, model_dir/'joint', seed=seed, epochs=cfg['joint_epochs'],
                    patience=cfg['patience'], graph_lr=spec['graph_lr'], phase='joint', batch_size=cfg['batch_size'],
                    micro_batch=cfg['micro_batch'], resume=args.resume, contract=contract)
        write_json(model_dir/'summary.json', dict(status='complete', model=args.model, seed=seed,
                                                shared_warmup=own, adapt=adapt, joint=joint, test_opened=False))
        del model
        torch.cuda.empty_cache()
    if args.model == 'both':
        write_json(args.run_dir/'summary.json', dict(status='complete', seeds=args.seeds, models=['qgnn','gnn'],
                                                     complete_six_runs=set(args.seeds)=={2026,2027,2028}, test_opened=False))


if __name__ == '__main__':
    main()
