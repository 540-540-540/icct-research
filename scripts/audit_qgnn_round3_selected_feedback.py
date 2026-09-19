"""Selected Round3 neutral-model dependency audit; no training or model selection."""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time, traceback, subprocess
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from frontend.sind_prediction_dataset import SinDPredictionDataset
from prediction.qgnn_adaptive.model import build_model
from prediction.q0.metrics import trajectory_metrics
from scripts.train_qgnn_adaptive import atomic_json, evaluate
from scripts.train_q0_motion_llm import token_loss

MODES = ('normal', 'no_K2_feedback', 'no_K3_feedback', 'no_feedback', 'neutral_controller')

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def strata(reference, features):
    assert len(reference) == len(features) == 1880
    assert all((x['scene_id'], x['start_frame']) == (y['scene_id'], y['start_frame']) for x, y in zip(reference, features))
    final = np.array([x['closing_pairs_30m_gt_0p5'] for x in reference])
    recent = np.array([x['closing_pairs_30m_gt_0p5'] for x in features])
    k3 = np.array([x['q_k3_rms'] for x in features])
    bounds = np.quantile(k3, np.linspace(0, 1, 6))
    groups = {'all': np.ones(1880, dtype=bool)}
    for name, v in [('final', final), ('recent5', recent)]:
        groups[name + '_closing_lt10'] = v < 10
        groups[name + '_closing_10_14'] = (v >= 10) & (v < 15)
        groups[name + '_closing_ge15'] = v >= 15
    groups['recent5_wedges_ge35'] = np.array([x['active_wedges'] for x in features]) >= 35
    groups['recent5_triangles_ge6'] = np.array([x['active_triangles'] for x in features]) >= 6
    for k in range(5):
        groups[f'frozen_K3_quintile_{k+1}'] = (k3 >= bounds[k]) & ((k3 <= bounds[k+1]) if k == 4 else (k3 < bounds[k+1]))
    return groups

def main(args):
    torch.set_num_threads(4)
    torch.manual_seed(7311)
    device = torch.device('cuda:0')
    directory, outdir = ROOT / args.run_dir, ROOT / args.out_dir
    outdir.mkdir(parents=True, exist_ok=True)
    output = outdir / 'summary.json'
    if output.exists():
        raise FileExistsError('Use a new output directory; do not overwrite an audit.')
    cp = torch.load(directory / 'best.pt', map_location='cpu', weights_only=False)
    c = cp['config']
    assert c['adaptive_mode'] == 'phase_feedback' and c['controller_init'] == 'neutral'
    assert c['snr'] == 0 and c['train_limit'] == 4096 and c['epochs'] == 12
    model = build_model(c['kind'], c['seed'], c['depth'], channels=c['channels'], snr_db=c['snr'], correction_cap_m=c['correction_cap'], adaptive_mode=c['adaptive_mode'], controller_init=c['controller_init']).to(device).eval()
    missing, extra = model.load_state_dict(cp['model_state'], strict=False)
    assert not extra and all(k.startswith('llm.gpt2.') and 'lora_' not in k for k in missing)
    core = model.graph
    reference = json.loads((directory / 'best_validation_rows.json').read_text())['rows']
    feature_path = ROOT / 'reports/qgnn/round3_strata_reference_20260919.json'
    groups = strata(reference, json.loads(feature_path.read_text())['rows'])
    selected_config = json.loads((directory / 'config.json').read_text())
    current_source = {p: sha(ROOT / p) for p in selected_config['source_sha256']}
    mismatches = {p: {'recorded': s, 'current': current_source[p]} for p, s in selected_config['source_sha256'].items() if s != current_source[p]}
    allowed_defaults = {
        'prediction/qgnn_adaptive/model.py': ('controller_init="neutral"):', 'controller_init="specialized"):'),
        'scripts/train_qgnn_adaptive.py': ("choices=['specialized','neutral'],default='neutral'", "choices=['specialized','neutral'],default='specialized'")}
    for path, mismatch in mismatches.items():
        if path not in allowed_defaults:
            raise ValueError('Unexpected checkpoint source drift: ' + path)
        current, recorded = allowed_defaults[path]
        source = (ROOT / path).read_text()
        assert source.count(current) == 1
        restored = source.replace(current, recorded, 1)
        assert hashlib.sha256(restored.encode()).hexdigest() == mismatch['recorded'], path
        mismatch['verified_reason'] = 'Only default changed specialized->neutral; replay passes explicit checkpoint config.'
    started = time.perf_counter()
    report = {'status': 'RUNNING', 'kind': c['kind'], 'test_set_used': False, 'new_training': False,
              'source_git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'checkpoint_path': str((directory / 'best.pt').relative_to(ROOT)), 'checkpoint_sha256': sha(directory / 'best.pt'),
              'checkpoint_epoch': cp['epoch'], 'checkpoint_config': c, 'source_sha256': current_source,
              'verified_default_only_source_changes': mismatches, 'audit_script_sha256': sha(Path(__file__)), 'frozen_strata_sha256': sha(feature_path), 'validation': {},
              'warning': 'Fixed-checkpoint OOD dependency interventions, not retrained causal ablations. No modified inference mode is selected. All Q scenes execute all circuits.'}
    def save(stage):
        report['stage'] = stage
        report['elapsed_seconds'] = time.perf_counter() - started
        atomic_json(output, report)
        atomic_json(outdir / 'heartbeat.json', {'pid': os.getpid(), 'stage': stage, 'elapsed_seconds': report['elapsed_seconds'], 'time': time.strftime('%Y-%m-%dT%H:%M:%S%z')})
        print(stage, flush=True)
    save('loading_validation')
    val = SinDPredictionDataset('val', 0., ROOT, True)
    loader = DataLoader(val, batch_size=32, shuffle=False)
    values = {}
    for mode in MODES:
        hook = None
        core.controller.enabled = mode != 'neutral_controller'
        core.feedback_enabled = mode != 'no_feedback'
        if mode in ('no_K2_feedback', 'no_K3_feedback'):
            order = 0 if mode == 'no_K2_feedback' else 1
            def remove_feedback(module, inputs, order=order):
                layer, desc, fb = inputs
                if fb is None:
                    return None
                fs = list(fb); fs[order] = torch.zeros_like(fs[order])
                return layer, desc, tuple(fs)
            hook = core.controller.register_forward_pre_hook(remove_feedback)
        try:
            metric, rows = evaluate(model, loader, device)
        finally:
            if hook is not None:
                hook.remove()
        assert all((x['index'], x['scene_id'], x['start_frame']) == (y['index'], y['scene_id'], y['start_frame']) for x, y in zip(rows, reference))
        arr = np.array([[x['ADE'], x['FDE'], x['ADE'] + .5*x['FDE']] for x in rows])
        values[mode] = arr
        if mode == 'normal':
            target = np.array([[x['ADE'], x['FDE'], x['ADE'] + .5*x['FDE']] for x in reference])
            error = float(np.abs(target - arr).max())
            assert error < 2e-5, error
            report['baseline_replay_max_metric_error_m'] = error
        report['validation'][mode] = {'overall': metric, 'strata': {name: {'windows': int(sel.sum()), **dict(zip(('ADE', 'FDE', 'J'), arr[sel].mean(0).tolist()))} for name, sel in groups.items()}, 'per_window_metric_sha256': hashlib.sha256(arr.tobytes()).hexdigest()}
        save('validation_' + mode)
    for mode in MODES[1:]:
        for name, sel in groups.items():
            base, changed = values['normal'][sel].mean(0), values[mode][sel].mean(0)
            report['validation'][mode]['strata'][name]['degradation_vs_normal_pct'] = dict(zip(('ADE', 'FDE', 'J'), (100*(changed/base-1)).tolist()))
    core.controller.enabled = True; core.feedback_enabled = True
    save('loading_training_audit')
    train = SinDPredictionDataset('train', 0., ROOT, True)
    trained_indices = np.random.default_rng(c['seed']).permutation(len(train))[:c['train_limit']]
    selected = np.random.default_rng(4811).choice(trained_indices, size=512, replace=False)
    train_loader = DataLoader(Subset(train, selected.tolist()), batch_size=32, shuffle=False)
    traces = {mode: {key: [[] for _ in range(c['depth'])] for key in ('lambda2', 'lambda3')} for mode in ('normal', 'no_feedback')}
    def capture(module, inputs, outputs):
        layer, desc, fb = inputs
        for key, val, weight in zip(('lambda2', 'lambda3'), outputs, (desc['pw'], desc['tw'])):
            assert torch.isfinite(val).all() and ((val >= .2) & (val <= 1.8)).all()
            weighted = (val*weight[:, None]).sum(-1)/weight.sum(-1)[:, None].clamp_min(1e-6)
            traces[trace_mode][key][layer].append(weighted.detach().cpu())
    hook = core.controller.register_forward_hook(capture)
    try:
        with torch.no_grad():
            for trace_mode in ('normal', 'no_feedback'):
                core.feedback_enabled = trace_mode == 'normal'
                for batch in train_loader:
                    core(batch['history_state'].to(device), batch['vehicle_mask'].to(device))
    finally:
        hook.remove(); core.feedback_enabled = True
    controller_stats = {'split': 'train', 'samples': 512, 'subset': '512 sampled from the exact4096 training subset', 'sampling_seed': 4811, 'indices_sha256': hashlib.sha256(selected.tobytes()).hexdigest()}
    for key in ('lambda2', 'lambda3'):
        x = torch.stack([torch.cat(v) for v in traces['normal'][key]], -1)
        y = torch.stack([torch.cat(v) for v in traces['no_feedback'][key]], -1)
        controller_stats[key] = {'mean_C_D': x.mean(0).tolist(), 'scene_std_C_D': x.std(0).tolist(),
                                 'mean_scene_std': float(x.std(0).mean()), 'mean_abs_deviation_from1': float((x-1).abs().mean()),
                                 'feedback_mean_abs_effect_C_D': (x-y).abs().mean(0).tolist(),
                                 'feedback_mean_abs_effect': float((x-y).abs().mean()), 'min_scene_weighted_mean': float(x.min()), 'max_scene_weighted_mean': float(x.max())}
    batch = next(iter(train_loader))
    h, m, y, ts = (batch[k].to(device) for k in ('history_state', 'vehicle_mask', 'future_state', 'history_timestamp'))
    model.zero_grad(set_to_none=True)
    out = model(h, m, ts)
    loss = trajectory_metrics(out['prediction'], y, m)['loss'] + .035*token_loss(out['token_logits'], model.llm.future_token_ids(h, y), m)
    loss.backward()
    controller_stats['loss_probe'] = float(loss.detach())
    controller_stats['gradients'] = {}
    for name, parameter in core.controller.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        controller_stats['gradients'][name] = {'parameter_norm': float(parameter.detach().norm()), 'gradient_norm': float(parameter.grad.norm())}
    report['train_controller'] = controller_stats
    report['peak_allocated_gib'] = torch.cuda.max_memory_allocated()/2**30
    report['status'] = 'COMPLETED'; save('complete')
    print(json.dumps({'kind': c['kind'], 'overall_by_mode': {k: v['overall'] for k, v in report['validation'].items()}, 'lambda_stats': {k: controller_stats[k] for k in ('lambda2', 'lambda3')}}), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    try:
        main(args)
    except Exception:
        directory = ROOT / args.out_dir; directory.mkdir(parents=True, exist_ok=True)
        atomic_json(directory / 'failure.json', {'status': 'FAILED', 'pid': os.getpid(), 'traceback': traceback.format_exc(), 'test_set_used': False})
        raise
