"""Frozen attention-weighted gate control on all 400 V_select scenes at 20 dB."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from experiments.qgnn_cases.diagnose import QPredictor, RUNS, configuration, evaluate, shared, summary_metrics, training

OUTPUT = ROOT / 'reports/qgnn_cases/weighted_gate_control.json'


class WeightedGateControl:
    """Replace neighbor gates, retaining their attention-weighted scalar total."""

    def __init__(self, model):
        self.handles = []
        self.statistics = [dict(layer=i, gate_calls=0, attention_checks=0,
                                mass_max_error=0.0, attention_max_error=0.0,
                                valid_receiver_heads=0) for i in range(len(model.graph.graph_layers))]
        for layer, statistics in zip(model.graph.graph_layers, self.statistics):
            cache = {}

            def before(module, args, cache=cache):
                cache.clear()
                adjacency = args[2]
                cache['adjacency'] = adjacency
                cache['selected'] = adjacency.reshape(-1).nonzero().flatten()

            def capture_score(module, args, output, cache=cache):
                cache['score'] = output

            def replace_gate(module, args, output, cache=cache, statistics=statistics):
                adj, selected, logits = cache['adjacency'], cache['selected'], cache['score']
                b, n, _ = adj.shape
                heads = output.shape[-1]
                score = logits.new_zeros(b * n * n, heads).index_copy(0, selected, logits).view(b, n, n, heads)
                attention = torch.softmax(score.masked_fill(~adj[..., None], -1e4), dim=2)
                cache['attention'] = attention
                masked_attention = attention * adj[..., None]
                gates = output.new_zeros(b * n * n, heads).index_copy(0, selected, 2 * torch.sigmoid(output)).view(b, n, n, heads)
                mass_before = (masked_attention * gates).sum(2)
                attention_sum = masked_attention.sum(2)
                mean = mass_before / attention_sum.clamp_min(torch.finfo(output.dtype).eps)
                receiver_rows = selected // n
                assigned = mean.reshape(b * n, heads).index_select(0, receiver_rows)
                # Finite inverse sigmoid. Any clipping must still meet the mass check.
                assigned = assigned.clamp(1e-7, 2 - 1e-7)
                replacement = torch.log(assigned / (2 - assigned))
                effective = 2 * torch.sigmoid(replacement)
                new_gates = output.new_zeros(b * n * n, heads).index_copy(0, selected, effective).view(b, n, n, heads)
                mass_after = (masked_attention * new_gates).sum(2)
                error = float((mass_before - mass_after).abs().max())
                assert error < 2e-6, (statistics['layer'], error)
                statistics['mass_max_error'] = max(statistics['mass_max_error'], error)
                statistics['gate_calls'] += 1
                statistics['valid_receiver_heads'] += int((attention_sum > 0).sum())
                return replacement

            def check_attention(module, args, cache=cache, statistics=statistics):
                # This shared dropout is also called on rank-3 residual/FFN tensors.
                if args[0].ndim != 4:
                    return
                error = float((args[0] - cache['attention']).abs().max())
                assert error == 0.0, (statistics['layer'], error)
                statistics['attention_max_error'] = max(statistics['attention_max_error'], error)
                statistics['attention_checks'] += 1

            self.handles.extend([layer.register_forward_pre_hook(before),
                                 layer.score.register_forward_hook(capture_score),
                                 layer.gate.register_forward_hook(replace_gate),
                                 layer.dropout.register_forward_pre_hook(check_attention)])

    def close(self):
        for handle in self.handles:
            handle.remove()


@torch.no_grad()
def main():
    busy = subprocess.run(['nvidia-smi', '-i', '0', '--query-compute-apps=pid', '--format=csv,noheader'],
                          capture_output=True, text=True, check=True).stdout.strip()
    assert not busy, 'GPU 0 is busy'
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_device(0)
    started = time.time()
    checkpoint_path = RUNS['qgnn'] / 'best.pt'
    before_stat = checkpoint_path.stat()
    cfg, valid = configuration(), shared.dataset('V_select')
    assert valid.n == 400
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    epoch = saved['progress']['epoch']
    assert epoch == 75, epoch
    assert all(saved['config']['validation_hashes'][key] == value for key, value in valid.input_hashes.items())
    model = QPredictor('original', cfg).to('cuda:0').eval()
    training.load_weights(model, saved['model'])
    baseline = evaluate(model, valid)
    reference = json.loads((RUNS['qgnn'] / f'validation_epoch_{epoch:02d}.json').read_text())
    rows = [row for row in reference['per_scene'] if row['snr_db'] == 20]
    assert [row['origin'] for row in rows] == list(range(400))
    archive_error = max(abs(float(baseline['scene_' + key][i]) - row[key])
                        for i, row in enumerate(rows) for key in ('ADE', 'FDE'))
    assert archive_error < 1e-5, archive_error
    for key, field in (('ADE_mask', 'ade_targets'), ('FDE_mask', 'fde_targets')):
        assert np.array_equal(baseline[key].sum(1), [row[field] for row in rows])
    print('BASELINE', summary_metrics(baseline), 'archive_error', archive_error, flush=True)

    control = WeightedGateControl(model)
    try:
        changed = evaluate(model, valid)
    finally:
        control.close()
    for key in ('ADE_mask', 'FDE_mask'):
        assert np.array_equal(baseline[key], changed[key])
    assert all(stat['gate_calls'] == 25 and stat['attention_checks'] == 25 for stat in control.statistics)
    assert all(torch.equal(value.detach().cpu(), saved['model'][key])
               for key, value in training.mutable_state(model).items())
    after_stat = checkpoint_path.stat()
    assert (before_stat.st_size, before_stat.st_mtime_ns) == (after_stat.st_size, after_stat.st_mtime_ns)
    baseline_metrics, control_metrics = summary_metrics(baseline), summary_metrics(changed)
    assert all(np.isfinite(value) for value in (*baseline_metrics.values(), *control_metrics.values()))
    report = dict(
        status='complete', seed=2026, split='V_select', snr_db=20, scenes=400,
        checkpoint=str(checkpoint_path.relative_to(ROOT)), checkpoint_epoch=epoch,
        baseline=baseline_metrics, attention_weighted_mean_gate=control_metrics,
        change={key: control_metrics[key] - baseline_metrics[key] for key in baseline_metrics},
        change_percent={key: 100 * (control_metrics[key] / baseline_metrics[key] - 1) for key in baseline_metrics},
        archived_scene_metrics_max_error=archive_error,
        denominators_identical=True,
        denominators=dict(scenes=400, ADE_targets=int(baseline['ADE_mask'].sum()),
                          FDE_targets=int(baseline['FDE_mask'].sum())),
        per_layer_checks=control.statistics,
        attention_weighted_gate_mass_max_error=max(stat['mass_max_error'] for stat in control.statistics),
        actual_attention_reconstruction_max_error=max(stat['attention_max_error'] for stat in control.statistics),
        control='For each receiver/head, replace existing adjacent gates (including self) by sum(a*g)/sum(a).',
        metric_aggregation='Mean of 400 per-scene metrics, matching the archived V_select evaluator.',
        inference='Direct no_grad model forward; batch 16; no evaluation cache.',
        optimizer_steps=0, model_parameters_unchanged=True, checkpoint_files_modified=False,
        V_confirm_accessed=False, test_accessed=False,
        script=str(Path(__file__).resolve().relative_to(ROOT)), script_source=Path(__file__).read_text(encoding='utf-8'),
        elapsed_seconds=time.time() - started,
        limitations=[
            'Preserves each receiver/head scalar sum(a*g); does not preserve aggregate message direction or norm.',
            'Gate replacement leaves that layer score/softmax untouched; changed first-layer nodes can change downstream attention.',
            'Frozen-model distribution intervention on development scenes; no retraining and no independent causal or quantum-advantage evidence.',
        ],
    )
    OUTPUT.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('CONTROL', control_metrics, 'change_percent', report['change_percent'],
          'mass_error', report['attention_weighted_gate_mass_max_error'], flush=True)
    del model
    torch.cuda.empty_cache()
    print('DONE', OUTPUT, flush=True)


if __name__ == '__main__':
    main()
