"""Outcome-blind interaction-complexity stratification for paired dev predictions."""
import json
from pathlib import Path

import numpy as np

ROOT = Path('/home/js_cn/sensing')
BASE = ROOT / 'diagnostics/core_message_seed2026_v2/physics_aligned_dual_seed2026_v1'


def ranks(x):
    order = np.argsort(np.argsort(x, kind='mergesort'), kind='mergesort')
    return order / max(1, len(x) - 1)


def scene_features(history, mask):
    count = mask.sum(1).astype(float)
    close_pairs, conflict_pairs, closing_strength, maneuver = [], [], [], []
    for h, m in zip(history, mask):
        z = h[:, m]
        pos, vel = z[-1, :, :2], z[-1, :, 2:]
        if len(pos) < 2:
            close_pairs.append(0.0); conflict_pairs.append(0.0); closing_strength.append(0.0)
        else:
            relp = pos[:, None] - pos[None, :]
            relv = vel[:, None] - vel[None, :]
            dist = np.linalg.norm(relp, axis=-1)
            upper = np.triu(np.ones_like(dist, dtype=bool), 1)
            close_pairs.append(float(((dist <= 20.0) & upper).sum()))
            closing = -np.sum(relp * relv, axis=-1) / np.maximum(dist, 1e-6)
            ttc = np.where(closing > 1e-3, dist / closing, np.inf)
            closest = np.linalg.norm(relp + np.clip(ttc, 0, 5)[..., None] * relv, axis=-1)
            conflict_pairs.append(float(((ttc > 0) & (ttc <= 5) & (closest <= 8) & upper).sum()))
            closing_strength.append(float(np.maximum(closing[upper], 0).mean()))
        if z.shape[0] > 1 and z.shape[1] > 0:
            maneuver.append(float(np.linalg.norm(np.diff(z[:, :, 2:], axis=0), axis=-1).mean()))
        else:
            maneuver.append(0.0)
    values = {'targets': count, 'close_pairs': np.asarray(close_pairs),
              'conflict_pairs': np.asarray(conflict_pairs), 'closing_strength': np.asarray(closing_strength),
              'maneuver': np.asarray(maneuver)}
    composite = np.mean(np.stack([ranks(v) for v in values.values()]), axis=0)
    values['composite'] = composite
    return values


def aggregate(arr, selected):
    x = arr[selected]
    den = x[:, 2].sum()
    return {'scenes': int(selected.sum()), 'targets': int(den),
            'ade_m': float(x[:, 0].sum() / (den * 20)), 'fde_m': float(x[:, 1].sum() / den)}


def main():
    q = np.load(BASE / 'physics_quantum_dual_graph_full_dev.npz')
    c = np.load(BASE / 'physics_classical_dual_graph_full_dev.npz')
    idx = q['indices']
    assert np.array_equal(idx, c['indices'])
    with np.load(ROOT / 'data/multitarget_lankershim_v1.npz') as d:
        history = d['train_states'][idx, :20]
        mask = d['train_mask'][idx]
    features = scene_features(history, mask)
    result = {'definition': 'tertiles of outcome-blind composite percentile rank over target count, close pairs, predicted conflicts, closing strength, and maneuver'}
    score = features['composite']
    cuts = np.quantile(score, [1/3, 2/3])
    groups = {'low': score <= cuts[0], 'medium': (score > cuts[0]) & (score <= cuts[1]), 'high': score > cuts[1]}
    result['composite_cuts'] = cuts.tolist(); result['groups'] = {}
    for name, selected in groups.items():
        result['groups'][name] = {}
        for snr in ('5', '10', '15', '20'):
            cm, qm = aggregate(c[snr], selected), aggregate(q[snr], selected)
            result['groups'][name][snr] = {'classical': cm, 'quantum': qm,
                'quantum_improvement_percent': {'ade': 100*(cm['ade_m']-qm['ade_m'])/cm['ade_m'],
                                                'fde': 100*(cm['fde_m']-qm['fde_m'])/cm['fde_m']}}
    result['feature_quantiles'] = {k: np.quantile(v, [0,.25,.5,.75,1]).tolist() for k,v in features.items()}
    out = BASE / 'interaction_stratification.json'
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
