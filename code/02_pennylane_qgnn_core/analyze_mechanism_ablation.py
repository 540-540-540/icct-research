"""Paired time-block uncertainty for mechanism ablations versus full circuit."""
import json
from pathlib import Path
import numpy as np

BASE = Path('/home/js_cn/sensing/diagnostics/core_message_seed2026_v2')
DEV = BASE / 'development'
ABL = BASE / 'mechanism_ablation_v1'
ARMS = ('quantum_no_entanglement', 'quantum_depth1', 'quantum_no_reupload')
SNR = ('5', '10', '15', '20')


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {s: z[s].copy() for s in SNR}, z['indices'].copy()


def metrics(arrays, idx):
    vals = []
    for s in SNR:
        a = arrays[s][idx]
        den = a[:, 2].sum()
        vals.append((a[:, 0].sum() / (den * 20), a[:, 1].sum() / den))
    return np.mean(vals, axis=0)


def main():
    with np.load('/home/js_cn/sensing/data/multitarget_lankershim_v1.npz', allow_pickle=False) as z:
        starts = z['train_start_index'].copy()
    rng = np.random.default_rng(20260906)
    out = {'bootstrap_repetitions': 10000, 'block_width_frames': 200,
           'warning': 'Intervals condition on one trained seed and quantify paired development time-block variation only.',
           'comparisons': {}}
    for phase in ('graph', 'llm'):
        full, idx = load(DEV / ('quantum_' + phase + '_full_dev.npz'))
        blocks = starts[idx] // 200
        unique = np.unique(blocks)
        members = {int(b): np.flatnonzero(blocks == b) for b in unique}
        out['unique_blocks'] = int(len(unique))
        for arm in ARMS:
            ablated, idx2 = load(ABL / (arm + '_' + phase + '_full_dev.npz'))
            assert np.array_equal(idx, idx2)
            f, a = metrics(full, np.arange(len(idx))), metrics(ablated, np.arange(len(idx)))
            point = 100 * (a - f) / f
            samples = np.empty((10000, 2))
            for i in range(10000):
                chosen = rng.choice(unique, len(unique), replace=True)
                take = np.concatenate([members[int(x)] for x in chosen])
                fb, ab = metrics(full, take), metrics(ablated, take)
                samples[i] = 100 * (ab - fb) / fb
            sf, sa = [], []
            for s in SNR:
                x, y = full[s], ablated[s]
                sf.append(np.stack([x[:, 0] / (x[:, 2] * 20), x[:, 1] / x[:, 2]], 1))
                sa.append(np.stack([y[:, 0] / (y[:, 2] * 20), y[:, 1] / y[:, 2]], 1))
            sf, sa = np.mean(sf, axis=0), np.mean(sa, axis=0)
            out['comparisons'][arm + '_' + phase] = {
                'ablation_error_increase_percent': {'ade_m': float(point[0]), 'fde_m': float(point[1])},
                'block_bootstrap_95_percentile_ci': {'ade_m': np.percentile(samples[:, 0], [2.5, 97.5]).tolist(),
                                                     'fde_m': np.percentile(samples[:, 1], [2.5, 97.5]).tolist()},
                'full_circuit_scene_win_fraction': {'ade_m': float(np.mean(sf[:, 0] < sa[:, 0])),
                                                    'fde_m': float(np.mean(sf[:, 1] < sa[:, 1]))}}
    (ABL / 'paired_block_analysis.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
