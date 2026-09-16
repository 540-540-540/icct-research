"""One fixed-checkpoint original-test evaluation for core-message v2."""
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/home/js_cn/sensing')
BASE = ROOT / 'diagnostics/core_message_seed2026_v2'
sys.path.insert(0, str(BASE))
import run as exp
from model import build_graph, CoreGraphLLM, restore_compact

SEED = 2026
ARMS = ('plain', 'classical', 'quantum')
SNR = (5, 10, 15, 20)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, obj):
    exp.r.atomic_json(path, obj)


def model_hash(model):
    return exp.r.tensor_mapping_sha256({n: p.detach().cpu() for n, p in model.named_parameters()})


def reduction(base, quantum):
    return {k: 100.0 * (base[k] - quantum[k]) / base[k] for k in ('ade_m', 'fde_m')}


def aggregate_arrays(arrays, indices):
    by_snr = {}
    for snr in SNR:
        a = arrays[str(snr)][indices]
        den = a[:, 2].sum()
        by_snr[str(snr)] = {
            'ade_m': float(a[:, 0].sum() / (den * 20.0)),
            'fde_m': float(a[:, 1].sum() / den),
        }
    return {k: float(np.mean([v[k] for v in by_snr.values()])) for k in ('ade_m', 'fde_m')}


def paired_bootstrap(all_arrays, starts, repetitions=10000):
    blocks = starts // 200
    unique = np.unique(blocks)
    members = {int(b): np.flatnonzero(blocks == b) for b in unique}
    rng = np.random.default_rng(20260906)
    result = {
        'training_seed': SEED,
        'observation_noise_seed': SEED + 100000,
        'block_width_frames': 200,
        'unique_blocks': int(len(unique)),
        'bootstrap_repetitions': repetitions,
        'warning': 'Intervals condition on one trained checkpoint and one observation-noise realization; they quantify paired test time-block variation only.',
        'comparisons': {},
    }
    for phase in ('graph', 'llm'):
        q = all_arrays[('quantum', phase)]
        for baseline in ('plain', 'classical'):
            b = all_arrays[(baseline, phase)]
            point_b, point_q = aggregate_arrays(b, np.arange(len(starts))), aggregate_arrays(q, np.arange(len(starts)))
            samples = np.empty((repetitions, 2), dtype=np.float64)
            for i in range(repetitions):
                chosen = rng.choice(unique, size=len(unique), replace=True)
                idx = np.concatenate([members[int(x)] for x in chosen])
                mb, mq = aggregate_arrays(b, idx), aggregate_arrays(q, idx)
                samples[i, 0] = 100.0 * (mb['ade_m'] - mq['ade_m']) / mb['ade_m']
                samples[i, 1] = 100.0 * (mb['fde_m'] - mq['fde_m']) / mb['fde_m']
            scene_b = []
            scene_q = []
            for snr in SNR:
                ba, qa = b[str(snr)], q[str(snr)]
                scene_b.append(np.stack([ba[:, 0] / (ba[:, 2] * 20), ba[:, 1] / ba[:, 2]], 1))
                scene_q.append(np.stack([qa[:, 0] / (qa[:, 2] * 20), qa[:, 1] / qa[:, 2]], 1))
            scene_b, scene_q = np.mean(scene_b, axis=0), np.mean(scene_q, axis=0)
            result['comparisons'][phase + '_vs_' + baseline] = {
                'reduction_percent': reduction(point_b, point_q),
                'block_bootstrap_95_percentile_ci': {
                    'ade_m': np.percentile(samples[:, 0], [2.5, 97.5]).tolist(),
                    'fde_m': np.percentile(samples[:, 1], [2.5, 97.5]).tolist(),
                },
                'scene_win_fraction_unweighted': {
                    'ade_m': float(np.mean(scene_q[:, 0] < scene_b[:, 0])),
                    'fde_m': float(np.mean(scene_q[:, 1] < scene_b[:, 1])),
                },
            }
    return result


def main():
    assert Path.cwd() == ROOT
    assert platform.node() == 'jscn' and torch.cuda.is_available() and '4090' in torch.cuda.get_device_name(0)
    torch.set_num_threads(4)
    out = BASE / 'test_fixed_v1'
    if out.exists():
        raise FileExistsError(out)
    out.mkdir()

    development = json.loads((BASE / 'development/completed.json').read_text())
    dev_protocol = json.loads((BASE / 'development/protocol.json').read_text())
    assert development['status'] == 'completed' and development['seed'] == SEED
    assert development['source_hashes_verified'] and development['no_test']
    for path, digest in dev_protocol['hashes'].items():
        assert sha(path) == digest, path

    graph_ck = {a: BASE / 'development' / (a + '_graph_selected.pt') for a in ARMS}
    llm_ck = {a: BASE / 'development' / (a + '_llm_selected.pt') for a in ARMS}
    ck_hashes = {a: {'graph': sha(graph_ck[a]), 'llm': sha(llm_ck[a])} for a in ARMS}

    cache = ROOT / 'data/multitarget_lankershim_v1.npz'
    with np.load(cache, allow_pickle=False) as z:
        states = z['test_states'].copy()
        masks = z['test_mask'].copy()
        starts = z['test_start_index'].copy()
        target_ids = z['test_target_ids'].copy()
        ranges = {s: [int(z[s + '_start_index'].min()), int(z[s + '_start_index'].max())] for s in ('train', 'val', 'test')}
        metadata = json.loads(str(z['metadata'].item()))
    length = int(metadata['history_length'] + metadata['prediction_length'])
    assert ranges['train'][1] + length <= ranges['val'][0]
    assert ranges['val'][1] + length <= ranges['test'][0]
    assert len(states) == 1200 and int(metadata['history_length']) == 20 and length == 40
    assert np.isfinite(states).all()

    bank = {
        'history': torch.from_numpy(states[:, :20]).float().cuda(),
        'future': torch.from_numpy(states[:, 20:]).float().cuda(),
        'mask': torch.from_numpy(masks).bool().cuda(),
    }
    exp.NOISE = exp.r.load_snr_noise_map(ROOT / 'results/multitarget_snr/snr_calibration.json')
    protocol = {
        'authorization': 'User requested test-set verification after reviewing development results.',
        'training_seed': SEED,
        'observation_noise_seed': SEED + 100000,
        'arms': ARMS,
        'snr_db': SNR,
        'scenes': len(states),
        'valid_targets': int(masks.sum()),
        'start_ranges': ranges,
        'temporal_windows_disjoint': True,
        'selection': 'Frozen development-selected checkpoints; no test training, tuning, calibration, or model selection.',
        'checkpoint_sha256': ck_hashes,
        'cache_sha256': sha(cache),
        'script_sha256': sha(Path(__file__)),
        'limitation': 'Existing project test set may have been inspected by historical experiments; it is not claimed as a new blind holdout.',
        'device': torch.cuda.get_device_name(0),
        'python': sys.executable,
    }
    atomic_json(out / 'protocol.json', protocol)

    rows, arrays = {}, {}
    for arm in ARMS:
        exp.r.set_seed(SEED)
        graph = build_graph(arm).cuda()
        selected = torch.load(graph_ck[arm], map_location='cpu', weights_only=True)
        graph.load_state_dict(selected['state'], strict=True)
        before = model_hash(graph)
        graph_result, scene = exp.evaluate(graph, bank, collect=True)
        assert model_hash(graph) == before and sha(graph_ck[arm]) == ck_hashes[arm]['graph']
        np.savez_compressed(out / (arm + '_graph_test.npz'), **scene, start_index=starts, target_ids=target_ids, target_mask=masks)
        arrays[(arm, 'graph')] = scene
        del graph
        torch.cuda.empty_cache()

        exp.r.set_seed(SEED)
        model = CoreGraphLLM(build_graph(arm)).cuda()
        selected = torch.load(llm_ck[arm], map_location='cpu', weights_only=True)
        restore_compact(model, selected['state'])
        before = model_hash(model)
        llm_result, scene = exp.evaluate(model, bank, collect=True)
        assert model_hash(model) == before and sha(llm_ck[arm]) == ck_hashes[arm]['llm']
        np.savez_compressed(out / (arm + '_llm_test.npz'), **scene, start_index=starts, target_ids=target_ids, target_mask=masks)
        arrays[(arm, 'llm')] = scene
        rows[arm] = {'graph': graph_result, 'llm': llm_result,
                     'graph_selected_epoch': int(torch.load(graph_ck[arm], map_location='cpu', weights_only=True)['epoch']),
                     'llm_selected_epoch': int(selected['epoch'])}
        atomic_json(out / 'progress.json', {'status': 'prediction_test', 'rows': rows})
        print(json.dumps({'arm': arm, 'graph': graph_result['aggregate'], 'llm': llm_result['aggregate']}, ensure_ascii=False), flush=True)
        del model
        torch.cuda.empty_cache()

    q = rows['quantum']
    reductions = {baseline: {phase: reduction(rows[baseline][phase]['aggregate'], q[phase]['aggregate'])
                             for phase in ('graph', 'llm')} for baseline in ('plain', 'classical')}
    paired = paired_bootstrap(arrays, starts)
    atomic_json(out / 'paired_block_analysis.json', paired)
    atomic_json(out / 'prediction_completed.json', {'status': 'completed', 'protocol': protocol, 'rows': rows, 'reductions_percent': reductions})

    # Full unknown-identity association -> Graph+LLM test chain. Reconstruct each SNR once and reuse it for every arm.
    assoc = exp.load_checkpoint(
        exp.MeasurementTrackAssociationNet(exp.AssociationGraphConfig(hidden_dim=96, graph_layers=2)),
        ROOT / 'results/measurement_track_association/association_gnn.pt').cuda().eval()
    simulation = exp.AssociationSimulationConfig(seed=SEED, dt=float(metadata['dt']))
    simnoise = exp.load_snr_noise(str(ROOT / 'results/multitarget_snr/snr_calibration.json'))
    reconstructed = {}
    for snr in SNR:
        atomic_json(out / 'progress.json', {'status': 'association_reconstruction', 'snr': snr})
        reconstructed[snr] = exp.reconstruct_histories('association_gnn', assoc, states, masks, simnoise, snr, simulation, torch.device('cuda'), 24)
        print(json.dumps({'association_snr': snr, 'history_position_rmse_m': reconstructed[snr]['history_position_rmse_m']}, ensure_ascii=False), flush=True)
    del assoc
    torch.cuda.empty_cache()
    association_rows = {}
    input_hashes = {str(s): hashlib.sha256(x['history'].tobytes()).hexdigest() for s, x in reconstructed.items()}
    for arm in ARMS:
        exp.r.set_seed(SEED)
        model = CoreGraphLLM(build_graph(arm)).cuda()
        selected = torch.load(llm_ck[arm], map_location='cpu', weights_only=True)
        restore_compact(model, selected['state'])
        model.eval()
        association_rows[arm] = {str(s): exp.forecast_metrics(model, x['history'], states[:, 20:], masks, torch.device('cuda'), 24)
                                 for s, x in reconstructed.items()}
        print(json.dumps({'association_arm': arm, 'metrics': association_rows[arm]}, ensure_ascii=False), flush=True)
        del model
        torch.cuda.empty_cache()
    atomic_json(out / 'association_test.json', {'results': association_rows, 'input_sha256': input_hashes,
        'reconstruction': {str(s): {k: v for k, v in x.items() if k != 'history'} for s, x in reconstructed.items()}})

    for arm in ARMS:
        assert sha(graph_ck[arm]) == ck_hashes[arm]['graph'] and sha(llm_ck[arm]) == ck_hashes[arm]['llm']
    assert sha(Path(__file__)) == protocol['script_sha256']
    atomic_json(out / 'completed.json', {'status': 'completed', 'prediction': rows, 'reductions_percent': reductions,
                                         'paired': paired, 'association': association_rows})
    atomic_json(out / 'progress.json', {'status': 'completed'})
    print('COMPLETED', flush=True)


if __name__ == '__main__':
    main()
