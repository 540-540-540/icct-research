"""Corrected full-1200-scene association-to-forecast test evaluation."""
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/home/js_cn/sensing')
BASE = ROOT / 'diagnostics/core_message_seed2026_v2'
OUT = BASE / 'test_fixed_v1'
sys.path.insert(0, str(BASE))
import run as exp
from model import build_graph, CoreGraphLLM, restore_compact

SEED = 2026
ARMS = ('plain', 'classical', 'quantum')
SNR = (5, 10, 15, 20)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert Path.cwd() == ROOT
    assert platform.node() == 'jscn' and torch.cuda.is_available() and '4090' in torch.cuda.get_device_name(0)
    prediction = json.loads((OUT / 'prediction_completed.json').read_text())
    assert prediction['status'] == 'completed' and prediction['protocol']['scenes'] == 1200
    cache = ROOT / 'data/multitarget_lankershim_v1.npz'
    assert sha(cache) == prediction['protocol']['cache_sha256']
    with np.load(cache, allow_pickle=False) as z:
        states = z['test_states'].copy()
        masks = z['test_mask'].copy()
        metadata = json.loads(str(z['metadata'].item()))
    assert len(states) == 1200 and int(masks.sum()) == 9070

    checkpoints = {a: BASE / 'development' / (a + '_llm_selected.pt') for a in ARMS}
    expected_hashes = {a: prediction['protocol']['checkpoint_sha256'][a]['llm'] for a in ARMS}
    for a in ARMS:
        assert sha(checkpoints[a]) == expected_hashes[a]
    protocol = {
        'status': 'running',
        'correction': 'The earlier debug association call passed 24 as number_scenes. This run passes all 1200 test scenes.',
        'training_seed': SEED,
        'scenes': len(states),
        'valid_targets': int(masks.sum()),
        'snr_db': SNR,
        'association_model': 'frozen classic association GNN',
        'shared_reconstructed_histories_across_forecasters': True,
        'checkpoint_sha256': expected_hashes,
        'script_sha256': sha(Path(__file__)),
    }
    exp.r.atomic_json(OUT / 'association_full_protocol.json', protocol)

    assoc = exp.load_checkpoint(
        exp.MeasurementTrackAssociationNet(exp.AssociationGraphConfig(hidden_dim=96, graph_layers=2)),
        ROOT / 'results/measurement_track_association/association_gnn.pt').cuda().eval()
    simulation = exp.AssociationSimulationConfig(seed=SEED, dt=float(metadata['dt']))
    noise = exp.load_snr_noise(str(ROOT / 'results/multitarget_snr/snr_calibration.json'))
    reconstructed = {}
    for snr in SNR:
        exp.r.atomic_json(OUT / 'progress.json', {'status': 'full_association_reconstruction', 'snr': snr, 'scenes': len(states)})
        reconstructed[snr] = exp.reconstruct_histories(
            'association_gnn', assoc, states, masks, noise, snr, simulation, torch.device('cuda'), len(states))
        assert reconstructed[snr]['history'].shape[0] == len(states)
        print(json.dumps({'association_snr': snr, 'scenes': len(states),
                          'history_position_rmse_m': reconstructed[snr]['history_position_rmse_m']}, ensure_ascii=False), flush=True)
    del assoc
    torch.cuda.empty_cache()

    rows = {}
    input_hashes = {str(s): hashlib.sha256(x['history'].tobytes()).hexdigest() for s, x in reconstructed.items()}
    for arm in ARMS:
        exp.r.set_seed(SEED)
        model = CoreGraphLLM(build_graph(arm)).cuda()
        checkpoint = torch.load(checkpoints[arm], map_location='cpu', weights_only=True)
        restore_compact(model, checkpoint['state'])
        model.eval()
        rows[arm] = {str(s): exp.forecast_metrics(model, x['history'], states[:, 20:], masks, torch.device('cuda'), 24)
                     for s, x in reconstructed.items()}
        for snr in SNR:
            assert rows[arm][str(snr)]['valid_target_trajectories'] == int(masks.sum())
        print(json.dumps({'association_arm': arm,
                          'aggregate': {m: float(np.mean([rows[arm][str(s)][m] for s in SNR])) for m in ('ade_m', 'fde_m')}}, ensure_ascii=False), flush=True)
        del model
        torch.cuda.empty_cache()

    aggregate = {a: {m: float(np.mean([rows[a][str(s)][m] for s in SNR])) for m in ('ade_m', 'fde_m')} for a in ARMS}
    q = aggregate['quantum']
    reductions = {a: {m: 100.0 * (aggregate[a][m] - q[m]) / aggregate[a][m] for m in ('ade_m', 'fde_m')}
                  for a in ('plain', 'classical')}
    protocol['status'] = 'completed'
    exp.r.atomic_json(OUT / 'association_full_protocol.json', protocol)
    exp.r.atomic_json(OUT / 'association_test_full.json', {
        'status': 'completed', 'protocol': protocol, 'aggregate': aggregate, 'reductions_percent': reductions,
        'results': rows, 'input_sha256': input_hashes,
        'reconstruction': {str(s): {k: v for k, v in x.items() if k != 'history'} for s, x in reconstructed.items()},
    })
    for a in ARMS:
        assert sha(checkpoints[a]) == expected_hashes[a]
    assert sha(Path(__file__)) == protocol['script_sha256']
    exp.r.atomic_json(OUT / 'completed.json', {
        'status': 'completed', 'prediction_test': 'prediction_completed.json',
        'paired_test': 'paired_block_analysis.json', 'association_test': 'association_test_full.json',
        'association_scenes': len(states), 'supersedes_association_test_json_debug_scenes': 24,
    })
    exp.r.atomic_json(OUT / 'progress.json', {'status': 'completed', 'association_scenes': len(states)})
    print('COMPLETED_FULL_ASSOCIATION', flush=True)


if __name__ == '__main__':
    main()
