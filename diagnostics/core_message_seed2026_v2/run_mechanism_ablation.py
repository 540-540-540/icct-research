"""Single-seed development-only circuit mechanism ablations."""
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
from model import CoreGraphLLM
from ablation_model import ABLATION_CONFIGS, build_ablation_graph

ARMS = tuple(ABLATION_CONFIGS)
SEED = 2026


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert Path.cwd() == ROOT
    assert platform.node() == 'jscn' and torch.cuda.is_available() and '4090' in torch.cuda.get_device_name(0)
    torch.set_num_threads(4)
    exp.r.set_seed(SEED)
    out = BASE / 'mechanism_ablation_v1'
    if out.exists():
        raise FileExistsError(out)
    out.mkdir()

    development = json.loads((BASE / 'development/completed.json').read_text())
    protocol0 = json.loads((BASE / 'development/protocol.json').read_text())
    assert development['status'] == 'completed' and development['seed'] == SEED
    for path, digest in protocol0['hashes'].items():
        assert sha(path) == digest, path
    train_idx = np.asarray(protocol0['train_indices'], dtype=np.int64)
    select_idx = np.asarray(protocol0['selection_indices'], dtype=np.int64)
    dev_idx = np.asarray(protocol0['full_dev_indices'], dtype=np.int64)
    with np.load(ROOT / 'data/multitarget_lankershim_v1.npz', allow_pickle=False) as z:
        states = z['train_states'].copy()
        masks = z['train_mask'].copy()

    def bank(idx):
        return {'history': torch.from_numpy(states[idx, :20]).float().cuda(),
                'future': torch.from_numpy(states[idx, 20:]).float().cuda(),
                'mask': torch.from_numpy(masks[idx]).bool().cuda()}
    train, select, full = bank(train_idx), bank(select_idx), bank(dev_idx)
    exp.NOISE = exp.r.load_snr_noise_map(ROOT / 'results/multitarget_snr/snr_calibration.json')

    full_q_graph = BASE / 'development/quantum_graph_selected.pt'
    full_q_llm = BASE / 'development/quantum_llm_selected.pt'
    protocol = {
        'status': 'running', 'seed': SEED, 'arms': ARMS, 'graph_epochs': 12, 'llm_epochs': 6,
        'selection': protocol0['selection'], 'training_indices_sha256': hashlib.sha256(train_idx.tobytes()).hexdigest(),
        'selection_indices_sha256': hashlib.sha256(select_idx.tobytes()).hexdigest(),
        'full_dev_indices_sha256': hashlib.sha256(dev_idx.tobytes()).hexdigest(),
        'full_quantum_reference_checkpoint_sha256': {'graph': sha(full_q_graph), 'llm': sha(full_q_llm)},
        'variants': {a: {'n_qubits': c.n_qubits, 'depth': c.depth,
                         'data_reuploading': c.data_reuploading, 'entanglement': c.entanglement}
                     for a, c in ABLATION_CONFIGS.items()},
        'scope': 'Development-only mechanism ablation; original test arrays are not loaded.',
        'script_sha256': sha(Path(__file__)), 'model_script_sha256': sha(BASE / 'ablation_model.py'),
    }
    exp.r.atomic_json(out / 'protocol.json', protocol)
    expected_inputs = {}
    summaries = {}
    llm_init = None
    for arm in ARMS:
        graph = build_ablation_graph(arm).cuda()
        graph_result = exp.train_stage(graph, arm, 'graph', 12, train, select, out, expected_inputs)
        graph_full, arr = exp.evaluate(graph, full, collect=True)
        np.savez_compressed(out / (arm + '_graph_full_dev.npz'), **arr, indices=dev_idx)
        exp.r.set_seed(SEED)
        model = CoreGraphLLM(graph).cuda()
        init = exp.r.tensor_mapping_sha256({n: p.detach().cpu() for n, p in model.named_parameters() if not n.startswith('graph_backbone.')})
        if llm_init is None:
            llm_init = init
        assert llm_init == init
        frozen = exp.r.tensor_mapping_sha256({n: p.detach().cpu() for n, p in model.named_parameters() if not p.requires_grad})
        llm_result = exp.train_stage(model, arm, 'llm', 6, train, select, out, expected_inputs)
        assert frozen == exp.r.tensor_mapping_sha256({n: p.detach().cpu() for n, p in model.named_parameters() if not p.requires_grad})
        llm_full, arr = exp.evaluate(model, full, collect=True)
        np.savez_compressed(out / (arm + '_llm_full_dev.npz'), **arr, indices=dev_idx)
        summaries[arm] = {'graph': graph_full, 'llm': llm_full,
                          'graph_selected_epoch': graph_result['selected_epoch'],
                          'llm_selected_epoch': llm_result['selected_epoch']}
        exp.r.atomic_json(out / 'summary.json', summaries)
        print(json.dumps({'arm': arm, 'graph': graph_full['aggregate'], 'llm': llm_full['aggregate']}, ensure_ascii=False), flush=True)
        del graph, model
        torch.cuda.empty_cache()

    reference = development['summaries']['quantum']
    changes = {}
    for arm in ARMS:
        changes[arm] = {}
        for phase in ('graph', 'llm'):
            changes[arm][phase] = {m: 100.0 * (summaries[arm][phase]['aggregate'][m] - reference[phase]['aggregate'][m]) /
                                   reference[phase]['aggregate'][m] for m in ('ade_m', 'fde_m')}
    # Positive values mean the ablated circuit has higher error than the complete circuit.
    protocol['status'] = 'completed'
    exp.r.atomic_json(out / 'protocol.json', protocol)
    exp.r.atomic_json(out / 'completed.json', {'status': 'completed', 'seed': SEED, 'summaries': summaries,
                                                'full_quantum_reference': reference,
                                                'ablation_error_increase_percent': changes,
                                                'no_test': True})
    exp.r.atomic_json(out / 'progress.json', {'status': 'completed'})
    assert sha(full_q_graph) == protocol['full_quantum_reference_checkpoint_sha256']['graph']
    assert sha(full_q_llm) == protocol['full_quantum_reference_checkpoint_sha256']['llm']
    assert sha(Path(__file__)) == protocol['script_sha256']
    print('COMPLETED', flush=True)


if __name__ == '__main__':
    main()
