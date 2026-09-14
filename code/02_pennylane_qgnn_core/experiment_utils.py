"""Self-contained deterministic utilities used by the retained experiment."""
import hashlib
import json
import math
import os
import random
import uuid
from pathlib import Path

import numpy as np
import torch

SNR_VALUES = (5, 10, 15, 20)


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_name('.%s.tmp-%s' % (path.name, uuid.uuid4().hex))
    try:
        with temporary.open('x', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def tensor_mapping_sha256(mapping):
    digest = hashlib.sha256()
    for name in sorted(mapping):
        tensor = mapping[name].detach().cpu().contiguous()
        digest.update(name.encode('utf-8'))
        digest.update(str(tensor.dtype).encode('ascii'))
        digest.update(json.dumps(list(tensor.shape), separators=(',', ':')).encode('ascii'))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def add_noise(history, mask, position_sigma, velocity_sigma, generator):
    observed = history.clone()
    if position_sigma > 0:
        observed[..., :2] += torch.randn(observed[..., :2].shape, dtype=observed.dtype,
                                         device=observed.device, generator=generator) * float(position_sigma)
    if velocity_sigma > 0:
        observed[..., 2:4] += torch.randn(observed[..., 2:4].shape, dtype=observed.dtype,
                                          device=observed.device, generator=generator) * float(velocity_sigma)
    return observed * mask[:, None, :, None].to(observed.dtype)


def load_snr_noise_map(path):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    position = {int(k): float(v['position_per_axis_rmse_m']) for k, v in payload['calibration'].items()}
    assert set(SNR_VALUES).issubset(position)
    reference = position[20]
    return {snr: {'position_sigma_m': position[snr],
                  'velocity_sigma_mps': .20 * position[snr] / reference} for snr in SNR_VALUES}


def select_profile_indices(indices, maximum):
    values = np.asarray(indices, dtype=np.int64)
    if maximum <= 0 or len(values) <= maximum:
        return values.copy()
    positions = np.linspace(0, len(values) - 1, num=maximum, dtype=np.int64)
    selected = np.unique(values[positions])
    if len(selected) != maximum:
        raise RuntimeError('Deterministic subsampling produced duplicates.')
    return selected


class InputDigest:
    def __init__(self):
        self._digest = hashlib.sha256()
    def update(self, tensor):
        value = tensor.detach().cpu().contiguous()
        self._digest.update(str(value.dtype).encode('ascii'))
        self._digest.update(json.dumps(list(value.shape), separators=(',', ':')).encode('ascii'))
        self._digest.update(value.numpy().tobytes())
    def hexdigest(self):
        return self._digest.hexdigest()
