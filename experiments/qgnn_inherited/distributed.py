"""Two processes share each global batch; shared training code stays unchanged.

Initialize the process group and set the local CUDA device before install().
Configure joint-phase requires_grad flags before constructing DDP. Keep passing
the raw model to training.fit so optimizer/checkpoint parameter names stay intact.
All ranks must call forward/loss, evaluation, RNG capture and checkpoint writes
in the same order. Launch with torchrun so a failed worker terminates its peer.
"""
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def rank_zero_call(function, *args, **kwargs):
    """Broadcast success or failure before any rank proceeds past a file write."""
    message = [None]
    if dist.get_rank() == 0:
        try:
            message[0] = (True, function(*args, **kwargs))
        except Exception as exc:
            message[0] = (False, f'{type(exc).__name__}: {exc}')
    dist.broadcast_object_list(message, src=0)
    success, value = message[0]
    if not success:
        raise RuntimeError(value)
    return value


def batch_slice(length, rank=None, world=None):
    rank = dist.get_rank() if rank is None else rank
    world = dist.get_world_size() if world is None else world
    size, extra = divmod(length, world)
    start = rank * size + min(rank, extra)
    return slice(start, start + size + int(rank < extra))


def install(model, cfg):
    from prediction import training

    if not dist.is_initialized() or dist.get_world_size() != 2:
        raise RuntimeError('Initialize a two-process group before installing QGNN DDP')
    if getattr(training, '_qgnn_distributed_installed', False):
        raise RuntimeError('Install QGNN DDP only once per process')
    rank, world = dist.get_rank(), dist.get_world_size()
    device = next(model.parameters()).device
    parallel = DistributedDataParallel(
        model, device_ids=[device.index] if device.type == 'cuda' else None,
        broadcast_buffers=False)
    original_forward = training.forward
    original_loss = training.masked_trajectory_loss
    original_evaluate = training.evaluate
    original_rng = training.rng_state
    original_restore = training.restore_rng
    original_seed = training.seed_all
    original_save = training.save_checkpoint
    original_json = training.write_json

    def forward(base, inputs, phase='joint'):
        if base is not model:
            raise ValueError('Distributed runtime was installed for a different model')
        if not base.training:
            return original_forward(base, inputs, phase)
        if phase != 'joint':
            raise ValueError('Inherited QGNN DDP only supports the fixed joint phase')
        length = len(inputs['state_hat'])
        if length < world:
            raise ValueError('Distributed microbatch must contain at least two scenes')
        part = batch_slice(length)
        # Set wrapper mode explicitly: fit toggles the raw model, not this wrapper.
        parallel.training = True
        return parallel(**{key: value[part] for key, value in inputs.items()})

    def loss(prediction, future_position, label_valid, origin_eligible):
        global_n, local_n = len(future_position), len(prediction)
        if local_n == global_n:
            return original_loss(prediction, future_position, label_valid, origin_eligible)
        part = batch_slice(global_n)
        if local_n != part.stop - part.start:
            raise ValueError('Prediction batch does not match this rank global-batch slice')
        metric = original_loss(prediction, future_position[part], label_valid[part], origin_eligible)
        # DDP averages gradients. For the 13-scene tail the weights are 14/13
        # and 12/13, giving each of the 7+6 scenes exactly weight 1/13.
        scaled = metric['loss'] * (world * local_n / global_n)
        global_loss = scaled.detach().clone()
        dist.all_reduce(global_loss)
        global_loss /= world
        # Log the global mean but differentiate the locally weighted objective.
        metric['loss'] = global_loss + (scaled - scaled.detach())
        supervised = torch.tensor(int(not metric['skip_optimizer']), device=device)
        dist.all_reduce(supervised, op=dist.ReduceOp.MAX)
        metric['skip_optimizer'] = not bool(supervised.item())
        return metric

    def evaluate(base, dataset, device=None, micro_batch=1, **kwargs):
        indices = kwargs.pop('indices', None)
        indices = list(range(dataset.n)) if indices is None else list(indices)
        was_training = base.training
        base.eval()
        try:
            if len(indices) < world:
                return rank_zero_call(original_evaluate, base, dataset, device,
                                      indices=indices, micro_batch=cfg['validation_micro_batch'], **kwargs)
            try:
                local = original_evaluate(base, dataset, device,
                                          indices=indices[batch_slice(len(indices))],
                                          micro_batch=cfg['validation_micro_batch'], **kwargs)
                message = (True, local)
            except Exception as exc:
                message = (False, f'{type(exc).__name__}: {exc}')
            messages = [None] * world
            dist.all_gather_object(messages, message)
            for success, value in messages:
                if not success:
                    raise RuntimeError(value)
            parts = [value for _, value in messages]
            by_snr = {}
            for snr in parts[0]['by_snr']:
                merged = {key: sum(part['by_snr'][snr][key] for part in parts)
                          for key in ('ade_sum', 'fde_sum', 'ade_scenes', 'fde_scenes', 'total_scenes')}
                ade = merged['ade_sum'] / merged['ade_scenes']
                fde = merged['fde_sum'] / merged['fde_scenes']
                by_snr[snr] = dict(merged, ADE=ade, FDE=fde, J=ade + .5 * fde,
                                   graph_evaluation_cache=None)
            ade = sum(row['ADE'] for row in by_snr.values()) / len(by_snr)
            fde = sum(row['FDE'] for row in by_snr.values()) / len(by_snr)
            rows = sorted((row for part in parts for row in part['per_scene']),
                          key=lambda row: (row['snr_db'], row['origin']))
            return dict(ADE=ade, FDE=fde, J=ade + .5 * fde, by_snr=by_snr,
                        per_scene=rows, split=dataset.split)
        finally:
            base.train(was_training)

    def rng_state(requested_device=None):
        states = [None] * world
        dist.all_gather_object(states, original_rng(device))
        return dict(distributed_rng=states, world_size=world)

    def restore_rng(state, requested_device=None):
        if state.get('world_size') != world or len(state.get('distributed_rng', [])) != world:
            raise ValueError('Resume requires both process RNG states from this DDP run')
        original_restore(state['distributed_rng'][rank], device)

    def seed_all(seed):
        original_seed(seed)
        if device.type == 'cuda':
            with torch.cuda.device(device):
                torch.cuda.manual_seed(seed + cfg['cuda_rng_offsets'][rank])

    def save_checkpoint(path, payload):
        return rank_zero_call(original_save, path, payload)

    def write_json(path, data):
        return rank_zero_call(original_json, path, data)

    def fit(base, train, validation, run_dir, **kwargs):
        import fcntl

        lock = None

        def acquire():
            nonlocal lock
            directory = Path(run_dir)
            directory.mkdir(parents=True, exist_ok=True)
            lock = (directory / '.training.lock').open('a')
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock.close()
                lock = None
                raise RuntimeError(f'Training is already active in {run_dir}')

        rank_zero_call(acquire)
        try:
            return training._fit_impl(base, train, validation, run_dir, **kwargs)
        finally:
            if lock is not None:
                fcntl.flock(lock, fcntl.LOCK_UN)
                lock.close()

    training.forward = forward
    training.masked_trajectory_loss = loss
    training.evaluate = evaluate
    training.rng_state = rng_state
    training.restore_rng = restore_rng
    training.seed_all = seed_all
    training.save_checkpoint = save_checkpoint
    training.write_json = write_json
    training.fit = fit
    training._qgnn_distributed_installed = True
    return parallel
