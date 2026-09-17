"""Call-local exact frame deduplication for frozen quantum evaluation only."""
import time
import numpy as np
import torch


def evaluation_graph_features(graph, state_hat, track_exists, frame_batch=20):
    """Return [B,T,N,D] readouts and cost statistics without retaining a cache.

    Invoke once per evaluation condition inside no_grad after model.eval().
    Keys contain every physical-state byte and every mask byte, so neither
    timestamps nor labels/identities participate. No values survive this call;
    a later weight update always requires a new evaluation computation.
    """
    if torch.is_grad_enabled() or graph.training:
        raise RuntimeError('Evaluation deduplication requires no_grad and graph.eval()')
    started = time.perf_counter()
    def array(value):
        return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    x, mask = array(state_hat), array(track_exists)
    if (x.ndim != 4 or x.shape[-1] != 4 or mask.shape != x.shape[:-1]
            or not 1 <= x.shape[-2] <= 8 or mask.dtype != np.bool_
            or x.dtype not in (np.float32, np.float64)):
        raise ValueError('Expected float32/64 [B,T,N,4] and bool [B,T,N]')
    if not x.shape[0] or not x.shape[1]:
        raise ValueError('Evaluation histories must be nonempty')
    n = x.shape[-2]
    flat = np.ascontiguousarray(x.reshape(-1,n,4))
    masks = np.ascontiguousarray(mask.reshape(-1,n))
    rows = np.concatenate((flat.view(np.uint8).reshape(len(flat),-1),
                           masks.view(np.uint8).reshape(len(flat),-1)),axis=1)
    keys = np.ascontiguousarray(rows).view(np.dtype((np.void,rows.shape[1]))).ravel()
    _, first, inverse = np.unique(keys,return_index=True,return_inverse=True)
    unique_x = torch.as_tensor(flat[first],device=graph.theta.device)
    unique_mask = torch.as_tensor(masks[first],device=graph.theta.device)
    values = graph.forward_history(unique_x.unsqueeze(0),unique_mask.unsqueeze(0),
                                   checkpoint=False,frame_batch=frame_batch)[0]
    features = values[torch.as_tensor(inverse,device=values.device)].reshape(
        *x.shape[:-1],graph.output_dim)
    # Returning the tensor on the graph device does not require a global cache.
    if features.is_cuda:
        torch.cuda.synchronize(features.device)
    stats = dict(frames=len(flat),unique_frames=len(first),reused_frames=len(flat)-len(first),
                 unique_fraction=len(first)/len(flat),frame_batch=frame_batch,
                 checkpoint=False,seconds=time.perf_counter()-started,
                 key='complete raw physical state and mask bytes',lifetime='single call')
    return features,stats
