"""One read-only model input path shared by QGNN and GNN; labels are not opened."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from .echo_source import ROOT

DATA=ROOT/'data/f01d'
FIELDS={'state_hat','track_exists','detected','timestamp'}

def input_digest(arrays):
    h=hashlib.sha256()
    for key in sorted(FIELDS):
        a=np.ascontiguousarray(arrays[key]);h.update(key.encode());h.update(a.dtype.str.encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()

def read_inputs(path):
    with np.load(path,allow_pickle=False) as f:
        if set(f.files)!=FIELDS:raise ValueError('Input cache fields must match the model-only whitelist')
        arrays={k:f[k] for k in FIELDS}
    x=arrays['state_hat'];m=arrays['track_exists'];d=arrays['detected'];t=arrays['timestamp']
    if x.ndim!=4 or x.shape[1:]!=(20,8,4) or x.dtype!=np.float32:raise ValueError('Expected float32 [S,20,8,4] state')
    if m.shape!=x.shape[:-1] or d.shape!=m.shape or m.dtype!=bool or d.dtype!=bool:raise ValueError('Invalid input masks')
    if t.shape!=x.shape[:2] or t.dtype!=np.float64:raise ValueError('Invalid timestamps')
    if not np.isfinite(x).all() or not np.isfinite(t).all() or np.any(d&~m):raise ValueError('Invalid finite values or mask relation')
    if np.any(x[~m]!=0):raise ValueError('Unavailable state must be zero padded')
    if not np.allclose(np.diff(t,axis=1),.1,rtol=0,atol=1e-6):raise ValueError('Nonuniform history clock')
    return arrays

def train_snr_files(data=DATA):
    """(level, path) pairs present in the train input cache, numeric order (F01-D/F01-E compatible)."""
    data=Path(data);entries=[]
    for path in (data/'inputs').glob('train_snr_*.npz'):
        tag=path.name[len('train_snr_'):-len('.npz')]
        try:level=int(tag.replace('m','-'))
        except ValueError:raise ValueError(f'Unparsable SNR cache name: {path.name}')
        entries.append((level,path))
    if not entries:raise ValueError('Training cache is empty')
    return sorted(set(entries))

def fit_normalization(data=DATA):
    data=Path(data);parts=[];counts={};digests={};entries=train_snr_files(data);levels=[level for level,_ in entries]
    for snr,path in entries:
        name=path.name;a=read_inputs(path)
        values=a['state_hat'][a['track_exists']].astype(np.float64)
        if not len(values):raise ValueError('Training cache is empty')
        parts.append(values);counts[str(snr)]=len(values);digests[name]=input_digest(a)
    values=np.concatenate(parts)
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for normalization; no CPU fallback')
    tensor=torch.from_numpy(values).to(device='cuda:0',dtype=torch.float64)
    mean=tensor.mean(0).cpu().tolist();std=tensor.std(0,correction=0).clamp_min(1e-3).cpu().tolist();scale=torch.quantile(tensor.abs(),.95,dim=0).clamp_min(1.).cpu().tolist()
    result=dict(fit_split='train',fields=['x','y','vx','vy'],snr_db=levels,weighting='Each valid cached training history entry counts once; repeated windows and SNR realizations retained equally; not independent traffic samples',valid_entries_by_snr=counts,valid_entries=len(values),mean=mean,std=std,quantum_scale=scale,compute_backend='torch_cuda_float64',quantum_rule='state/quantum_scale; no mean subtraction, preserves physical zero velocity',standard_rule='(state-mean)/std; unavailable entries reset to zero',input_hashes=digests)
    out=data/'normalization.json';out.write_text(json.dumps(result,indent=2)+'\n');return result

class SharedPredictionInputs:
    """Only state, causal masks and clock are exposed; identity and labels stay external."""
    def __init__(self,path,normalization=None):
        self.path=Path(path);self.arrays=read_inputs(self.path);self.input_hash=input_digest(self.arrays)
        normal=Path(normalization) if normalization else self.path.parent.parent/'normalization.json'
        self.normalization=json.loads(normal.read_text())
        if self.normalization['fit_split']!='train':raise ValueError('Normalization must be trained on train only')
        for key in ['mean','std','quantum_scale']:
            a=np.asarray(self.normalization[key],float)
            if a.shape!=(4,) or not np.isfinite(a).all():raise ValueError('Invalid normalization')
        if min(self.normalization['std'])<=0 or min(self.normalization['quantum_scale'])<=0:raise ValueError('Nonpositive normalization scale')
    def __len__(self):return len(self.arrays['state_hat'])
    def __getitem__(self,index):
        x=self.arrays['state_hat'][index].copy();m=self.arrays['track_exists'][index].copy();d=self.arrays['detected'][index].copy()
        z=((x-np.asarray(self.normalization['mean'],np.float32))/np.asarray(self.normalization['std'],np.float32)).astype(np.float32);z[~m]=0
        # Both graph models receive physical estimates; each owns its declared encoding.
        return dict(state_hat=x,standardized_state=z,track_exists=m,detected=d,timestamp=self.arrays['timestamp'][index].copy(),origin_eligible=m[-1]&(m.sum(axis=0)>=3))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['normalize']);p.add_argument('--data',type=Path,default=DATA);args=p.parse_args()
    print(json.dumps(fit_normalization(args.data),indent=2))
