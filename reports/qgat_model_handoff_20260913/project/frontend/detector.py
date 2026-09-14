"""Anonymous fixed-waveform 3D CA-CFAR detector; no truth-side inputs."""
from functools import lru_cache
from pathlib import Path
import json
import numpy as np
from scipy import fft, ndimage
import torch
import torch.nn.functional as TF
from .ofdm_echo import DEFAULT_WAVEFORM as W, processing_windows

ROOT=Path(__file__).resolve().parents[1]
DEFAULTS=dict(threshold_multiplier=1., target_false_alarms_per_cpi=.1,
              spatial_fft_size=64, max_candidates=32, p_D=.9, device='cuda:0',
              R=np.diag([1.5**2,np.deg2rad(5.)**2,1.]).tolist(),
              training_half_width=[4,4,8], guard_half_width=[1,1,4], revision='C01',
              nms_half_width=[1,1,1], sidelobe_power_margin=1.5,
              lambda_c=1e-8)


def load_config(config=None):
    if config is None:
        path=ROOT/'configs/detector.json'
        config=json.loads(path.read_text()) if path.exists() else {}
    return dict(DEFAULTS,**config)


def _box_sum(x,size):
    return ndimage.uniform_filter(x,size=size,mode='constant',cval=0.)*np.prod(size)


@lru_cache(maxsize=2)
def _grid(outer_angle=8,guard_angle=4):
    ranges=np.arange(W.K)*W.c/(2*W.B)
    idx=np.flatnonzero((ranges>=10)&(ranges<=300))
    ranges=ranges[idx]
    u=2*fft.fftshift(fft.fftfreq(64))
    velocity=-fft.fftshift(fft.fftfreq(W.M,W.Tr))*W.wavelength/2
    elevation=np.sqrt(1-25/ranges**2)
    valid_ru=np.abs(u[None,:])<=elevation[:,None]*np.sin(np.deg2rad(70.))
    valid=np.broadcast_to(valid_ru[:,None,:],(len(idx),W.M,64)).copy()
    count=_box_sum(valid.astype(float),(9,9,2*outer_angle+1))-_box_sum(valid.astype(float),(3,3,2*guard_angle+1))
    count=np.rint(count).astype(np.float32)
    nominal_pfa=.1/np.count_nonzero(valid)
    alpha=np.where(count>0,count*np.expm1(-np.log(nominal_pfa)/np.maximum(count,1)),np.inf)
    return idx,ranges,u,velocity,valid,count,alpha,nominal_pfa


def power_map(observation):
    Y=np.asarray(observation['Y']); X=np.asarray(observation['X'])
    if Y.shape!=(W.A,W.K,W.M) or X.shape!=(W.K,W.M):
        raise ValueError('Observation must contain fixed 16x512x256 Y and 512x256 X')
    if not np.isfinite(Y).all() or not np.isfinite(X).all() or np.any(np.abs(X)<1e-12):
        raise ValueError('Nonfinite observation or zero reference symbol')
    _,wr,wm=processing_windows()
    cube=(Y/X[None,:,:])*(wr[None,:,None]*wm[None,None,:]).astype(np.float32)
    cube=fft.ifft(cube,axis=1,workers=1)[:,_grid()[0],:]
    cube=fft.fftshift(fft.fft(cube,axis=2,workers=1),axes=2)
    cube=fft.fftshift(fft.fft(cube,n=64,axis=0,workers=1),axes=0)
    cube=np.transpose(cube,(1,2,0))
    return (cube.real*cube.real+cube.imag*cube.imag).astype(np.float32)


def candidate_map_cpu(observation,config=None):
    """Local maxima scored against CA-CFAR, usable for noise calibration."""
    config=load_config(config)
    outer=config['training_half_width'][2];guard=config['guard_half_width'][2]
    power=power_map(observation)
    idx,r,u,v,valid,count,alpha,pfa=_grid(outer,guard)
    masked=np.where(valid,power,0.)
    total=_box_sum(masked.astype(float),(9,9,2*outer+1))-_box_sum(masked.astype(float),(3,3,2*guard+1))
    mean=np.maximum(total/np.maximum(count,1),np.finfo(np.float32).tiny)
    score=np.where(valid,power/(mean*alpha),0.)
    # Candidate prefilter .1 is far below the calibrated threshold, not a
    # target-count selector. NMS operates in physical resolution neighbourhoods.
    peak=(power==ndimage.maximum_filter(np.where(valid,power,-np.inf),size=tuple(2*n+1 for n in config['nms_half_width']),mode='constant',cval=-np.inf))&valid&(score>.1)
    coords=np.argwhere(peak)
    values=score[peak]
    order=np.argsort(values)[::-1]
    return coords[order],values[order],power,mean



def _torch_box(x, sizes):
    for axis,size in enumerate(sizes):
        pad=[0]*6
        pad[2*(2-axis)]=size//2
        pad[2*(2-axis)+1]=size//2
        t=TF.pad(x,pad).cumsum(axis)
        pad0=[0]*6;pad0[2*(2-axis)]=1
        t=TF.pad(t,pad0)
        x=t.narrow(axis,size,x.shape[axis])-t.narrow(axis,0,x.shape[axis])
    return x


@lru_cache(maxsize=4)
def _gpu_grid(device,outer,guard):
    idx,r,u,v,valid,count,alpha,pfa=_grid(outer,guard)
    return (torch.as_tensor(idx,device=device),torch.as_tensor(valid,device=device),
            torch.as_tensor(count,device=device),torch.as_tensor(alpha,device=device))


def candidate_map(observation,device='cuda:0',config=None):
    config=load_config(config)
    outer=config['training_half_width'][2];guard=config['guard_half_width'][2]
    if device=='cpu' or not torch.cuda.is_available(): return candidate_map_cpu(observation,config)
    Y=np.asarray(observation['Y']);X=np.asarray(observation['X'])
    if Y.shape!=(W.A,W.K,W.M) or X.shape!=(W.K,W.M):
        raise ValueError('Observation must contain fixed Y/X dimensions')
    if not np.isfinite(Y).all() or not np.isfinite(X).all() or np.any(np.abs(X)<1e-12):
        raise ValueError('Invalid observation/reference')
    with torch.no_grad():
        idx,valid,count,alpha=_gpu_grid(device,outer,guard)
        y=torch.as_tensor(Y,device=device); x=torch.as_tensor(X,device=device)
        _,wr,wm=processing_windows()
        weights=torch.as_tensor((wr[:,None]*wm[None,:]).astype(np.float32),device=device)
        cube=torch.fft.ifft(y/x[None]*weights[None],dim=1)[:,idx,:]
        cube=torch.fft.fftshift(torch.fft.fft(cube,dim=2),dim=2)
        cube=torch.fft.fftshift(torch.fft.fft(cube,n=64,dim=0),dim=0)
        power=cube.abs().square().permute(1,2,0).contiguous()
        masked=torch.where(valid,power,0.)
        # Float64 sums protect weak cells beside high dynamic-range near echoes.
        total=(_torch_box(masked.double(),(9,9,2*outer+1))-_torch_box(masked.double(),(3,3,2*guard+1))).float()
        mean=(total/count.clamp_min(1)).clamp_min(torch.finfo(torch.float32).tiny)
        score=torch.where(valid,power/(mean*alpha),0.)
        maxima=TF.max_pool3d(torch.where(valid,power,-torch.inf)[None,None],kernel_size=tuple(2*n+1 for n in config['nms_half_width']),stride=1,padding=tuple(config['nms_half_width']))[0,0]
        peaks=(power==maxima)&valid&(score>.1)
        coords=torch.nonzero(peaks)
        values=score[peaks]
        order=torch.argsort(values,descending=True)
        return coords[order].cpu().numpy(),values[order].cpu().numpy(),power.cpu().numpy(),mean.cpu().numpy()

def _parabolic(log_power,index,axis):
    i=index[axis]
    if i<=0 or i>=log_power.shape[axis]-1: return 0.
    left=list(index); right=list(index)
    left[axis]-=1; right[axis]+=1
    lm,lc,lp=float(log_power[tuple(left)]),float(log_power[index]),float(log_power[tuple(right)])
    denom=lm-2*lc+lp
    return float(np.clip(.5*(lm-lp)/denom,-.5,.5)) if denom<0 else 0.


def _spatial_leakage(delta):
    # Max over half a beam bin around the strong peak: a conservative ULA
    # sidelobe envelope, with 16 physical elements and 64 scan points.
    offsets=delta+np.linspace(-.5,.5,9)
    a=np.arange(W.A)
    return float(np.max(np.abs(np.exp(2j*np.pi*offsets[:,None]*a/64).mean(axis=1))**2))


def _select(coords,scores,power,config):
    accepted=[]; suppressed=0
    for point,score in zip(coords,scores):
        if score<config['threshold_multiplier']: break
        point=tuple(map(int,point)); discard=False
        for previous in accepted:
            delta=np.abs(np.array(point)-np.array(previous))
            if np.all(delta<=np.array(config['nms_half_width'])):
                discard=True; break
            # Merge a rectangular-array angular sidelobe only when range and
            # Doppler coincide. Different resolvable range/velocity peaks survive.
            if delta[0]<=1 and delta[1]<=1 and delta[2]>4:
                predicted=power[previous]*_spatial_leakage(point[2]-previous[2])
                if power[point]<=config['sidelobe_power_margin']*predicted:
                    discard=True; break
        if discard: suppressed+=1
        else: accepted.append(point)
    return accepted,suppressed


def detect(observation,config=None):
    config=load_config(config)
    if not isinstance(config['max_candidates'],int) or not 1<=config['max_candidates']<=32:
        raise ValueError('Candidate safety cap must be between 1 and 32')
    if config['spatial_fft_size']!=64 or (config['training_half_width'],config['guard_half_width']) not in (([4,4,2],[1,1,1]),([4,4,8],[1,1,4])):
        raise ValueError('Only the frozen waveform/CFAR dimensions are implemented')
    if not np.isfinite(config['threshold_multiplier']) or config['threshold_multiplier']<=0:
        raise ValueError('Invalid threshold multiplier')
    covariance=np.asarray(config['R'],float)
    if covariance.shape!=(3,3) or not np.isfinite(covariance).all() or not np.allclose(covariance,covariance.T) or np.linalg.eigvalsh(covariance).min()<=0:
        raise ValueError('R must be finite symmetric positive definite 3x3')
    coords,scores,power,noise=candidate_map(observation,config['device'],config)
    accepted,suppressed=_select(coords,scores,power,config)
    idx,ranges,u,velocity,valid,count,alpha,pfa=_grid(config['training_half_width'][2],config['guard_half_width'][2])
    # Only log once; interpolation uses neighbouring measured bins, never truth.
    logp=np.log(np.maximum(power,np.finfo(np.float32).tiny))
    detections=[]; invalid=0
    for point in accepted:
        dr,dd,du=(_parabolic(logp,point,axis) for axis in range(3))
        r=float(ranges[point[0]]+dr*W.c/(2*W.B))
        direction=float(u[point[2]]+du*2/64)
        if not 10<=r<=300 or r<=5:
            invalid+=1; continue
        projection=np.sqrt(1-25/r**2)
        if abs(direction)>projection:
            invalid+=1; continue
        angle=float(np.arcsin(direction/projection))
        if abs(angle)>np.deg2rad(70):
            invalid+=1; continue
        vr=float(velocity[point[1]]-dd*W.wavelength/(2*W.CPI))
        detections.append(dict(station_id=int(observation['station_id']),
            measurement_time_ns=int(observation['measurement_time_ns']),
            z=[r,angle,vr],R=covariance.tolist(),
            quality=dict(peak_power=float(power[point]),noise_floor=float(noise[point]),
                peak_to_noise=float(power[point]/noise[point]),peak_to_noise_db=float(10*np.log10(power[point]/noise[point])),cfar_score=float(power[point]/(noise[point]*alpha[point])),
                training_cells=int(count[point]),grid_index=list(point))))
    overflow=len(detections)>config['max_candidates']
    diagnostics=dict(candidate_count_before_cap=len(detections),returned_count=min(len(detections),config['max_candidates']),
        overflow=overflow,overflow_count=max(0,len(detections)-config['max_candidates']),
        sidelobe_or_nms_suppressed=suppressed,interpolation_geometry_rejected=invalid,
        nominal_cell_pfa=float(pfa),search_cells=int(valid.sum()),threshold_multiplier=config['threshold_multiplier'],
        measurement_time_ns=int(observation['measurement_time_ns']),station_id=int(observation['station_id']))
    return detections[:config['max_candidates']],diagnostics
