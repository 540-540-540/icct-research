"""CUDA-only batched sensing production; CPU is used for source routing and file IO."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse,hashlib,json,multiprocessing,time
import numpy as np
from .echo_source import ROOT,SourceEpisodes
from .pack_symbol_dataset import DATA,CONFIG,sequence_path,load_sequence,write_json,write_npz,snr_name


def measurement_seed(episode,frame,slot):
    raw=f'2026:{episode}:{frame}:{slot}:101'.encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8],'little') & ((1<<63)-1)


def frozen_manifest():
    names=['frontend/symbol_level_gpu.py','frontend/generate_symbol_gpu.py','frontend/echo_source.py','configs/symbol_frontend.json','reports/f01a/episodes.jsonl','reports/f01a/geometry.json']
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}


def run_device(device_id,indices,batch_size):
    import torch
    from .symbol_level_gpu import synthesize_batch,divide_symbols,estimate_batch
    if not torch.cuda.is_available() or device_id>=torch.cuda.device_count():raise RuntimeError('CUDA device required; no CPU fallback')
    torch.cuda.set_device(device_id);device=f'cuda:{device_id}';source=SourceEpisodes();config=json.loads(CONFIG.read_text())
    if config['snr_db']!=[5,10,15,20] or config['nominal_snr_db']!=20:raise ValueError('Expected positive historical SNR levels')
    hashes=frozen_manifest();config_hash=hashes['configs/symbol_frontend.json'];completed=[]
    for index in indices:
        started=time.perf_counter();ep=source.episodes[index];deadlines=int(ep['start_ms'])+np.arange(1,200,dtype=np.int64)*100
        states=[];frames=[];slots=[];seeds=[]
        for f,deadline in enumerate(deadlines):
            value,slot,_=source.at_time(ep,int(deadline)*1000000)
            for s,k in zip(value,slot):
                states.append(s);frames.append(f);slots.append(int(k));seeds.append(measurement_seed(index,f+1,int(k)))
        states=np.asarray(states,np.float64).reshape(-1,4);frame_index=np.asarray(frames);slot_index=np.asarray(slots);conditions=[]
        for snr in config['snr_db']:
            path=sequence_path(index,snr);diagnostic=DATA/'diagnostics'/snr_name(snr)/f'episode_{index:03d}.json'
            if path.exists() and diagnostic.exists():
                old=json.loads(diagnostic.read_text());a=load_sequence(path)
                if old.get('frontend_hashes')!=hashes or not old.get('complete') or not np.array_equal(np.rint(a['timestamp']*1000).astype(np.int64),deadlines):raise ValueError('Stale GPU cache: '+str(path))
                conditions.append(dict(snr_db=snr,resumed=True));continue
            a=dict(state_hat=np.zeros((199,8,4),np.float32),track_exists=np.zeros((199,8),bool),detected=np.zeros((199,8),bool),timestamp=deadlines.astype(np.float64)/1000)
            report=dict(episode_index=index,split=ep['split'],snr_db=snr,revision=config['revision'],config_sha256=config_hash,frontend_hashes=hashes,backend='torch_cuda_complex128',device=device,complete=False,frames=199,outputs=len(states),diagnostic_counts={},failures=[])
            try:
                for start in range(0,len(states),batch_size):
                    stop=min(start+batch_size,len(states));data=states[start:stop]
                    with torch.inference_mode():
                        observation=synthesize_batch(data[:,:2],data[:,2:],config['stations'],waveform=config['waveform'],snr_db=snr,seeds=seeds[start:stop],device=device)
                        b=divide_symbols(**observation)
                        result=estimate_batch(b,config['stations'],config['xy_bounds'],config['v_bounds'],waveform=config['waveform'],search=config['search'])
                        values=result['state_hat'].to(device='cpu',dtype=torch.float32).numpy()
                    if values.shape!=(stop-start,4) or not np.isfinite(values).all():raise ValueError('Invalid CUDA estimates')
                    ff=frame_index[start:stop];ss=slot_index[start:stop];a['state_hat'][ff,ss]=values;a['track_exists'][ff,ss]=True;a['detected'][ff,ss]=True
                    diagnostics=result['diagnostics']
                    if len(diagnostics)!=stop-start:raise ValueError('Diagnostic batch mismatch')
                    for d in diagnostics:
                        for key,val in d.items():
                            if isinstance(val,(bool,np.bool_)):report['diagnostic_counts'][key]=report['diagnostic_counts'].get(key,0)+int(val)
                    del observation,b,result
                report['complete']=True;write_npz(path,a);write_json(diagnostic,report)
            except Exception as error:
                report['failures'].append(dict(error=repr(error)));write_json(diagnostic,report);raise
            conditions.append(dict(snr_db=snr,outputs=len(states)))
        row=dict(episode_index=index,device=device,seconds=time.perf_counter()-started,conditions=conditions);completed.append(row)
        print(json.dumps(row),flush=True)
    return completed


def generate(devices,batch_size,indices=None):
    if len(devices)!=len(set(devices)) or not devices or batch_size<1:raise ValueError('Invalid CUDA devices/batch')
    source=SourceEpisodes();indices=list(range(len(source.episodes))) if indices is None else list(indices)
    if len(indices)!=len(set(indices)) or any(i<0 or i>=len(source.episodes) for i in indices):raise ValueError('Invalid episode selection')
    started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=len(devices),mp_context=multiprocessing.get_context('spawn')) as pool:
        jobs=[pool.submit(run_device,d,indices[j::len(devices)],batch_size) for j,d in enumerate(devices)]
        results=[row for job in jobs for row in job.result()]
    report=dict(complete=True,backend='torch_cuda_complex128',devices=devices,batch_size=batch_size,elapsed_seconds=time.perf_counter()-started,episodes=sorted(results,key=lambda x:x['episode_index']),frontend_hashes=frozen_manifest(),scope='Full generation' if len(indices)==len(source.episodes) else 'Selected-episode smoke generation')
    out=ROOT/'reports/f01d'/('generation.json' if len(indices)==len(source.episodes) else 'gpu_smoke_generation.json');write_json(out,report)
    print(json.dumps(dict(complete=True,episodes=len(results),seconds=report['elapsed_seconds'])),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--devices',type=int,nargs='+',default=[0,1]);p.add_argument('--batch-size',type=int,default=16);p.add_argument('--episodes',type=int,nargs='+');a=p.parse_args();generate(a.devices,a.batch_size,a.episodes)
