"""Causal source adapter and streaming three-station observations for F01-B."""
from pathlib import Path
import json
import numpy as np
from .ofdm_echo import DEFAULT_WAVEFORM as W, echo_cube

ROOT=Path(__file__).resolve().parents[1]

def cpi_schedule(deadline_ms, station):
    if station not in (0,1,2): raise ValueError('station must be 0,1,2')
    deadline_ns=int(deadline_ms)*1_000_000
    span_ns=round(W.CPI*1e9); period_ns=round(W.Tr*1e9)
    sample_ns=round(1e9/W.B); cp_ns=round(W.cp*1e9)
    start=deadline_ns-(3-station)*span_ns
    first=start+cp_ns
    last=first+(W.M-1)*period_ns+(W.K-1)*sample_ns
    centre=first+W.K*sample_ns//2+(W.M-1)*period_ns//2
    return dict(station_id=station,deadline_ns=deadline_ns,slot_start_ns=start,
                slot_end_ns=start+span_ns,reference_first_sample_ns=first,
                reference_last_sample_end_ns=last+sample_ns,measurement_time_ns=centre,
                reference_period_ns=period_ns,
                centre_definition='mean of reference useful-symbol centres, not full slot midpoint')

class SourceEpisodes:
    def __init__(self, root=ROOT):
        self.root=Path(root)
        self.rows=np.load(self.root/'data/f01_source/source_states.npy',mmap_mode='r',allow_pickle=False)
        self.episodes=[json.loads(x) for x in (self.root/'reports/f01a/episodes.jsonl').read_text().splitlines()]
        self.geometry=json.loads((self.root/'reports/f01a/geometry.json').read_text())
        keys,first,counts=np.unique(self.rows['source_key'],return_index=True,return_counts=True)
        self.tracks={int(k):self.rows[i:i+n] for k,i,n in zip(keys,first,counts)}
        self.stations=np.array(self.geometry['stations_xy_m'])
        self.bores=np.array(self.geometry['boresight_rad'])

    def at_time(self, episode, query_ns, tracks=None):
        if query_ns<int(episode['start_ms'])*1_000_000: raise ValueError('Cohort not yet selected')
        tracks=self.tracks if tracks is None else tracks
        states=[]; slots=[]; used=[]
        for slot,key in enumerate(episode['source_keys']):
            track=tracks[key]
            idx=np.searchsorted(track['time_ms'],query_ns//1_000_000,side='right')-1
            if idx<0: continue
            row=track[idx]; age_ns=query_ns-int(row['time_ms'])*1_000_000
            # Hold/extrapolate for at most one source sample interval. No future
            # existence flag or last-track-end lookup can decide activation.
            if age_ns<0 or age_ns>=100_000_000 or row['past_frames']<3: continue
            age=age_ns/1e9
            states.append([row['x']+row['vx']*age,row['y']+row['vy']*age,row['vx'],row['vy']])
            slots.append(slot); used.append(int(row['time_ms']))
        return np.asarray(states,dtype=float).reshape(-1,4),np.asarray(slots,dtype=int),used

    def _rng(self, episode_index, frame_index, station, stream):
        return np.random.default_rng(np.random.SeedSequence([2026,int(episode_index),int(frame_index),int(station),int(stream)]))

    def pressure_states(self, episode_index):
        episode=self.episodes[episode_index]
        rng=self._rng(episode_index,0,0,44)
        n=len(episode['source_keys'])
        bad=np.empty((199,3,n),dtype=bool)
        bad[0]=rng.random((3,n))<.1
        for i in range(1,len(bad)):
            u=rng.random((3,n))
            bad[i]=np.where(bad[i-1],u>=1/3,u<1/27)
        return bad

    def observation(self, episode_index, deadline_ms, station, snr_db=20., pressure=False, tracks=None):
        episode=self.episodes[episode_index]
        elapsed=int(deadline_ms)-int(episode['start_ms'])
        if elapsed%100 or not 100<=elapsed<20000: raise ValueError('Deadline must be on episode +0.1..19.9s grid')
        if pressure and snr_db!=20: raise ValueError('Predeclared pressure condition is 20dB only')
        frame_index=elapsed//100-1
        schedule=cpi_schedule(deadline_ms,station)
        states,slots,used=self.at_time(episode,schedule['measurement_time_ns'],tracks=tracks)
        # Separate streams: receiver noise never depends on how many targets exist.
        rx=self._rng(episode_index,frame_index,station,0)
        X=np.exp(1j*(np.pi/4+np.pi/2*rx.integers(0,4,(W.K,W.M))))
        phases=self._rng(episode_index,frame_index,station,1).uniform(-np.pi,np.pi,len(episode['source_keys']))[slots]
        scales=np.ones(len(states))
        if pressure:
            scales=np.where(self.pressure_states(episode_index)[frame_index,station,slots],.1,1.)
        Y,X,audit=echo_cube(states,self.stations[station],self.bores[station],X=X,phases=phases,
                           rng=self._rng(episode_index,frame_index,station,2),
                           snr_db=snr_db,per_target_amplitude_scale=scales,dtype=np.complex64)
        audit.update(schedule=schedule,episode_id=episode['episode_id'],source_slots=slots.tolist(),
                     source_sample_ms=used,pressure=pressure,
                     random_seed_recipe=[2026,episode_index,frame_index,station],
                     random_streams={'QPSK':0,'scattering_phase':1,'receiver_noise':2,'pressure':44})
        # This observation is all the downstream detector receives. audit is a
        # physically separate simulation-side artifact, forbidden to prediction.
        observation={'Y':Y,'X':X,'station_id':np.int16(station),
                     'measurement_time_ns':np.int64(schedule['measurement_time_ns']),
                     'deadline_ns':np.int64(schedule['deadline_ns'])}
        return observation,audit

    def iter_episode(self, episode_index, snr_db=20., pressure=False):
        episode=self.episodes[episode_index]
        for deadline in range(episode['start_ms']+100,episode['end_exclusive_ms'],100):
            for station in range(3):
                yield self.observation(episode_index,deadline,station,snr_db,pressure)

