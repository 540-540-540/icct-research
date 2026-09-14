"""Independent F01-D contract checks; source truth is used only in this audit."""
import argparse
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
from .echo_source import ROOT,SourceEpisodes
from .symbol_dataset import read_inputs,SharedPredictionInputs,input_digest
from . import symbol_dataset

SNRS=(5,10,15,20)
FIELDS={'state_hat','track_exists','detected','timestamp'}

def _same(a,b):
    return set(a)==set(b) and all(np.array_equal(a[k],b[k]) for k in a)


def check_loader_and_normalization(data):
    normal=json.loads((data/'normalization.json').read_text())
    train=[];calls=[]
    for snr in SNRS:
        arrays=read_inputs(data/'inputs'/f'train_snr_{snr}.npz')
        train.append(arrays['state_hat'][arrays['track_exists']].astype(np.float64))
    values=np.concatenate(train)
    assert normal['fit_split']=='train' and normal['snr_db']==list(SNRS)
    assert normal['valid_entries']==len(values)
    np.testing.assert_allclose(normal['mean'],values.mean(0),rtol=0,atol=1e-12)
    np.testing.assert_allclose(normal['std'],np.maximum(values.std(0),1e-3),rtol=0,atol=1e-12)
    np.testing.assert_allclose(normal['quantum_scale'],np.maximum(np.quantile(abs(values),.95,axis=0),1),rtol=0,atol=1e-12)
    def guarded_read(path):
        assert Path(path).name in {f'train_snr_{s}.npz' for s in SNRS}
        calls.append(Path(path).name)
        return read_inputs(path)
    # Run the actual fitter, refusing any attempt to open a nontraining input.
    # Its write is intercepted so the sealed normalization file stays untouched.
    with patch.object(symbol_dataset,'read_inputs',guarded_read),patch.object(Path,'write_text',return_value=0):
        recomputed=symbol_dataset.fit_normalization(data)
    assert recomputed==normal and len(calls)==4
    path=data/'inputs'/'train_snr_20.npz'
    quantum_loader=SharedPredictionInputs(path)
    classical_loader=SharedPredictionInputs(path)
    assert quantum_loader.input_hash==classical_loader.input_hash
    for index in sorted({0,len(quantum_loader)//2,len(quantum_loader)-1}):
        q=quantum_loader[index];g=classical_loader[index]
        assert _same(q,g)
        assert set(q)=={'state_hat','standardized_state','track_exists','detected','timestamp','origin_eligible'}
        assert np.array_equal(q['origin_eligible'],q['track_exists'][-1]&(q['track_exists'].sum(0)>=3))
        assert not q['standardized_state'][~q['track_exists']].any()
    work=ROOT/'.codex-work';work.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='f01d_contract_',dir=work) as temporary:
        bad=Path(temporary)/'invalid_input.npz'
        arrays=read_inputs(path)
        np.savez(bad,**{k:v[:1] for k,v in arrays.items()},source_key=np.array([[123]],dtype=np.int64))
        try:
            read_inputs(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Source identity field unexpectedly entered model loader')
    assert np.array_equal(np.zeros(4)/np.asarray(normal['quantum_scale']),np.zeros(4))
    return dict(normalization_training_only=True,normalization_entry_count=len(values),
                four_training_files_opened=calls,normalization_exact_recalculation=True,
                shared_Q_G_loader_inputs_equal=True,input_identity_extra_field_rejected=True,quantum_scaling_preserves_zero=True,
                model_forward_qualification='Both instantiate the same public input loader; no QGNN/GNN forward or training is claimed')


def fixture_checks(pack_history,build_labels):
    origin_ms=10_000
    sequence={'state_hat':np.zeros((25,8,4),np.float32),
              'track_exists':np.zeros((25,8),bool),'detected':np.zeros((25,8),bool),
              'timestamp':np.arange(8_000,10_500,100,dtype=np.float64)/1000}
    for slot in range(3):
        sequence['state_hat'][:,slot,:]=[slot+1,slot+10,0,0]
    sequence['track_exists'][5:21,0]=True  # legal partial history with left padding
    sequence['track_exists'][19:21,1]=True # only two observations, not eligible
    sequence['track_exists'][:20,2]=True  # currently absent despite earlier history
    sequence['detected']=sequence['track_exists'].copy()
    sequence['detected'][10,0]=False
    sample=pack_history(sequence,origin_ms)
    assert set(sample)==FIELDS and sample['state_hat'].shape==(20,8,4)
    assert not sample['track_exists'][:4,0].any() and sample['track_exists'][4:,0].all()
    assert sample['track_exists'][:,1].sum()==2 and not sample['track_exists'][:-2,1].any()
    assert sample['track_exists'][:,2].sum()==19 and not sample['track_exists'][-1,2]
    assert not sample['track_exists'][:,3:].any()
    assert sample['track_exists'][9,0] and not sample['detected'][9,0] and sample['state_hat'][9,0,0]==1
    assert not sample['state_hat'][~sample['track_exists']].any()
    assert np.allclose(sample['timestamp'],np.arange(8_100,10_001,100)/1000)
    assert np.all(sample['state_hat'][sample['track_exists'][:,0],0,0]==1)
    changed={k:v.copy() for k,v in sequence.items()}
    changed['state_hat'][21:]+=9999;changed['track_exists'][21:]=True;changed['detected'][21:]=True
    assert _same(sample,pack_history(changed,origin_ms))
    return dict(left_padding_and_missing_slots=True,fewer_than_three_history_retained_as_context=True,
                currently_absent_slot_past_context_retained=True,exists_without_detection_state_preserved=True,fixed_slot_values_preserved=True,
                future_estimate_rows_do_not_enter_history=True)


def check_disk_cache(data,source):
    from .pack_symbol_dataset import load_sequence,sequence_path
    summaries={};hashes={};groups={};checked_labels=0;checked_histories=0
    splits=list(dict.fromkeys(ep['split'] for ep in source.episodes))
    assert {p.name for p in (data/'inputs').glob('*.npz')}=={f'{split}_snr_{snr}.npz' for split in splits for snr in SNRS}
    for split in splits:
        metadata=json.loads((data/'metadata'/f'{split}.json').read_text())
        samples=metadata['samples']
        expected=[(i,int(t)) for i,ep in enumerate(source.episodes) if ep['split']==split for t in ep['prediction_grid_ms']]
        assert len(samples)==len(expected) and metadata['split']==split
        if split=='train':assert len(samples)<=6000
        groups[split]=set()
        for j,(record,(index,origin)) in enumerate(zip(samples,expected)):
            ep=source.episodes[index]
            assert origin-1900>=ep['start_ms']+1000 and origin+2000<ep['end_exclusive_ms']
            assert record['sample_index']==j and record['episode_index']==index and record['origin_ms']==origin
            assert record['source_keys']==ep['source_keys'] and record['episode_id']==ep['episode_id']
            assert record['source_block']==ep['source_block'] and record['source_groups']==ep['source_groups']
            assert record['track_keys']==list(range(len(ep['source_keys'])))
            groups[split].update(record['source_groups'])
        with np.load(data/'labels'/f'{split}.npz',allow_pickle=False) as file:
            assert set(file.files)=={'future_position','label_valid'}
            future=file['future_position'];valid=file['label_valid']
        assert future.shape==(len(samples),20,8,2) and future.dtype==np.float32
        assert valid.shape==(len(samples),20,8) and valid.dtype==bool
        assert np.isfinite(future).all() and not future[~valid].any()
        # Independent exact-time lookup, not the packing label helper.
        for row,(index,origin) in enumerate(expected):
            ep=source.episodes[index]
            for slot in range(8):
                if slot>=len(ep['source_keys']):
                    assert not valid[row,:,slot].any()
                    continue
                track=source.tracks[ep['source_keys'][slot]]
                for h in range(20):
                    t=origin+(h+1)*100;position=int(np.searchsorted(track['time_ms'],t))
                    found=position<len(track) and int(track[position]['time_ms'])==t
                    assert bool(valid[row,h,slot])==found
                    if found:
                        reference=np.array([track[position]['x'],track[position]['y']],dtype=np.float32)
                        assert np.array_equal(future[row,h,slot],reference)
                        checked_labels+=1
        common_masks=None
        for snr in SNRS:
            path=data/'inputs'/f'{split}_snr_{snr}.npz';arrays=read_inputs(path)
            assert len(arrays['state_hat'])==len(samples)
            assert np.array_equal(np.rint(arrays['timestamp'][:,-1]*1000).astype(np.int64),np.array([t for _,t in expected]))
            if common_masks is None:common_masks=(arrays['track_exists'],arrays['detected'])
            else:
                assert np.array_equal(common_masks[0],arrays['track_exists'])
                assert np.array_equal(common_masks[1],arrays['detected'])
            hashes[path.name]=input_digest(arrays)
            cached_index=None;sequence=None
            for row,(index,origin) in enumerate(expected):
                if index!=cached_index:
                    sequence=load_sequence(sequence_path(index,snr,data));cached_index=index
                    times=np.rint(sequence['timestamp']*1000).astype(np.int64)
                wanted=origin+np.arange(-19,1)*100
                positions=np.searchsorted(times,wanted)
                assert np.array_equal(times[positions],wanted)
                mask=sequence['track_exists'][positions]
                detected=sequence['detected'][positions]&mask
                values=sequence['state_hat'][positions].copy();values[~mask]=0
                assert np.array_equal(arrays['state_hat'][row],values)
                assert np.array_equal(arrays['track_exists'][row],mask)
                assert np.array_equal(arrays['detected'][row],detected)
                assert np.array_equal(arrays['timestamp'][row],sequence['timestamp'][positions])
                assert not mask[:,len(source.episodes[index]['source_keys']):].any()
                checked_histories+=1
        summaries[split]=dict(samples=len(samples),episodes=len(set(i for i,t in expected)),
                              label_valid_count=int(valid.sum()),invalid_future_slot_steps_including_padding=int(valid.size-valid.sum()))
    for i,split in enumerate(splits):
        for other in splits[i+1:]:assert not groups[split].intersection(groups[other])
    return dict(split_summaries=summaries,exact_original_future_xy_checks=checked_labels,
                independently_reconstructed_history_checks=checked_histories,
                four_SNR_same_origins_and_masks=True,fixed_slots_and_metadata=True,
                input_npz_whitelist=True,source_groups_disjoint=True,input_hashes=hashes)


def check_future_pack_loader(data,source):
    from .pack_symbol_dataset import pack_history,build_labels,load_sequence,sequence_path,write_npz
    indices=[i for i,ep in enumerate(source.episodes) if ep['split']=='train'][:2]
    assert len(indices)==2
    evidence=[];work=ROOT/'.codex-work';work.mkdir(exist_ok=True)
    for index in indices:
        ep=source.episodes[index];origin=int(ep['prediction_grid_ms'][0])
        sequence=load_sequence(sequence_path(index,20,data))
        baseline=pack_history(sequence,origin)
        label=build_labels(source,index,[origin])
        assert label['label_valid'].any()
        for mode in ('mutate_future','truncate_future'):
            tracks=dict(source.tracks)
            for key in ep['source_keys']:
                track=source.tracks[key].copy();is_future=track['time_ms']>origin
                if mode=='mutate_future':
                    for field in ('x','y','vx','vy'):track[field][is_future]+=12345.
                else:track=track[~is_future]
                tracks[key]=track
            seq={k:v.copy() for k,v in sequence.items()}
            future_rows=seq['timestamp']>origin/1000
            if mode=='mutate_future':
                seq['state_hat'][future_rows]+=9999;seq['track_exists'][future_rows]=False;seq['detected'][future_rows]=False
            else:seq={k:v[~future_rows] for k,v in seq.items()}
            sample=pack_history(seq,origin)
            changed_label=build_labels(source,index,[origin],tracks=tracks)
            assert _same(baseline,sample)
            assert not _same(label,changed_label)
            if mode=='truncate_future':assert not changed_label['label_valid'].any()
            with tempfile.TemporaryDirectory(prefix='f01d_future_',dir=work) as temporary:
                temp=Path(temporary)
                write_npz(temp/'original.npz',{k:v[None] for k,v in baseline.items()})
                write_npz(temp/'changed.npz',{k:v[None] for k,v in sample.items()})
                write_npz(temp/'changed_labels.npz',changed_label)
                original=SharedPredictionInputs(temp/'original.npz',data/'normalization.json')
                changed=SharedPredictionInputs(temp/'changed.npz',data/'normalization.json')
                assert len(original)==len(changed)==1 and original.input_hash==changed.input_hash
                assert _same(original[0],changed[0])
            evidence.append(dict(episode_index=index,origin_ms=origin,mode=mode,
                                 packed_inputs_identical=True,actual_loader_outputs_identical=True,
                                 label_sidecar_changed=True,origin_retained=True))
    return dict(future_pack_loader_checks=evidence,
                qualification='Fixed sealed frontend sequence/past estimates and preprocessing; alter or truncate all later source rows and cached estimate rows, repack with production pack_history/build_labels and read with actual loader. Upstream observation re-generation is separately audited in symbol_level causal_checks.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,default=ROOT/'data/f01d')
    parser.add_argument('--contracts-only',action='store_true');args=parser.parse_args()
    from .pack_symbol_dataset import pack_history,build_labels
    result=dict(stage='F01-D',fixture_checks=fixture_checks(pack_history,build_labels))
    if not args.contracts_only:
        source=SourceEpisodes()
        result.update(cache_checks=check_disk_cache(args.data,source),
                      loader_normalization_checks=check_loader_and_normalization(args.data),
                      causal_checks=check_future_pack_loader(args.data,source))
        result['scope']='All packed splits/four SNRs, raw-label checks and final pack/loader future tests; no model training'
        result['passed']=True
        out=ROOT/'reports/f01d/validation.json';out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(result,indent=2)+'\n')
    else:result['scope']='Contract fixtures only; full cache not yet validated'
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
