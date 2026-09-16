"""Extend the frozen Lankershim scenes from 2 s to 4 s without resampling IDs."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path('/home/js_cn/sensing')
sys.path.insert(0,str(ROOT))
SOURCE=ROOT/'data/multitarget_lankershim_v1.npz'
OUTPUT=ROOT/'data/multitarget_lankershim_h20_p40_matched_v1.npz'
SPLIT_SOURCE=ROOT/'diagnostics/core_message_seed2026_v2/retained_inputs/circuit_split_indices.npz'
SPLIT_OUTPUT=ROOT/'diagnostics/core_message_seed2026_v2/retained_inputs/long40_circuit_split_indices.npz'
HISTORY=20; PREDICTION=40; TOTAL=60

from multitarget_scene_dataset import _vehicle_record, _extract_contiguous, _to_anchor_frame


def main():
    with np.load(SOURCE,allow_pickle=False) as old:
        metadata=json.loads(str(old['metadata'].item()))
        old_arrays={name:old[name].copy() for name in old.files if name!='metadata'}
    csv_path=Path(metadata['source_csv'])
    if not csv_path.is_absolute():csv_path=ROOT/csv_path
    columns=['Vehicle_ID','Global_Time','Local_X','Local_Y']
    frame=pd.read_csv(csv_path,usecols=columns)
    for column in columns:frame[column]=pd.to_numeric(frame[column],errors='coerce')
    frame=frame.dropna().drop_duplicates(['Vehicle_ID','Global_Time'],keep='first')
    frame=frame.sort_values(['Global_Time','Vehicle_ID'],kind='mergesort').reset_index(drop=True)
    frame['Vehicle_ID']=frame['Vehicle_ID'].astype(np.int64)
    unique_times=np.sort(frame['Global_Time'].unique());dt=float(np.median(np.diff(unique_times)))/1000.0
    time_index=pd.Series(np.arange(len(unique_times),dtype=np.int32),index=unique_times)
    frame['time_index']=frame['Global_Time'].map(time_index).astype(np.int32)
    frame['x_m']=frame['Local_X'].astype(np.float32)*0.3048;frame['y_m']=frame['Local_Y'].astype(np.float32)*0.3048
    records={int(vehicle_id):_vehicle_record(group['time_index'].to_numpy(np.int32),group[['x_m','y_m']].to_numpy(np.float32),dt)
             for vehicle_id,group in frame.groupby('Vehicle_ID',sort=False)}
    arrays={};counts={};rechecks={}
    for split in ('train','val','test'):
        states=[];masks=[];ids=[];starts=[];original=[];maximum=0.0
        old_states=old_arrays[split+'_states'];old_masks=old_arrays[split+'_mask'];old_ids=old_arrays[split+'_target_ids'];old_starts=old_arrays[split+'_start_index']
        for old_index,(mask,target_ids,start) in enumerate(zip(old_masks,old_ids,old_starts)):
            trajectories=[];complete=True
            for target_index in np.flatnonzero(mask):
                trajectory=_extract_contiguous(records[int(target_ids[target_index])],int(start),TOTAL)
                if trajectory is None:complete=False;break
                trajectories.append((int(target_index),trajectory))
            if not complete:continue
            state=np.zeros((TOTAL,old_states.shape[2],4),dtype=np.float32)
            for target_index,trajectory in trajectories:state[:,target_index]=trajectory
            state=_to_anchor_frame(state,HISTORY)
            maximum=max(maximum,float(np.max(np.abs(state[:40][:,mask]-old_states[old_index][:,mask]))))
            states.append(state);masks.append(mask);ids.append(target_ids);starts.append(start);original.append(old_index)
        arrays[split+'_states']=np.stack(states);arrays[split+'_mask']=np.stack(masks);arrays[split+'_target_ids']=np.stack(ids)
        arrays[split+'_start_index']=np.asarray(starts,dtype=np.int32);arrays[split+'_original_index']=np.asarray(original,dtype=np.int32)
        counts[split]=len(states);rechecks[split]=maximum
    metadata.update({'prediction_length':PREDICTION,'matched_source':str(SOURCE),'counts':counts,
                     'matching':'same start and all original target IDs; scenes lacking complete 4 s futures removed',
                     'first_40_max_abs_recheck_m':rechecks})
    arrays['metadata']=np.asarray(json.dumps(metadata,ensure_ascii=False));np.savez_compressed(OUTPUT,**arrays)
    with np.load(SPLIT_SOURCE,allow_pickle=False) as split:
        train_source=split['circuit_train_indices'];dev_source=split['circuit_dev_indices']
    lookup={int(old):new for new,old in enumerate(arrays['train_original_index'])}
    train=np.asarray([lookup[int(x)] for x in train_source if int(x) in lookup],dtype=np.int64)
    dev=np.asarray([lookup[int(x)] for x in dev_source if int(x) in lookup],dtype=np.int64)
    np.savez_compressed(SPLIT_OUTPUT,circuit_train_indices=train,circuit_dev_indices=dev,
                        original_train_indices=np.asarray([int(x) for x in train_source if int(x) in lookup]),
                        original_dev_indices=np.asarray([int(x) for x in dev_source if int(x) in lookup]))
    print(json.dumps({'output':str(OUTPUT),'counts':counts,'split_counts':{'train':len(train),'dev':len(dev)},
                      'first_40_max_abs_recheck_m':rechecks},indent=2))


if __name__=='__main__':main()
