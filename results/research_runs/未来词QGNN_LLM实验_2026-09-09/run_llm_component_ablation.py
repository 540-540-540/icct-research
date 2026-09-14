"""Retrained future-LLM component ablations with an identical QGNN source."""
from __future__ import annotations
import argparse,json,platform,sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path('/home/js_cn/sensing');BASE=ROOT/'diagnostics/core_message_seed2026_v2'
HERE=ROOT/'双图研究框架/未来词QGNN_LLM实验_2026-09-09'
sys.path[:0]=[str(HERE),str(BASE),str(ROOT)]
import experiment_utils as utils
import run as retained
from future_token_model import future_compact_state
import future_token_model
import run_future_token_qgnn_llm as future_run

SEED=2026


def set_variant(model,variant):
    model.future_use_motion_tokens=variant!='no_motion'
    model.future_use_graph_token=variant!='no_graph'
    model.future_use_queries=variant!='no_queries'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--variant',choices=('no_motion','no_graph','no_queries'),required=True)
    parser.add_argument('--output-name',required=True);parser.add_argument('--epochs',type=int,default=16);parser.add_argument('--warmup',type=int,default=4)
    parser.add_argument('--smoke-only',action='store_true');args=parser.parse_args()
    assert Path.cwd()==ROOT and platform.node()=='jscn' and torch.cuda.is_available() and '4090' in torch.cuda.get_device_name(0)
    torch.set_num_threads(4);utils.set_seed(SEED);retained.NOISE=utils.load_snr_noise_map(ROOT/'results/multitarget_snr/snr_calibration.json')
    output=BASE/args.output_name;output.mkdir(exist_ok=False)
    train,selection,dev,train_indices,selection_indices,dev_indices=future_run.load_banks()
    assert Path(future_token_model.__file__).resolve()==(HERE/'future_token_model.py').resolve(),future_token_model.__file__
    old,model,source_checkpoint,source_dir,graph_name=future_run.restore_models('quantum')
    old=old.cuda().eval();model=model.cuda().eval();history=selection['history'][:2];mask=selection['mask'][:2]
    with torch.no_grad():full_branch=model(history,mask)['token_logits'].detach().clone()
    set_variant(model,args.variant)
    with torch.no_grad():
        ablated_output=model(history,mask)
        difference=float((old(history,mask)['future_position']-ablated_output['future_position']).abs().max())
        branch_difference=float((full_branch-ablated_output['token_logits']).abs().max())
    assert difference<2e-6,difference
    assert branch_difference>1e-6,branch_difference
    future_run.set_trainable(model,joint=True);model.train();result=model(history,mask);result['future_position'].sum().backward()
    gradient=float(model.fusion_strength.grad.abs());assert np.isfinite(gradient) and gradient>0
    utils.atomic_json(output/'smoke.json',{'variant':args.variant,'epoch0_exact_max_abs_m':difference,
       'ablated_branch_max_logit_change':branch_difference,'fusion_gradient':gradient,
       'future_token_model_source':future_token_model.__file__})
    if args.smoke_only:utils.atomic_json(output/'completed.json',{'status':'smoke_completed'});return
    utils.atomic_json(output/'protocol.json',{'seed':SEED,'variant':args.variant,'arm':'physics_quantum_dual',
       'source':str(source_dir/f'{graph_name}_llm_selected.pt'),'epochs':args.epochs,'warmup':args.warmup,
       'selection':'fixed 256 four-SNR ADE+0.35FDE; epoch0 included','retrained':True,'no_test':True})
    del old;torch.cuda.empty_cache();selected=future_run.train(model,train,selection,output,args.epochs,args.warmup)
    full,arrays=retained.evaluate(model,dev,collect=True);np.savez_compressed(output/'full_dev_predictions.npz',**arrays,indices=dev_indices)
    summary={'variant':args.variant,'selected_epoch':int(selected['epoch']),'selection':selected['metrics'],'full_dev':full,'no_test':True}
    utils.atomic_json(output/'summary.json',summary);utils.atomic_json(output/'completed.json',{'status':'completed','summary':summary});print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
