"""Resumable train/validation harness for TRC/RTCN and TO-JQGNN/TR-TGN finalists. Never opens test."""
from __future__ import annotations
import argparse,hashlib,json,os,random,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from frontend.sind_prediction_dataset import SinDPredictionDataset
from prediction.qgnn_finalists.model import build_model
from prediction.q0.metrics import trajectory_metrics
from scripts.train_q0_motion_llm import token_loss
from scripts.evaluate_q0_motion_llm_val import interaction_stats

def atomic_json(path,payload):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(payload,indent=2)+'\n');tmp.replace(path)
def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
@torch.no_grad()
def evaluate(model,loader,device):
    model.eval();rows=[];index=0
    for batch in loader:
        h=batch['history_state'].to(device,non_blocking=True).float();m=batch['vehicle_mask'].to(device,non_blocking=True).bool();y=batch['future_state'].to(device,non_blocking=True).float();ts=batch['history_timestamp'].to(device,non_blocking=True)
        out=model(h,m,ts);metric=trajectory_metrics(out['prediction'],y,m);stats=interaction_stats(h,m)
        for j in range(h.shape[0]):rows.append({'index':index+j,'scene_id':int(batch['scene_id'][j]),'start_frame':int(batch['start_frame'][j]),'ADE':float(metric['scene_ade'][j]),'FDE':float(metric['scene_fde'][j]),'closing_pairs_30m_gt_0p5':int(stats['closing_pairs_30m_gt_0p5'][j])})
        index+=h.shape[0]
    ade=float(np.mean([r['ADE'] for r in rows]));fde=float(np.mean([r['FDE'] for r in rows]));return {'ADE':ade,'FDE':fde,'J':ade+.5*fde,'scenes':len(rows)},rows

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--kind',choices=['trc','rtcn','toj','trtgn'],required=True)
    p.add_argument('--seed',type=int,default=2026);p.add_argument('--snr',type=float,default=0.)
    p.add_argument('--epochs',type=int,default=20);p.add_argument('--batch-size',type=int,default=32);p.add_argument('--train-limit',type=int);p.add_argument('--val-limit',type=int)
    p.add_argument('--lr',type=float,default=3e-4);p.add_argument('--correction-cap',type=float,default=16.)
    p.add_argument('--classical-width',type=int,choices=[64,128],default=64);p.add_argument('--no-checkpoint',action='store_true')
    p.add_argument('--run-dir',required=True);p.add_argument('--resume',action='store_true');p.add_argument('--max-seconds',type=float,default=14400.);p.add_argument('--save-steps',type=int,default=100)
    args=p.parse_args();torch.set_num_threads(4);seed_all(args.seed)
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    outdir=ROOT/args.run_dir;outdir.mkdir(parents=True,exist_ok=True)
    if (outdir/'last.pt').exists() and not args.resume:raise FileExistsError('Use --resume or a new run directory')
    srcfiles=list((ROOT/'prediction/qgnn_finalists').glob('*.py'))+[Path(__file__)]
    source={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in srcfiles}
    token_hash=hashlib.sha256((ROOT/'configs/qgnn_final_tokens.json').read_bytes()).hexdigest()
    config=vars(args)|{'source_sha256':source,'token_sha256':token_hash,'pid':os.getpid(),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'test_set_used':False}
    if not args.resume:atomic_json(outdir/'config.json',config)
    model=build_model(args.kind,args.seed,args.correction_cap,args.classical_width,not args.no_checkpoint).to(device)
    digest=hashlib.sha256()
    for name,param in model.llm.named_parameters():
        if param.requires_grad and ('base.gpt2' in name or 'motion_embedding' in name or 'history_adapter' in name or 'future_queries' in name or 'token_head' in name):digest.update(name.encode());digest.update(param.detach().cpu().numpy().tobytes())
    shared_hash=digest.hexdigest()
    train=SinDPredictionDataset('train',args.snr,ROOT,True);val=SinDPredictionDataset('val',args.snr,ROOT,True)
    if args.val_limit:
        vids=np.random.default_rng(args.seed+991).permutation(len(val))[:args.val_limit].tolist();val=Subset(val,vids)
    selected=np.random.default_rng(args.seed).permutation(len(train));
    if args.train_limit:selected=selected[:args.train_limit]
    train=Subset(train,selected.tolist())
    val_loader=DataLoader(val,batch_size=args.batch_size,shuffle=False,num_workers=0,pin_memory=device.type=='cuda')
    groups=[{'params':[v for n,v in model.named_parameters() if v.requires_grad and 'lora_' not in n],'lr':args.lr},{'params':[v for n,v in model.named_parameters() if v.requires_grad and 'lora_' in n],'lr':args.lr*.25}]
    opt=torch.optim.AdamW(groups,weight_decay=2e-4);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs,eta_min=args.lr*.08)
    best=None;best_epoch=0;records=[];start_epoch=1;skip_batches=0;global_step=0;elapsed_before=0.;resumed_loss=0.;resumed_seen=0
    if args.resume:
        saved=torch.load(outdir/'last.pt',map_location='cpu',weights_only=False)
        if saved['source_sha256']!=source or saved['token_sha256']!=token_hash:raise ValueError('Resume code/token hash mismatch')
        for key in ['kind','seed','snr','epochs','batch_size','train_limit','val_limit','lr','correction_cap','classical_width','no_checkpoint']:
            if saved['config'][key]!=vars(args)[key]:raise ValueError('Resume config mismatch: '+key)
        model.load_state_dict(saved['model_state'],strict=False);opt.load_state_dict(saved['optimizer']);sched.load_state_dict(saved['scheduler'])
        random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng']);torch.set_rng_state(saved['torch_rng']);
        if torch.cuda.is_available():torch.cuda.set_rng_state_all(saved['cuda_rng'])
        start_epoch=saved['next_epoch'];skip_batches=saved['next_batch'];global_step=saved['global_step'];best=saved['best'];best_epoch=saved['best_epoch'];records=saved['records'];elapsed_before=saved['elapsed_seconds'];resumed_loss=saved['epoch_loss_sum'];resumed_seen=saved['epoch_seen']
    started=time.perf_counter()
    def elapsed():return elapsed_before+time.perf_counter()-started
    def checkpoint_save(ne,nb,ls=0.,seen=0):
        data={'model_state':{k:v.detach().cpu() for k,v in model.state_dict().items() if not ((k.startswith('llm.base.gpt2.') or k.startswith('llm.gpt2.')) and 'lora_' not in k)},'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'config':vars(args),'source_sha256':source,'token_sha256':token_hash,'next_epoch':ne,'next_batch':nb,'global_step':global_step,'best':best,'best_epoch':best_epoch,'records':records,'elapsed_seconds':elapsed(),'epoch_loss_sum':ls,'epoch_seen':seen,'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
        tmp=outdir/'last.tmp';torch.save(data,tmp);tmp.replace(outdir/'last.pt')
    def summary(status):atomic_json(outdir/'summary.json',{'status':status,'kind':args.kind,'best_validation':best,'best_epoch':best_epoch,'epochs_completed':len(records),'global_step':global_step,'elapsed_seconds':elapsed(),'parameters':model.parameter_summary(),'shared_initialization_sha256':shared_hash,'train_indices_sha256':hashlib.sha256(selected.tobytes()).hexdigest(),'train_samples':len(train),'validation_samples':len(val),'test_set_used':False})
    summary('RUNNING');print('START',json.dumps(config),flush=True)
    for epoch in range(start_epoch,args.epochs+1):
        order=torch.randperm(len(train),generator=torch.Generator().manual_seed(args.seed+1009*epoch)).tolist();skip=skip_batches if epoch==start_epoch else 0
        epoch_data=Subset(train,order[skip*args.batch_size:]);loader=DataLoader(epoch_data,batch_size=args.batch_size,shuffle=False,num_workers=0,pin_memory=device.type=='cuda')
        loss_sum=resumed_loss if epoch==start_epoch else 0.;seen=resumed_seen if epoch==start_epoch else 0;epoch_started=time.perf_counter();model.train()
        for step,batch in enumerate(loader,start=skip):
            tick=time.perf_counter();h=batch['history_state'].to(device,non_blocking=True).float();m=batch['vehicle_mask'].to(device,non_blocking=True).bool();y=batch['future_state'].to(device,non_blocking=True).float();ts=batch['history_timestamp'].to(device,non_blocking=True)
            opt.zero_grad(set_to_none=True);out=model(h,m,ts);met=trajectory_metrics(out['prediction'],y,m);tok=token_loss(out['token_logits'],model.llm.future_token_ids(h,y),m);loss=met['loss']+.035*tok
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),3.)
            if not torch.isfinite(grad):raise FloatingPointError('Nonfinite gradient')
            opt.step();
            if device.type=='cuda':torch.cuda.synchronize()
            global_step+=1;loss_sum+=float(loss.detach())*len(h);seen+=len(h);step_s=time.perf_counter()-tick
            if step%25==0:
                batches=(len(train)+args.batch_size-1)//args.batch_size;frac=epoch-1+(step+1)/max(batches,1);remain=elapsed()/max(frac,1e-6)*(args.epochs-frac)
                hb={'pid':os.getpid(),'kind':args.kind,'epoch':epoch,'total_epochs':args.epochs,'batch':step+1,'total_batches':batches,'global_step':global_step,'loss':float(loss.detach()),'ADE':float(met['ADE'].detach()),'FDE':float(met['FDE'].detach()),'gradient_norm':float(grad),'step_seconds':step_s,'elapsed_seconds':elapsed(),'estimated_remaining_seconds':remain,'estimated_finish_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(time.time()+remain)),'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30 if device.type=='cuda' else 0,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30 if device.type=='cuda' else 0}
                atomic_json(outdir/'heartbeat.json',hb);print('HEARTBEAT',json.dumps(hb),flush=True)
            if global_step%args.save_steps==0:checkpoint_save(epoch,step+1,loss_sum,seen)
            if elapsed()>args.max_seconds:checkpoint_save(epoch,step+1,loss_sum,seen);summary('PAUSED_BUDGET');print('PAUSED_BUDGET',flush=True);return
        sched.step();validation,rows=evaluate(model,val_loader,device);rec={'epoch':epoch,'train_loss':loss_sum/max(seen,1),'validation':validation,'training_seconds':time.perf_counter()-epoch_started};records.append(rec)
        if best is None or validation['J']<best['J']:
            best=validation;best_epoch=epoch;tmp=outdir/'best.tmp';torch.save({'model_state':{k:v.detach().cpu() for k,v in model.state_dict().items() if not ((k.startswith('llm.base.gpt2.') or k.startswith('llm.gpt2.')) and 'lora_' not in k)},'validation':best,'epoch':epoch,'config':vars(args),'source_sha256':source},tmp);tmp.replace(outdir/'best.pt');atomic_json(outdir/'best_validation_rows.json',{'summary':validation,'rows':rows,'test_set_used':False})
        checkpoint_save(epoch+1,0);atomic_json(outdir/'training.json',records);summary('RUNNING');print('EPOCH',json.dumps(rec),flush=True)
    summary('COMPLETED');print('COMPLETED',json.dumps(best),flush=True)
if __name__=='__main__':
    try:main()
    except Exception as exc:
        traceback.print_exc()
        if '--run-dir' in sys.argv:
            d=ROOT/sys.argv[sys.argv.index('--run-dir')+1];d.mkdir(parents=True,exist_ok=True);atomic_json(d/'failure.json',{'status':'FAILED','error':repr(exc),'traceback':traceback.format_exc()})
        raise
