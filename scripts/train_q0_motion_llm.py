"""Train Automatum Graph + Motion-Token GPT-2 co-core diagnostic."""
from __future__ import annotations

import argparse, json, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from frontend.automatum_prediction_dataset import AutomatumPredictionDataset
from prediction.q0.graph_motion_llm import build_graph_motion_llm
from prediction.q0.metrics import trajectory_metrics
from prediction.q0.normalization import load_normalization,normalization_path


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def mutable_state(model):
    return {k:v.detach().cpu() for k,v in model.state_dict().items()
            if not (k.startswith("llm.gpt2.") and "lora_" not in k)}


def token_loss(logits, targets, mask):
    valid=mask[:,None,:].expand_as(targets)
    losses=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),targets.reshape(-1),reduction="none")
    return losses[valid.reshape(-1)].mean()


@torch.no_grad()
def evaluate(model,loader,device):
    model.eval(); a=f=0.; n=0
    for batch in loader:
        h=batch["history_state"].to(device,non_blocking=True).float()
        m=batch["vehicle_mask"].to(device,non_blocking=True).bool()
        y=batch["future_state"].to(device,non_blocking=True).float()
        out=model(h,m)
        met=trajectory_metrics(out["prediction"],y,m)
        b=h.shape[0]; a+=float(met["scene_ade"].sum().cpu()); f+=float(met["scene_fde"].sum().cpu()); n+=b
    a/=n; f/=n
    return {"ADE":a,"FDE":f,"J":a+.5*f,"scenes":n}


def subset(ds,limit,seed):
    if limit is None or limit>=len(ds): return ds
    ids=np.random.default_rng(seed).permutation(len(ds))[:limit].tolist()
    return Subset(ds,ids)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--graph",choices=["nograph","mpnn","routed_mpnn","pair_triplet"],required=True)
    p.add_argument("--snr",type=float,default=0.)
    p.add_argument("--epochs",type=int,default=20)
    p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--lr",type=float,default=3e-4)
    p.add_argument("--token-weight",type=float,default=0.035)
    p.add_argument("--seed",type=int,default=2026)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--train-limit",type=int)
    p.add_argument("--val-limit",type=int)
    p.add_argument("--graph-pretrain-checkpoint")
    p.add_argument("--freeze-graph", action="store_true")
    p.add_argument("--run-dir",required=True)
    args=p.parse_args()

    seed_all(args.seed)
    device=torch.device(args.device if torch.cuda.is_available() else "cpu")
    stats=load_normalization(normalization_path(ROOT,args.snr))
    model=build_graph_motion_llm(args.graph,stats,init_seed=args.seed)
    if args.graph_pretrain_checkpoint:
        payload=torch.load(args.graph_pretrain_checkpoint,map_location="cpu",weights_only=False)
        source={k[len("graph."):]:v for k,v in payload["model_state"].items() if k.startswith("graph.")}
        missing,unexpected=model.graph.load_state_dict(source,strict=True)
        if missing or unexpected:
            raise ValueError(f"graph pretrain mismatch missing={missing} unexpected={unexpected}")
    if args.freeze_graph:
        for param in model.graph.parameters():
            param.requires_grad=False
    model=model.to(device)

    tr=AutomatumPredictionDataset("train",args.snr,ROOT,True)
    va=AutomatumPredictionDataset("val",args.snr,ROOT,True)
    tr=subset(tr,args.train_limit,args.seed); va=subset(va,args.val_limit,args.seed+1)
    train_loader=DataLoader(tr,batch_size=args.batch_size,shuffle=True,num_workers=0,
        pin_memory=device.type=="cuda",generator=torch.Generator().manual_seed(args.seed))
    val_loader=DataLoader(va,batch_size=args.batch_size,shuffle=False,num_workers=0,
        pin_memory=device.type=="cuda")

    graph=[]; lora=[]; other=[]
    for name,param in model.named_parameters():
        if not param.requires_grad: continue
        if name.startswith("graph."): graph.append(param)
        elif "lora_" in name: lora.append(param)
        else: other.append(param)
    groups=[]
    if graph: groups.append({"params":graph,"lr":args.lr})
    if other: groups.append({"params":other,"lr":args.lr})
    if lora: groups.append({"params":lora,"lr":args.lr*.25})
    opt=torch.optim.AdamW(groups,weight_decay=2e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=max(args.epochs,1),eta_min=args.lr*.08)

    outdir=ROOT/args.run_dir; outdir.mkdir(parents=True,exist_ok=True)
    (outdir/"config.json").write_text(json.dumps(vars(args),indent=2)+"\n")
    (outdir/"parameters.json").write_text(json.dumps(model.parameter_summary(),indent=2)+"\n")

    best=None; best_epoch=0; logs=[]; steps=0; total=args.epochs*len(train_loader); started=time.perf_counter()
    for epoch in range(1,args.epochs+1):
        model.train()
        for step,batch in enumerate(train_loader,1):
            tick=time.perf_counter()
            h=batch["history_state"].to(device,non_blocking=True).float()
            m=batch["vehicle_mask"].to(device,non_blocking=True).bool()
            y=batch["future_state"].to(device,non_blocking=True).float()
            opt.zero_grad(set_to_none=True)
            out=model(h,m)
            coord=trajectory_metrics(out["prediction"],y,m)
            targets=model.llm.future_token_ids(h,y)
            tok=token_loss(out["token_logits"],targets,m)
            loss=coord["loss"]+args.token_weight*tok
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],3.)
            opt.step()
            if device.type=="cuda": torch.cuda.synchronize(device)
            steps+=1
            if step==1 or step==len(train_loader) or step%50==0:
                elapsed=time.perf_counter()-started; eta=elapsed/steps*(total-steps)
                vram=torch.cuda.max_memory_allocated(device)/2**30 if device.type=="cuda" else 0.
                print(f"{args.graph}/motionLLM epoch={epoch}/{args.epochs} step={step}/{len(train_loader)} "
                      f"loss={loss.item():.5f} coord={coord['loss'].item():.5f} tokenCE={tok.item():.4f} "
                      f"ADE={coord['ADE'].item():.5f} FDE={coord['FDE'].item():.5f} "
                      f"step_s={time.perf_counter()-tick:.3f} elapsed={elapsed:.1f}s eta={eta:.1f}s vram={vram:.2f}GB",
                      flush=True)
        scheduler.step()
        val=evaluate(model,val_loader,device)
        rec={"epoch":epoch,"validation":val}; logs.append(rec); print("validation="+json.dumps(rec),flush=True)
        if best is None or val["J"]<best["J"]:
            best=val; best_epoch=epoch
            torch.save({"model_state":mutable_state(model),"validation":val,"config":vars(args)},
                       outdir/"best.pt")

    summary={"best_validation":best,"best_epoch":best_epoch,
             "elapsed_seconds":time.perf_counter()-started,
             "parameters":model.parameter_summary(),"test_set_used":False}
    (outdir/"training.json").write_text(json.dumps(logs,indent=2)+"\n")
    (outdir/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()

