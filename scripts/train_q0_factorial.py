"""Train one cell of the 2x2 Graph x Downstream LLM marginal-effect diagnostic."""
from __future__ import annotations

import argparse, json, random, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from frontend.automatum_prediction_dataset import AutomatumPredictionDataset
from prediction.q0.factorial import build_factorial_model
from prediction.q0.metrics import trajectory_metrics
from prediction.q0.normalization import load_normalization, normalization_path


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def mutable_state(model):
    return {k:v.detach().cpu() for k,v in model.state_dict().items()
            if not (k.startswith("temporal.llm.") and "lora_" not in k)}

@torch.no_grad()
def evaluate(model,loader,device):
    model.eval(); a=f=0.; n=0
    for batch in loader:
        h=batch["history_state"].to(device).float()
        m=batch["vehicle_mask"].to(device).bool()
        y=batch["future_state"].to(device).float()
        out=model(h,m); met=trajectory_metrics(out["prediction"],y,m)
        b=h.shape[0]; a+=float(met["scene_ade"].sum().cpu()); f+=float(met["scene_fde"].sum().cpu()); n+=b
    a/=n; f/=n
    return {"ADE":a,"FDE":f,"J":a+.5*f,"scenes":n}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--graph",choices=["nograph","mpnn"],required=True)
    p.add_argument("--downstream",choices=["simple","gpt2"],required=True)
    p.add_argument("--snr",type=float,default=0.)
    p.add_argument("--epochs",type=int,default=20)
    p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--lr",type=float,default=3e-4)
    p.add_argument("--seed",type=int,default=2026)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--run-dir",required=True)
    args=p.parse_args()

    seed_all(args.seed)
    device=torch.device(args.device if torch.cuda.is_available() else "cpu")
    stats=load_normalization(normalization_path(ROOT,args.snr))
    model=build_factorial_model(args.graph,args.downstream,stats,init_seed=args.seed).to(device)
    train=AutomatumPredictionDataset("train",args.snr,ROOT,True)
    val=AutomatumPredictionDataset("val",args.snr,ROOT,True)
    train_loader=DataLoader(train,batch_size=args.batch_size,shuffle=True,num_workers=0,
        pin_memory=device.type=="cuda",generator=torch.Generator().manual_seed(args.seed))
    val_loader=DataLoader(val,batch_size=args.batch_size,shuffle=False,num_workers=0)
    optim=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.lr,weight_decay=1e-4)

    outdir=ROOT/args.run_dir; outdir.mkdir(parents=True,exist_ok=True)
    (outdir/"config.json").write_text(json.dumps(vars(args),indent=2)+"\n")
    (outdir/"parameters.json").write_text(json.dumps(model.parameter_summary(),indent=2)+"\n")
    best=None; best_epoch=0; logs=[]; total=args.epochs*len(train_loader); gs=0; started=time.perf_counter()
    for epoch in range(1,args.epochs+1):
        model.train()
        for step,batch in enumerate(train_loader,1):
            h=batch["history_state"].to(device,non_blocking=True).float()
            m=batch["vehicle_mask"].to(device,non_blocking=True).bool()
            y=batch["future_state"].to(device,non_blocking=True).float()
            optim.zero_grad(set_to_none=True)
            pred=model(h,m)
            met=trajectory_metrics(pred["prediction"],y,m)
            met["loss"].backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5.)
            optim.step(); gs+=1
            if device.type=="cuda": torch.cuda.synchronize(device)
            if step==1 or step==len(train_loader) or step%50==0:
                elapsed=time.perf_counter()-started; eta=elapsed/gs*(total-gs)
                print(f"{args.graph}/{args.downstream} epoch={epoch}/{args.epochs} step={step}/{len(train_loader)} loss={met['loss'].item():.5f} ADE={met['ADE'].item():.5f} FDE={met['FDE'].item():.5f} elapsed={elapsed:.1f}s eta={eta:.1f}s",flush=True)
        v=evaluate(model,val_loader,device); rec={"epoch":epoch,"validation":v}
        logs.append(rec); print("validation="+json.dumps(rec),flush=True)
        if best is None or v["J"]<best["J"]:
            best=v; best_epoch=epoch
            torch.save({"model_state":mutable_state(model),"validation":v,"config":vars(args)},outdir/"best.pt")
    summary={"best_validation":best,"best_epoch":best_epoch,
      "elapsed_seconds":time.perf_counter()-started,"parameters":model.parameter_summary(),"test_set_used":False}
    (outdir/"training.json").write_text(json.dumps(logs,indent=2)+"\n")
    (outdir/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))

if __name__=="__main__": main()

