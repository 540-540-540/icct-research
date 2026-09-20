#!/usr/bin/env python3
"""Bounded train-only numerical checks; this does not run a formal experiment."""
from __future__ import annotations
import argparse, gc, json, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from experiments.gpt2_self_diagnostic.model import build_diagnostic_model, parameter_groups
from frontend.sind_target_dataset import SinDTargetPredictionDataset
from prediction.qgnn_raj_pennylane import build_model, build_self_baseline
from prediction.q0.metrics import trajectory_metrics
from scripts.train_raj_residual_self import batch_inputs, seed_all, atomic_json

ROOT = Path(__file__).resolve().parents[2]

def clean():
    gc.collect()
    torch.cuda.empty_cache()

def load_old(model, state):
    # Existing frozen GPT weights are reconstructed from the same local checkpoint.
    state = {k:v for k,v in state.items() if not k.startswith("core.")}
    missing, extra = model.load_state_dict(state, strict=False)
    allowed = lambda k: (k.startswith("llm.base.gpt2.") and "lora_" not in k) or "heading_adapter" in k or k.startswith("core.")
    assert not extra and all(allowed(k) for k in missing), (missing, extra)

def loss_for(model, batch, device):
    history, mask, future, timestamps = batch_inputs(batch, device)
    out = model(history, mask, timestamps)
    scores = trajectory_metrics(out["prediction"], future, mask)
    weights = batch["target_weight"].to(device)
    coordinate = ((scores["scene_ade"] + .5*scores["scene_fde"])*weights).mean()
    ce = coordinate.new_zeros(())
    if hasattr(model, "llm"):
        targets = model.llm.future_token_ids(history, future)
        logits = out["token_logits"]
        values = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
        ).reshape_as(targets)
        valid = mask[:,None,:].expand_as(targets)
        ce = (((values*valid).sum((1,2))/valid.sum((1,2)))*weights).mean()
    return coordinate + .035*ce


def reference_update_check(batch, device):
    """Old and new ego controls must update alike under paired dropout."""
    old=build_model(seed=2026,phase="self",self_frame="ego_v1").to(device).train()
    new=build_diagnostic_model(kind="llm",seed=2026,frame="ego",
                              initialization="pretrained",adaptation="lora").to(device).train()
    trainable=[(n,p) for n,p in old.named_parameters() if p.requires_grad]
    original_optimizer=torch.optim.AdamW([
        {"params":[p for n,p in trainable if "lora_" not in n],"lr":3e-4},
        {"params":[p for n,p in trainable if "lora_" in n],"lr":7.5e-5}],weight_decay=2e-4)
    new_optimizer=torch.optim.AdamW(parameter_groups(new,7.5e-5,3e-4),weight_decay=2e-4)
    for step in range(3):
        for model,optimizer in ((old,original_optimizer),(new,new_optimizer)):
            torch.cuda.manual_seed_all(302029+step)
            optimizer.zero_grad(set_to_none=True)
            loss=loss_for(model,batch,device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],3.)
            optimizer.step()
    original=dict(old.named_parameters())
    error=max(float((p.detach()-original[n].detach()).abs().max()) for n,p in new.named_parameters()
              if n in original and not n.startswith("core."))
    assert error<=2e-6,("old-new ego updates",error)
    old.eval();new.eval()
    history,mask,_,timestamps=batch_inputs(batch,device)
    with torch.no_grad():
        prediction_error=float((old(history,mask,timestamps)["prediction"]-
                                new(history,mask,timestamps)["prediction"]).abs().max())
    assert prediction_error<=2e-5,("old-new ego predictions",prediction_error)
    result={"steps":3,"targets":len(history),"parameter_max_error":error,
            "prediction_max_error_m":prediction_error,"dropout_paired":True}
    del old,new,original,original_optimizer,new_optimizer,trainable,loss,model,optimizer
    clean()
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--report", default="reports/qgnn/gpt2_self_diagnostic/preflight.json")
    args=p.parse_args()
    if not 1 <= args.steps <= 8: p.error("bounded preflight requires 1..8 steps")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA profile required")
    torch.set_num_threads(4)
    seed_all(2026)
    device=torch.device("cuda:0")
    data=SinDTargetPredictionDataset("train",0.,ROOT,True)
    ids=np.random.default_rng(20260920).choice(len(data),max(args.batch_size,32),replace=False)
    batch=next(iter(DataLoader(Subset(data,ids[:args.batch_size].tolist()),batch_size=args.batch_size)))
    small=next(iter(DataLoader(Subset(data,ids[:32].tolist()),batch_size=32)))
    h,m,f,t=batch_inputs(small,device)
    results={"status":"RUNNING","test_set_used":False,"data":"train only",
             "train_targets":len(data),"batch_size":args.batch_size,"optimizer_steps_per_profile":args.steps,
             "scope":"engineering checks and short-step timings, not validation results",
             "old_replay":{}, "profiles":{}, "checks":{}}
    for kind,frame in (("tcn","global"),("llm","ego"),("llm","ego_heading")):
        old=build_self_baseline("tcn",seed=2026) if kind=="tcn" else build_model(seed=2026,phase="self",self_frame="ego_v1")
        saved=torch.load(ROOT/f"results/qgnn/self_repair_v1/target_{kind}_ego_v1_0db_seed2026/best.pt",map_location="cpu",weights_only=False)
        load_old(old,saved["model_state"])
        old.to(device).eval()
        with torch.no_grad(): expected=old(h,m,t)["prediction"].cpu()
        del old;clean()
        new=build_diagnostic_model(kind=kind,seed=2026,frame=frame,initialization="pretrained",adaptation="lora").to(device).eval()
        load_old(new,saved["model_state"])
        with torch.no_grad(): actual=new(h,m,t)["prediction"].cpu()
        error=float((actual-expected).abs().max())
        assert error <= 2e-5, (kind,frame,error)
        results["old_replay"][f"{kind}_{frame}"]={"targets":len(h),"max_prediction_error_m":error}
        del new,saved;clean()

    arms=[("tcn","ego","pretrained","lora"),
          ("llm","ego_heading","pretrained","lora"),
          ("llm","ego_heading","pretrained","full"),
          ("llm","ego_heading","random","lora"),
          ("llm","ego_heading","random","full")]
    results["checks"]["legacy_update_parity"]=reference_update_check(small,device)
    reference_peripheral=None;reference_gpt=None
    for kind,frame,initialization,adaptation in arms:
        name=f"{kind}_{frame}_{initialization}_{adaptation}"
        model=build_diagnostic_model(kind=kind,seed=2026,frame=frame,
                    initialization=initialization,adaptation=adaptation)
        if kind=="llm":
            peripheral={k:v.detach().clone() for k,v in model.state_dict().items()
                        if not k.startswith("llm.base.gpt2.") and not k.startswith("core.")}
            sample_gpt=model.state_dict()["llm.base.gpt2.h.0.attn.c_attn.base.weight"].clone()
            if reference_peripheral is None:
                reference_peripheral=peripheral
                reference_gpt=sample_gpt
            else:
                assert peripheral.keys()==reference_peripheral.keys()
                assert all(torch.equal(v,reference_peripheral[k]) for k,v in peripheral.items())
            assert torch.equal(sample_gpt,reference_gpt) == (initialization=="pretrained")
            assert not model.llm.gpt2.wte.weight.requires_grad
            if adaptation=="full":
                assert all(not p.requires_grad for k,p in model.named_parameters() if "lora_" in k)
            del peripheral,sample_gpt
        frozen={k:v.detach().cpu().clone() for k,v in model.named_parameters()
                if not v.requires_grad and not k.endswith("wte.weight")}
        trainable={k:v.detach().cpu().clone() for k,v in model.named_parameters() if v.requires_grad}
        model.to(device).eval()
        with torch.no_grad():
            zero=model(h,m,t)["prediction"]
            dt=((t[:,-1]-t[:,0])/19).to(h.dtype)
            steps=torch.arange(1,21,device=device,dtype=h.dtype)[None,:,None,None]
            cv=h[:,-1:,:,:2]+steps*dt[:,None,None,None]*h[:,-1:,:,2:]
            cv=torch.where(m[:,None,:,None],cv,0.)
            err=float((zero-cv).abs().max())
            assert err < 5e-5,(name,"epoch0 CV",err)
        optimizer=torch.optim.AdamW(parameter_groups(model,backbone_lr=7.5e-5,peripheral_lr=3e-4),weight_decay=2e-4)
        model.train()
        torch.cuda.reset_peak_memory_stats()
        timings=[]
        for i in range(args.steps):
            torch.cuda.synchronize();start=time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss=loss_for(model,batch,device)
            assert torch.isfinite(loss)
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],3.)
            assert torch.isfinite(norm)
            optimizer.step()
            torch.cuda.synchronize();timings.append(time.perf_counter()-start)
        named=dict(model.named_parameters())
        assert all(torch.equal(named[k].detach().cpu(),v) for k,v in frozen.items()),(name,"frozen changed")
        changed=[k for k,v in trainable.items() if not torch.equal(named[k].detach().cpu(),v)]
        assert changed,(name,"no update")
        if kind=="llm":
            assert any("heading_adapter" in k for k in changed)
            if adaptation=="full":
                for component in (".attn.",".mlp.",".ln_",".wpe."):
                    assert any(component in k and "lora_" not in k for k in changed),(name,component)
            else: assert any("lora_B" in k for k in changed)
        model.eval()
        with torch.no_grad():
            pred=model(h,m,t)["prediction"]
            altered=h.clone()
            altered[:,:,1:,:]=torch.randn_like(altered[:,:,1:,:])*1e4
            altered[:,:,~m[0],:]=float("nan") if bool((~m[0]).any()) else 0.
            changed_pred=model(altered,m,t)["prediction"]
            assert torch.equal(pred,changed_pred),(name,"neighbor leakage")
            if frame=="ego":
                angle=.8
                r=h.new_tensor([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
                rotated=torch.cat((h[...,:2]@r.T,h[...,2:]@r.T),-1)
                rpred=model(rotated,m,t)["prediction"]
                equiv=float((rpred-pred@r.T).abs().max())
                assert equiv<2e-4,(name,"rotation",equiv)
        used_steps=timings[1:] or timings
        results["profiles"][name]={
            "seconds_per_step":sum(used_steps)/len(used_steps),"step_seconds":timings,
            "peak_allocated_gib":torch.cuda.max_memory_allocated()/2**30,
            "peak_reserved_gib":torch.cuda.max_memory_reserved()/2**30,
            "trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),
            "total_parameters":sum(p.numel() for p in model.parameters()),
            "updated_parameter_tensors":len(changed),"frozen_unchanged":True,
            "epoch0_cv_max_error_m":err,"neighbor_invariant":True,
            "rotation_max_error_m":equiv if frame=="ego" else None}
        print(name,json.dumps(results["profiles"][name]),flush=True)
        del model,optimizer,named,frozen,trainable,pred,changed_pred,zero,loss,norm
        clean()
    results["checks"].update(shared_llm_peripheral_initialization=True,
                              initialization_factor_changes_gpt=True,unused_word_embeddings_frozen=True)
    results["status"]="PASSED"
    report=ROOT/args.report;report.parent.mkdir(parents=True,exist_ok=True);atomic_json(report,results)
    print("PASSED",report,flush=True)

if __name__=="__main__": main()
