#!/usr/bin/env python3
"""Short parallel-run and full-GPT checkpoint/resume acceptance test."""
from __future__ import annotations
import json, os, subprocess, sys, time, shutil
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
TEMP=ROOT/".codex-work/gpt2-self-diagnostic-20260920"
REPORT=ROOT/"reports/qgnn/gpt2_self_diagnostic/runner_preflight.json"
ARMS=[
 ("0","tcn_ego","tcn","ego","pretrained","lora"),
 ("0","gpt_heading_pretrained_lora","llm","ego_heading","pretrained","lora"),
 ("0","gpt_heading_pretrained_full","llm","ego_heading","pretrained","full"),
 ("1","gpt_heading_random_lora","llm","ego_heading","random","lora"),
 ("1","gpt_heading_random_full","llm","ego_heading","random","full"),
]
def command(arm,path,steps,resume=False):
    gpu,name,kind,frame,init,adapt=arm
    cmd=[sys.executable,"-u","-m","experiments.gpt2_self_diagnostic.run",
         "--model",kind,"--frame",frame,"--initialization",init,"--adaptation",adapt,
         "--seed","2026","--epochs","20","--batch-size","256",
         "--train-limit","512","--val-limit","16","--max-steps",str(steps),
         "--save-steps","500","--max-seconds","150","--run-dir",str(path)]
    if resume: cmd.append("--resume")
    return cmd,dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,PYTHONDONTWRITEBYTECODE="1")

def run_one(arm,path,steps,resume=False):
    cmd,env=command(arm,path,steps,resume)
    log=TEMP/(path.name+("_resume" if resume else "")+".log")
    with log.open("w") as stream:
        result=subprocess.run(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=180)
    if result.returncode: raise RuntimeError(log.read_text()[-8000:])

def compare(left,right,path=""):
    if torch.is_tensor(left):
        assert left.shape==right.shape,(path,"shape")
        error=float((left-right).abs().max()) if left.numel() else 0.
        assert torch.allclose(left,right,rtol=1e-5,atol=2e-6),(path,error)
        return error
    if isinstance(left,dict):
        assert left.keys()==right.keys(),path
        return max((compare(v,right[k],path+"/"+str(k)) for k,v in left.items()),default=0.)
    if isinstance(left,(list,tuple)):
        assert len(left)==len(right),path
        return max((compare(a,b,path+f"/{i}") for i,(a,b) in enumerate(zip(left,right))),default=0.)
    assert left==right,(path,left,right)
    return 0.

def main():
    if TEMP.exists(): raise FileExistsError(f"Preserve prior preflight: {TEMP}")
    TEMP.mkdir(parents=True)
    jobs=[]
    start=time.perf_counter()
    peak={0:0.,1:0.}
    try:
        for arm in ARMS:
            path=TEMP/arm[1]
            cmd,env=command(arm,path,3)
            stream=(TEMP/(arm[1]+".log")).open("w")
            process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            jobs.append((arm,process,stream))
        print("Started five bounded workers: three optimizer steps each.",flush=True)
        while any(p.poll() is None for _,p,_ in jobs):
            if time.perf_counter()-start>180: raise TimeoutError("parallel preflight exceeded 180 seconds")
            usage=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used","--format=csv,noheader,nounits"],text=True)
            for line in usage.splitlines():
                gpu,memory=line.split(",")
                peak[int(gpu)]=max(peak[int(gpu)],float(memory)/1024)
            time.sleep(1)
        for arm,process,stream in jobs:
            stream.close()
            if process.returncode: raise RuntimeError((TEMP/(arm[1]+".log")).read_text()[-8000:])
        profiles={}
        for arm,_,_ in jobs:
            summary=json.loads((TEMP/arm[1]/"summary.json").read_text())
            assert summary["status"]=="PAUSED_STEPS" and summary["global_step"]==3,summary
            assert summary["train_samples"]==512 and summary["validation_samples"]==16
            profiles[arm[1]]={"status":summary["status"],"steps":3,"elapsed_seconds":summary["elapsed_seconds"],
                            "first_complete_epoch":json.loads((TEMP/arm[1]/"training.json").read_text())[1]}
        elapsed=time.perf_counter()-start
        print("All five bounded workers stopped. Checking random/full resume across an epoch boundary.",flush=True)
        arm=ARMS[-1]
        split=TEMP/"resume_random_full"
        run_one(arm,split,1)
        run_one(arm,split,3,True)
        baseline=torch.load(TEMP/arm[1]/"last.pt",map_location="cpu",weights_only=False)
        resumed=torch.load(split/"last.pt",map_location="cpu",weights_only=False)
        for key in ("global_step","next_epoch","next_batch","training_config","data_manifest","scheduler"):
            assert baseline[key]==resumed[key],key
        model_error=compare(baseline["model_state"],resumed["model_state"],"model")
        optimizer_error=compare(baseline["optimizer"],resumed["optimizer"],"optimizer")
        assert torch.equal(baseline["torch_rng"],resumed["torch_rng"])
        assert len(baseline["cuda_rng"])==len(resumed["cuda_rng"])
        assert all(torch.equal(a,b) for a,b in zip(baseline["cuda_rng"],resumed["cuda_rng"]))
        assert baseline["python_rng"]==resumed["python_rng"]
        assert baseline["numpy_rng"][0]==resumed["numpy_rng"][0]
        assert np.array_equal(baseline["numpy_rng"][1],resumed["numpy_rng"][1])
        assert baseline["numpy_rng"][2:]==resumed["numpy_rng"][2:]
        changed=[k for k in baseline["model_state"] if k.startswith("llm.base.gpt2.") and "wte" not in k]
        assert len(changed)>40
        report={"status":"PASSED","test_set_used":False,"scope":"bounded engineering preflight, no formal runs",
                "parallel_workers":5,"parallel_wall_seconds":elapsed,"peak_device_used_gib":peak,
                "profiles":profiles,"resume":{"mode":"random/full","global_step":3,
                  "across_epoch_boundary":True,"full_model_state_entries":len(baseline["model_state"]),
                  "model_max_abs_difference":model_error,"optimizer_max_abs_difference":optimizer_error,
                  "tensor_comparison_atol":2e-6,"tensor_comparison_rtol":1e-5,"rng_exact":True}}
        REPORT.parent.mkdir(parents=True,exist_ok=True)
        REPORT.write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps({k:v for k,v in report.items() if k!="profiles"}),flush=True)
    finally:
        for _,process,stream in jobs:
            if process.poll() is None:
                process.terminate()
                try: process.wait(timeout=30)
                except subprocess.TimeoutExpired: process.kill();process.wait()
            stream.close()
    # Remove only this explicitly verified, newly-created preflight directory.
    assert TEMP.resolve().parent==(ROOT/".codex-work").resolve()
    assert TEMP.name=="gpt2-self-diagnostic-20260920"
    shutil.rmtree(TEMP)
    print("Preflight temporary checkpoints removed; durable report preserved.",flush=True)

if __name__=="__main__": main()
