"""Fresh-seed neutral initialization reproduces frozen Q/C functions, not trained weights."""
import sys,json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from prediction.qgnn_final.model import build_model as frozen
from prediction.qgnn_adaptive.model import build_model as adaptive
torch.set_num_threads(4);torch.manual_seed(4918)
x=torch.randn(2,20,8,4);m=torch.ones(2,8,dtype=torch.bool);m[0,5:]=False
r={"test_set_used":False,"fresh_initialization_only":True,"models":{}}
for kind in ["quantum","classical"]:
    old=frozen(kind).eval();new=adaptive(kind,adaptive_mode="phase_feedback",controller_init="neutral").eval()
    with torch.no_grad():
        a=old(x,m);b=new(x,m)
        errors={k:float((a[k]-b[k]).abs().max()) for k in ["prediction","token_logits","graph_features"]}
    assert max(errors.values())<2e-4
    assert new.graph.controller.enabled
    r["models"][kind]=errors;print(kind,json.dumps(errors),flush=True)
    del old,new
r["status"]="PASS";(ROOT/"reports/qgnn/round3_neutral_init_preflight.json").write_text(json.dumps(r,indent=2))
