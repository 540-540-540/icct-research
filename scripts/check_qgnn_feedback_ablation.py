"""CPU core sanity for fixed feedback on/off ablation; no validation/test data."""
import json, sys, hashlib
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from prediction.qgnn_adaptive.quantum import SceneAdaptiveQuantumCore
from prediction.qgnn_adaptive.classical import SceneAdaptiveClassicalCore

def neutral(core):
    with torch.no_grad():
        for module in (core.controller.scene_head,core.controller.feedback_global):
            for param in module.parameters():param.zero_()
        core.controller.relation_gain.zero_();core.controller.feedback_local.zero_()

def main():
    torch.set_num_threads(2);torch.manual_seed(4206)
    report={'status':'RUNNING','test_set_used':False,'synthetic_only':True,'checks':{}}
    for kind in ['quantum','classical']:
        torch.manual_seed(102029)
        core=(SceneAdaptiveQuantumCore(phase_mode=True) if kind=='quantum' else SceneAdaptiveClassicalCore()).double()
        neutral(core)
        x=torch.randn(2,20,8,4,dtype=torch.float64);mask=torch.ones(2,8,dtype=torch.bool);mask[0,5:]=False
        before={k:p.detach().clone() for k,p in core.state_dict().items()}
        outputs={};gradients={}
        for feedback in [True,False]:
            core.feedback_enabled=feedback;core.zero_grad(set_to_none=True)
            result=core(x,mask)
            assert result.shape==(2,8,64) and torch.isfinite(result).all() and (result[~mask]==0).all()
            (result.square().mean()).backward()
            assert all(torch.isfinite(p.grad).all() for p in core.parameters() if p.grad is not None)
            f=core.controller.feedback_global.weight
            gradients[str(feedback)]=None if f.grad is None else float(f.grad.norm())
            outputs[str(feedback)]=result.detach()
        assert torch.equal(outputs['True'],outputs['False'])
        assert gradients['True']>0 and gradients['False'] is None
        assert all(torch.equal(before[k],p) for k,p in core.state_dict().items())
        perm=torch.randperm(8)
        with torch.no_grad():
            reference=core(x,mask);permuted=core(x[:,:,perm],mask[:,perm])
            err=float((permuted-reference[:,perm]).abs().max());assert err<1e-8
        if kind=='quantum':
            xx=x[:1].clone().requires_grad_();mm=mask[:1].clone()
            cross=torch.autograd.grad(core(xx,mm)[0,0].square().sum(),xx)[0][:,:,1:].norm()
            assert cross>1e-8
        report['checks'][kind]={'initial_on_off_exactly_equal':True,'feedback_global_gradient':gradients,'off_permutation_error':err,'core_parameters':sum(p.numel() for p in core.parameters()),'no_parameter_updates':True}
    report['status']='PASS'
    out=ROOT/'reports/qgnn/round3_feedback_retrain_preflight.json'
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
