"""History-only diagnostic for where the frozen RC-HQGNN helps or hurts.
Uses validation labels only for offline diagnosis. Candidate explanatory features use sensing history only.
Never loads the test split.
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name=="scripts" else Path("/home/dell/YrM/ICCT")
sys.path.insert(0,str(ROOT))
from frontend.sind_prediction_dataset import SinDPredictionDataset
from prediction.qgnn_final.model import build_model
from prediction.qgnn_final.quantum import evolve
from prediction.qgnn_final.relational import cumulants, rooted_to_global

QROWS=ROOT/"reports/qgnn/r2_final_full_quantum_0db_seed2026/best_validation_rows.json"
CROWS=ROOT/"reports/qgnn/r2_final_full_classical_0db_seed2026/best_validation_rows.json"
QCKPT=ROOT/"reports/qgnn/r2_final_full_quantum_0db_seed2026/best.pt"
OUT=ROOT/"reports/qgnn/history_only_quantum_mode_diagnostic_20260919.json"
DOC=ROOT/"docs/qgnn/HISTORY_ONLY_QUANTUM_MODE_DIAGNOSTIC_20260919.md"

def load_rows(path):
    d=json.loads(path.read_text())
    return d["rows"] if isinstance(d,dict) else d

def pair_snapshot(state, mask):
    p=state[:,:2]; v=state[:,2:]; n=len(mask)
    ii,jj=np.triu_indices(n,1)
    valid=mask[ii]&mask[jj]
    ii,jj=ii[valid],jj[valid]
    if len(ii)==0:
        return dict(ii=ii,jj=jj,d=np.array([]),c=np.array([]),tau=np.array([]),dcpa=np.array([]),rel=np.array([]),risk=np.array([]))
    r=p[jj]-p[ii]; u=v[jj]-v[ii]
    d=np.sqrt(np.maximum((r*r).sum(1),1e-6))
    dot=(r*u).sum(1); u2=np.maximum((u*u).sum(1),1e-4)
    c=-dot/d; tau_raw=-dot/u2; tau=np.clip(tau_raw,0,4)
    dcpa=np.sqrt(np.maximum(((r+tau[:,None]*u)**2).sum(1),1e-6))
    rel=np.sqrt(np.maximum((u*u).sum(1),1e-6))
    risk=np.exp(-d/30)*(0.25+1/(1+np.exp(-c/2)))*np.exp(-dcpa/20)
    return dict(ii=ii,jj=jj,d=d,c=c,tau=tau_raw,dcpa=dcpa,rel=rel,risk=risk)

def comp_max(n, ii, jj, active):
    adj=[set() for _ in range(n)]
    for a,b,on in zip(ii,jj,active):
        if on: adj[a].add(b); adj[b].add(a)
    best=0; seen=set()
    for i in range(n):
        if i in seen: continue
        stack=[i]; seen.add(i); size=0
        while stack:
            x=stack.pop(); size+=1
            for y in adj[x]:
                if y not in seen: seen.add(y); stack.append(y)
        best=max(best,size)
    return best

def scene_features(history, mask):
    n=int(mask.sum()); h=history[:,mask]
    early=h[:5].mean(0); recent=h[-5:].mean(0)
    pe=pair_snapshot(early,np.ones(n,dtype=bool)); pr=pair_snapshot(recent,np.ones(n,dtype=bool))
    d,c,tau,dcpa,rel,risk=[pr[k] for k in ("d","c","tau","dcpa","rel","risk")]
    active=(d<=30)&((c>0.5)|((tau>0)&(tau<=4)&(dcpa<=10)))
    deg=np.zeros(n,dtype=int)
    for a,b,on in zip(pr["ii"],pr["jj"],active):
        if on: deg[a]+=1;deg[b]+=1
    tri=0
    aset={(int(a),int(b)) for a,b,on in zip(pr["ii"],pr["jj"],active) if on}
    for i in range(n):
        for j in range(i+1,n):
            for k in range(j+1,n):
                if (i,j) in aset and (i,k) in aset and (j,k) in aset: tri+=1
    wedges=int(sum(x*(x-1)//2 for x in deg))
    rmat=np.zeros((n,n),dtype=float)
    for a,b,x in zip(pr["ii"],pr["jj"],risk): rmat[a,b]=rmat[b,a]=x
    tw=[]
    for i in range(n):
        for j in range(i+1,n):
            for k in range(j+1,n):
                tw.append((rmat[i,j]*rmat[i,k]+rmat[i,j]*rmat[j,k]+rmat[i,k]*rmat[j,k])/3)
    tw=np.asarray(tw,float)
    def stat(x,fn,default=0.):
        return float(fn(x)) if len(x) else float(default)
    positive=np.maximum(c,0)
    rs=risk.sum()
    rp=risk/rs if rs>0 else risk
    entropy=float(-(rp*np.log(rp+1e-12)).sum()/math.log(max(len(rp),2))) if len(rp)>1 else 0.
    speeds=np.linalg.norm(recent[:,2:],axis=1)
    out={
      "vehicle_count":n,
      "close_pairs_10m":int((d<10).sum()),"close_pairs_15m":int((d<15).sum()),
      "close_pairs_20m":int((d<20).sum()),"close_pairs_30m":int((d<30).sum()),
      "closing_pairs_30m_gt_0p5":int(((d<30)&(c>.5)).sum()),
      "closing_pairs_30m_gt_1p0":int(((d<30)&(c>1.)).sum()),
      "closing_pairs_30m_gt_2p0":int(((d<30)&(c>2.)).sum()),
      "active_edges":int(active.sum()),"max_active_degree":int(deg.max(initial=0)),
      "mean_active_degree":float(deg.mean()) if n else 0.,"nodes_deg_ge2":int((deg>=2).sum()),
      "nodes_deg_ge3":int((deg>=3).sum()),"nodes_deg_ge4":int((deg>=4).sum()),
      "active_wedges":wedges,"active_triangles":tri,"active_component_max":comp_max(n,pr["ii"],pr["jj"],active),
      "min_distance":stat(d,np.min,99.),"mean_distance":stat(d,np.mean,99.),
      "min_dcpa":stat(dcpa,np.min,99.),"mean_dcpa":stat(dcpa,np.mean,99.),
      "max_closing":stat(c,np.max,0.),"mean_positive_closing":stat(positive,np.mean,0.),
      "closing_mass_30m":float(np.maximum(c[d<30],0).sum()) if len(c) else 0.,
      "mean_relative_speed":stat(rel,np.mean,0.),"speed_std_agents":float(speeds.std()) if n else 0.,
      "pair_risk_mean":stat(risk,np.mean,0.),"pair_risk_max":stat(risk,np.max,0.),
      "pair_risk_std":stat(risk,np.std,0.),"pair_risk_entropy":entropy,
      "triple_risk_mean":stat(tw,np.mean,0.),"triple_risk_max":stat(tw,np.max,0.),
      "triple_risk_std":stat(tw,np.std,0.),"triple_risk_sum":float(tw.sum()),
    }
    if len(pe["d"])==len(d) and len(d):
        out.update({
          "distance_change_mean":float((d-pe["d"]).mean()),
          "closing_change_mean":float((c-pe["c"]).mean()),
          "risk_change_mean":float((risk-pe["risk"]).mean()),
          "risk_change_abs_mean":float(np.abs(risk-pe["risk"]).mean()),
        })
    else:
        out.update(distance_change_mean=0.,closing_change_mean=0.,risk_change_mean=0.,risk_change_abs_mean=0.)
    return out

def make_folds(scene, frame, k=5, embargo=39):
    scene=np.asarray(scene); frame=np.asarray(frame); folds=[]
    for fold in range(k):
        test=np.zeros(len(scene),bool)
        ranges=[]
        for s in np.unique(scene):
            ids=np.where(scene==s)[0]; order=ids[np.argsort(frame[ids])]
            chunks=np.array_split(order,k); part=chunks[fold]
            test[part]=True
            if len(part): ranges.append((s,int(frame[part].min()),int(frame[part].max())))
        train=~test
        for s,lo,hi in ranges:
            train &= ~((scene==s)&(frame>=lo-embargo)&(frame<=hi+embargo))
        folds.append((np.where(train)[0],np.where(test)[0]))
    return folds

def cv_predict(X,y,folds,kind="hgb",reg=False):
    pred=np.full(len(y),np.nan)
    importances=[]
    for tr,te in folds:
        if reg:
            model=make_pipeline(SimpleImputer(strategy="median"),HistGradientBoostingRegressor(max_depth=3,learning_rate=.05,max_iter=180,l2_regularization=1.,random_state=2026))
        elif kind=="linear":
            model=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),LogisticRegression(C=1.,class_weight="balanced",max_iter=3000,random_state=2026))
        else:
            model=make_pipeline(SimpleImputer(strategy="median"),HistGradientBoostingClassifier(max_depth=3,learning_rate=.05,max_iter=180,l2_regularization=1.,random_state=2026))
        model.fit(X[tr],y[tr])
        pred[te]=model.predict(X[te]) if reg else model.predict_proba(X[te])[:,1]
        if not reg and len(np.unique(y[te]))>1:
            pi=permutation_importance(model,X[te],y[te],scoring="roc_auc",n_repeats=6,random_state=2026)
            importances.append(pi.importances_mean)
    return pred, np.mean(importances,0) if importances else np.zeros(X.shape[1])

def quantile_bins(x,delta,win):
    qs=np.unique(np.quantile(x,[0,.2,.4,.6,.8,1]))
    rows=[]
    if len(qs)<3:return rows
    for a,b in zip(qs[:-1],qs[1:]):
        sel=(x>=a)&(x< b if b<qs[-1] else x<=b)
        rows.append({"lo":float(a),"hi":float(b),"n":int(sel.sum()),"deltaJ_mean":float(delta[sel].mean()),"q_win_rate":float(win[sel].mean())})
    return rows

def main():
    q=load_rows(QROWS); c=load_rows(CROWS)
    assert len(q)==len(c)==1880
    for a,b in zip(q,c):
        assert (a["index"],a["scene_id"],a["start_frame"])==(b["index"],b["scene_id"],b["start_frame"])
    ds=SinDPredictionDataset("val",0.,ROOT,False)
    feats=[]
    for i in range(len(ds)):
        d=ds[i]; feats.append(scene_features(d["history_state"],d["vehicle_mask"]))

    # Learned quantum internal signals, all deterministic functions of history + frozen checkpoint.
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model=build_model("quantum",2026,3,channels=4,quantum_version=3,correction_cap_m=16.).to(device)
    payload=torch.load(QCKPT,map_location="cpu",weights_only=False)
    model.load_state_dict(payload["model_state"],strict=False); model.eval(); core=model.graph
    dsg=SinDPredictionDataset("val",0.,ROOT,True)
    dl=DataLoader(dsg,batch_size=64,shuffle=False,num_workers=0)
    cursor=0
    with torch.no_grad():
      for b in dl:
        h=b["history_state"].to(device); m=b["vehicle_mask"].to(device).bool()
        own,risk,tw,ry,rz,pa,ta,rx,em=core.circuit_inputs(h,m)
        states=evolve(ry,rz,pa,ta,rx,em)
        base=core.readout_features(states,em,risk,tw,own.shape[0])
        rel=core.relational_features(states,own,h,m); ff=torch.cat((base,rel),-1)
        gate=torch.sigmoid(core.gate(torch.cat((own,ff),-1))).squeeze(-1)
        read=core.readout(ff)
        B=h.shape[0]; N=h.shape[2]; CH=core.channels
        pair_r=[]; tri_r=[]; k2r=[]; k3r=[]
        for st in states:
            c2,c3=cumulants(st,N)
            c2=c2.reshape(B,CH,N,N)
            pair_r.append(c2.square().mean((1,2,3)).sqrt())
            if N>=3: k3r.append(c3.reshape(B,CH,-1).square().mean((1,2)).sqrt())
        k2=torch.stack(pair_r).mean(0)
        k3=torch.stack(k3r).mean(0) if k3r else torch.zeros(B,device=device)
        for j in range(B):
            vm=m[j]
            f=feats[cursor+j]
            f["q_gate_mean"]=float(gate[j,vm].mean().cpu())
            f["q_gate_std"]=float(gate[j,vm].std(unbiased=False).cpu())
            f["q_readout_rms"]=float(read[j,vm].square().mean().sqrt().cpu())
            f["q_pair_phase_rms"]=float(pa[j*CH:(j+1)*CH].square().mean().sqrt().cpu())
            f["q_triple_phase_rms"]=float(ta[j*CH:(j+1)*CH].square().mean().sqrt().cpu())
            f["q_k2_rms"]=float(k2[j].cpu()); f["q_k3_rms"]=float(k3[j].cpu())
        cursor+=B
    assert cursor==len(feats)

    names=list(feats[0]); X=np.array([[f[n] for n in names] for f in feats],float)
    qad=np.array([r["ADE"] for r in q]); qfd=np.array([r["FDE"] for r in q])
    cad=np.array([r["ADE"] for r in c]); cfd=np.array([r["FDE"] for r in c])
    delta_ade=cad-qad; delta_fde=cfd-qfd; delta=(cad+.5*cfd)-(qad+.5*qfd); win=(delta>0).astype(int)
    scene=np.array([r["scene_id"] for r in q]); frame=np.array([r["start_frame"] for r in q])
    folds=make_folds(scene,frame)
    phys=[i for i,n in enumerate(names) if not n.startswith("q_")]
    allidx=list(range(len(names)))
    results={}
    for label,idx in [("physics_only",phys),("physics_plus_quantum_signals",allidx)]:
        xx=X[:,idx]
        p_lin,_=cv_predict(xx,win,folds,"linear",False)
        p_hgb,imp=cv_predict(xx,win,folds,"hgb",False)
        p_reg,_=cv_predict(xx,delta,folds,"hgb",True)
        results[label]={
          "logistic_auc":float(roc_auc_score(win,p_lin)),
          "hgb_auc":float(roc_auc_score(win,p_hgb)),
          "hgb_average_precision":float(average_precision_score(win,p_hgb)),
          "hgb_balanced_accuracy_at_0p5":float(balanced_accuracy_score(win,p_hgb>=.5)),
          "deltaJ_regression_spearman":float(spearmanr(delta,p_reg).statistic),
          "deltaJ_regression_r2":float(r2_score(delta,p_reg)),
          "top_permutation_importance":[{"feature":names[idx[j]],"auc_drop":float(imp[j])} for j in np.argsort(-imp)[:15]],
        }
    uni=[]
    for j,n in enumerate(names):
        rho=float(spearmanr(X[:,j],delta).statistic)
        if not np.isfinite(rho): rho=0.
        try: auc=float(roc_auc_score(win,X[:,j]))
        except: auc=.5
        auc_dir=max(auc,1-auc)
        uni.append({"feature":n,"spearman_deltaJ":rho,"directional_auc":auc_dir})
    uni=sorted(uni,key=lambda z:abs(z["spearman_deltaJ"]),reverse=True)

    bin_features=["closing_pairs_30m_gt_0p5","active_edges","active_wedges","active_triangles",
                  "max_active_degree","nodes_deg_ge3","pair_risk_entropy","triple_risk_max",
                  "triple_risk_sum","risk_change_abs_mean","q_pair_phase_rms","q_triple_phase_rms",
                  "q_k2_rms","q_k3_rms","q_gate_mean"]
    bins={n:quantile_bins(X[:,names.index(n)],delta,win) for n in bin_features}
    report={
      "scope":{"split":"val","snr_db":0.0,"scenes":len(feats),"test_set_used":False,
               "label_definition":"deltaJ=(classical ADE+0.5FDE)-(quantum ADE+0.5FDE); positive means frozen RC-HQGNN wins",
               "feature_rule":"all candidate features use sensing history only; q_* are deterministic frozen-QGNN internal signals from history",
               "cv":"5 chronological blocks per scene with 39-frame embargo around each held-out block; diagnostic only, not a deployable router estimate"},
      "global":{"quantum_win_rate_J":float(win.mean()),"deltaJ_mean":float(delta.mean()),
                "deltaADE_mean":float(delta_ade.mean()),"deltaFDE_mean":float(delta_fde.mean()),
                "oracle_min_gain_ADE_pct":float(100*(1-np.minimum(qad,cad).mean()/cad.mean())),
                "oracle_min_gain_FDE_pct":float(100*(1-np.minimum(qfd,cfd).mean()/cfd.mean()))},
      "predictability":results,
      "univariate_top20":uni[:20],
      "quantile_bins":bins,
      "feature_names":names,
      "per_scene":[dict(scene_id=int(scene[i]),start_frame=int(frame[i]),deltaJ=float(delta[i]),deltaADE=float(delta_ade[i]),deltaFDE=float(delta_fde[i]),q_win_J=bool(win[i]),**feats[i]) for i in range(len(feats))],
    }
    OUT.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2)[:50000])

if __name__=="__main__": main()
