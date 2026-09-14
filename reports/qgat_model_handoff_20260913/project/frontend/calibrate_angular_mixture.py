"""Fit a frozen two-scale angular error model using training calibration residuals."""
import numpy as np
from .run_frontend import ROOT,OUT,load,dump

def fit_rows(rows,base_R):
    e=np.array([x['residual'][1] for x in rows]);wide=np.deg2rad(5.);floor=np.deg2rad(.5)
    sigma=max(float(np.median(abs(e))/.67448975),floor);sigma=min(sigma,wide);alpha=.1
    for iteration in range(200):
        core=np.exp(-.5*(e/sigma)**2)/sigma;tail=np.exp(-.5*(e/wide)**2)/wide
        responsibilities=alpha*tail/((1-alpha)*core+alpha*tail)
        a=float(np.mean(responsibilities));s=max(floor,float(np.sqrt(np.sum((1-responsibilities)*e*e)/np.sum(1-responsibilities))))
        s=min(s,wide)
        change=max(abs(a-alpha),abs(s-sigma));alpha=a;sigma=s
        if change<1e-9:break
    narrow=np.array(base_R);broad=np.array(base_R);narrow[1,1]=sigma**2;broad[1,1]=wide**2
    moment=(1-alpha)*narrow+alpha*broad
    logpdf=np.log(((1-alpha)*np.exp(-.5*(e/sigma)**2)/sigma+alpha*np.exp(-.5*(e/wide)**2)/wide)/np.sqrt(2*np.pi))
    return dict(count=len(rows),core_std_deg=float(np.rad2deg(sigma)),wide_weight=alpha,iterations=iteration+1,angular_log_likelihood=float(logpdf.sum()),components=[dict(weight=1-alpha,R=narrow.tolist()),dict(weight=alpha,R=broad.tolist())],R=moment.tolist())

def main():
    original=load(OUT/'C03_before_angular_revision/R_calibration.json');config=load(ROOT/'configs/frontend_config.json');residuals=original['residuals'];base=original['config']
    global_fit=fit_rows(residuals,base['R_global']);bins=[]
    for b in range(3):
        rows=[x for x in residuals if x['quality_bin']==b];fit=fit_rows(rows,base['R_quality_bins'][b]['R']) if len(rows)>=30 else global_fit.copy();fit['bin']=b;fit['fallback_global']=len(rows)<30;bins.append(fit)
    config['frontend_revision']='C05';config['measurement_noise_mixture_bins']=[dict(bin=b['bin'],components=b['components']) for b in bins]
    config['R_global']=global_fit['R']
    for b,fit in zip(config['R_quality_bins'],bins):b['R']=fit['R']
    config['R_revision']='C05 zero-mean narrow-plus-5deg angular Gaussian mixture; training-fitted core and weight; R interface is prior moment'
    config['calibration']['residual_estimator']='Training-only EM angular mixture; core sigma floor0.5deg, fixed wide5deg; original range/Doppler variances unchanged'
    config['q_a']=4.
    dump(ROOT/'configs/frontend_config.json',config)
    dump(OUT/'angular_mixture_calibration.json',dict(revision='C05',scope='same fixed846 training residuals, no tracking-metric optimization',global_fit=global_fit,bins=bins,qualification='Residuals retain original matching censoring; fitted mixture is an approximation'))
    original['config']=config;original['revision']='C05';original['original_empirical_evidence']='C03_before_angular_revision/R_calibration.json';dump(OUT/'R_calibration.json',original)
    print([{k:v for k,v in b.items() if k not in ('components','R')} for b in bins])
if __name__=='__main__':main()
