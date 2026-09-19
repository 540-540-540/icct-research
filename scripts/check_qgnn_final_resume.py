"""Bounded GPU resume regression: interrupt after one step and compare parameters."""
import sys,os,json,time,subprocess
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT)
report={'status':'RUNNING','test_set_used':False,'phases':[]}
OUT=ROOT/'reports/qgnn/resume_regression.json'
def save():
    t=OUT.with_suffix('.tmp');t.write_text(json.dumps(report,indent=2)+'\n');t.replace(OUT)
base=[sys.executable,'scripts/train_qgnn_final.py','--kind','quantum','--seed','2299','--snr','0','--epochs','1','--batch-size','16','--train-limit','64','--quantum-version','3','--correction-cap','16']
straight='reports/qgnn/resume_straight_20260919'
interrupted='reports/qgnn/resume_interrupted_20260919'
try:
    for name,directory,extra in [('straight',straight,[]),('pause',interrupted,['--max-seconds','0.001']),('resume',interrupted,['--resume','--max-seconds','300'])]:
        Path(directory).mkdir(parents=True,exist_ok=True)
        command=base+['--run-dir',directory]+extra
        logfile=Path(directory)/('resumecheck_'+name+'.log')
        with logfile.open('w') as f:
            child=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            phase={'name':name,'pid':child.pid,'command':command,'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'log':str(logfile)}
            report['phases'].append(phase);save();print('START',phase,flush=True)
            code=child.wait(timeout=180)
        phase['returncode']=code;assert code==0,logfile.read_text()[-2000:]
        phase['summary']=json.loads((Path(directory)/'summary.json').read_text());save()
    torch.set_num_threads(2)
    a=torch.load(Path(straight)/'last.pt',map_location='cpu',weights_only=False)
    b=torch.load(Path(interrupted)/'last.pt',map_location='cpu',weights_only=False)
    assert a['source_sha256']==b['source_sha256']
    assert a['global_step']==b['global_step']==4
    assert 0<report['phases'][1]['summary']['global_step']<4
    assert report['phases'][1]['summary']['status']=='PAUSED_BUDGET'
    errors={k:float((a['model_state'][k]-b['model_state'][k]).abs().max()) for k in a['model_state']}
    report['max_parameter_error']=max(errors.values());assert report['max_parameter_error']<1e-6
    report['global_step']=a['global_step'];report['source_sha256']=a['source_sha256']
    report['validation_ADE_difference']=abs(a['best']['ADE']-b['best']['ADE'])
    report['validation_FDE_difference']=abs(a['best']['FDE']-b['best']['FDE'])
    report['status']='PASS';save();print(json.dumps(report),flush=True)
except Exception as exc:
    report['status']='FAIL';report['error']=repr(exc);save()
    if 'child' in globals() and child.poll() is None:
        child.terminate()
        try:child.wait(timeout=10)
        except subprocess.TimeoutExpired:child.kill()
    raise
