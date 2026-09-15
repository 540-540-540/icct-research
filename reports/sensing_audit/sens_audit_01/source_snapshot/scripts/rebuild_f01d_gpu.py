"""User-invoked rebuild after numerical solver repair; retain prior cache for audit."""
from pathlib import Path
import shutil,sys,subprocess,json
R=Path(__file__).resolve().parents[1]
def main():
    config=json.loads((R/'configs/symbol_frontend.json').read_text())
    if config['revision']!='A03-symbol-v4-cuda-stable-solver':raise RuntimeError('Unexpected solver version')
    source=R/'data/f01d';backup=R/'data/f01d_before_solver_fix'
    reports=R/'reports/f01d';report_backup=R/'reports/f01d_before_solver_fix'
    if not backup.exists():
        if report_backup.exists():raise RuntimeError('Review partial previous archive before retry')
        if reports.exists():shutil.copytree(reports,report_backup)
        if source.exists():source.rename(backup)
        for name in ['generation.json','packing.json','validation.json','run_status.json','F01D_REPORT.md','gpu_smoke_generation.json']:
            p=reports/name
            if p.exists():p.unlink()
        print('Previous cache and reports retained; starting repaired GPU generation',flush=True)
    else:print('Previous cache already retained; resume current repaired GPU generation',flush=True)
    subprocess.run([sys.executable,'-u',str(R/'scripts/run_f01d_gpu.py')],cwd=R,check=True)
if __name__=='__main__':main()
