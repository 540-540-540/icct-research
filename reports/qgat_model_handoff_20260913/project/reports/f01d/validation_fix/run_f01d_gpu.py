"""Run the GPU F01-D pipeline and leave durable status without interactive polling."""
from pathlib import Path
import sys,subprocess,json,time,os
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from frontend.pack_symbol_dataset import write_json
O=R/'reports/f01d';O.mkdir(parents=True,exist_ok=True)

def main():
    gpu=json.loads((R/'reports/symbol_gpu/validation.json').read_text())
    if gpu.get('all_required_float64_checks_passed') is not True:raise RuntimeError('GPU float64 validation must pass first')
    state=dict(status='running',stage='generate',pid=os.getpid(),snr_db=[5,10,15,20],devices=[0,1],dtype='float64/complex128',started_unix=time.time(),steps=[])
    write_json(O/'run_status.json',state)
    try:
        for stage,args in [('generate',['frontend.generate_symbol_gpu','--devices','0','1','--batch-size','16']),('pack',['frontend.pack_symbol_dataset','pack']),('normalize',['frontend.symbol_dataset','normalize']),('validate',['frontend.check_symbol_dataset'])]:
            state['stage']=stage;write_json(O/'run_status.json',state);print('STAGE '+stage,flush=True);start=time.perf_counter()
            subprocess.run([sys.executable,'-u','-m']+args,cwd=R,check=True)
            state['steps'].append(dict(stage=stage,seconds=time.perf_counter()-start))
        validation=json.loads((O/'validation.json').read_text());assert validation['passed']
        packing=json.loads((O/'packing.json').read_text());generation=json.loads((O/'generation.json').read_text())
        failures={};outputs={}
        for p in (R/'data/f01d/diagnostics').glob('snr_*/episode_*.json'):
            a=json.loads(p.read_text());key=str(a['snr_db']);failures[key]=failures.get(key,0)+a['diagnostic_counts'].get('search_failure',0);outputs[key]=outputs.get(key,0)+a['outputs']
        state.update(status='needs_review' if any(failures.values()) else 'complete',stage='finished',finished_unix=time.time(),structural_validation_passed=True,search_failures=failures,estimated_target_states=outputs,prediction_training_started=False)
        write_json(O/'run_status.json',state)
        rows='\n'.join(f"|{k}|{v['episodes']}|{v['samples']}|" for k,v in packing['splits'].items())
        report='# F01-D：GPU共同预测数据缓存\n\n'
        report+='状态：'+('结构验收通过，感知搜索异常需复核。' if any(failures.values()) else '生成、打包、训练归一化及独立结构/因果验收通过。')+'\n\n'
        report+='SNR：5、10、15、20dB，标称20dB。两个RTX4090，批大小16，双精度符号计算；CPU负责源记录路由与文件读写。旧负SNR未完成缓存已清理。\n\n'
        report+='|集合|片段数|预测起点数|\n|---|---:|---:|\n'+rows+'\n\n'
        report+='每档SNR共280份199帧序列；每个预测样本含20帧、最多8车的四维估计状态和两组因果标记。四档共用同一运动起点，不算四倍独立交通样本。未来20帧原始位置及label_valid独立保存，身份路由不进入模型输入。\n\n'
        report+='输入、标签、元数据分别位于data/f01d/inputs、labels、metadata；normalization.json只由训练四档有效缓存项拟合，数值尺度在CUDA计算。GPU随机流按每目标固定种子可复现，与历史NumPy噪声不作逐样本相同声明。\n\n'
        report+='独立验收覆盖完整缓存重建、原始未来位置逐点核对、四SNR样本/槽位一致、训练尺度权限、Q/G公共loader及未来改删后的最终输入不变。详见validation.json。\n\n'
        report+='各档估计目标状态计数：'+json.dumps(outputs)+'；搜索异常计数：'+json.dumps(failures)+'。\n\n'
        report+='GPU核验见../symbol_gpu/validation.json；生产参数/代码版本见generation.json；各阶段时间见run_status.json。未进行预测训练，未据V_confirm/test性能选择配置。\n'
        (O/'F01D_REPORT.md').write_text(report)
        print('F01D '+state['status'],flush=True)
    except Exception as error:
        state.update(status='failed',error=repr(error),finished_unix=time.time());write_json(O/'run_status.json',state);raise
if __name__=='__main__':main()
