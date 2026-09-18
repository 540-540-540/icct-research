"""Publish scoped C06 evidence only after current-version validation."""
import argparse
import hashlib
import json
import sys
import numpy as np
import yaml
from .run_frontend import ROOT,OUT,DATA,load,dump,summarize,quality_bin

REVISION='C06'


def checks_passed(record):
    flags=[value for value in record.values() if isinstance(value,bool)]
    return bool(flags) and all(flags)


def require_current(record,config_sha,label):
    if record.get('frontend_revision')!=REVISION or record.get('frontend_config_sha256')!=config_sha:
        raise ValueError(f'{label}: missing or stale frontend revision/config provenance')


def stage_decision(scope,engineering,other_checks):
    if not engineering or not other_checks:
        return False,'failed_stage_not_accepted'
    if scope=='train-only':
        return False,'pending_development_validation'
    return True,'accepted'


def _number(value,digits=3):
    return '无样本' if value is None else f'{value:.{digits}f}'


def _percent(value):
    return '无样本' if value is None else f'{value:.2%}'


def _check_summary(stored,derived,split):
    for key,value in derived.items():
        if key not in stored:raise ValueError(f'{split}: summary is missing {key}')
        observed=stored[key]
        if isinstance(value,(int,float)) and not isinstance(value,bool):
            equal=isinstance(observed,(int,float)) and np.isclose(value,observed,rtol=1e-10,atol=1e-10)
        else:equal=value==observed
        if not equal:raise ValueError(f'{split}: stored summary differs from current episodes at {key}')


def finalize(q_a=9.,scope='train-only',causal_file='causal_checks.json',validate_only=False):
    if q_a!=9.:raise ValueError('Final C06 configuration is selected q_a=9; other candidates remain historical development evidence')
    if scope not in ('train-only','full'):raise ValueError('scope must be train-only or full')
    manifest=load(OUT/'development_manifest.json')
    config_path=ROOT/'configs/frontend_config.json'
    config_bytes=config_path.read_bytes();config=json.loads(config_bytes)
    config_sha=hashlib.sha256(config_bytes).hexdigest()
    if config.get('frontend_revision')!=REVISION or config.get('association_mode')!='jpda' or config.get('q_a')!=9 or config.get('max_missed_cycles')!=7:
        raise ValueError('C06 finalizer requires current C06 JPDA q9 and seven-miss configuration')
    quality=load(OUT/'quality_q9.json')
    expected=list(manifest['train_tracking_audit_episodes']);splits=['train']
    if scope=='full':expected+=manifest['development_episodes'];splits.append('V_select')
    if len(expected)!=len(set(expected)):raise ValueError('Duplicate episodes in requested scope')
    if len(manifest['train_tracking_audit_episodes'])!=12 or len(manifest['development_episodes'])!=25:
        raise ValueError('Unexpected fixed train/development scope')
    causal_path=OUT/causal_file;causal=load(causal_path)
    require_current(causal,config_sha,'causal evidence')
    detector=load(OUT/'detector_calibration.json')
    tracker=load(OUT/'tracker_checks.json');jpda=load(OUT/'jpda_checks.json')
    mixture_checks=load(OUT/'measurement_mixture_checks.json')
    lifecycle=load(OUT/'lifecycle_checks.json');require_current(lifecycle,config_sha,'lifecycle evidence')
    if lifecycle.get('max_missed_cycles')!=7:raise ValueError('Lifecycle check is not for selected seven-miss rule')
    angular_calibration=load(OUT/'angular_mixture_calibration.json')
    from .measurement_mixture import validate_bins
    mixture_bins=validate_bins(config['measurement_noise_mixture_bins'],config['quality_bin_edges_db'])
    if angular_calibration.get('revision')!='C05' or angular_calibration['global_fit']['count']!=846:
        raise ValueError('Frozen training angular-mixture calibration is missing or changed')
    for index,components in mixture_bins.items():
        prior=sum(weight*R for weight,R in components)
        if not np.allclose(prior,config['R_quality_bins'][index]['R'],rtol=1e-10,atol=1e-12):
            raise ValueError('Detection R must equal the zero-mean mixture prior covariance moment')
        fitted=angular_calibration['bins'][index]['components']
        if len(components)!=len(fitted) or any(not np.isclose(w,row['weight']) or not np.allclose(R,row['R']) for (w,R),row in zip(components,fitted)):
            raise ValueError('Mixture parameters differ from the frozen training fit')
    covariance_path=OUT/'covariance_consistency_C06.json';covariance=load(covariance_path)
    if not covariance.get('scope','').startswith('C06 JPDA q9 '):raise ValueError('Covariance evidence is not current C06')
    overflow=[];generations=[];components=[];evaluations=[]
    for index in expected:
        generation=load(DATA/'generation'/f'episode_{index:03d}.json')
        if generation['episode_index']!=index or generation['cubes']!=597:
            raise ValueError('Incomplete or wrong episode generation')
        generations.append(generation)
        records=[json.loads(line) for line in (DATA/'detections'/f'episode_{index:03d}.jsonl').read_text().splitlines()]
        if len(records)!=597:raise ValueError('Expected 199 cycles x three station observations')
        for j,row in enumerate(records):
            if row['station_id']!=j%3 or row['measurement_time_ns']>row['deadline_ns'] or len(row['detections'])>32:
                raise ValueError('Invalid station order, causal timestamp or candidate cap')
            if any(d['station_id']!=row['station_id'] or d['measurement_time_ns']!=row['measurement_time_ns'] for d in row['detections']):
                raise ValueError('Detection observation metadata mismatch')
            for detection in row['detections']:
                expected_R=np.asarray(config['R_quality_bins'][quality_bin(detection)]['R'],float)
                recorded_R=np.asarray(detection.get('R'),float)
                if recorded_R.shape!=(3,3) or not np.allclose(recorded_R,expected_R,rtol=1e-12,atol=1e-12):
                    raise ValueError(f'Episode {index}: cached detection covariance is not current C06 quality-bin R')
            if row['diagnostics'].get('overflow',False) or row['diagnostics'].get('overflow_count',0)>0:
                overflow.append(dict(episode_index=index,**row['diagnostics']))
        evaluation=load(OUT/f'evaluation/q9/episode_{index:03d}.json')
        require_current(evaluation,config_sha,f'episode {index} evaluation')
        if evaluation.get('association_mode')!='jpda' or evaluation['q_a']!=q_a or evaluation['episode_index']!=index:
            raise ValueError('Wrong episode or association configuration')
        split='train' if index in manifest['train_tracking_audit_episodes'] else 'V_select'
        if evaluation['split']!=split:raise ValueError('Episode split mismatch')
        evaluations.append(evaluation);components.extend(evaluation['jpda_components'])
        tracks=(DATA/'tracks/q9'/f'episode_{index:03d}.jsonl').read_text().splitlines()
        if len(tracks)!=199:raise ValueError('Incomplete current track cache')
    for split in splits:
        if split not in quality:raise ValueError(f'Missing current {split} summary')
        derived=summarize([record for record in evaluations if record['split']==split])
        _check_summary(quality[split],derived,split)
    if covariance['state_nees']['n']!=quality['train']['matched_object_origins']:
        raise ValueError('Covariance diagnostic sample count does not match current training evaluation')
    history=[]
    def historical(revision,misses,candidate,path,phase):
        result=load(path)
        if result.get('q_a')!=candidate:raise ValueError('Historical q_a record mismatch')
        history.append(dict(revision=revision,max_missed_cycles=misses,q_a=candidate,phase=phase,
                            source=str(path.relative_to(ROOT)),metrics={k:result[k] for k in ('train','V_select') if k in result}))
    for revision,directory in [('C01',OUT/'C01_before_R_floor'),('C02',OUT/'C02_conservative_R'),('C03',OUT/'C03_before_angular_revision')]:
        for candidate in (1,4,9):historical(revision,5,candidate,directory/f'quality_q{candidate}.json','固定训练比较')
    historical('C04',5,4,OUT/'C04_before_mixture_revision/quality_q4.json','完整开发验证')
    for candidate in (1,4,9):historical('C05',5,candidate,OUT/f'C05_before_lifecycle_revision/quality_q{candidate}.json','完整开发比较')
    for candidate in (4,9):historical('C06',8,candidate,OUT/f'C06_miss8/quality_q{candidate}.json','先前8次漏检试验')
    historical('C06',6,9,OUT/'C06_miss6/quality_q9.json','限定范围训练选择')
    historical('C06',7,9,OUT/'C06_miss7/quality_q9.json','限定范围训练选择')
    history.append(dict(revision=REVISION,max_missed_cycles=7,q_a=9,phase='当前最终配置',
                        source='reports/f01c/quality_q9.json',metrics={split:quality[split] for split in splits}))
    prior=next(item['metrics']['train'] for item in history if item['revision']=='C03' and item['q_a']==4)
    train=quality['train']
    tradeoff=dict(comparison='C03 q4 vs final C06 q9, same fixed12train; multiple engineering revisions',
        coverage_change_percentage_points=100*(train['coverage']-prior['coverage']),
        entire_confirmed_false_change_percentage_points=100*(train['false_confirmed_fraction']-prior['false_confirmed_fraction']),
        IDSW_per1000_change=train['IDSW_per_1000_reference_object_frames']-prior['IDSW_per_1000_reference_object_frames'],
        C03_V_select_comparison_available=False)
    declaration=load(OUT/'C06_predeclared_revision.json')
    bound=declaration['bounded_lifetime_calibration']
    if bound['allowed_max_missed_cycles']!=[5,6,7,8] or bound['q_a']!=9:
        raise ValueError('C06 finite lifetime selection record is inconsistent')
    finalists=[item for item in history if item['revision']=='C06' and item['phase']=='限定范围训练选择' and item['metrics']['train']['passed']]
    if not finalists:raise ValueError('No training-qualified finite lifetime candidate')
    selected=max(finalists,key=lambda item:(item['metrics']['train']['coverage'],-item['metrics']['train']['false_confirmed_fraction']))
    if selected['max_missed_cycles']!=7:raise ValueError('Final lifetime does not follow the declared training selection rule')
    truncated=sum(c['truncated'] for c in components)
    stats=dict(component_count=len(components),truncated_components=truncated,
        truncated_fraction=truncated/len(components) if components else 0.,
        max_tracks=max((c['tracks'] for c in components),default=0),
        max_detections=max((c['detections'] for c in components),default=0),
        hypotheses_evaluated=sum(c['hypothesis_count'] for c in components),
        ESS_mean=float(np.mean([c['effective_sample_size'] for c in components])) if components else None,
        ESS_max=max((c['effective_sample_size'] for c in components),default=0),
        qualification='Large clusters use genuine Murty top50; truncated posterior is conditional on retained hypotheses, omitted mass unknown')
    yaml_text=yaml.safe_dump(config,sort_keys=False,allow_unicode=True)
    names=('frontend/ofdm_echo.py','frontend/echo_source.py','frontend/detector.py',
           'frontend/calibrate_detector.py','frontend/tracker.py','frontend/jpda.py',
           'frontend/measurement_mixture.py','frontend/calibrate_angular_mixture.py',
           'frontend/run_frontend.py','frontend/finalize_frontend.py','configs/frontend_config.json',
           'reports/f01c/R_calibration.json','reports/f01c/covariance_consistency_C06.json',
           'reports/f01c/measurement_mixture_checks.json','reports/f01c/lifecycle_checks.json',
           'reports/f01c/angular_mixture_calibration.json','reports/f01c/C06_predeclared_revision.json')
    provenance={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}
    provenance['configs/frontend_config.yaml']=hashlib.sha256(yaml_text.encode('utf-8')).hexdigest()
    engineering=all(quality[split]['passed'] for split in splits)
    other_checks=bool(causal.get('passed',False) and detector['all_synthetic_checks_passed'] and
        detector['calibration']['holdout_target_in_poisson95ci'] and checks_passed(tracker) and checks_passed(jpda) and
        checks_passed(mixture_checks) and checks_passed(lifecycle) and not overflow)
    passed,status=stage_decision(scope,engineering,other_checks)
    not_executed=(['C06 V_select not included in this scope'] if scope=='train-only' else [])+[
        'V_confirm','test','prediction model training','F01-D input/label packing',
        'Complete current-frontend quality validation at other SNRs and pressure conditions']
    report=dict(stage='F01-C',revision=REVISION,frontend_revision=REVISION,frontend_config_sha256=config_sha,
        scope=scope,scope_description='12 fixed training audit episodes' if scope=='train-only' else '12 fixed training audit and all25 current C06 V_select episodes',
        status=status,q_a=q_a,max_missed_cycles=7,nominal_snr_db=20,
        q_a_role='Final selected q9; earlier finite q1/q4/q9 development comparisons and bounded lifetime selection are preserved',
        validation_interpretation='Adaptive engineering development, not blind held-out model-performance evidence',
        **{split:quality[split] for split in splits},
        V_select_status='not_in_current_scope' if scope=='train-only' else 'evaluated_for_C06',
        historical_development='Earlier-revision results remain historical; no C03 V_select comparison is claimed',
        engineering_gate_passed=bool(scope=='full' and engineering),scope_engineering_gate_passed=bool(engineering),
        training_engineering_gate_passed=bool(train['passed']),stage_passed=passed,
        causality_passed=bool(causal.get('passed',False)),causal_evidence=str(causal_path.relative_to(ROOT)),
        detector_synthetic_passed=detector['all_synthetic_checks_passed'],noise_holdout_target_verified=detector['calibration']['holdout_target_in_poisson95ci'],
        tracker_selfchecks_passed=checks_passed(tracker),jpda_selfchecks_passed=checks_passed(jpda),
        measurement_mixture_selfchecks_passed=checks_passed(mixture_checks),lifecycle_selfchecks_passed=checks_passed(lifecycle),
        runtime=sys.executable,config='configs/frontend_config.json',source_provenance=provenance,
        generation_cubes=sum(s['cubes'] for s in generations),overflow_events=len(overflow),
        false_confirmed_definition='Unmatched entire extant confirmed pool at scored origins / all confirmed pool; selected-eight false fraction is separate',
        evaluation_binding='Unchanged 5m current-position maximum cardinality then minimum distance; evaluation never edits tracks',
        raw_detection_cache_use='Fixed C01 detector; every detection R matches the frozen mixture-prior covariance moment used by the current tracker',
        cached_detection_R_verified=True,
        R_status='C05 zero-mean two-Gaussian angle model fitted once on846 training residuals; broad sigma5deg fixed, core sigma and prior weight learned; range/Doppler unchanged',
        R_covariance_consistency_validated=False,jpda=stats,
        covariance_consistency_evidence=str(covariance_path.relative_to(ROOT)),covariance_consistency=covariance,
        angular_mixture_calibration_evidence='reports/f01c/angular_mixture_calibration.json',
        finite_selection_record=declaration,revision_history=history,C03_C06_training_tradeoff=tradeoff,
        not_executed=not_executed)
    rows=[]
    for item in history:
        m=item['metrics'];t=m['train'];v=m.get('V_select')
        joint='通过' if t['passed'] and v is not None and v['passed'] else ('仅训练通过' if t['passed'] and v is None else '未通过')
        rows.append(f"| {item['revision']} / {item['phase']} | {item['q_a']} | {item['max_missed_cycles']} | {_percent(t['coverage'])} | {_percent(t['false_confirmed_fraction'])} | {_percent(v['coverage']) if v else '未运行'} | {_percent(v['false_confirmed_fraction']) if v else '未运行'} | {joint} |")
    scope_rows=[]
    for split in splits:
        m=quality[split]
        scope_rows.append(f"| {split} | {m['episodes']} | {_percent(m['coverage'])} | {_number(m['position_rmse_m'])} | {_number(m['velocity_rmse_mps'])} | {_percent(m['false_confirmed_fraction'])} | {_percent(m['stationary_coverage'])} | {_number(m['IDSW_per_1000_reference_object_frames'])} | {'通过' if m['passed'] else '未通过'} |")
    if passed:
        status_text='C06最终配置在本次名义20dB、固定训练审计集与全部V_select上的工程质量及必要检查通过，F01-C验收通过。'
        next_text='可以进入F01-D预测输入与标签打包；本次尚未执行F01-D，没有最终测试或预测模型性能结论。'
    elif status=='pending_development_validation':
        status_text='C06当前训练范围通过，但本报告未包含完整开发集验收，F01-C阶段尚未验收。'
        next_text='production_ready为false；须完成所选配置的全部25个V_select评价后再判断是否进入F01-D。'
    else:
        status_text='C06当前范围未同时满足工程质量门槛或必要检查，F01-C验收未通过。'
        next_text='production_ready为false，F01-D继续阻断；当前失败证据保留，不自动扩展参数搜索。'
    cal=detector['calibration'];pos_nees=covariance['position_nees'];state_nees=covariance['state_nees']
    truncation_text='发生截断时，只对保留假设归一化，省略的后验概率质量未知。' if truncated else '本次范围未发生前50项截断。'
    global_fit=angular_calibration['global_fit']
    text=f'''# F01-C：检测与匿名追踪工程验收

**{status_text}** 最终参数为q_a=9、连续第7次漏检删除，前6次漏检保留；确认条件仍为三周期中两次命中。V_select已用于多轮开发判断，本次结果属于自适应工程开发证据，不是盲测泛化结论。

## 最终配置与当前范围

共核对{len(expected)}个场景、{report['generation_cubes']}个站级回波周期，{len(overflow)}次候选溢出。当前逐场景评价、因果检查和生命周期检查携带相同C06及配置SHA256；每条检测R均与当前质量档混合分布的先验协方差矩一致。

| 划分 | 场景 | 覆盖率 | 位置RMSE/m | 速度RMSE/(m/s) | 全确认池虚假比例 | 静止覆盖率 | IDSW/千对象帧 | 联合门槛 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(scope_rows)}

四项质量门槛保持不变：覆盖率≥90%，匹配位置与速度RMSE分别≤3m及3m/s，全确认池虚假比例≤5%。评价仍使用当前真实对象、5m一对一最大匹配，随后最小化距离；标签、对象分母与八槽选择不变。RMSE仅包含匹配对象；八槽虚假比例不能替代全确认池指标。阶段通过仅覆盖这里列出的名义20dB场景，不扩展到其他信噪比、压力条件或预测性能。

## 修订与有限参数选择记录

| 版本/步骤 | q_a | 删除漏检次数 | 训练覆盖 | 训练全池虚假 | V_select覆盖 | V_select全池虚假 | 联合结果 |
|---|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(rows)}

C01修正检测保护区和局部峰值抑制；C02同时采用保守距离、角度和速度R下限；C03恢复经验R并采用JPDA；C04仅保留5°角度标准差下限。C03只有上述训练比较，不虚构C03开发集对照。所有历史完整数据与配置均按版本保留，当前清单只接受当前配置证据。

C05采用零均值双高斯角度噪声：宽分量标准差固定5°，窄分量标准差及权重只对原有846条训练残差进行EM拟合，距离/径向速度误差模型保持不变。全局窄分量标准差{global_fit['core_std_deg']:.3f}°，宽分量权重{global_fit['wide_weight']:.2%}；实际采用分质量档的训练拟合，样本不足档使用全局结果。检测R接口表示混合先验协方差矩，不把混合分量当作车辆身份。5°是宽尾不确定性的建模选择，不等于实际估计精度或阵列分辨率。

C05 q_a=4开发覆盖未过门槛后，才登记并比较原协议有限q_a=1/4/9，三者均未通过完整联合门槛。C06依据训练检出间断的95%—99%区间限定漏检删除次数为5—8；8次方案实际只比较q_a=4/9，没有运行或补写q_a=1，且未通过训练联合门槛。随后在事先登记的范围内，仅对q_a=9的6/7次方案做训练比较；两者训练通过，依登记规则选择训练覆盖更高的7次，再只对选定7次做完整V_select。没有继续搜索其他角度参数或扩大门槛。

C03 q_a=4到最终C06 q_a=9的同批训练比较：覆盖率变化{tradeoff['coverage_change_percentage_points']:+.2f}个百分点、全确认池虚假比例变化{tradeoff['entire_confirmed_false_change_percentage_points']:+.2f}个百分点、IDSW/千对象帧变化{tradeoff['IDSW_per1000_change']:+.3f}。这反映多项工程修订后的取舍，不把全部变化归因于单一模块。最终开发集指标直接列于首表，不与不存在的C03开发结果比较。

## 算法、自检与不确定性边界

JPDA小簇精确枚举，较大簇采用真正Murty前50个完整分配；每个关联及噪声分量分支使用Joseph EKF，并保留假设内及均值间协方差散布。门控不会重新归一化被排除噪声分量的先验权重。当前范围有{len(components)}个连通簇、{truncated}个截断，占{stats['truncated_fraction']:.3%}，最大航迹/检测数{stats['max_tracks']}/{stats['max_detections']}，ESS均值{_number(stats['ESS_mean'],6)}。{truncation_text}

当前训练匹配对象状态NEES均值{_number(state_nees['mean'])}、位置NEES均值{_number(pos_nees['mean'])}，位置NEES超过二维卡方95%阈值的比例{_percent(pos_nees['fraction_above_chi95'])}。后验加权、经过门控的分支NIS均值{_number(covariance['association_weighted_nis']['mean'])}。这些统计受到匹配与门控筛选影响，后验不确定性仍偏乐观，不能宣称协方差已完全校准；工程质量通过不消除该局限。

检测、原追踪器、JPDA、混合观测及生命周期自检均通过：{all([report['detector_synthetic_passed'],report['tracker_selfchecks_passed'],report['jpda_selfchecks_passed'],report['measurement_mixture_selfchecks_passed'],report['lifecycle_selfchecks_passed']])}。噪声设计/独立holdout虚警均值分别{cal['design']['mean']:.6f}/{cal['holdout']['mean']:.6f}次/CPI，holdout Poisson近似95%区间{cal['holdout']['poisson_95ci']}覆盖0.1目标；较早小样本失败统计保留。当前因果检查通过：{report['causality_passed']}，证据为{report['causal_evidence']}，覆盖两个场景的300个站级周期、100个追踪周期，修改未来记录后的当前前缀完全一致。

## 交付与下一阶段

frontend_quality.json状态为{status}，production_manifest.json的production_ready为{str(passed).lower()}；YAML与当前JSON一致，overflow_log.json覆盖本次核对清单。SHA256版本追溯包含混合观测实现及标定脚本，当前配置为{config_sha}。

{next_text} V_confirm、测试集、QGNN/GNN/LLM训练均未运行；F01-D输入序列化还需独立验收。完整版本与参数选择证据保留在revision_history及对应归档目录。
'''
    if config_path.read_bytes()!=config_bytes:raise ValueError('Configuration changed during finalization')
    if not validate_only:
        (ROOT/'configs/frontend_config.yaml').write_text(yaml_text)
        dump(OUT/'frontend_quality.json',report)
        dump(OUT/'overflow_log.json',dict(revision=REVISION,scope=scope,episode_indices=expected,cubes=report['generation_cubes'],events=overflow,empty_means_all_listed_cpis_checked=True))
        dump(OUT/'production_manifest.json',dict(stage_status=status,production_ready=passed,revision=REVISION,
            frontend_revision=REVISION,frontend_config_sha256=config_sha,scope=scope,episode_indices=expected,q_a=q_a,
            max_missed_cycles=7,q_a_role=report['q_a_role'],nominal_snr_db=20,shared_for_all_predictors=True,
            validation_interpretation=report['validation_interpretation'],data_directory=str(DATA),
            per_episode=generations,source_provenance=provenance,R_covariance_consistency_validated=False,
            V_select_status=report['V_select_status'],not_executed=not_executed))
        (OUT/'F01C_REPORT.md').write_text(text,encoding='utf-8')
    print(json.dumps(dict(validate_only=validate_only,**{k:report[k] for k in ('status','q_a','max_missed_cycles','stage_passed','training_engineering_gate_passed','causality_passed','generation_cubes','overflow_events','V_select_status')}),indent=2))
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--q-a',type=float,default=9.)
    parser.add_argument('--scope',choices=['train-only','full'],default='train-only')
    parser.add_argument('--causal-file',default='causal_checks.json');parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args();finalize(args.q_a,args.scope,args.causal_file,args.validate_only)
