# REBUILD_03_WORKPLAN.md — 下一轮施工计划（可直接执行）

前置：本文件由 SENS-DESIGN-02 冻结；执行者无需重做架构判断。
基线：`sens-design-02` 分支（从 `sens-audit-01` @ `0cac17ba` 派生）。
目标：实现最小可信的 3-BS 共享回波 → 检测 → 关联/融合 → CV-KF → `[x_hat,y_hat,vx_hat,vy_hat]`
→ `data/f01e` 缓存；**不训练下游模型，不覆盖 F01-D，不读 V_confirm/test**。

## 0. 全局约束（REBUILD-03 必须遵守）

1. 不改 `data/f01d/**`、`configs/symbol_frontend.json`、旧入口脚本、旧检测/跟踪文件行为。
2. 新代码只落 `frontend/sensing/`、`frontend/fusion/`、`frontend/tracking/`、
   `frontend/pack_shared_dataset.py`、`frontend/check_shared_dataset.py`、`scripts/run_f01e_shared.py`、
   `scripts/calibrate_f01e_snr.py`、`configs/shared_frontend.json`、`code/07_shared_frontend/`（诊断）。
3. B 域模块禁止 import A 域（见 `GT_ISOLATION_SPEC.md`）；测试 T2 强制执行。
4. 计算后端：torch CUDA + float64/complex128（推理/仿真）；落盘 float32/bool。
5. 用户手动启动正式全量；助手只准备入口与验收。
6. 任何 NIST 算法移植需在代码注释与 `EXTERNAL_REUSE_MATRIX.md` 登记来源与改动。

## 1. 数据流（实现时按此顺序）

```text
for episode in A01 episodes (split-based):
  for frame (199 deadlines):
    A: position/velocity from SourceEpisodes.at_time(deadline)          # 仅 simulator 内部
    for snr_ref in level_set:                                            # 校准阶段为 sweep 子集
      simulator.synthesize_shared(...) -> Y[3,A,K,N], X[3,K,N]           # 求和后一次噪声
      for b in 0..2:
        detector.process_baseline(Y[b], X[b], b) -> detections[b]        # CFAR/NMS/几何
      observations = association.fuse_frame(detections_by_bs)            # 门控+匹配+融合
      tracker.step(observations, time_ns) -> TrackRecords                # CV-KF + slot
    write sequences/snr_{s}/episode_{NNN}.npz（199 帧槽位表）
    （可选）保存选定帧的 Y 到 echoes/（仅诊断，默认关闭）
  pack -> inputs/labels/metadata
  normalize（train 四档，train-only）
  check_shared_dataset（T1–T7 结构/隔离子集）
```

内存与存储：Y 单帧单 BS 为 `16·256·256·16B = 16.8 MB`；**不落盘**，逐帧丢弃。
序列/诊断落盘量 ≈ 7 MB/SNR（280 episodes）+ 可选 detections npz；禁止转存原始回波全量。

## 2. 阶段与验收

### P0 脚手架（0.5 会话）

产出：`configs/shared_frontend.json`（`INTERFACE_SPEC.md` §4.1 模式，`snr_ref_db=null` 占位）、
新目录与空模块、`code/07_shared_frontend/baseline_hash.py`（记录旧文件/缓存 hash）。

命令：

```bash
cd /home/dell/YrM/ICCT
/home/dell/YrM/envs/ICCT/bin/python code/07_shared_frontend/baseline_hash.py --write reports/f01e/baseline_hashes.json
```

验收：旧文件/缓存 hash 清单生成；新目录无副作用；`git status` 只出现新文件。

### P1 simulator + T1（1–2 会话）

文件：`frontend/sensing/waveform.py`、`frontend/sensing/simulator.py`、
`code/07_shared_frontend/check_shared_echo.py`。

实现：SYSTEM_MODEL §2-3 公式；种子流 `waveform/phase/noise` 分离；
`x/y/‖` 全部 float64 torch；可见性硬门控；`snr_ref_db` 由 config 传入（可为任意值）。

T1 断言（`check_shared_echo.py`）：求和线性、噪声一次、顺序无关、经验噪声方差。
诊断：把 1 帧 3 BS 的 Y/X 存 `reports/f01e/diagnostics/shared_echo_frame.npz`（小样本）。

验收：T1 全过；单帧生成 GPU 时间 <50 ms/BS。

### P2 detector + T2 + CFAR 底噪标定（2–3 会话）

文件：`frontend/sensing/detector.py`、`frontend/sensing/coords.py`、
`code/07_shared_frontend/calibrate_cfar.py`、`code/07_shared_frontend/check_no_gt.py`。

实现：3D CA-CFAR（torch 盒滤波，需 float64 累加）、NMS、抛物线插值、单站几何、
检测记录 schema（INTERFACE_SPEC §3.2）。

标定：**噪声-only**（不注入目标，Y=W）跑 ≥512 帧/BS，测量不同 `Pfa_cell` 下的虚警数，
选定使目标虚警 ≈ `target_false_alarms_per_bs_frame`（默认 0.5）的阈乘数；
输出 `reports/f01e/cfar_calibration.json`（不含 GT）。

T2：AST/签名/import 哨兵/置换测试全部落 `check_no_gt.py`。

验收：单目标合成场景（已知解析位置）检测误差在栅格内；T2 全过；噪声-only 虚警率符合目标。

### P3 association/fusion（1–2 会话）

文件：`frontend/fusion/association.py`、`code/07_shared_frontend/check_association.py`。

实现：Mahalanobis χ²(2) 门控、分量枚举、分量内最大基数/最小代价匹配（scipy）、
逆协方差融合；观测 schema（INTERFACE_SPEC §3.3）。

单元：两 BS/三 BS 合成检测的融合位置误差应 ≈ 单站误差/√n（within 20%）；
漏一站时用剩余站；同站两个近邻检测不串配（构造 2×1 案例）。

验收：合成用例全过；T6 的 1/2/3 BS 消融脚本可运行（数字待 P5 汇总）。

### P4 tracker + slots + T7（1–2 会话）

文件：`frontend/tracking/cv_kf.py`、`code/07_shared_frontend/check_tracker.py`。

实现：STATE_MODEL §7-8；确认规则（≥2 次更新/3 帧 + ≥1 次 n_bs≥2）；coast 5 帧；
slot 分配/回收（冷却 10 帧）；>8 时的非 GT 淘汰；`detected`/`track_exists` 语义。

T7 用例：丢弃检测、恢复、删除、slot 复用、mask 关系断言。

验收：T7 全过；单目标恒速合成场景速度收敛（10 帧后速度误差 <0.5 m/s，配置待标定）。

### P5 打包/loader/结构验收 + 小规模端到端（2 会话）

文件：`frontend/pack_shared_dataset.py`、`frontend/check_shared_dataset.py`、
`scripts/run_f01e_shared.py`（先支持 `--episodes 0 1 --snr-ref-db 20` 冒烟）、
`frontend/symbol_dataset.py`（加 `--data` 根）。

实现：20 帧窗口、labels 分离、metadata 路由、C 域 track↔GT 匹配（cost/gate/诊断）、
train-only normalization、输入 hash 清单。

验收：2 episodes × 1 个临时 `snr_ref_db` 的 f01e 冒烟缓存通过结构与 T1–T7 子集；
`data/f01d` hash 不变；T4/T5 压力脚本产出报告。

### P6 校准 sweep + STOP 点（1 会话 + 用户评审）

文件：`scripts/calibrate_f01e_snr.py`、`reports/f01e/calibration/**`。

执行：`snr_ref_db = -10..40 step 5`，train 子集（≥3 episodes，含多密度帧），T3 全指标；
输出曲线与候选档位（clean/nominal/challenging）。

**硬停**：正式 `snr_levels_db` 与 `snr_ref_db` 由用户在评审后决定（D20）。
不得为了让下游好看而反向调 sensing；不得提前冻结。

### P7 正式全量（用户手动启动）

入口：

```bash
cd /home/dell/YrM/ICCT
nohup /home/dell/YrM/envs/ICCT/bin/python -u scripts/run_f01e_shared.py \
  --stages generate,sense,track,pack,normalize,validate > reports/f01e/run.log 2>&1 &
```

（实际由用户执行；助手不代启动。）产出 `data/f01e/**`、`reports/f01e/F01E_REPORT.md`、
`run_status.json`。验收：`check_shared_dataset` 全过、`search_failure` 类计数器清零或解释、
输入 hash 清单封存、旧资产 hash 复核。

### P8 下游切换（不属于 REBUILD-03）

ADE_oracle vs ADE_sensing（T8）与 GNN/QGNN 重训在独立任务中决策；本轮只交付 sensing 侧。

## 3. 运行时间与资源预算（估算，RTX4090×2）

| 阶段 | 工作量 | 估计 |
|---|---|---|
| P1 单帧生成 | 199 帧×3 BS×280 episodes×1 档 | ~0.5–1.5 h/档 |
| P2 检测（含 3D CFAR） | 同上，主导开销 | ~1–3 h/档（oversample=2） |
| P6 校准 sweep | 11 档×子集（3 episodes） | ~20–60 min |
| P7 正式全量 | 4–6 档×280 episodes | ~6–20 h（可双卡并行） |

若超标：先降 CSV 诊断输出频率、再降 CFAR 训练窗；不得降低 float64 或跳过 NMS。

## 4. 风险登记

| 风险 | 影响 | 缓解 |
|---|---|---|
| 角度分辨率不足（16 元 6.35°） | 远端横向位置误差大 | 3 BS 融合 + KF；T6 量化；必要时 V1.1 增大阵列 |
| CFAR 虚警目标值标定不稳 | 假航迹/确认率异常 | P2 噪声-only 标定；T3/T4 回归 |
| 近邻目标合并 | 短时少检 | 记录为预期局限（T5）；V1.1 DBSCAN/多散点 |
| Doppler 分辨率 1.97 m/s | 径向速度粗 | V1 不用 vr；V1.1 WLS 再用 |
| 运行时间 | 全量延迟 | 流式、双卡、诊断降频 |
| 误把临时档位当正式档位 | 违反 §13 | P6 硬停；config 中 `snr_levels_db=null` 直到用户批准 |

## 5. 交付清单（REBUILD-03 完成时）

```text
frontend/sensing/{waveform,simulator,detector,coords}.py
frontend/fusion/association.py
frontend/tracking/cv_kf.py
frontend/{pack_shared_dataset,check_shared_dataset}.py
scripts/{run_f01e_shared,calibrate_f01e_snr}.py
configs/shared_frontend.json
code/07_shared_frontend/*（诊断与测试）
reports/f01e/{calibration/**, F01E_REPORT.md, run_status.json, baseline_hashes.json}
data/f01e/**（正式缓存，用户启动生成）
```

不变更：`data/f01d/**`、`configs/symbol_frontend.json`、旧 frontend 文件行为、下游模型。