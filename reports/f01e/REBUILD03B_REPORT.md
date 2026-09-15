# REBUILD-03B：3-BS 多目标关联、融合与跟踪实现报告

- 分支：`sens-rebuild-03b`（基线 `sens-rebuild-03a` @ `e93a9875eef6d613df3735aea383f9fe77067725`）
- 范围：P2.5 低质量协方差补充、P3 3-BS 顺序 Hungarian 关联/融合、P4 CV-KF 多目标跟踪、
  匿名 track→固定 8 slots、T6-A/T6-B 与 T7 基础验收
- 未进入：P5 packing、正式 `data/f01e` 缓存、正式 SNR 档位、下游训练、V_confirm/test
- 运行环境：服务器 `YrM_TwYhB`，`/home/dell/YrM/envs/ICCT/bin/python`（CUDA float64）

## 1. 结论总表

| 项 | 结果 | 证据 |
|---|---|---|
| P2.5 covariance extension | **PASS** | `reports/f01e/covariance_calibration.json`、`reports/f01e/p25_covariance/{summary.json,bin_coverage.csv}` |
| P3 association/fusion | **PASS** | `frontend/fusion/association.py`、`reports/f01e/p3_association/{summary.json,checks.json}`（A–H 8/8） |
| Association no-GT | **PASS** | T2 8/8 + P3 置换/接口检查（无 GT 输入，仅 `x,y,C_xy,station_id,peak_to_noise_db`） |
| T6-A | **PASS** | `reports/f01e/t6a_bs_ablation/summary.json`（多站不劣化，见 §4） |
| P4 tracker | **PASS** | `frontend/tracking/cv_kf.py`、`reports/f01e/p4_tracker/{summary.json,checks.json}`（5/5） |
| T7 dropout | **PASS** | `p4_tracker/checks.json:lifecycle_and_dropout`（dropout 10–13 帧，coast 语义正确，恢复后继续同一 track） |
| 8-slot policy | **PASS** | `p4_tracker/checks.json:eight_slot_policy`（10 confirmed → 8 slots + 2 slotless；删除/冷却/按排名回收） |
| T6-B | **PASS** | `reports/f01e/t6b_bs_tracker/summary.json`（ablation mode，见 §4） |
| 修改 `data/f01d` | **NO** | `baseline_hashes.json` 与 03A 提交逐文件 SHA256 一致（26/26） |
| 读取 V_confirm/test | **NO** | 全部为 A01 几何 + 合成场景 |
| 下游训练 | **NO** | 未调用任何预测模型 |

## 2. P2.5 低质量协方差补充

扩展 sweep：`snr_ref_db = [-20..20] step 5`、range 30–300 m、bearing ±69°/±45°/0°、4 realizations；
仅统计几何可见的 BS（visible attempts 2556），全部为合成单目标。

| q 箱 | samples | detection share | sigma_r (m) | sigma_u | fallback |
|---|---:|---:|---:|---:|---|
| <6 dB | 5 | 0.002 | 0.0718 | 8.58e-3 | **是**（样本 <30，取 pooled/上包络） |
| 6–12 dB | 85 | 0.033 | 0.0718 | 4.75e-3 | 否 |
| 12–20 dB | 316 | 0.124 | 0.02795 | 1.86e-3 | 否 |
| >20 dB | 2022 | 0.791 | 0.00269 | 1.73e-4 | 否 |

- 总体可见检测率 0.950（按 SNR：0.940 @ -20 dB → 0.972 @ 20 dB）
- 低 q 样本仍偏少：`6–12 dB` 85 条、`<6 dB` 5 条。**如实声明**：该区间 detector 有效检出本身稀疏，
  covariance 使用保守 pooled/上包络回退，未制造假样本、未降低 detector 标准。
- 真实同目标跨站对的 gating 统计（13 491 对）：固定 χ²(2, 0.99)=9.21 门下命中率 97.40%；
  p99(D²)=17.51 → 由此标定 **covariance inflation = 1.90**，命中率提升至 **99.01%**。
  inflation 写入 LUT（`covariance_inflation`），由 `coords.covariance_from_lut` 应用；
  **gate 数值保持冻结的 9.21**，只校准协方差尺度。
- B 域 inference 仍只按 observed `peak_to_noise_db` 查表（T2 `covariance_depends_only_on_observed_q` 通过）。

## 3. P3 顺序 Hungarian 关联/融合

实现与冻结算法一致：检测按 `(peak_power 降序, grid_index 升序)` 排序 →
Stage 1 BS0↔BS1 Hungarian（Mahalanobis² + 9.21 门控，出格 BIG）→ 每次 assignment 显式 post-filter
（BIG/超门限恢复 unmatched）→ group state 逆协方差融合 → Stage 2 groups↔BS2 同规则 →
输出 `{time_ns,x_m,y_m,C_xy,bs_mask,n_bs,quality_db}`；无参考传感器依赖、无 GT。

单元测试 A–H 全部通过（`p3_association/checks.json`）：
- A 两站融合（n_bs=2）、B 三站融合（n_bs=3）
- C 出格对即使被 Hungarian 返回，也被 post-filter 撤销为 singleton
- D BS0 缺失（BS1+BS2 正常融合）、E BS1 缺失（BS0 singleton 在 Stage 2 与 BS2 融合）
- F 全 singleton、G 近邻冲突一对一且确定性、H 每站列表随机打乱 10 次 group set 不变
- 融合协方差正定、无 NaN

## 4. T6-A / T6-B（1/2/3 BS 消融）

3 个合成场景 × 3 目标 × 16 帧，`snr_ref=25 dB`；1 BS/2 BS 指标为所有单站/配对子集的平均。

| 档位 | detection recall | false alarms / frame | observation RMSE (m) |
|---|---:|---:|---:|
| 1 BS | 0.875 | 0.50 | 0.509 |
| 2 BS | 0.984 | 1.14 | 0.0191 |
| 3 BS | 1.000 | 1.71 | 0.0140 |

- 结论：多站信息没有系统性恶化；融合后位置精度提升约 36×（1BS→3BS），recall 提升。
- 1 BS 的 0.509 m 主要来自单站跨距/角度量化，未出现异常。

T6-B（tracker sanity，三档统一 `confirm_requires_nbs2=false`，仅诊断；正式 mainline 为 true）：

| 档位 | track RMSE (m) | continuity (frame≥3) | ID switches |
|---|---:|---:|---:|
| 1 BS | 0.168 | 93.2% | 0 |
| 2 BS | 0.014 | 100% | 0 |
| 3 BS | 0.013 | 100% | 0 |

- **ablation mode 不改变正式 3-BS mainline tracker**（报告与 summary 均已标注）。
- 弱测量开发点（非正式 SNR sweep）：`snr_ref=5 dB` 与 `-10 dB` 下，检测/融合/跟踪均保持
  （每 BS 约 3.0–3.4 detections/帧，obs ≈4.7/帧，无 coast）。**如实说明**：本组合成场景距离较近、
  相干积累增益高，-10 dB 仍不构成困难点；正式 SNR 档位与困难场景属 P6，本轮不选值。

## 5. P4 CV-KF 跟踪与 8-slot

- 状态 `[x,y,vx,vy]`、`F`/`H` 按冻结式，`R=C_xy`；CV white-noise acceleration `Q(q_a)`
- `q_a` 独立合成标定（CV + 温和 CA，10 Hz，LUT 派生量测协方差；无下游指标）：
  **q_a = 1.0 m²/s³**（`reports/f01e/p4_tracker/tracker_calibration.json`，来源可复现）
- 生命周期：birth→tentative→confirmed（3 帧内 ≥2 次更新且 ≥1 次 `n_bs≥2`）→coast（≤5）→delete（>5）；
  tentative 3 帧未确认即删除；tentative 不输出、不占 slot
- coast 语义：`track_exists=1, detected=0`；全流程恒有 `not (detected and not track_exists)`
- 8-slot：confirmed 时分配最小空闲 slot；删除后冷却 10 帧回收；
  confirmed>8 时只输出持 slot 的 8 条，slotless confirmed 按 `(misses↑, trace(P_xy)↑, age↓, track_key↑)`
  在 slot 可用时回收（全部为非 GT 信息）
- 合成精度（同 `q_a`，量测协方差来自 LUT）：CV 位置 RMSE 0.111 m / 速度 1.17 m/s；
  温和 CA 位置 0.130 m / 速度 1.28 m/s；无 NaN、单一 track key。
  判定界以物理分辨率为准：位置 <0.5 m（<距离分辨 1.61 m）、速度 <2 m/s（<一个多普勒分辨格 1.97 m/s）

## 6. 实现中发现的问题与处理（非静默重设计）

1. **真对 gating 命中率不足（实现级）**：固定 9.21 门下真实同目标对命中 97.4%，p99(D²)=17.5。
   处理：按 P2.5 gating 统计标定 covariance inflation=1.90（gate 不变），命中率 99.0%；
   记录于 LUT 与 `p25_covariance/summary.json`。**非阻断**。
2. **tracker gate 未在冻结文档给数值**：采用与 association 相同的 χ²(2,0.99)=9.21，
   记录为 implementation calibration。**非阻断**。
3. **矩形 Hungarian 未匹配行未记 miss（实现 bug）**：修复为对未分配行显式记 miss；
   修复后 coast/delete/T7 全部按设计工作。**非阻断**。
4. **确认计数是否含 birth 观测**：按"3 帧内 ≥2 次 measurement update"语义，birth 记 1 次 hit；
   生命周期测试随之确定（首帧出生、第 2 帧确认）。**非阻断**。
5. **>8 confirmed 的 slot 语义**：冻结文档要求"slot alive 期间固定"与"超 8 按排名只输出 8 个"；
   实现为"持 slot 输出 + slotless 按排名在冷却结束后补位"，全部非 GT 信息。**非阻断，记录解释。**

## 7. 复现命令

```bash
cd /home/dell/YrM/ICCT
/home/dell/YrM/envs/ICCT/bin/python -u code/07_shared_frontend/calibrate_covariance.py
/home/dell/YrM/envs/ICCT/bin/python -u code/07_shared_frontend/calibrate_tracker.py
/home/dell/YrM/envs/ICCT/bin/python -u code/07_shared_frontend/check_association.py
/home/dell/YrM/envs/ICCT/bin/python -u code/07_shared_frontend/check_tracker.py
/home/dell/YrM/envs/ICCT/bin/python -u code/07_shared_frontend/check_bs_ablation.py
/home/dell/YrM/envs/ICCT/bin/python -u code/07_shared_frontend/check_no_gt.py
```

## 8. 停止状态

P4 已完成并验收；**在 P4 停止**，未进入 P5 packing、P6 正式 SNR 校准、P7 全量缓存或下游训练。
阻断项：无。