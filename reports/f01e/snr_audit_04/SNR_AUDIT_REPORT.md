# SENS-SNR-AUDIT-04：正常 SNR 区间感知响应审计报告

- 分支：`sens-snr-audit-04`（基线 `sens-rebuild-03b` @ `71a9abfeff056345e59dbe2c9576335b8c146643`）
- 审计区间：`snr_ref = [-10, -5, 0, 5, 10, 15, 20] dB`（audit points，非正式档位）
- 场景：41 个 controlled synthetic snapshots（count 1/3/5/8 × near/medium/far × center/mid/edge，
  含 2 个 close-target）+ 3 个 train-subset snapshots；6 条 12 帧流（3 合成 + 3 train-subset）
- 只读审计：未改正式 frontend/config/cache，未进入 P5/P6，未训练下游

## 1. Phase A：03B calibration 修正（已完成并合入 03B）

| 修正 | 结果 |
|---|---|
| covariance true-pair grouping | pair key 修正为 `(scenario, realization, snr_ref)`；true pairs **1499 = 组合数理论上限**（修正前 13 491 为跨 SNR 混合） |
| covariance inflation 重标定 | 对每个候选 inflation 调用真实 `covariance_from_lut()` 重算 D²（不用除法近似）；倍增+二分求最小必要值 **3.9255**（1.0 仅 97.40% 命中）；命中率 **99.07%**，p99(D²)=9.21 |
| tracker q_a 真值 | 合成轨迹直接保存解析 position+velocity（删除末帧差分=0）；选择准则 = min(CA pos + CA vel + CV vel)；最终 **q_a = 1.0 m²/s³** |
| no-GT full B-domain | 审计扩展到 detector/coords/association/cv_kf，禁止 A 域 import 与 GT/truth/source/vehicle/count/source_key/future 参数；T2 8/8 PASS |
| D13 sticky slot | 冻结文档更新：alive 期 slot identity 固定，slotless confirmed 冷却后按非 GT 排名补位 |
| 重跑 | covariance / tracker calibration / P3 8/8 / P4 5/5 / T6-A / T6-B / T2 全部 PASS |

03B fix commit：`71a9abfeff056345e59dbe2c9576335b8c146643`

## 2. SNR 语义（Q1）

三个量在本项目中**严格区分**：

```text
snr_ref_db      : 参考距离 R0=100 m、参考 RCS σ0=10 m² 下，逐接收 entry 的处理前 SNR（控制变量）
received_snr_db : 具体目标/BS 的实际处理前接收 SNR
                  = snr_ref + 40·log10(100/r) + 10·log10(σ/σ0)，V1 σ=σ0
peak_to_noise_db: RD 处理与 CFAR 之后的检测峰值与局部噪声底之比（检测统计量，含处理增益）
```

`SENS-SNR-AUDIT-04` 中 `snr_ref=-10 dB` 绝不等于"接收 SNR=-10 dB"。

## 3. Level-1/2：输入质量与 RD 处理（Q2, Q3）

真实 train-subset 几何（2671 个可见 target×BS）：

```text
range (m):        p10 65.1 / p50 129.4 / p90 218.0
received_snr - snr_ref (dB): p10 -13.54 / p50 -4.48 / p90 +7.44
```

审计场景的 received SNR 分布：

| snr_ref | received p10 | received p50 | received p90 |
|---:|---:|---:|---:|
| -10 | -27.9 | -16.6 | -0.7 |
| 0 | -17.9 | -6.6 | +9.3 |
| 10 | -7.9 | +3.4 | +19.3 |
| 20 | +2.1 | +13.4 | +29.3 |

即 `snr_ref=-10 dB` 时中位接收 SNR 约 **-16.6 dB**，p10 到 **-27.9 dB**。

经验处理增益（仅诊断量 `G_emp = peak_to_noise_db − received_snr_db`，不宣称理论相干增益）：

| snr_ref | G_emp median | peak_to_noise median |
|---:|---:|---:|
| -10 | **+37.0 dB** | 19.8 dB |
| 0 | +27.6 dB | 23.3 dB |
| 10 | +17.7 dB | 23.8 dB |
| 20 | +7.7 dB | 23.8 dB |

说明：低 SNR 端 peak-to-noise 仍在 20–24 dB，且 G_emp 在低端高达 ~37 dB；
`peak_to_noise` 在高端封顶 ~24 dB 是因为 CFAR 局部噪声底被强目标自身旁瓣/泄漏抬高（非热噪声底）。

## 4. Level-3：Detection（Q4, Q5）

| snr_ref | station recall | target recall | det/BS/frame | FA/BS/frame |
|---:|---:|---:|---:|---:|
| -10 | 0.8727 | 0.9127 | 2.69 | 0.000 |
| 0 | 0.8727 | 0.9127 | 2.69 | 0.000 |
| 10 | 0.8727 | 0.9127 | 2.69 | 0.000 |
| 20 | 0.8753 | 0.9127 | 2.76 | 0.000 |

按 range bin 的 recall 在各 SNR 下完全相同（结构性差异，非 SNR 驱动）：

```text
10–75 m: 0.864–0.879 | 75–150 m: 0.825 | 150–225 m: 0.938 | 225–300 m: 0.873
```

结论：**-10–20 dB 内 detector 完全饱和**；recall 变化 0.26 pp；未匹配的主要原因是
close-target 合并/密集场景匹配（结构性），与 SNR 无关。FA 在目标场景为 0（噪声-only 标定值 0.5/BS/帧
是空场景上限；目标场景 CFAR 局部底被抬高）。

## 5. Level-4/5：单站估计与跨站融合

| snr_ref | range median abs | u median abs | single-BS xy median | fusion xy median | fusion xy RMSE |
|---:|---:|---:|---:|---:|---:|
| -10 | 8.1 mm | 5.6e-4 | 11.7 cm | 5.6 cm | 0.644 m |
| 0 | 3.3 mm | 2.2e-4 | 4.8 cm | 2.9 cm | 0.614 m |
| 10 | 2.4 mm | 1.5e-4 | 2.9 cm | 1.7 cm | 0.612 m |
| 20 | 2.3 mm | 1.3e-4 | 2.6 cm | 1.5 cm | 0.605 m |

- 匹配成功后估计精度确有单调响应（single-BS 中位 11.7→2.6 cm，融合中位 5.6→1.5 cm），
  **但绝对量级为厘米**，且 RMSE 被结构性 outlier 主导（0.60–0.64 m，几乎不随 SNR 变化）。
- association/fusion 结构量完全饱和：n_bs 分布 ~0.57/0.34/0.09，gate 拒绝 ~20.6/帧，
  post-filter ~1.4/帧，gt_match 0.913，obs precision 0.697 —— 各 SNR 下不变。

## 6. Level-6：Tracker

| snr_ref | continuity | coast ratio | ID switches | track pos RMSE | track vel RMSE |
|---:|---:|---:|---:|---:|---:|
| -10 | 0.9693 | 6.4% | 16 | 0.446 m | 1.454 m/s |
| 0 | 0.9878 | 7.5% | 13 | 0.454 m | 1.051 m/s |
| 10 | 0.9878 | 7.5% | 13 | 0.450 m | 1.033 m/s |
| 20 | 0.9878 | 7.5% | 13 | 0.450 m | 1.031 m/s |

- continuity 变化 1.85 pp；coast ratio 6.4%→7.5%（结构性，非 SNR 驱动）；
  position RMSE 完全平坦；velocity RMSE 仅在 -10 dB 略差（1.45 vs 1.03，来自个别流），0–20 dB 无差异。
- `detected=0 & track_exists=1` 在所有 SNR 下存在（6–8%），不随 SNR 下降而增加。

## 7. Saturation 判定（Q4/Q5/Q6）

按本轮工程判据（仅内部使用）：

| 判据 | 实测 | 通过 |
|---|---|---|
| detection recall 变化 < 5 pp | 0.8727→0.8753 = **0.26 pp** | ✓ |
| fusion RMSE max/min < 1.25 | 0.6437/0.6049 = **1.064** | ✓ |
| track continuity 变化 < 5 pp | 0.9693→0.9878 = **1.85 pp** | ✓ |
| coast ratio 基本为 0/无 SNR 依赖 | 6.4%→7.5%，平坦 | ✓（结构常数） |

**Verdict：SATURATED（在 -10–20 dB 内）**

保留说明（不改变判定）：匹配成功后的单站/融合**中位**估计误差仍随 SNR 单调减小
（厘米→毫米量级）；这只是饱和平台上的精度边角变化，不影响检测/跟踪/融合结构指标。

### 饱和来源（Q6）

1. **相干处理预算过大**：256×256 处理 + 逐峰 64 点 AoA；`snr_ref=-10 dB` 时 median G_emp 仍有 +37 dB，
   peak-to-noise 中位 19.8 dB（远高于 CFAR 阈值）。
2. **距离几何有利**：train-subset p50 距离 129 m，received 相对 ref 仅 -4.5 dB（p10 -13.5）；
   多数目标接收 SNR 并不低。
3. **无任何非理想因素 + 阵列/处理余量**：无 clutter/multipath/shadowing/RCS 起伏；
   内部 probe 显示即使 **K=N=64 或 4 阵元**，peak-to-noise 也仅下降 ~0.5–8.5 dB（见
   `processing_probe.csv`），说明整个预算对当前阈值是"过剩"的。
   （probe 中 peak-to-noise 变化小还因为局部噪声底随旁瓣升降；检测余量本身极大。）

## 8. Q7：现有系统能否支撑"SNR → sensing degradation → GNN/QGNN"因果链？

**NO**。

证据：-10–20 dB 内 detection/tracking/fusion 全部 KPIs 平坦（§4–§6）；唯一随 SNR 变化的是
厘米级估计中位数（0.117→0.026 m），远低于车辆轨迹预测的工作尺度（米级），
下游模型在相同训练流程下几乎无法观察到有意义的 ADE 差异。
若仅保留"估计精度"通道，可勉强称 CONDITIONAL；但按本轮"先证明链路"的标准，判 **NO**。

## 9. Q8：下一轮最小修改候选（本轮不实施）

1. **A. 处理预算与链路预算一致性**：把 coherent processing 的 K/N（及等效积分时间/噪声带宽）
   纳入链路预算，使 `snr_ref` 与处理增益之间不再"免费"; 例如 K=N=64/64 或按论文日照条件
   重新定义 reference SNR（需重标定 CFAR/covariance，成本低）。
2. **C. 一种最小且文献支撑的非理想因素**：按 TR 38.901 给车辆 RCS 加入对数正态起伏
   （σ_S=3.41 dB，均值 10 dBsm）+ 固定双向波束损失；产生帧间起伏，使低端出现自然 miss。
3. **B. 阵列/处理二级旋钮**：在 (1) 之后，用 4/8/16 阵元与不同 CFAR 训练窗做一次
   calibration-style 消融，选择让 -10–20 dB 出现"困难→过渡→饱和"的最小配置。
   不改正式 16 阵元配置，除非 (1)+(2) 仍无法产生响应。

排序理由：先修定义一致性（A）→ 再加唯一必要非理想因素（C）→ 最后才动阵列/处理旋钮（B）。

## 10. 输出文件

```text
reports/f01e/snr_audit_04/
  SNR_AUDIT_REPORT.md      （本文件）
  summary.json
  response_table.csv       （§4–§6 汇总）
  range_stratified.csv     （按 range bin 的 recall）
  processing_gain.csv      （逐 detection 的 received/G_emp）
  processing_probe.csv     （K/N 与阵列 probe）
  geometry_probe.json      （train 距离/接收 SNR 分布）
  tracker_response.csv     （逐 stream×SNR tracker 指标）
  figures/01..08.png       （8 条诊断曲线）
```

## 11. 停止状态

审计完成并停止：未修改正式 sensing 模型，未冻结正式 SNR levels（`snr_levels_db` 仍为 `null`），
未进入 P5/P6，未训练 GNN/QGNN。阻断项：无。