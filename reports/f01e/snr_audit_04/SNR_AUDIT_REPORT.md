# SENS-SNR-AUDIT-04：正常 SNR 区间感知响应审计报告（rev2，evaluator 修正版）

- 分支：`sens-snr-audit-04`
- rev1 commit：`5b575a459df54fdc783f37e47b15ef85f18c508a`；本 rev2 修正 audit evaluator 后重新生成全部结果
- 审计区间：`snr_ref = [-10, -5, 0, 5, 10, 15, 20] dB`（audit points，非正式档位）
- 场景：41 snapshots + 6 条 12 帧流（3 合成 + 3 train-subset，identity 采用 A01 slot）
- 未修改 production sensing/config/K/N/阵列/CFAR/RCS/clutter；未进入 P5/P6；未训练下游

## 1. Evaluator 修正（本轮唯一实质改动）

1. **Detection**：每个 BS/frame/SNR 独立构造 GT↔detection 的欧氏距离 cost matrix，
   Hungarian + 5 m gate，严格 one-to-one；FA = 真正未匹配 detection 数（不再用 detections−重复GT匹配）。
2. **Fusion**：GT↔observation 同样严格 one-to-one（5 m gate）；target recall / observation precision /
   fusion RMSE / n_bs / association correctness 全部重算。
3. **Tracker**：每帧 GT↔TrackRecord 严格 one-to-one；train stream 保留 `at_time` 的 slot 作为稳定车辆
   identity（不再默认 `target_index` 跨帧同车）；synthetic stream 使用固定 target key（不含 frame）；
   ID switch 按稳定 identity 统计。
4. **Density stratification**：按场景真实 target count 分 low 1–3 / mid 4–6 / high 7–8，
   分别统计 recall/miss/估计质量。
5. **Detector-independent 诊断量**：对 controlled single-target scenes，用 clean 与 noise 分量在
   GT 对应 RD cell（±1 cell）计算 `gt_rd_snr_db = 10log10(max(clean 3×3)/median(noise 3×3))`，
   对 miss 也可计算；`G_emp` 保留但注明为 conditional-on-detection。

## 2. 修正前后对比

| 指标 | rev1（旧 matching） | rev2（严格 one-to-one） | 变化 |
|---|---|---|---|
| station recall（-10…15 dB） | 0.8727 | **0.6921** | −18.1 pp（旧值把同一 detection 重复计入多个 GT） |
| false alarms / BS / frame | 0.000 | **0.48**（-10…15）/ 0.545（20） | 旧 FA 定义错误；新值与 CFAR 目标 0.5 一致 |
| target recall | 0.9127 | 0.9012 | −1.2 pp |
| fusion RMSE（-10 / 20 dB） | 0.6437 / 0.6049 | 0.6744 / 0.6494 | +0.03（量级不变） |
| observation precision | 0.697 | 0.711 / 0.686 | 定义修正 |
| tracker continuity | 0.9693 / 0.9878 | 0.9207 / 0.9184–0.9392 | 严格 identity 匹配后下降 |
| tracker coast ratio | 6.4% / 7.5% | 5.1% / 6.6–7.9% | 量级不变 |
| tracker pos RMSE | 0.446–0.454 m | 0.319–0.356 m | 略降 |
| tracker vel RMSE | 1.03–1.45 m/s | 1.10–1.20 m/s | 量级不变 |
| ID switches（合计） | 13–16 | 6–8 | 稳定 identity 后更少 |

**SATURATED 结论未发生变化**（判据见 §5）。

## 3. 修正后 Level-1/2/3（Q2–Q5）

| snr_ref | received p10/p50/p90 | peak-to-noise med | G_emp med | station recall | miss | target recall | FA/BS/f |
|---:|---|---:|---:|---:|---:|---:|---:|
| -10 | -27.9 / -16.6 / -0.7 | 21.4 | 37.9 | 0.6921 | 0.3079 | 0.9012 | 0.480 |
| 0 | -17.9 / -6.6 / +9.3 | 23.9 | 30.5 | 0.6921 | 0.3079 | 0.9012 | 0.480 |
| 10 | -7.9 / +3.4 / +19.3 | 24.3 | 20.9 | 0.6921 | 0.3079 | 0.9012 | 0.480 |
| 20 | +2.1 / +13.4 / +29.3 | 24.4 | 11.0 | 0.6947 | 0.3053 | 0.9012 | 0.545 |

**检测独立的 GT-cell RD 诊断（单目标 controlled scenes，9 个几何）**：

```text
snr_ref=-10: 16.5–51.5 dB（近/中/远三簇 50.9 / 29.3 / 17.1）
snr_ref=  0: 26.5–61.5 dB
snr_ref= 20: 46.5–81.5 dB
```

`gt_rd_snr_db` 随 snr_ref **严格线性 +1 dB/dB**（偏移约 +39 dB 由 256×256 RD 处理与信噪定义给出）。
即：**RD 处理层没有饱和**；饱和发生在之后的决策/关联/跟踪层。

按 range bin 的 recall（各 SNR 下完全相同，结构性）：10–75 m 0.803；75–150 m 0.636；
150–225 m 0.823；225–300 m 0.507。中段偏低主要由 close-target scenes（2 m 间距合并）造成。

按 density（各 SNR 下相同，结构性）：low 0.706 / mid 0.696 / high 0.682。

## 4. Level-4/5/6（修正后）

| snr_ref | range med | u med | single-BS xy med | fusion med | fusion RMSE | nbs(1/2/3) | gate/f | post/f |
|---:|---:|---:|---:|---:|---:|---|---:|---:|
| -10 | 4.8 mm | 4.1e-4 | 6.5 cm | 5.5 cm | 0.674 | 0.53/0.38/0.09 | 21.5 | 1.4 |
| 0 | 2.5 mm | 1.9e-4 | 2.8 cm | 2.9 cm | 0.649 | 0.53/0.38/0.09 | 21.5 | 1.4 |
| 10 | 2.1 mm | 1.3e-4 | 1.9 cm | 1.7 cm | 0.647 | 0.53/0.38/0.09 | 21.5 | 1.4 |
| 20 | 1.9 mm | 1.2e-4 | 1.8 cm | 1.5 cm | 0.649 | 0.53/0.38/0.09 | 22.4 | 1.5 |

Tracker（严格 one-to-one，稳定 identity）：

| snr_ref | continuity | coast | ID switches | pos RMSE | vel RMSE |
|---:|---:|---:|---:|---:|---:|
| -10 | 0.9207 | 5.1% | 6 | 0.330 m | 1.202 m/s |
| 0 | 0.9392 | 6.6% | 6 | 0.319 m | 1.097 m/s |
| 10 | 0.9369 | 7.9% | 8 | 0.345 m | 1.119 m/s |
| 20 | 0.9184 | 7.9% | 8 | 0.356 m | 1.205 m/s |

## 5. Saturation 判定（修正后，未调整判据）

| 判据 | 实测（rev2） | 通过 |
|---|---|---|
| detection recall 变化 < 5 pp | 0.6921→0.6947 = **0.26 pp** | ✓ |
| fusion RMSE max/min < 1.25 | 0.6744/0.6473 = **1.042** | ✓ |
| track continuity 变化 < 5 pp | 0.9207→0.9392 波动 = **≤2.1 pp** | ✓ |
| coast ratio 基本为 0 / 无 SNR 依赖 | 5.1–7.9%，非单调、幅度小 | ✓ |

**Verdict：SATURATED（-10–20 dB 内），与 rev1 一致。**

保留说明：匹配成功后的 mm/cm 级估计中位数仍随 SNR 单调改善
（single-BS 6.5→1.8 cm、fusion 5.5→1.5 cm）；这不改变检测/跟踪/融合结构指标的平台结论。

## 6. 主要平台来源（修正后的定位）

1. **决策层余量过大**：`gt_rd_snr_db` 线性响应且最低簇在 -10 dB 仍有 ~17 dB、近距 ~51 dB；
   CFAR 峰值/底噪中位 21–24 dB，远高于阈值 → 判决结果与 SNR 无关。
2. **结构性 miss 主导残余误差**（与 SNR 无关）：station miss 30.8%，在全部 SNR 下相同；
   close-target 合并 + 严格 one-to-one 使同一 RD cell 内多目标只能匹配一个 GT；
   range bin 75–150 m recall 0.636、225–300 m 0.507 为几何/合并效应。
3. **融合与跟踪继承**：n_bs 分布、gate/post-filter 率、continuity/coast/RMSE 全平坦；
   只有 mm/cm 级估计中位数响应。

## 7. Q1–Q8（rev2 结论）

- **Q1**：`snr_ref_db` = 100 m 参考距离、10 m² 参考 RCS 下逐接收 entry 的处理前 SNR 控制变量；
  与实际 `received_snr_db = snr_ref + 40log10(100/r)` 和 `peak_to_noise_db` 严格区分。
- **Q2**：-10 dB 时 received p10/p50/p90 = -27.9/-16.6/-0.7 dB；train 几何相对偏移 p50 -4.5 dB。
- **Q3**：`G_emp` 中位 37.9 → 11.0 dB（conditional-on-detection）；检测独立的 `gt_rd_snr_db`
  偏移约 +39 dB，逐 dB 线性。
- **Q4**：压平发生在 **CFAR 决策层**（阈值余量），不在 RD 处理（gt_rd 线性），也不在 fusion/tracker
  （两者继承平台）。
- **Q5**：detector 在 -10–20 dB 已饱和（recall 变化 0.26 pp；miss 结构恒定）。
- **Q6**：几何 + 处理预算使实际 RD SNR 远高于阈值（+39 dB 级偏移）；即使降到 K=N=64 或 4 阵元
  仍不足 8.5 dB 落差（rev1 probe）；距离 p50 129 m 仅 -4.5 dB 损失。
- **Q7**：**NO**（检测/跟踪/融合指标平坦；唯一 SNR 响应是 mm/cm 估计精度与原始 RD 诊断，
  下游轨迹预测尺度为米级，因果链不可观察）。
- **Q8**：见 §8。

## 8. 下一轮最小修改候选（本轮不实施，顺序不变）

1. **A. 处理/链路预算一致性**：把 K/N（等效积分时间/噪声带宽）纳入 `snr_ref` 定义，
   去掉"免费"的 ~39 dB 处理偏移（需重标定 CFAR/covariance）。
2. **C. 一种最小文献支撑非理想因素**：TR 38.901 车辆 RCS 对数正态起伏（σ_S=3.41 dB）+ 固定双向波束损失。
3. **B. 阵列/处理二级旋钮**：在 (1)(2) 之后做 4/8/16 阵元与 CFAR 窗的标定式消融，
   选择让 -10–20 dB 出现"困难→过渡→饱和"的最小配置（不改正式 16 阵元，除非必要）。

## 9. 输出文件（全部已按 rev2 重生成）

```text
reports/f01e/snr_audit_04/
  SNR_AUDIT_REPORT.md   summary.json
  response_table.csv    range_stratified.csv（含 density 与 range_bin 两个维度）
  processing_gain.csv   （kind=input/detection/gt_rd 三类行）
  gt_rd_probe.json      processing_probe.csv（rev1 保留）geometry_probe.json（rev1 保留）
  tracker_response.csv  figures/01..08.png
```

## 10. 停止状态

已 push、停止。未修改正式 sensing 模型，未冻结正式 SNR levels（仍为 `null`），未进入 P5/P6，
未训练 GNN/QGNN。阻断项：无。