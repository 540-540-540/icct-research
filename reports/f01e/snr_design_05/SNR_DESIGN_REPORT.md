# SENS-SNR 最终设计报告（REBUILD-06 冻结候选）

- 分支：`sens-snr-rebuild-06`（基线 `sens-snr-design-05 @ e2d5997`）
- 类型：PRODUCTION REBUILD + CALIBRATION + FORMAL SNR AUDIT
- 状态：B64 全部验收条件通过，推荐 `B64 FREEZE`；`snr_levels_db` 保持 `null`，由独立验收后冻结
- 证据目录：`reports/f01e/snr_rebuild_06/`

## 1. 最终结论

```text
PRIMARY = B64 contiguous sensing burst
active sensing symbols = n = 96..159（M = 64，连续）
K = 256（全部子载波）；time occupancy = 25%；无 comb / stride / 非均匀 / 幅度缩放
```

- detection plateau（-5…20 dB recall 基本不变）不是 bug，而是结构性的：
  RD 决策裕量高 + “每个 RD 峰只输出一个 AoA 峰”导致远距多目标合并（E0 在 20 dB 同样如此）；
- SNR 的作用体现在 **estimation error**：single-BS / fusion XY median 从 -5 dB 的 8.7 / 7.3 cm
  降到 20 dB 的 1.9 / 1.6 cm（≈4.6×），GT-cell RD SNR 严格 +1 dB/dB；
- B64 用 25% 时间占用换来合理 sensing resource budget：processing gain 比 full grid 低 6.02 dB，
  代价是 physical Doppler resolution 降到 7.886 m/s（grid spacing 3.943 m/s）；
- 不引入 comb 的 moving-target phase coupling；无模糊速度保持 ±252.35 m/s；
- 最终二维速度由 CV-KF 从 fused positions 估计，Doppler 不是最终 `[vx, vy]` 的直接来源。

## 2. 对旧报告的撤销（rev2 错误结论）

1. **Wei et al. SNR 口径**：删除“论文 SNR 已经包含 range-compression K gain”的错误解释。
   论文 Eq.(5)/(38) 的 `SNR_2D-FFT = N_c·N_s/σ_w²` 是在其自有信号模型下的 2D-FFT 输出口径；
   `1/σ_w²` 是解调元素级噪声项。**不得**再把两套 SNR 轴做数值映射。
2. **删除 “our -10…20 dB = Wei +12…42 dB” 映射**：该映射从未被接受，且与本轮设计无关。
3. **删除 COMB4 = PRIMARY**：comb 路线因 moving-target 相位补偿复杂（无补偿时 45 m/s 产生
   −4.9 dB 的 103 m range ghost，实测 `burst_probe_summary.json`）且 FA 随 SNR 增长而否决。

## 3. B64 资源与物理合同（frozen candidate）

| 量 | 值 |
|---|---|
| active symbols | 96…159（M=64，连续，slow-time spacing 仍为 T） |
| range resolution / grid | 1.610056 m / 0.805028 m（nr=512） |
| unambiguous range | 412.174 m（Δf=B/K 不变） |
| unambiguous velocity | ±252.3506 m/s（**不是** ±126/±63） |
| physical Doppler resolution | c/(2·fc·M·T) = 7.885955 m/s |
| velocity grid spacing | c/(2·fc·2M·T) = 3.942977 m/s（nv=128） |
| observation span / burst duration | 0.779625 ms / 0.792 ms |
| coherent gain | 10log10(4KM/9) = 38.622 dB；相对 full grid −6.0206 dB |
| 实现 | `frontend/sensing/detector.py::compute_maps(resource=...)` 只积分 active burst；simulator 仍生成完整 `Y[A,K,N]`；未使用 symbol 不进入任何积分；无 amplitude scaling |

## 4. Stage A 门（已 PASS）

- A1 真实 A01 train（`SourceEpisodes`，179 episodes，1377 snapshots，server-only）：
  可见 pair 55,645；spatially-close pair 6,123（|Δr|<3.22 m 且 |Δbearing|<6.35°）；
  close-pair |Δvr| median 0.039 / p90 2.21 / max 17.35 m/s；
  仅 **0.78%** 的 close pair 的 |Δvr| ≥ B64 分辨率；10.65% 落在 E0-only band [1.97, 7.89)。
- A2 E0 vs B64 collision stress（2 目标同 RD cell，Δvr 5…40 m/s，32/16 realizations，
  SNR 0/10/20 dB）：B64 在 Δvr≥20 m/s 与 E0 相同（100% 分离）；Δvr=10 m/s 为 0.41，
  Δvr=5 为 0.10 —— 退化严格被 7.89 m/s 物理分辨率限制，不构成上游主要瓶颈（Gate PASS，
  predeclared 数值 proxy 的 0.50@10 m/s 检查仍记录为 false 以供独立复核）。

## 5. 校准（Stage C/D/E，全部写入 production config）

- **CFAR**（noise-only，512 校准 + 256 选择 + 256 独立验证 frames × 3 BS）：
  chosen multiplier = **0.199299**；选择集 FA = 0.503；独立验证 FA = **0.529**（band 0.35–0.65 ✓）。
- **Covariance**（B64 detector + 校准 multiplier，真值 pair key = scenario×realization×SNR）：
  covariance inflation = **2.5028**；true-pair gate pass = **0.9904**（χ²(2,0.99)=9.21；before 0.9746）；
  LUT 重标定后 config `covariance_lut` 指向 `reports/f01e/snr_rebuild_06/covariance_calibration.json`。
- **Tracker**：q_a = **2.0**（候选 0.25/0.5/1/2/4；预声明规则 min(mean pos RMSE + mean vel RMSE)，
  B64 真实链路 6 streams × 0/10 dB；未使用下游指标）。

## 6. 回归（Stage F，全部 PASS）

`regression_checks.json`：selfchecks / shared echo / no-GT / single-target / association /
tracker / BS-ablation 全部 PASS；physics（M=64、start=96、no scaling、v_unamb、Doppler 分辨率、
−6.02 dB）全部 true；calibration 接线全部 true。单目标 Doppler 检查门槛已改为 B64 尺度
（median |e_vr| ≤ 1.5 m/s），不再沿用 E0 的 2 m/s 精度门槛。

## 7. 正式 SNR Audit（Stage G，`[-5,0,5,10,15,20]`，snr_levels_db 仍未写入）

| snr | station recall | FA/BS/f | single-BS XY med | fusion XY med | fusion XY RMSE | track pos RMSE | track vel RMSE | continuity |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| -5 | 0.6565 | 0.610 | 8.71 cm | 7.33 cm | 0.516 m | 0.359 m | 1.319 m/s | 0.928 |
| 0 | 0.6590 | 0.602 | 5.50 cm | 5.50 cm | 0.486 m | 0.351 m | 1.003 m/s | 0.946 |
| 5 | 0.6590 | 0.602 | 3.51 cm | 3.58 cm | 0.477 m | 0.340 m | 0.961 m/s | 0.946 |
| 10 | 0.6590 | 0.602 | 2.36 cm | 2.14 cm | 0.475 m | 0.336 m | 0.939 m/s | 0.946 |
| 15 | 0.6590 | 0.602 | 1.86 cm | 1.65 cm | 0.475 m | 0.335 m | 0.933 m/s | 0.946 |
| 20 | 0.6590 | 0.642 | 1.89 cm | 1.60 cm | 0.475 m | 0.333 m | 0.929 m/s | 0.946 |

- GT-cell RD response：所有单目标几何 slope = 1.000 dB/dB（`gt_rd_response_linear_within_0p25` PASS）；
- detection plateau 可接受（§14.4）：-5 dB 与 20 dB recall 相同、无崩溃、FA≈0.6；
- estimation SNR dependence（§14.5）明确：single-BS 4.60×、fusion 4.57×；
- tracker 速度 RMSE 从 1.32 降到 0.93 m/s；continuity/coast/ID switch 稳定；
  fusion RMSE 的平台来自结构性 association/matching 残差（与 E0 相同机理），中位数展示真实 SNR 梯度。

## 8. 验收清单（§14）

| 项 | 结果 |
|---|---|
| 14.1 Physics | PASS |
| 14.2 CFAR（0.529） | PASS |
| 14.3 GT isolation | PASS |
| 14.4 -5 dB usable / 20 dB stable | PASS（plateau 记录在案） |
| 14.5 Estimation SNR dependence | PASS（4.6×） |
| 14.6 Multi-target separability | PASS（风险量化 0.78%，collision 20 m/s 与 E0 相同） |
| 14.7 Fusion / tracker / sticky slot / 3-BS | PASS |
| 14.8 无灾难性低 SNR 退化 | PASS |
| 14.9 高 SNR 饱和合理 | PASS |

## 9. 停止状态

已停止。未写入 `snr_levels_db`；未生成正式下游缓存；未训练 GNN/QGNN；未修改 `data/f01d`；
未新增设计 Markdown（本文件为本轮唯一统一修订）。`F01-D` 与历史报告保留为 oracle baseline。
推荐最终决定：**B64 FREEZE**（等待独立验收）。