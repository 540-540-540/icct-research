# Automatum ISAC Route-B Final Audit 报告

- 基线：`module/isac` @ `c2b023be13cc31abb59eba936c34ed1c76fcec62`（审计开始与结束时工作树干净）
- 性质：**AUDIT ONLY**。未修改任何 sensing 参数：Candidate B、noise floors、sigma 公式、SNR 档位、geometry、fusion、state_quality、samples.npz、split、dataset 与 frontend 数值路径均与 HEAD 逐字节一致（selfcheck 5–12 验证）。
- 数据规则：**train** 为主审计；**val** 独立确认；**test 完全未使用**（selfcheck 13；`decision_summary.json → test_used_for_parameter_decisions = false`）。
- 去重规则：prediction-history unique states = `samples.npz` 的 20-frame history 内 `(scene_id, frame, vehicle_id)` 唯一集合。train 769,400 行 → **49,765** 唯一状态（重复系数 15.46）；val 69,520 行 → **4,779**（14.55）。同一真实状态无论被多少重叠窗口使用只计一次（selfcheck 2–3 用独立 `np.unique` 重建验证）。
- 审计对象：Candidate B `a_range=0.10 m, a_bearing=0.004 rad, a_radial_velocity=0.16 m/s`；SNR `[-10,-5,0,+5,+10] dB`。

---

## 0. 人话结论（§27）

### 1. 典型车的数据尺度（train prediction cohort；val 括号）

| 量 | median | mean | P90 | P95 |
|---|---|---|---|---|
| 速度 | **13.08 m/s** (17.25) | 12.64 (14.18) | 22.44 (21.74) | 23.45 (22.88) |
| 0.1 s 位移 | **1.31 m** (1.73) | 1.27 (1.42) | 2.25 (2.18) | 2.35 (2.29) |
| 1 s 位移（10 frame = 1.001 s） | **13.06 m** (17.45) | 12.65 (14.17) | 22.47 (21.85) | 23.52 (22.94) |
| 2 s history 跨度 | **20.00 m** (31.38) | 21.46 (25.52) | 41.28 (40.90) | 43.67 (42.92) |
| 最近邻距离 | **16.77 m** (14.40) | 21.00 (19.02) | 41.35 (37.01) | 51.60 (47.75) |

补充：train 有 9.4% 状态 speed<0.5 m/s、10.9% <1 m/s（静止/刚起步占比不低）；5.1% 状态最近邻 ≤5 m，26.4% ≤10 m。2 s future 跨度 median 20.15 m，与 history 几乎相同。

### 2. Candidate B 的误差相当于什么程度（n_bs≥2 汇总，train）

| SNR | position RMSE | 占单帧位移 | velocity RMSE | 占中位速度 | 2 s 累积速度影响 |
|---|---|---|---|---|---|
| +10 dB | **0.040 m** | 3.0% | **0.083 m/s** | 0.63% | 0.8% of 2 s 跨度 |
| 0 dB | **0.125 m** | 9.5% | **0.261 m/s** | 2.0% | 2.6% |
| −10 dB | **0.395 m** | 30.1% | **0.827 m/s** | 6.3% | 8.3% |

即：+10 dB 只是轻微扰动；0 dB 已有可见误差但不主导轨迹；−10 dB 误差明显（约 0.4 m / 0.83 m/s），车辆整体位置、运动方向（方向角中位 1.87°、翻转率 0%）与交互结构（P95 位置误差仅为中位最近邻的 5%）都保留。

### 3. 当前误差偏小/合理/偏大

**Position：REASONABLE（偏轻一侧）**；**Velocity：REASONABLE（偏轻一侧）**。依据见 §12：−10 dB 位置误差 = 0.30× 单帧位移、2.3% of 2 s 跨度；速度误差 = 6.3% of 中位速度，2 s 累积 8.3% of 2 s 跨度。高低档相差 10×，物理上可辨识但不算“重”。

### 4. SNR 梯度

**REASONABLE**。n_bs≥2 口径下相邻档比值恒为 **0.5623 = 10^(−5/20)**（位置与速度都是），+10→−10 = **10.0×**，与噪声模型 `10^(−SNR/20)` 的物理预期完全一致。全量（含 1-BS）速度梯度在 +5→+10 段偏平，那是 scene 0 的 1-BS 几何尾巴（见第 5 条），不是梯度形状问题。

### 5. 3-BS 现状

- **无盲区**：prediction-history 全部唯一状态 0-BS 数 = 0（train 与 val）。`geometry_blind_states.csv` 为空；`GEOMETRY HARD ISSUE = NO`。
- ≥2 BS 覆盖：train **97.04%**（scene 0 93.96%，scene 1 100%）；val 95.84%。
- 1-BS 比例：train **2.96%**（**只出现在 scene 0，占 scene 0 的 6.04%**；scene 1 为 0）；val 4.16%（scene 0 8.08%）。
- 1-BS 速度惩罚（+10 dB）：1-BS velocity RMSE **2.263 m/s** vs 3-BS **0.067 m/s** → **33.8×**；位置 0.143 vs 0.029 m → 5.0×。且 1-BS 速度误差在所有 SNR 恒为 ~2.26 m/s：这是**几何可观测性**问题（rank=1 + 弱先验），不是通信噪声问题。它把全量 +10 dB 速度 RMSE 从 0.083 拉到 0.398。

### 6. 当前 90% / 45% 为什么不能当 accuracy

`quality = 100·exp(−sqrt((Δp/1 m)²+(Δv/1 m/s)²))` 是一个**自声明评分**：分母 1 m / 1 m/s 是人为尺子（分别约为中位单帧位移的 0.76×、中位速度的 7.6%），不来自硬件、不来自下游容忍度，也不是任何“正确率”。同一批物理误差换尺子会整体平移：−10 dB 在 (0.25,0.25) 尺子下只有 **2.6%**，在 (1,1) 下 40.0%，在 (2,2) 下 63.2%；+10 dB 对应 69.3% / 91.2% / 95.5%。详细审计见 `quality_audit.md`；第二阶段建议改用“预测时域影响”归一化候选 C（NOT FROZEN）。

---

## 1. Audit A：真实速度尺度（unique prediction-history states）

train（n=49,765）：mean 12.64 ± 7.69，P01 0.01，P05 0.06，P10 0.66，P25 6.40，**P50 13.08**，P75 19.08，P90 22.44，P95 23.45，P99 25.90，max 30.37 m/s。

| 速度区间 | train overall | scene 0 | scene 1 | val overall |
|---|---|---|---|---|
| <0.5 m/s | 9.4% | 10.4% | 8.5% | 4.6% |
| <1 m/s | 10.9% | 12.0% | 9.9% | 6.0% |
| <2 m/s | 13.2% | 14.2% | 12.2% | 8.9% |
| <5 m/s | 20.8% | 21.6% | 20.1% | 12.8% |
| <10 m/s | 39.2% | 41.9% | 36.5% | 30.8% |
| 10–20 m/s | 40.3% | 38.7% | 41.9% | 50.4% |
| ≥20 m/s | 20.4% | 19.4% | 21.5% | 18.8% |

scene 0 median 11.69 m/s，scene 1 median 14.68 m/s；val median 17.25 m/s（val 更快）。图：`plots/speed_distribution.png`。

## 2. Audit B：真实位移尺度

| 量 | split | mean | P10 | P25 | P50 | P75 | P90 | P95 |
|---|---|---|---|---|---|---|---|---|
| 单帧位移 ‖p(t+1)−p(t)‖ (m) | train | 1.266 | 0.059 | 0.643 | **1.312** | 1.910 | 2.247 | 2.349 |
| | val | 1.419 | 0.265 | 0.884 | **1.728** | 1.932 | 2.175 | 2.289 |
| 10-frame ≈1 s 位移 (m) | train | 12.65 | 0.63 | 6.42 | **13.06** | 19.12 | 22.47 | 23.52 |
| | val | 14.17 | 2.54 | 8.75 | **17.45** | 19.35 | 21.85 | 22.94 |
| 20-frame history 跨度 (m) | train | 21.46 | 0.44 | 9.20 | **20.00** | 34.71 | 41.28 | 43.67 |
| history path length (m) | train | 21.50 | 0.45 | 9.28 | 20.05 | 34.71 | 41.29 | 43.67 |
| 20-frame future 跨度 (m) | train | 21.68 | 0.44 | 9.31 | **20.15** | 34.87 | 41.50 | 43.91 |
| future path length (m) | train | 21.72 | 0.45 | 9.38 | 20.20 | 34.87 | 41.51 | 43.92 |
| 20-frame history 跨度 (m) | val | 25.52 | 4.30 | 15.14 | **31.38** | 36.44 | 40.90 | 42.92 |

真实时间：`dt = 3/29.97 = 0.1001001… s`，10 frame = **1.001 s**；history/future 各 20 frame = 2.002 s。future 只用于统计，不进入 estimator。图：`plots/single_frame_displacement_distribution.png`。

## 3. Audit C：真实速度变化尺度

| 量 | split | mean | P25 | P50 | P75 | P90 | P95 |
|---|---|---|---|---|---|---|---|
| ‖v(t+1)−v(t)‖ (m/s) | train | 0.100 | 0.034 | **0.071** | 0.146 | 0.232 | 0.283 |
| ≈1 s ‖Δv‖ (m/s) | train | 0.995 | 0.336 | **0.713** | 1.456 | 2.299 | 2.797 |
| 2 s ‖Δv‖ (m/s) | train | 1.983 | 0.617 | **1.441** | 3.060 | 4.500 | 5.417 |
| 单帧加速度 ≈‖Δv‖/dt (m/s²) | train | 0.999 | 0.337 | **0.707** | 1.462 | 2.320 | 2.823 |
| val 对应值 | val | — | — | 0.071 / 0.714 / 1.390 / 0.711 | — | — | — |

对比意义：Candidate B 的 ge2 velocity RMSE 为 **0.083（+10）/ 0.261（0）/ 0.827（−10）m/s**，即 −10 dB 的速度误差与车辆 **1 s 自身速度变化中位数（0.713 m/s）同量级**，约 2 s 速度变化中位数（1.44 m/s）的 57%；+10 dB 则远低于任何真实速度变化（0.083 ≪ 0.071×?，约为单帧速度变化中位数的 1.2 倍）。注：canonical 速度差分含 GT 视觉跟踪噪声（单帧 ‖Δv‖ P10≈0.013 m/s），上表“加速度”是含噪估计，不应解读为物理加速度分布。

## 4. Audit D：车辆交互空间尺度

基于 prediction target 所在真实物理 frame 的**全部**车辆（不是 N≤8 cohort）：

| 量 | split/scene | P05 | P10 | P25 | P50 | P75 | P90 | P95 |
|---|---|---|---|---|---|---|---|---|
| 最近邻距离 (m) | train overall | 4.99 | 6.38 | 9.62 | **16.77** | 27.56 | 41.35 | 51.60 |
| | train scene 0 | 4.49 | — | — | 14.32 | — | — | — |
| | train scene 1 | — | — | — | 19.12 | — | — | — |
| | val overall | 4.35 | 4.81 | 8.39 | 14.40 | 23.88 | 37.01 | 47.75 |

阈值占比（train）：≤5 m **5.1%**、≤10 m **26.4%**、≤20 m **59.1%**、≤30 m **78.9%**。图：`plots/nearest_neighbor_distribution.png`。

## 5. Audit E：Candidate B 误差（prediction-history unique states）

### 5.1 n_bs≥2 汇总（主口径，train / val）

| SNR | n | position MAE | position RMSE | position median | position P95 | velocity MAE | velocity RMSE | velocity median | velocity P95 |
|---|---|---|---|---|---|---|---|---|---|
| +10 | 48,293 / 4,580 | 0.030 / 0.033 | **0.040 / 0.043** | 0.024 / 0.025 | 0.071 / 0.075 | 0.062 / 0.068 | **0.083 / 0.091** | 0.043 / 0.045 | 0.147 / 0.155 |
| +5 | 48,293 / 4,580 | 0.054 / 0.059 | 0.070 / 0.077 | 0.043 / 0.045 | 0.125 / 0.133 | 0.109 / 0.120 | 0.147 / 0.162 | 0.076 / 0.080 | 0.260 / 0.272 |
| 0 | 48,293 / 4,580 | 0.095 / 0.104 | 0.125 / 0.137 | 0.077 / 0.079 | 0.220 / 0.238 | 0.195 / 0.214 | 0.261 / 0.288 | 0.137 / 0.143 | 0.457 / 0.483 |
| −5 | 48,293 / 4,580 | 0.168 / 0.184 | 0.222 / 0.243 | 0.136 / 0.140 | 0.391 / 0.425 | 0.348 / 0.383 | 0.465 / 0.512 | 0.245 / 0.256 | 0.755 / 0.805 |
| −10 | 48,293 / 4,580 | 0.300 / 0.328 | 0.395 / 0.433 | 0.243 / 0.251 | 0.696 / 0.754 | 0.620 / 0.682 | 0.827 / 0.910 | 0.438 / 0.457 | 1.320 / 1.401 |

全量（含 1-BS）train：−10 pos RMSE 0.460 / vel 0.907；0 0.146 / 0.467；+10 0.046 / **0.398**（速度被 1-BS 尾巴污染，P99 2.75 m/s）。完整 MAE/RMSE/median/P75/P90/P95/P99 见 `sensing_scale_by_snr.csv`。

### 5.2 归一化解释（§10，train；ge2 与 all 对照）

| 比值 | +10 | 0 | −10 |
|---|---|---|---|
| position RMSE / median 单帧位移 | 0.030 | 0.095 | **0.301** |
| position RMSE / median 1 s 位移 | 0.0030 | 0.0096 | 0.030 |
| position RMSE / median 2 s 跨度 | 0.0020 | 0.0063 | 0.0198 |
| position RMSE / median 最近邻 | 0.0024 | 0.0075 | 0.0236 |
| position P95 / median 单帧位移 | 0.054 | 0.168 | **0.531** |
| position P95 / median 最近邻 | 0.0042 | 0.0131 | 0.0415 |
| velocity RMSE / median speed | 0.0063 | 0.0200 | 0.0632 |
| velocity RMSE / P25 speed | 0.0129 | 0.0409 | 0.129 |
| velocity RMSE / P75 speed | 0.0043 | 0.0137 | 0.0433 |
| velocity×2 s / median 2 s 跨度 | 0.0083 | 0.0262 | 0.0828 |

（若用全量口径，+10 的 velocity/median speed = 0.030，同样被 1-BS 几何放大；低速车辆也用绝对 velocity error 单独报告，不计算速度百分比。）

### 5.3 Audit F：按速度区间（train overall，n_bs≥2 的 velocity RMSE；全速度区间 n）

| 速度区间 | n | pos RMSE +10/0/−10 | vel RMSE +10/0/−10 | vel/speed median @−10 | direction flip @−10 |
|---|---|---|---|---|---|
| 0–0.5 m/s | 4,696 | 0.031 / 0.097 / 0.306 | 0.065 / 0.204 / 0.646 | 不适用（低速用绝对误差） | 不适用 |
| 0.5–1 | 763 | 0.032 / 0.102 / 0.323 | 0.066 / 0.208 / 0.659 | 不适用 | 不适用 |
| 1–2 | 1,156 | 0.032 / 0.102 / 0.323 | 0.067 / 0.211 / 0.668 | 不适用 | 不适用 |
| 2–5 | 3,771 | 0.031 / 0.099 / 0.313 | 0.068 / 0.214 / 0.678 | 14.9% | 0.0 |
| 5–10 | 9,129 | 0.034 / 0.106 / 0.335 | 0.073 / 0.230 / 0.726 | 7.4% | 0.0 |
| 10–20 | 18,810 | 0.043 / 0.135 / 0.428 | 0.088 / 0.279 / 0.882 | 4.3% | 0.0 |
| ≥20 | 9,968 | 0.046 / 0.144 / 0.456 | 0.095 / 0.301 / 0.952 | 3.0% | 0.0 |

低速端（<2 m/s，13.2% 状态）只报绝对误差：−10 dB 时 velocity error 中位 0.53 m/s、RMSE 0.65–0.67 m/s，方向上（估计速度模长接近 0）不做百分比判断。图：`plots/velocity_error_by_speed_bin.png`。

## 6. Audit G：SNR 梯度形状

n_bs≥2 相邻档比值（train，position / velocity）：

| 档位 | position ratio | velocity ratio |
|---|---|---|
| +10→+5 | 0.56274 | 0.56246 |
| +5→0 | 0.56268 | 0.56243 |
| 0→−5 | 0.56243 | 0.56237 |
| −5→−10 | 0.56230 | 0.56233 |
| **+10→−10** | **0.10026（10.0×）** | **0.10009（10.0×）** |

- G1：position **REASONABLE**；velocity（ge2）**REASONABLE**。全量 velocity 的“平台”来自 1-BS 几何尾巴（+5→+10 ratio 0.957），不是 SNR 模型。
- G2：position 与 velocity 退化协调（both reasonable）——两者比值逐档完全一致（0.562）。
- 该梯度形状与 `sigma ∝ 10^(−SNR/20)` 的设计完全吻合；断面比值≈0.5623=10^(−5/20) 可作为第二阶段 sanity check。

## 7. Audit H：Good / Medium / Poor-but-usable 语义

判定 rubric（写入 `decision_summary.json` 的代码，数据见 §5、§10）：

- Good：pos RMSE ≤10% 单帧位移，且 vel RMSE ≤5% 中位速度，且 normal-speed 方向翻转 = 0；
- Medium：pos 5–30% 单帧位移，且 vel 1–10% 中位速度；
- Poor-but-usable：pos 20–100% 单帧位移，且 vel 3–30% 中位速度，且翻转 <1%，且方向角中位 <10°。

| 档位 | 证据 | 判定 |
|---|---|---|
| +10 dB | pos 3.0% 单帧位移；vel 0.63% 中位速度；翻转 0，方向角中位 0.19° | **PASS（Good）** |
| 0 dB | pos 9.5%；vel 2.0%；翻转 0，方向角中位 0.59° | **PASS（Medium）** |
| −10 dB | pos 30.1%；vel 6.3%；2 s 累积 8.3% of 2 s 跨度；翻转 0，方向角中位 1.87°（P95 8.9°） | **PASS（Poor-but-usable）** |

## 8. Audit I：时间稳定性（无 smoothing，逐帧独立）

normal-speed（≥2 m/s）状态：

| SNR | 帧间位移误差 median / P95 (m) | 二阶差分误差 median / P95 (m) | 速度方向 vs 真值 median / P95 (°) | 方向翻转 | 速度模长比 median | 速度异常 |
|---|---|---|---|---|---|---|
| +10 | 0.041 / 0.124 | 0.061 / 0.203 | 0.19 / 1.27 | 0 | 1.000 | 0 |
| 0 | 0.130 / 0.391 | 0.214 / 0.662 | 0.59 / 3.82 | 0 | 1.000 | 0 |
| −10 | 0.412 / 1.237 | 0.700 / 2.121 | 1.87 / 8.85 | 0 | 1.000 | 0.019% |

low-speed（<2 m/s）：+10 / 0 / −10 帧间位移误差 median 0.034 / 0.108 / 0.341 m；方向指标不适用（速度模长本身无稳定方向），不参与 flip 统计。结论：**无速度方向翻转、无速度模长异常、无位置跳变异常**；−10 dB 的帧间抖动（P95 1.24 m）是独立量测噪声的固有表现，不随时间平滑。

## 9. Audit J/K/L：几何覆盖、0-BS、1-BS 与 condition ratio

### 9.1 覆盖（unique prediction-history states）

| split | scope | n | 0 BS | 1 BS | 2 BS | 3 BS | P(≥2) |
|---|---|---|---|---|---|---|---|
| train | overall | 49,765 | 0 | 1,472 (2.96%) | 17,613 (35.4%) | 30,680 (61.7%) | **97.04%** |
| train | scene 0 | 24,352 | 0 | 1,472 (6.04%) | 7,508 | 15,372 | 93.96% |
| train | scene 1 | 25,413 | 0 | 0 | 10,105 | 15,308 | 100% |
| val | overall | 4,779 | 0 | 199 (4.16%) | 2,028 | 2,552 | 95.84% |
| val | scene 0 | 2,464 | 0 | 199 (8.08%) | 829 | 1,436 | 91.92% |
| val | scene 1 | 2,315 | 0 | 0 | 1,199 | 1,116 | 100% |

图：`plots/bs_coverage_by_scene.png`；表：`geometry_coverage.csv`。

### 9.2 0-BS 专项（§16）

**0 个 0-BS 状态**（train 与 val）。`geometry_blind_states.csv` 仅有表头；不存在连续盲区，`GEOMETRY HARD ISSUE = NO`。

### 9.3 1-BS 专项（§17）

- 占比：train 2.96%（全部在 scene 0；scene 1 = 0），val 4.16%。
- 误差对照（train）：见下表（+10 dB 最关键，因为此时剩余误差应主要来自几何而非噪声）：

| n_bs | +10 pos RMSE | +10 vel RMSE | −10 pos RMSE | −10 vel RMSE | 各档 vel RMSE |
|---|---|---|---|---|---|
| 1 BS | 0.143 m | **2.263 m/s** | 1.425 m | 2.323 m/s | 2.32 / 2.28 / 2.27 / 2.26 / 2.26 |
| 2 BS | 0.053 m | 0.105 m/s | 0.533 m | 1.046 m/s | 1.046 / 0.588 / 0.331 / 0.186 / 0.105 |
| 3 BS | 0.029 m | 0.067 m/s | 0.288 m | 0.670 m/s | 0.670 / 0.377 / 0.212 / 0.119 / 0.067 |

**结论**：1-BS 速度误差与 SNR 完全无关（≈2.26 m/s 恒定），是高 SNR 下全量速度指标的支配项（penalty 33.8×）、也是 §6 中全量速度梯度“平台”的唯一原因。它属于 geometry observability（rank=1 + 弱先验），必须在第二阶段通过 geometry 调整解决，不能靠 SNR 或 fusion 参数解决。

### 9.4 condition ratio（§18，未修改任何阈值）

| n_bs | P01 | P05 | P10 | P25 | P50 | P75 | P90 | min | <0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 2 BS | 0.050 | 0.066 | 0.086 | 0.126 | 0.182 | 0.334 | 0.606 | 0.046 | **1.04%** |
| 3 BS | 0.175 | 0.204 | 0.251 | 0.367 | 0.463 | 0.512 | 0.616 | 0.144 | 0 |

不存在“n_bs≥2 但视线几乎共线”的普遍问题：2-BS 中仅 1.04% 状态低于既有 0.05 阈值（其在 −10 dB 的误差贡献已包含在 2-BS 统计内）。图：`plots/condition_ratio_distribution.png`；表：`geometry_conditioning.csv`。

## 10. Audit L/M：quality 公式审计摘要

完整版：`quality_audit.md`；参考尺度敏感性（train，n_bs≥2 汇总 RMSE）：

| SNR | pos RMSE | vel RMSE | Q(0.25/0.25) | Q(0.5/0.5) | Q(1/1) | Q(2/2) |
|---|---|---|---|---|---|---|
| +10 | 0.0396 | 0.0828 | 69.3 | 83.2 | **91.2** | 95.5 |
| 0 | 0.1250 | 0.2615 | 31.4 | 56.0 | **74.8** | 86.5 |
| −10 | 0.3953 | 0.8269 | **2.6** | **16.0** | **40.0** | 63.2 |

（校准报告里逐状态平均的 45.1 / 76.0 / 91.4 与上表 40.0 / 74.8 / 91.2 的差异来自口径：前者是校准帧逐状态平均，后者是完整 prediction history 的汇总 RMSE 反算。）

- 不是 accuracy；1 m / 1 m/s 是人为尺度（约等于 0.76× 单帧位移、7.6% 中位速度）；换尺子百分数从 2.6% 变到 63.2%。
- 第二阶段候选：A 固定任务容忍度；B 数据运动尺度；**C（推荐讨论）预测时域影响 `s = sqrt((Δp/H)² + (Δv·T_h/H)²)`，H=20 m、T_h=2.002 s**。全部 NOT FROZEN。

## 11. Audit M：Candidate B 总体判定

| 项目 | 判定 | 关键证据 |
|---|---|---|
| Position error | **REASONABLE（偏轻）** | −10：0.395 m = 0.30× 单帧位移、0.02× 2 s 跨度；+10：0.040 m = 0.030× |
| Velocity error | **REASONABLE（偏轻）** | ge2：0.827→0.083 m/s，−10 = 6.3% 中位速度、2 s 累积 8.3%；翻转 0 |
| Gradient shape | **REASONABLE** | ge2 逐档 0.5623，+10→−10 = 10.0×，与 10^(−ΔSNR/20) 完全一致 |
| +10 dB Good | **PASS** | 3.0% / 0.63%，翻转 0，方向角 0.19° |
| 0 dB Medium | **PASS** | 9.5% / 2.0%，翻转 0，方向角 0.59° |
| −10 dB Poor-but-usable | **PASS** | 30.1% / 6.3%，翻转 0，方向角 1.87°，轨迹趋势与交互结构保留 |

**Final Audit 总判定：WARN**。Candidate B 的物理行为与语义全部可用（无 FAIL），但三项需要在第二阶段处理：① 绝对误差相对数据尺度偏轻（尤其 −10 dB 的 2 s 影响仍只有 ~8%）；② scene 0 的 1-BS 几何使高 SNR 全量速度指标失真（33.8× 惩罚）；③ quality 百分比依赖人为尺子。

## 12. 第二阶段建议方向（NO PARAMETER CHANGES MADE）

- range 通道适度增强：`a_range` 建议搜索区间 **[0.15, 0.25] m**（当前 0.10）；
- velocity 通道适度增强：`a_radial_velocity` 建议搜索区间 **[0.20, 0.30] m/s**（当前 0.16）；
- bearing 通道表现合适：维持 `a_bearing = 0.004 rad`；
- geometry：修复 scene 0 的 1-BS 区域（约 6% 状态，速度误差 2.26 m/s 恒定），可将 BS0/BS2 位置或 FOV 微调后重跑本审计；
- quality：改用候选 C（预测时域影响）或明确声明候选 A 的尺子来源；不要直接沿用 91%/45% 作为科学指标；
- 梯度：当前形状已与物理噪声模型一致；如需更强的低 SNR 退化，通过上调节 range/velocity 通道自然获得，不需要改 SNR 档位。
- **以上全部为建议方向，本轮未修改任何配置或代码数值路径，NOT FROZEN。**

## 13. 交付与验证

- selfcheck：**15/15 PASS**（dt、去重、无重复计数、vehicle_id 仅索引、Candidate B/SNR/geometry/fusion/quality 未变、samples/split SHA、frontend 未动、test 未用、确定性重建、train/val 独立）。
- 确定性：完整审计连续两次运行的全部输出文件 SHA256 一致（见末尾说明）；`decision_summary.json` 不含 wall-clock 字段。
- 数据声明：**test not used for audit-driven parameter decisions**；train 为主、val 仅确认；未生成全量 sensing cache；未修改 samples.npz / split / dataset。
- 复现：

```bash
python experiments/automatum_controlled_isac_final_audit/selfcheck.py
python experiments/automatum_controlled_isac_final_audit/run_audit.py           # ~170 s
python experiments/automatum_controlled_isac_final_audit/run_audit.py --assemble-only
```

## 14. 输出文件索引

| 内容 | 路径 |
|---|---|
| 本报告 | `reports/isac_final_audit/final_audit_report.md` |
| 尺度汇总 | `reports/isac_final_audit/prediction_cohort_scale.json` |
| 逐 SNR/逐速度区间误差 | `sensing_scale_by_snr.csv`、`sensing_scale_by_speed.csv` |
| 梯度 | `snr_gradient.csv`、`normalized_error_scale.json` |
| 几何 | `geometry_coverage.csv`、`geometry_blind_states.csv`、`geometry_conditioning.csv`、`bs_count_error_scale.csv` |
| 稳定性 | `temporal_stability.csv` |
| quality 审计 | `quality_audit.md`、`quality_pooled_ge2.json` |
| 结论 | `decision_summary.json` |
| 图 | `plots/`（9 张，含 speed / displacement / nearest-neighbor / SNR 双指标 / normalized / speed-bin / coverage / condition ratio） |

轨迹叠图沿用 Calibration V1 的 `reports/isac_calibration/automatum_controlled_isac_v1/plots/trajectories_scene{0,1}_good_medium_poor.png`（同一批车辆、同一时间窗，+10/0/−10 dB）。