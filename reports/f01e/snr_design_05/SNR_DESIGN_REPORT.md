# SENS-SNR-DESIGN-05：正常 SNR 工作区间重设计报告

- 分支：`sens-snr-design-05`（基线 `sens-snr-audit-04` @ `495af522ebaea5140067abdea4f62d68c34e59fc`）
- 类型：**DESIGN**（未修改 production sensing；未冻结正式 SNR levels；未进入 REBUILD）
- 证据：`energy_budget.csv`、`energy_golden_checks.json`、`literature_matrix.csv`、
  `code/07_shared_frontend/check_energy_budget.py`（全部 PASS），以及 SNR-AUDIT-04 rev2 实测

## 1. 能量账本（Q1）

### 1.1 OFDM symbol 频域能量

```text
E_f = Σ_{k=0}^{K-1} |X[k,n]|²
当前 |X|=1（unit-modulus QPSK） → E_f = K = 256   （每个 OFDM symbol）
```

### 1.2 三种 IFFT convention 下的时域账本（Parseval）

| IFFT convention | Σ_n |x[n]|² | average symbol power | 与频域关系 |
|---|---|---:|---:|---|
| unnormalized `Σ_k X e^{j2πkn/K}` | K·E_f = K² | K | 总能量 = K² |
| `1/√K` unitary | E_f = K | 1 | 能量守恒（Parseval 等式） |
| `1/K` (numpy default) | E_f/K = 1 | 1/K | 能量被 1/K 缩放 |

- 归一化 convention 不改变任何 SNR 比值；它只改变绝对能量账本的“单位”。
- **物理 Tx 功率**由 PA 决定，不由 FFT convention 决定。当前模型不是从 Tx 功率 + PA 出发,
  而是直接规定"每个 received RE 的 SNR"。

### 1.3 CPI 能量

```text
E_CPI = N · E_symbol
当前：E_CPI = N·K = 65536（相对 per-RE 单位能量）
```

### 1.4 关键结论：K 与 N 的物理含义不同

| 操作 | 固定 per-RE SNR（当前 E1） | 固定 OFDM-symbol 总功率（E2） |
|---|---|---|
| K 增加 | 总 symbol 能量 ∝ K（隐式增加 Tx 能量） | symbol 能量不变；per-RE SNR ∝ 1/K |
| N 增加 | CPI 能量 ∝ N | CPI 能量 ∝ N（观测时间变长，真实积分增益） |

**当前 `snr_ref` 固定时，增加 K 会隐式增加总 sensing 能量（∝K）；增加 N 则对应真实的观测时间
积分增益。** 这是本轮最重要的发现（golden test Test1/Test2/Test3 全部验证）。

## 2. Range-Doppler 处理增益推导（严格按当前实现）

实现：Range Hann → IFFT(K→2K) → Doppler Hann → FFT(N→2N) → `P_RD = Σ_a |S|²`。

```text
单维加窗相干增益：  G_w(L) = (Σ w)² / Σ w²
periodic Hann:      Σw = L/2, Σw² = 3L/8  →  G_w = 2L/3
Range 增益：        G_r = 2K/3 = 170.67  → 22.32 dB
Doppler 增益：      G_d = 2N/3 = 170.67  → 22.32 dB
RD 合计（对 per-RE SNR）：G_RD = G_r·G_d = 4KN/9 = 29127 → 44.64 dB

zero padding：K→2K、N→2N 只做插值，不提供任何 SNR gain（已由 Test4 Monte-Carlo 验证：
theory 44.64 dB vs 实测 44.43 dB，偏差 0.21 dB）。
阵列非相干求和 Σ_a|S|²：信号与噪声同比例累加，peak-to-floor 比值不变；
   阵列只服务 AoA，不贡献 detection SNR（修正 rev1 报告中"12 dB 阵列增益"的说法）。
```

**与 SNR-AUDIT-04 实测对照**：

- 审计的 `gt_rd_snr_db - received_snr_db`（单目标、GT cell、3×3 clean max / noise median）
  实测为 **44.0–45.2 dB**（按近/中/远三簇换算），与理论 44.64 dB 一致；
- 报告中曾写"约 39 dB"的量是审计的 `G_emp` **conditional-on-detection 中位数**（在 −10 dB 时 37.9 dB），
  它被以下因素压低：离格损失（Hann 主瓣）、噪声底用 3×3 median（比 mean 低 ~1.5 dB）、
  以及 detection 选择偏差。**39 dB 不是处理增益本身。**

## 3. SNR 四个口径（Q2 术语）

| 术语 | 定义 | 当前值 |
|---|---|---|
| `SNR_RE` | 每个 RE、处理前 SNR | E1: `snr_ref + 40log10(100/r)`；E2: `snr_ref − 10log10(K) + 40log10(100/r)` |
| `SNR_symbol` | 一个整 OFDM symbol 的接收信号能量/噪声能量 | 等于 `SNR_RE`（信号与噪声都随 K 等比例累加） |
| `SNR_CPI` | 整个 CPI 的能量比 | 等于 `SNR_RE`（能量比口径） |
| `SNR_RD` | RD 匹配处理后的输出 SNR | `SNR_RE + 44.64 dB`（Hann 窗，K=N=256） |

**Q2 判定：`PARTIALLY CONSISTENT`。**
- 一致处：都属"处理前参考 SNR"；NIST 5GNRad 也使用链路预算口径
  （`SNR_dB = 10log10(P_tx/(kB·T0·systemBw·NF))`，`precompute.m:132-138`），
  与 OFDM radar 经典理论（Sturm & Wiesbeck 2011：matched filter 输出 SNR = 总信号能量/噪声 PSD）一致；
- 不一致处：当前 `snr_ref` 是 **per-RE** 值，而 K=256 全资源 unit-modulus 意味着隐含"每增加一个
  子载波就增加一份发射能量"；文献/NIST 的 SNR 是**总带宽/链路预算**口径，
  子载波数不改变总功率。因此当前 `snr_ref` 与主流口径差一个 `10log10(K_active) = 24.08 dB`。

## 4. 文献矩阵要点（详见 `literature_matrix.csv`，12 条）

- **NIST 5GNRad（CODE INFERENCE）**：PRS comb-2、每 slot 2（示例）/4（默认）个 PRS symbol、
  PRS periodicity 跨 slot；噪声按链路预算 SNR 加到时域波形（方差 1/snrvar）；
  3GPP 全量路损+阴影；RCS 用 TR 38.901 表（vehicle single 基均值 −20 dBsm + 方向性增益 +
  对数正态 σ_S=3.41 dB）；检测链 2D CFAR+NMS+DBSCAN+DoA。
  **对照：NIST 用稀疏 PRS（约 1/2 子载波 × 2–4/14 symbol，≈7% 资源）；
  我们 V1 用 256×256 全资源连续 sensing。** 这放大了"资源占用"差异，但 primary 问题仍是功率口径。
- **3GPP TR 38.901 Rel-19**：RCS/path-loss 模型权威来源；`3.41 dB` 核实为 vehicle
  single/multi scatterer 的 **lognormal fluctuation σ_S**（方向性增益为独立分量，不能省略后仍称"完整标准"）。
- **Sturm & Wiesbeck 2011（基础理论）**：matched filter 输出 SNR = 能量/N0，与子载波分摊方式无关。
- **Favarelli ICC 2023**：多基站+融合中心架构依据；SNR 口径 UNKNOWN。
- **jose 2026 / MathWorks 官方示例 / arXiv 2509.08504 / arXiv 2609.00559 / IEEE 11271832 /
  IEEE 10771629 / NIST 3GPP 贡献**：作为工程与资源占用/非理想因素的旁证；关键字段未证实处标 UNKNOWN。

## 5. 三个能量预算模型（Q3, Q5）

| 模型 | per-RE 口径 | RD 输出 SNR（对 `snr_ref`） | 实测验证 |
|---|---|---|---|
| **E1 当前** | 固定 `snr_ref` | `snr_ref + 40log10(100/r) + 44.64` | 实测 44.43 dB（0 dB 点），偏差 0.21 dB |
| **E2 固定 symbol 功率** | `snr_ref − 10log10(K)` | `snr_ref + 40log10(100/r) + 20.56` | E2−E1 实测位移 = **−24.08 dB**（= −10log10(K)） |
| **E3 固定 CPI 能量** | `snr_ref − 10log10(KN)` | `snr_ref + 40log10(100/r) − 3.52` | 理论参照，不采用 |

E2 下预测的检测余量（CFAR 阈值 ≈ 13.79×0.19 = 2.62 → 4.18 dB，按近/中/远）：

| snr_ref | 40 m margin | 140 m margin | 280 m margin |
|---:|---:|---:|---:|
| -10 | +24.1 dB | +2.3 dB（临界） | −9.7 dB（漏检） |
| -5 | +29.1 | +7.3 | −4.7（漏检） |
| 0 | +34.1 | +12.3 | +0.3（临界） |
| 5 | +39.1 | +17.3 | +5.3 |
| 10 | +44.1 | +22.3 | +10.3 |
| 20 | +54.1 | +32.3 | +20.3 |

**预计 E2 下 −10–20 dB 自然覆盖 challenging（远距/低端）→ transition（0–10 dB）→ nominal/saturation
（≥10 dB）**，且不需要改 K/N、阵列、CFAR、RCS 或 SNR 区间。

## 6. Q3：当前增益的分解

```text
A. 合理 processing gain：N=256 的 CPI 相干积分（Hann 加权）          = +22.32 dB
B. 由于资源功率口径导致的额外 gain：per-RE SNR 未按 1/K 归一化，
   即总发射能量随 K 线性增长（K=256）                                = +24.08 dB（应扣除）
C. 更长 CPI 的真实 energy gain（N）                                  已含在 A
D. zero padding：0 dB（仅插值，Test4 验证）
E. array 非相干求和：0 dB peak-to-floor（只服务 AoA）
   合计 per-RE→RD：44.64 dB（其中 24.08 dB 属口径问题）
```

## 7. K/N 判定（Q4, Q6）

- **K=256（全子载波、unit-modulus）**：对"固定总功率"的物理系统而言资源过强 —— 它隐含 24 dB 的
  额外能量；文献/NIST 的稀疏 PRS 或资源分配结果说明 sensing 资源占用应计入预算。
- **N=256（CPI=3.168 ms）**：其增益是真实的观测时间积分（22.32 dB），不算过强；且
  Doppler 分辨 1.97 m/s 对车辆场景并不奢侈。
- **Q6 是否需要改 K/N：`CONDITIONAL`** ——
  若采用 E2 功率口径（PRIMARY），**不需要改 K/N**；
  若 E2 后 transition 仍不足（预计不会），再用资源占用（稀疏 PRS 式）作二级旋钮。

## 8. 候选方案（最多 3 个）

### Candidate A（PRIMARY）：固定 OFDM-symbol 总功率的幅度归一化

- 修改内容：simulator 目标幅度 `α = 10^{snr_ref/20}·(R0/r)²·(1/√K_active)`；
  或等价地把 `snr_ref` 定义为总带宽接收 SNR（`SNR_RE = snr_ref − 10log10(K)`）。
  不改 K/N/阵列/CFAR 阈值定义/RCS/clutter。
- 物理依据：固定 PA 总功率下，子载波数不改变总 received energy；matched filter 输出 SNR = 总能量/N0
  （Sturm & Wiesbeck 2011；NIST 链路预算口径 precompute.m:132-138）。
- 预期 SNR response：RD 输出整体下移 24.08 dB，−10–20 dB 覆盖 challenging→transition→saturation（§5 表）。
- 分辨率/模糊：不变（K/N 未动）。
- 需重标定：CFAR 噪声路径不变（0.19 可复用，需 verify）；**covariance LUT 必须重跑**（q 分布改变）；
  T2/T6/T7 重跑。
- 复杂度：simulator 1 行 + 文档/注释；论文解释难度低（"link-budget SNR with fixed Tx power"）。
- 风险：低。唯一风险是用户审阅后认为"改变输入幅度尺度"需要同步修改 config 语义描述。

### Candidate B（BACKUP-1）：Candidate A + 3GPP RCS 对数正态起伏（最小简化）

- 修改内容：在 A 基础上给每目标每帧 RCS 加 lognormal 波动 `σ_S=3.41 dB`（TR 38.901 vehicle），
  均值固定；简化省略方向性增益并在报告标注。
- 物理/文献依据：TR 38.901 Rel-19 / NIST `getSigmaRCS.m`（CODE INFERENCE）。
- 预期：在 transition 区产生帧间起伏（更真实的 miss/coast 统计），不改变平均工作点。
- 复杂度低；风险：被质疑"未完整实现方向性 RCS"——需明确标注为简化。

### Candidate C（BACKUP-2）：Candidate A + 现实 sensing 资源占用

- 修改内容：在 A 基础上改为 PRS 式稀疏资源（例如 comb-2 子载波或部分 sensing symbol），
  保持 B 不变时需重查 df/无模糊距离；或减少 N。
- 依据：NIST 稀疏 PRS（≈7% 资源）；IEEE 11271832 资源分配。
- 复杂度中高、改变分辨率/CPI，论文解释最重；**仅在 A 不足时使用**。

### 明确否决

- 固定 link-budget loss（实现损失/NF）：本质是 `SNR' = SNR − C` 的横轴平移，不解决口径问题；
- clutter/multipath/shadowing：本轮无证据表明必要，先不上；
- 提高 CFAR 阈值制造 miss、把正式 SNR 扩到 −40、按下游 ADE 反调：全部违反工单。

## 9. 向下游尺度的匹配（Q7 相关，仅尺度分析）

误差传播 `Δp(T) ≈ Δp0 + T·Δv`，T=2 s：

```text
Δv=0.1 m/s → 0.2 m    Δv=0.5 m/s → 1.0 m    Δv=1 m/s → 2.0 m
Δp0=1 cm /10 cm /50 cm 在 2 s 尺度可忽略；米级需要 Δv ~ 0.5 m/s
```

当前饱和系统下游看到的状态几乎与 SNR 无关（mm–cm 精度）→ 下游不可能观察到 sensing 退化。
采用 A 后，低 SNR 端远距目标将出现 miss/singleton/coast 与更大状态误差（由 RD margin 表推断可达
数米/1–2 m/s 量级），足以对 2 s 预测产生可观察影响。**这不是按 ADE 调参，而是恢复物理链路。**

## 10. 十个问题的直接回答

- **Q1**：见 §1（E_f=K，E_CPI=NK；K 增加隐式增加能量，N 增加是真实时间积分；三种 IFFT convention 不改变 SNR）。
- **Q2**：`PARTIALLY CONSISTENT`（§3；与 NIST/经典理论差 10log10(K)=24.08 dB 的归一化口径）。
- **Q3**：44.64 dB 中，+22.32 dB 为 N 的真实 CPI 积分（含 Hann 损失后），+24.08 dB 为 K 的口径增益（应扣除），
  zero padding 与阵列非相干求和均为 0 dB；审计"39 dB"为 conditional 中位数而非处理增益（§2、§6）。
- **Q4**：K=256 全资源在固定功率口径下过强；N=256 合理（§7）。
- **Q5**：E2 下 −10–20 dB 覆盖 challenging→transition→saturation（§5 表）。
- **Q6**：`CONDITIONAL`（A 方案下无需改 K/N）。
- **Q7**：RCS fluctuation 回答 `BACKUP`（A 已预计足够；用于平滑 transition 的帧间统计）。
- **Q8**：clutter/multipath 回答 `NO`（本轮无必要；会显著增加复杂度与论文解释负担）。
- **Q9**：PRIMARY = A；BACKUP-1 = A+B；BACKUP-2 = A+C（§8）。
- **Q10**：见 §11。

## 11. 下一轮 REBUILD 最小改动与重跑清单（Q10）

```text
代码（仅 1 个最小改动）：
  frontend/sensing/simulator.py
    amplitude *= 1/sqrt(K_active)（或按总带宽 SNR 重定义；K_active = K = 256）
  configs/shared_frontend.json
    更新 power/snr_definition 字段文字（语义：reference total-band SNR at 100 m），
    K/N/array/CFAR 数值一律不动
必须重跑（不改算法）：
  1) CFAR noise-only verify（预期 0.19 不变，仅确认噪声路径未受影响）
  2) covariance calibration（LUT 的 q 区间会整体下移，必须重标定）
  3) P3/P4/T2/T6-A/T6-B/T7 回归
  4) 新一版 SNR audit（-10…20 dB，确认 challenging/transition/saturation 出现）
不做：正式 SNR levels 冻结（仍 null）、data/f01e、下游训练
```

## 12. 停止状态

已停止：未修改正式 sensing、未改正式 config、未降 SNR 区间、未冻结 SNR levels、未生成 data/f01e、
未训练 GNN/QGNN。阻断项：无。等待独立验收。