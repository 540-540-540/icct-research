# SENS-SNR-DESIGN-05（rev2）：正常 SNR 工作区间重设计报告

- 分支：`sens-snr-design-05`；rev1 `ac9bc86`，本 rev2 修正核心物理判断后重新生成全部结果
- 类型：DESIGN；未修改 production sensing / config / cache；未冻结 SNR levels；未训练下游
- 证据：`energy_budget.csv`、`energy_golden_checks.json`（5/5 PASS）、`literature_matrix.csv`（12 条）

## 0. rev1 撤销声明

撤销 rev1 的 PRIMARY「`amplitude *= 1/√K` = 固定总发射功率」。该结论把三个不同的量混为一谈：

- **frequency-domain vector energy** `E_f = Σ|X[k]|² = K`（unit-modulus 全网格）；
- **time-domain average transmit power**：unitary IFFT 下 `Σ|x[n]|² = K`，但
  `mean|x[n]|² = 1`，**与 K 无关**；`E_f=K` 不能证明 Tx 平均功率随 K 增长；
- **OFDM symbol physical duration** `T_u = 1/Δf = K/B`（固定 B），因此
  `signal energy = P·T_u ∝ K`，`matched-filter gain ∝ time-bandwidth product B·T_u = K`。

K 方向的 `10log10(K)` 是**真实的相干积累增益**（TB 积），不是功率归一化错误。golden test
Test1/Test1b 严格验证（`mean|x|²=1` 与 `B·T_u=K` 对所有 K 成立）。

## 1. 能量账本（Q1，修正版）

| 量 | 表达式 | 当前值（K=256, N=256） |
|---|---|---|
| 频域向量能量 | `E_f = Σ_k |X[k]|²` | 256 / symbol |
| 时域平均功率（unitary IFFT） | `mean|x[n]|² = 1` | 1，与 K 无关 |
| OFDM useful 时长 | `T_u = 1/Δf = K/B` | 2.75 µs |
| symbol 信号能量 | `E_sym = P·T_u = P·K/B` | ∝K |
| CPI 能量 | `E_CPI = N·E_sym` | ∝K·N |
| matched-filter 输出 SNR | `SNR_link · K · N`（Hann 窗后 ×4/9） | SNR_link + 44.64 dB |

**增加 K（固定 B）**：symbol 变长（T_u∝K），TB 积增大 → 真实增益；
**增加 N**：观测时间变长（CPI∝N）→ 真实时间积分增益。两者都不是"免费"的记账问题。
（F/IFFT normalization 不改变任何 SNR 比值，只改变绝对能量单位。）

## 2. K 方向增益的物理来源（Q3 前置）

`Δf=B/K`，`T_u=1/Δf=K/B`，`B·T_u=K`。matched filter 输出 SNR = 信号能量/N0 = `P·T_u/N0 = P·K/(B·N0)`
= `SNR_link·K`（`SNR_link=P/(B·N0)` 为总带宽链路 SNR，与 K 无关）。
因此 RD 处理对每 RE 看到的"增益"中：range 维 = `2K/3`（Hann 加权后的 K，−1.76 dB 窗损），
Doppler 维 = `2N/3`，合计 **44.64 dB**；zero padding 0 dB；array 非相干求和 0 dB peak-to-floor。
Monte-Carlo 实测 44.43 dB（偏差 0.21 dB），与理论一致。

**基底论文佐证**：Wei et al.（arXiv:2308.06702）Eq.(5) `U' = (U/2)·N_c·e^{…}`，
即解调符号幅度正比于 `N_c`；Eq.(38) `SNR_2D-FFT = N_c·N_s/σ_w²`。与我们的推导完全同构。

## 3. 基底论文核查（Q4，已消除 UNKNOWN）

Wei et al., "Symbol-Level ISAC Enabled Multiple Base Stations Cooperative Sensing", IEEE TVT 2024 /
arXiv:2308.06702，TABLE II 与正文（ar5iv 全文提取）：

```text
f_c = 24 GHz;  B = 93.1 MHz;  T = 12.375 µs;  N_c = 128;  N_s = 256;
SNR sweep = -5 ~ -20 dB（Table II："SNR of the received echo signal"）;  目标速度 27 m/s;  BS 数 2~4
SNR 定义位置：解调后的元素 SNR = 1/σ_w²（元素幅度 k_w=|U'| 已含 N_c 相干增益）；
             2D-FFT 再增益 N_c·N_s（Eq.38）
归一化：论文全文无 Tx 功率/归一化声明；隐含 unit-modulus QAM；接收端除以数据符号
资源：全部 N_c×N_s RE 都参与 sensing（矩阵 B_w 为完整网格，无导频/子集选择）
```

两个要点：
1. **全网格 sensing 是基底论文的原设计**，我们的 256×256 假设并非"比论文更过分"；
   差别仅是 K=256 vs 论文 N_c=128（原因见 §5A）。
2. 论文的 SNR 轴位于**距离压缩之后**的元素上（已含 N_c 增益），其 −5…−20 dB 与我们
   `snr_ref`（原始 per-RE）不在同一测量点；映射到我们的链路，我们的 −10…+20 dB 相当于
   论文口径的 +12…+42 dB，即**高出论文工作区约 17–47 dB**。这是"当前系统看起来饱和"的
   定义层原因，而不是功率记账错误。

## 4. NIST 5GNRad 校验（Q3 直接回答）

CODE INFERENCE（repo commit e10114e）：

```text
precompute.m:134-137  SNR_dB = 10log10(P_tx/(kB*T0*systemBw*NF))     ← 链路预算 SNR
precompute.m:240      freqDomainSymbol = sqrt(prs.CombSize)*ofdmGrid  ← 稀疏 PRS 的 RE 功率补偿
precompute.m:242      timeDomainNoCP   = ifft(...)*sqrt(Nfft)         ← unitary 一致缩放
estimateChannel.m:94-115  偶/奇 comb 合并（destaggering）+ (1/sqrt(Nfft))*fft(...)
getRxWaveform.m:98-101    noisePower = 1/snrvar，AWGN 加到时域波形
```

解释：NIST 的 `sqrt(CombSize)` 把稀疏 comb 的每个 active RE 功率提升 CombSize 倍，
使**每个 PRS symbol 的总 sensing 能量与全网格 symbol 相当**；因此 NIST 的链路预算 SNR
**不意味着**我们的 per-RE SNR 需要减 `10log10(K)`（rev1 的推断错误）。
NIST 中唯一真正减少 sensing 能量的是**时间占用率**（每 slot 2–4 个 PRS symbol + periodicity）。

## 5. 资源预算重审（A–E）

| 项 | 判断 |
|---|---|
| A. K=256 全带宽 | **合理且必需**：`Δf=B/K=363.7 kHz`，无模糊距离 `c/(2Δf)=412 m ≥ 场景 376 m`；K=128 时 `Δf=727 kHz` → 无模糊距离 206 m，远距目标会混叠。基底论文 N_c=128 是因为其 Δf=1/T=80.8 kHz（无模糊 1855 m），参数集与我们不同。 |
| B. N=256 全慢时间 | **与论文一致**（N_s=256、T=12.375 µs、CPI=3.168 ms），保留。 |
| C. 资源占用 | **唯一真正过强处**：我们用 100% 时频资源 sensing；标准/NIST 用 comb + 有限 symbol。但降低占用受**速度无模糊约束**：慢时间采样 `T_eff ≤ c/(4·f_c·v_max)`；train 数据实际 max 45.1 m/s、p99 19.4 m/s，故 `T_eff ≤ 49.5 µs`（m=4）仍安全（±63 m/s），m=6（74 µs→±42 m/s）会对 45 m/s 离群目标混叠。comb 若不做 destaggering 会牺牲无模糊距离。 |
| D. CPI=3.168 ms | 与论文一致，保留。 |
| E. 44.64 dB 分解 | 22.32 dB（range：TB 积 K，含 Hann 损失）+ 22.32 dB（Doppler：CPI N，含 Hann 损失）；zero padding 0；array 非相干 0；**无虚假增益**；rev1 的"A/B/C/D/E"分解作废。 |

## 6. 候选（最多 3 个）

### Candidate A（PRIMARY）：PRS 式稀疏子载波占用（comb-4 + destaggering，无功率补偿）

- 修改内容：每 sensing symbol 只用 1/4 子载波（comb-4，NIST PRS 合法 comb 值之一），
  连续 4 个 symbol 的 4 个 comb parity 通过 destaggering 合并，重建完整 Δf 采样；
  不施加 `sqrt(CombSize)` 功率补偿（sensing 与通信共享 PA，未占用 RE 承载数据）。
- 物理依据：NIST `precompute.m:240`、`estimateChannel.m:94-115`；TR 38.901 PRS comb 选项。
- 预期 SNR response：sensing 能量 −**6.02 dB**（K 增益经 destagger 恢复，慢时间速率降为 N/4）；
  远距目标（280 m）在 −10 dB 的检测余量由 ~13 dB 降至 ~7 dB，进入 transition 边缘；
  −5…0 dB 恢复检测；高 SNR 端保持饱和。**预计是含糊安全范围内可取得的最大占用缩减**。
- 分辨率：range/Doppler resolution **不变**（B 与 CPI 不变）；无模糊距离不变（destagger 恢复 Δf）；
  无模糊速度 ±252 → **±63 m/s**（≥ 实测 max 45.1 m/s，余量 18 m/s；声明）。
- 重标定：CFAR 噪声-only 重跑（P_RD 噪声统计变化，阈值乘数需重标）；covariance LUT 重跑；回归 T2/T6/T7。
- 复杂度：中（comb 选择 + 4-parity destagger）；论文解释难度低（标准参考信号占用）。
- 风险：实现须保证 destagger 的相位一致性；慢时间速率下降会改变 Doppler 模糊语义。

### Candidate B（BACKUP-1）：A + 3GPP 车辆 RCS 对数正态起伏

- 修改内容：在 A 之上给每目标每帧 RCS 加 lognormal 波动 `σ_S=3.41 dB`；
  **标准事实修正**：TR 38.901 Rel-19 vehicle single-scatterer `10log10(σ_M)=11.25 dBsm` +
  σ_S=3.41 dB（另有角度相关方向性增益分量）。NIST `getSigmaRCS.m` 中 monostatic 实现的
  `sigmaMdBsm=-20` 是**实现值**，不是标准值，不得引用为标准。
- 预期：在 threshold 附近产生帧级 miss/coast 起伏（更接近真实雷达），均值工作点不变。
- 定位：仅当 A 的远距 transition 仍不足以产生可观察的 sensing 退化统计时启用。

### Candidate C（BACKUP-2）：SNR 轴口径对齐（**需用户审批的定义决策**）

- 修改内容：把 `snr_ref` 定义对齐基底论文的测量点（距离压缩后元素 SNR，即
  `snr_ref_perRE = snr_ref − 10log10(2K/3)` ≈ −22.3 dB 的等效重定义）。
- 预期：−10…20 dB 完整覆盖 challenging→transition→saturation（远距 −9.8/+0.2/+10.2 dB 余量于 −10/0/+10）。
- **明确标注**：这在数值上等价于横轴平移，属于"SNR 定义位置"的科研决策，不是物理修复；
  本轮不作为 PRIMARY、不自动实施。

### 明确否决

`1/√K` 幅度重归一化（本轮撤销）；固定 link loss/NF（横轴平移）；clutter/multipath（无证据必要）；
减小 K（破坏无模糊距离）；扩展到 −30/−40 dB（违反工单）。

## 7. Golden Test（`energy_golden_checks.json`，5/5 PASS）

```text
Test1  unitary IFFT mean power = 1（K=64…512）            PASS
Test1b B*T_useful = K                                     PASS
Test2  NIST sqrt(CombSize) 功率补偿保持 symbol 能量        PASS
Test2b 稀疏损耗 comb-4 = -6.02 dB / comb-2 = -3.01 dB      PASS
Test3  理论 44.64 dB vs Monte-Carlo 44.43 dB（Δ0.21 dB）   PASS
```

`energy_budget.csv` 重新生成：E0 全网格 / A comb-4 destagger / B comb-2 destagger / C=A+RCS
在 7 个 SNR × 近/中/远的 predicted detection margin。

## 8. 向下游尺度匹配（不变）

`Δp(T) ≈ Δp0 + T·Δv`，T=2 s：Δv=0.1/0.5/1 m/s → 0.2/1.0/2.0 m。
当前饱和系统给下游的状态几乎与 SNR 无关（mm–cm）；Candidate A 使远距/低 SNR 目标出现
miss/singleton/coast 与更大的状态误差，恢复可观察的因果链；不影响高 SNR 工作点。

## 9. Q1–Q10 直接回答

- **Q1**：见 §1。unitary IFFT 下 `mean|x|²=1`，`E_f=K` 不表示平均功率随 K 增长；
  `E_sym=P·K/B`、`E_CPI=N·E_sym`。
- **Q2**：`PARTIALLY CONSISTENT`。相干增益结构（N_c·N_s）与基底论文一致；
  但论文 SNR 测量点在距离压缩后的元素上（含 N_c），我们的 `snr_ref` 在原始 per-RE 上，
  两者相差约 10log10(N_c)（论文 N_c=128 → 21 dB），因此工作区整体偏高。
- **Q3**：44.64 dB = 22.32（range TB 积，Hann）+ 22.32（Doppler CPI，Hann）；
  zero padding 与阵列非相干求和为 0 dB；**没有需要扣除的"功率记账增益"**（rev1 的 24.08 dB 撤销）；
  审计"39 dB"是 conditional-on-detection 中位数，非处理增益。
- **Q4**：K=256 合理（无模糊距离要求）；N=256 与论文一致；100% 资源占用是唯一过强处，
  但其可缩减量受速度无模糊约束（最大含糊安全 −6.02 dB）。
- **Q5**：Candidate A 下 −10…0 dB 的远距目标进入 transition，−5 dB 起恢复；中/近距仍在饱和区；
  完整 −10…20 全覆盖需要 B（RCS 起伏）或 C（口径对齐，需审批）。
- **Q6**：是否必须改 K/N：**NO**（K 因无模糊距离必须 256；N 与论文一致；PRIMARY 只改资源占用与
  相干资源选择，不改网格分辨率）。
- **Q7**：RCS fluctuation：**BACKUP**（标准值 11.25 dBsm + σ_S=3.41 dB；NIST −20 dBsm 为实现值）。
- **Q8**：clutter/multipath：**NO**。
- **Q9**：PRIMARY = A（comb-4 destaggered 稀疏占用）；BACKUP-1 = A + RCS 起伏；BACKUP-2 = SNR 轴口径对齐（需审批）。
- **Q10**：见 §10。

## 10. 下一轮 REBUILD 最小改动与重跑（Q10）

```text
代码（最小）：
  frontend/sensing/simulator.py + detector 资源选择：
    1) 每 sensing symbol 仅启用 comb-4 parity 子载波（未占用 RE 不参与相干处理）；
    2) 每 4 个 symbol 的 parity 组合做 destaggering，保持 Δf 采样（NIST estimateChannel 思路）；
    3) 不施加 sqrt(CombSize) 功率补偿；
    4) K/N/B/CPI/阵列/CFAR 阈值定义/RCS/clutter 均不改。
  configs/shared_frontend.json：仅更新 sensing_resource 字段文字（comb=4, parity_span=4）
重跑：CFAR noise-only 标定（必须）→ covariance calibration（必须）→ T2/T6-A/T6-B/T7 回归 →
      新一版 −10…20 dB SNR audit（确认远距 transition 与 miss/coast 响应）
不做：正式 SNR levels 冻结、data/f01e、下游训练
```

## 11. 停止状态

已停止。未修改 production sensing/config/cache，未扩展或冻结正式 SNR levels，未训练 GNN/QGNN。
阻断项：无。等待独立验收。