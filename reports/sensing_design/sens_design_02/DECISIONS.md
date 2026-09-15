# DECISIONS.md — SENS-DESIGN-02 冻结决定

每条决定状态固定为 **FROZEN**（本轮冻结，REBUILD-03 按此执行）。
"改动须理由"：若 REBUILD-03 需要偏离，必须在本文件新增一条带证据的修订记录，而不是静默修改。

---

**D01 Shared echo 组织**
FROZEN：每个 BS 每帧一份共享观测 `Y_b:[A,K,N]`；先对所有目标求和，再加**一份**接收机噪声；
目标维不得出现在观测张量中。不同 BS 各自一份观测与噪声。
理由：审计判定 A/B/C 的直接根因是逐目标独立回波。
替代被否：逐目标信号（现状）、共享求和但保留每目标噪声（物理不一致）。

**D02 AoA V1**
FROZEN：**采用 Candidate A**。每 BS `A=16` 元 ULA（λ/2）；**只对 2D CFAR/NMS 后的 RD 峰**
做零填充 64 点 FFT/Bartlett AoA 扫描（逐峰取最大），不使用 MUSIC；
`aoa_max_peaks` 预留多峰接口，V1 冻结为 1（最强 AoA 峰，T5 记录该限制）。
理由：每 BS 直接得到局部 (x,y)，跨站关联退化为空间门控；NIST 与旧码均有可复用实现；
3 BS × ≤32 检测规模下 A 的总代价低于 B（三站 range 组合关联）。
替代被否：Candidate B（仅 range/Doppler + 三站几何联合定位）——匿名多目标组合爆炸、
无 AoA 时单站无法形成局部点、关联与虚警控制更难。V1.1 可加 MUSIC/更大阵列/多 AoA 峰。

**D03 波形**
FROZEN：沿用 A03 冻结常量 `fc=24 GHz, B=93.1 MHz, K=N=256, T=12.375 µs`；
不模拟 CP/ICI。
**T 语义（P1 澄清）**：`1/df = 2.75 µs` 是子载波间隔对应的 useful-time 尺度；
`T` 是 slow-time 采样间隔 / PRI，**不是** "OFDM useful symbol duration"；
模型为可分离简化 OFDM sensing，不声称复刻 5G NR numerology。
理由：距离分辨率 1.61 m、无模糊距离 412 m（> 场景 376 m）、无模糊速度 252 m/s（> 57 m/s）
全部满足；换波形会改变所有几何/边界与缓存语义，收益不足。
替代被否：5G NR PRS 标准链（NIST 风格）、30 GHz/50 MHz 等新参数。

**D04 Tx/Rx 阵列**
FROZEN：Tx 单全向单元；Rx 每站 16 元 ULA（V1 不做 Tx 波束成形与 Tx 增益图）。
理由：AoA 只需 Rx；Tx 增益会引入额外面向性建模但没有必要。

**D05 功率与 SNR 定义**
FROZEN：`α = 10^(snr_ref/20)·(R0/r)²·sqrt(σ/σ0)`；`R0=100 m`、`σ0=σ=10 m²`（V1 固定 RCS）；
每 BS 每帧一份 `CN(0,1)` 噪声（逐 entry 方差 1）；控制变量是 `snr_ref`。
禁止逐目标 SNR 注入估计/检测/跟踪任何阶段。TR 38.901 方向性/起伏 RCS 留 V1.1。
理由：两行公式即可产生真实的 per-target/per-BS 功率差异；可校准、可解释。
替代被否：每目标同 SNR（旧错误）、完整链路预算（无增益价值）。

**D06 可见性与杂波**
FROZEN：几何硬门控 `10 ≤ r ≤ 300 m`、`|θ| ≤ 70°`（沿用 A01/旧几何）；无静态杂波、
无多径、无阴影；不做均值去杂波。
理由：最小模型；本场景无杂波真值，硬造杂波反而不可信。

**D07 检测器（P0-1 修订）**
FROZEN：**2D Range-Doppler CA-CFAR**（边缘自适应）→ 2D NMS → RD 抛物线插值 →
**逐 RD 峰 64 点 AoA**（1D NMS + 抛物线插值，`aoa_max_peaks=1`）；
检测数即目标数来源；候选上限 32/BS。主路径不再构造全尺寸 3D 功率立方（约 1.68e7 单元），
AoA 只作用于少量 RD 峰。**DBSCAN V1 不启用**；触发条件：若 T4/T5 显示
单目标平均产生 >1.5 个 NMS 后检测（旁瓣/扩散），则在 V1.1 启用 DBSCAN 聚类。
理由：与 NIST 5GNRad 主链一致（2D RD CFAR + peak-wise AoA），成本低约一个量级，
仍保留真实检测与 AoA。
替代被否：全尺寸 3D CFAR（成本不成比例）、学习式检测、每 RD 峰多 AoA（V1 先限制，T5 声明）。

**D08 CFAR 虚警标定**
FROZEN：按"目标虚警数/BS/帧"反算 cell 级 `Pfa_cell = target_false_alarms / N_cells`
（默认目标 0.5），用**噪声-only** CPI 标定阈乘数；禁止用 GT 标定。
`N_cells = Nr·Nv = 512·512 = 262 144`（2D RD map；P0-1 后不再使用 1.68e7 的 3D 单元数）。
理由：直接设 cell Pfa 会得到大量虚警；2D 后单元数下降 64 倍。

**D09 跨 BS 关联（P0-5 修订：顺序 Hungarian grouping）**
FROZEN：检测按 `(peak_power 降序, grid_index 升序)` 确定性排序；固定传感器顺序 `[0,1,2]`：
Stage 1 BS0↔BS1 Hungarian（Mahalanobis² + χ²(2,0.99)=9.21 门控，出格 BIG）；
未匹配者成为 singleton group；group state = 逆协方差融合 `(p_g,C_g)`；
Stage 2 groups↔BS2 再 Hungarian；最终每组每 BS 至多 1 检测；
`n_bs≥2` → `C_f=(ΣC⁻¹)⁻¹`，`p_f=C_fΣC⁻¹p`；`n_bs=1` 保留单站观测。
匹配器只用 `scipy.optimize.linear_sum_assignment`；不用 PHD/MHT/JPDA；不用 GT。
每次 Hungarian 后必须显式 post-filter：逐对复核 `Mahalanobis² ≤ 9.21`，
BIG/超门限 assignment 一律恢复为 unmatched（回到 singleton），Stage 1/Stage 2 都要执行。
理由：唯一、可直接编码、确定性；固定顺序只依赖站点几何与观测排序（BS0/BS2 同侧视差弱，
先配异侧 BS1 最稳）。
已知局限：Stage 1 局部最优不可回溯（T4/T5 量化；V1.1 可全三元枚举）。
替代被否：连通分量+枚举（实现细节仍需临场决定）、JPDA/MHT（超 V1 范围）。

**D10 单站观测策略**
FROZEN：单站观测可用于**更新已有航迹**；但**不能单独确认新航迹**（D12）。
理由：A01 统计 ≥2 BS 覆盖率 99.95%，限制代价很小；可压制单站虚警形成假航迹。

**D11 速度来源**
FROZEN：融合位置序列 → 恒速卡尔曼滤波给出 `[vx,vy]`；V1 不融合径向多普勒
（径向速度保留在 detection payload 供诊断）。
V1.1 触发：若 T3/T6 显示速度 RMSE 明显不达标，再启用
`(vr1,vr2,vr3)→2D WLS` 或作为 KF 附加量测（单站径向速度量测）。
理由：KF 简单、可解释、可测；多普勒分辨率 1.97 m/s 本身较粗。

**D12 航迹状态机**
FROZEN：birth→tentative；最近 3 帧 ≥2 次更新且有 ≥1 次 `n_bs≥2` → confirmed；
tentative 3 帧未确认即删除；coast 最多 5 帧（`detected=0`，`track_exists=1`）；
misses>5 删除；只输出 confirmed alive 航迹。
**T6-B 消融例外（P0-4）**：仅 1/2/3-BS 公平比较的诊断模式允许 `confirm_requires_nbs2=false`
（3 帧 ≥2 hits 即可确认）；正式 mainline 恒为 true，报告须标注 ablation mode。
替代被否：立即确认、无 coast 直接删除。

**D13 8 槽策略**
FROZEN：slot 在**确认时**分配（最小空闲号）；删除后冷却 10 帧再回收；
confirmed>8 时按 `(misses↑, trace(P_xy)↑, age↓, track_key↑)` 保留 8 个（全部为非真值信息）。
理由：接口固定 8 槽；车辆会进出生灭，必须回收；排槽不得用 GT。

**D14 `detected` / `track_exists` 新语义**
FROZEN：`detected=1 ⇔ 本帧获得真实量测更新`；`track_exists=1 ⇔ 航迹 alive`；
允许 `track_exists=1, detected=0`；tentative 两项均 0。

**D15 监督标签**
FROZEN：感知与打包完成并封存后，按 prediction origin 做离线 track↔GT 匹配：
cost=历史公共帧平均位置距离，要求 ≥3 公共帧；Hungarian；原点距离 >5 m 视为不匹配；
匹配上则 label=该车未来 20 帧位置，`label_valid` 取 GT 存在性；未匹配 slot →
`label_valid` 全 False；输入、track_key、slot、detector 输出一律不可回写。
替代被否：在线匹配（会引入 GT 反馈）、按 A01 slot 直接继承（L3 泄漏）。

**D16 三域隔离**
FROZEN：见 `GT_ISOLATION_SPEC.md`；B 域不得 import A 域；C 域在 B 封存后运行。
强制：AST/签名/import 哨兵/哈希复核（T2 + pack 前后 hash）。

**D17 计算与落盘**
FROZEN：仿真/检测/融合 float64（CUDA）；缓存 float32/bool；
原始 Y 不落盘（仅少量诊断样本）；流式逐帧处理；序列/诊断落 `data/f01e`。

**D18 种子合同（P0-2 修订：三类独立）**
FROZEN：
`seed_waveform = SHA256("2026:{episode}:{frame}:{bs}:waveform")`（每 BS 每帧一份共享波形）；
`seed_noise = SHA256("2026:{episode}:{frame}:{bs}:noise")`（每 BS 每帧一份共享噪声）；
`seed_phase = SHA256("2026:{episode}:{frame}:{bs}:{source_key}:phase")`（逐目标散射相位）。
waveform/noise **不含** target slot 或 source_key；`source_key` 仅 A 域 simulator 内部；
同一车辆跨 SNR 复用同一 phase 种子（配对比较）。与旧 `...:101` 单一流不同，
新旧缓存不要求逐样本一致。

**D19 数据路径与兼容**
FROZEN：新 config `configs/shared_frontend.json`（`revision=B01-shared-3bs-v1`）；
新缓存 `data/f01e`；`data/f01d` 与旧 config/入口**冻结保留**为 oracle baseline；
下游四键契约、0.1 s 网格、20 帧窗口、A01 origins 不变；normalization 新 train 重拟合。
旧 `scripts/run_f01d_gpu.py` 不再修改，只用于复现历史。

**D20 SNR 档位**
FROZEN（流程）：**不继承 `[5,10,15,20]`，不预冻结**。REBUILD-03 先宽范围 sweep
（`snr_ref_db = -10…40`），给出 clean/nominal/challenging 候选与全部质量曲线；
正式档位由用户在评审后决定（T3 硬停点）。禁止反向调 sensing 迎合下游。

**D21 BS 间干扰**
FROZEN：V1 假设每 BS 单基地自观测、无跨 BS 干扰与自干扰残留；
（V1.1 如需可加简单的 BS 间泄漏模型。）

**D22 目标模型**
FROZEN：每辆车 V1 为**单主散射点**（固定 RCS=10 m²，逐 (BS,目标,帧) 随机相位）；
扩展目标（多散点/长度）留 V1.1+。理由：最小可信、可解释；近邻合并作为已知局限由 T5 量化。

**D23 测量协方差 `C_xy` 标定 LUT（P0-3 新增）**
FROZEN：`C_xy` 由离线 calibration LUT 产生：单目标合成扫 range/angle/snr_ref，
按 detector 可观察量 `q = peak_to_noise_db` 分箱，取残差 MAD 稳健 σ：
`sigma_r(q), sigma_u(q)`；`R_ru = diag(σ_r², σ_u²)`；
`C_xy = J·R_ru·Jᵀ + εI`，`ε = 0.01 m²`；Jacobian 见 `SYSTEM_MODEL.md` §5.1（`coords.py` 实现并单测）。
B 域 inference 只用 q 查表，禁止 GT/SNR truth；样本不足时取相邻箱上包络常数并在报告声明。
标定必须在 association 参数冻结前完成（`REBUILD_03_WORKPLAN.md` P2）。
理由：关联/融合/KF 的 R 必须有可追溯、可复现来源；这是实现阻断项。

---

## 回传速查（SENS-DESIGN-02-R1）

- detector path：**2D RD CA-CFAR + 2D NMS + 逐峰 64 点 AoA（V1 最强单峰）** — D07
- seed contract：**waveform/noise 按 (episode,frame,bs)；phase 按 (episode,frame,bs,source_key)** — D18
- covariance calibration：**离线 σ_r(q)/σ_u(q) LUT（按 observed peak_to_noise_db 分箱）+ Jacobian 传播** — D23
- T6 ablation rule：**T6-A 融合前消融；T6-B 三档统一 `confirm_requires_nbs2=false`（仅诊断）** — D12 + VALIDATION_PLAN
- 3-BS association：**顺序 Hungarian grouping：BS0↔BS1 → groups↔BS2，逆协方差融合** — D09
- OFDM T semantic：**`1/df` 为 useful-time 尺度；`T` 为 slow-time/PRI；可分离简化模型，不声称 5G NR numerology** — D03
- AoA V1：**采用**（16 元 ULA，逐 RD 峰 FFT/Bartlett）— D02
- DBSCAN V1：**不启用**（触发条件已定义）— D07
- target model：**单主散射点，固定 RCS=10 m²，随机相位** — D22
- velocity source：**融合位置序列 → 恒速卡尔曼滤波** — D11
- SNR/power definition：**雷达方程简化（1/r⁴ × 固定 RCS），`snr_ref`@100 m，每 BS 单噪声底** — D05
- track slot policy：**确认时分配、死亡冷却回收、非真值评分淘汰超 8** — D13
- GT label matching policy：**离线逐 origin 历史平均距离 + Hungarian，≥3 公共帧、5 m 门控** — D15