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
FROZEN：**采用 Candidate A**。每 BS `A=16` 元 ULA（λ/2），零填充 FFT 64 点扫描
（Bartlett/beamspace FFT，逐峰取最大），不使用 MUSIC。
理由：每 BS 直接得到局部 (x,y)，跨站关联退化为空间门控；NIST 与旧码均有可复用实现；
3 BS × ≤32 检测规模下 A 的总代价低于 B（三站 range 组合关联）。
替代被否：Candidate B（仅 range/Doppler + 三站几何联合定位）——匿名多目标组合爆炸、
无 AoA 时单站无法形成局部点、关联与虚警控制更难。V1.1 可加 MUSIC/更大阵列。

**D03 波形**
FROZEN：沿用 A03 冻结常量 `fc=24 GHz, B=93.1 MHz, K=N=256, T=12.375 µs`；
不模拟 CP/ICI；`T` 与 `df` 独立声明为建模简化。
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

**D07 检测器**
FROZEN：3D（range, doppler, angle）CA-CFAR（边缘自适应）+ NMS + log 抛物线插值；
检测数即目标数来源；候选上限 32/BS。**DBSCAN V1 不启用**；触发条件：若 T4/T5 显示
单目标平均产生 >1.5 个 NMS 后检测（旁瓣/扩散），则在 V1.1 启用 DBSCAN 聚类。
理由：CA-CFAR/NMS 是审计推荐的最小实现；旧码与 NIST 都有现成算法结构。
替代被否：学习式检测、2D 投影 CFAR + 逐峰角度（保留为可选，不默认）。

**D08 CFAR 虚警标定**
FROZEN：按"目标虚警数/BS/帧"反算 cell 级 `Pfa_cell = target_false_alarms / N_cells`
（默认目标 0.5），用**噪声-only** CPI 标定阈乘数；禁止用 GT 标定。
理由：`N_cells≈1.68e7`，直接设 cell Pfa 会得到上千虚警。

**D09 跨 BS 关联**
FROZEN：检测→全局 (x,y)+协方差；两两 Mahalanobis 门控 `χ²(2,0.99)=9.21`；
连通分量内做"最大基数、最小总代价"一对一匹配；≥2 站匹配→逆协方差加权融合
（`C=(ΣC⁻¹)⁻¹`）；落单检测保留为单站观测。不设参考传感器，不用 PHD/MHT/JPDA。
理由：规模小、可枚举、可测试；门限来自测量协方差而非 GT。
替代被否：JPDA/MHT（超 V1 范围）、单纯 Hungarian 跨三站（冲突消解复杂）。

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

**D18 种子与配对**
FROZEN：`seed(aspect) = SHA256("2026:{episode}:{frame}:{slot}:{aspect}")`，
`aspect ∈ {waveform, phase, noise}`；同一目标跨 SNR 复用同组种子（配对比较）。
与旧 `...:101` 单一流不同，但新旧缓存不要求逐样本一致。

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

---

## GitHub 回传速查（对应工单 §17）

- AoA V1：**采用**（16 元 ULA，FFT/Bartlett 扫描）— D02
- DBSCAN V1：**不启用**（触发条件已定义）— D07
- target model：**单主散射点，固定 RCS=10 m²，随机相位** — D22
- cross-BS association：**Mahalanobis 门控 + 分量匹配 + 逆协方差融合** — D09
- velocity source：**融合位置序列 → 恒速卡尔曼滤波** — D11
- SNR/power definition：**雷达方程简化（1/r⁴ × 固定 RCS），`snr_ref`@100 m，每 BS 单噪声底** — D05
- track slot policy：**确认时分配、死亡冷却回收、非真值评分淘汰超 8** — D13
- GT label matching policy：**离线逐 origin 历史平均距离 + Hungarian，≥3 公共帧、5 m 门控** — D15