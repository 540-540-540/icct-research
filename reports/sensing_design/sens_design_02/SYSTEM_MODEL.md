# SYSTEM_MODEL.md — 三基站多目标 ISAC 前端数学模型（V1 冻结草案）

所属：SENS-DESIGN-02。基线：`sens-audit-01` @ `0cac17ba21c32e5c28548d7bc7f391f9e0345760`。
本文只做设计，不实现；所有"冻结"条目最终汇总在 `DECISIONS.md`。

## 1. 符号与坐标系

| 符号 | 含义 | 单位 |
|---|---|---|
| `b ∈ {0,1,2}` | 基站（BS）编号；0 左、1 右、2 左上（A01 几何） | — |
| `i` | 目标（车辆）编号；仅模拟器侧存在 | — |
| `t` | 帧序号；`Δt = 0.1 s` | — |
| `a ∈ [0,A)` | Rx 均匀线阵（ULA）单元 | — |
| `k ∈ [0,K)` | 子载波 | — |
| `n ∈ [0,N)` | 慢时间 OFDM 符号 | — |
| `x_b,y_b` | BS 平面坐标 | m |
| `p_i=(x_i,y_i)` | 车辆平面位置（2D 公共平面，BS 高 6 m、车高 1 m） | m |
| `v_i=(vx_i,vy_i)` | 车辆速度 | m/s |
| `r_{b,i}` | 斜距 `sqrt(|p_i-s_b|²+h²)`，`h=5 m` | m |
| `u_{b,i}` | 方向余弦 `sin(θ)`，θ 为相对 BS 视轴的方位角 | — |
| `ṙ_{b,i}` | 径向速度，正方向指向 BS | m/s |

坐标框架：与 A01 一致（本地笛卡尔米制，`reports/f01a/geometry.json`）。
BS 坐标 `[(-64.234,-106.907),(114.234,0),(-64.234,106.907)]`，视轴 `[0°,180°,0°]`。
平面 2D 模型（无俯仰向检测）；高度差只进入斜距。

波形冻结常量（沿用 `configs/symbol_frontend.json`，`revision=A03-symbol-v4-cuda-stable-solver`）：

| 量 | 值 | 派生 |
|---|---:|---|
| `fc` | 24 GHz | λ = 12.49 mm |
| `B` | 93.1 MHz | 距离分辨率 `c/2B` = 1.610 m |
| `K` | 256 | `df = B/K` = 363.672 kHz；无模糊距离 `c/2df` = 412.17 m |
| `N` | 256 | 速度分辨率 `c/(2fc·N·T)` = 1.971 m/s |
| `T` | 12.375 µs | slow-time / PRI 采样间隔：无模糊速度 `c/(4fc·T)` = 252.35 m/s；CPI = `N·T` = 3.168 ms |
| `c` | 299 792 458 m/s | — |

**时间参数语义（P1 澄清）**：`1/df = K/B = 2.75 µs` 是子载波间隔对应的 useful-time 尺度；
`T` 在本项目中**不是** "OFDM useful symbol duration"，而是 **slow-time 采样间隔 / PRI**
（相邻慢时间符号之间的等效重复间隔）。本项目使用**可分离的简化 OFDM sensing 模型**
（range 相位只用 `df`，Doppler 相位只用 `fc` 与 `T`），不声称严格复刻 5G NR numerology，
也不模拟 CP/ICI。

场景最大角点距离 376.4 m < 无模糊距离 412.2 m；目标速度 ≤56.6 m/s < 无模糊速度。

Rx 阵列（V1 新增）：每 BS 一个 `A=16` 元 ULA，间距 `λ/2`，沿站点局部视轴横向摆放；
扫描 FFT 长度 64（零填充），角栅格 `Δu = 2/64 = 0.03125`；物理半功率波束宽度
`0.886·2/A ≈ 6.35°`。Tx 在 V1 为单全向单元（不做发射波束成形，见 `DECISIONS.md` D04）。

## 2. Shared multi-target echo（每 BS 一份共享观测）

一个 BS、一帧，收到全部可视车辆回波的**叠加**，再叠加**一份**接收机噪声：

```text
Y_{b,t}[a,k,n] = Σ_i  α_{b,i,t} · e^{jφ_{b,i,t}} · D(r_{b,i,t},k) · V(ṙ_{b,i,t},n) · U(u_{b,i,t},a) · X_b[k,n]
                 + W_{b,t}[a,k,n]
```

其中：

```text
D(r,k)   = exp(-j·4π·df/c · r · k)                 距离相位（k 向）
V(ṙ,n)   = exp(+j·4π·fc·T/c · ṙ · n)               多普勒相位（n 向，正 ṙ 指向 BS）
U(u,a)   = exp(+j·π·a·u)                           阵列相位（a 向，λ/2 ULA）
α_{b,i,t}= 10^(snr_ref/20) · (R0/r_{b,i,t})² · sqrt(σ_i/σ0)   幅度（见 §3）
φ_{b,i,t}~ U(-π,π)                                  每 (BS,目标,帧) 独立散射相位
W_{b,t}  ~ CN(0,1)（逐 entry，E|W|²=1）              每 BS 每帧一份共享噪声
X_b[k,n] = exp(j(π/4 + π/2·q))，q∈{0,1,2,3}          该 BS 自己的单位模 QPSK 网格
```

要点：

1. **目标维求和发生在回波生成内部**，`Y` 中没有目标维。
2. `X_b` 对同一 BS 的所有目标**共享**（同一份发射波形）；不同 BS 可各自不同。
3. `W` 只在求和之后加一次；拒绝"每目标独立噪声"。
4. 目标参数在 CPI 内冻结（快照模型，沿用 A03 时域约定：deadline 快照 + ≤100 ms 因果保持）。
5. 可视性硬门控（V1 声明）：仅当 `10 ≤ r ≤ 300 m` 且 `|θ| ≤ 70°` 时 `α>0`，否则该 BS 看不到该目标
   （沿用旧 `ofdm_echo`/`station_geometry` 的几何可见性，不是检测能力声明）。
6. V1 不建模：BS 间互干扰、多径、阴影衰落、Tx/Rx 主瓣增益（除阵列接收响应本身）、
   近场、双基地。

### 2.1 随机种子合同（P0-2 修订，target slot 完全解耦）

```text
seed_waveform = SHA256("2026:{episode}:{frame}:{bs}:waveform")            # X_b：每 BS 每帧一份
seed_noise    = SHA256("2026:{episode}:{frame}:{bs}:noise")               # W_b：每 BS 每帧一份
seed_phase    = SHA256("2026:{episode}:{frame}:{bs}:{source_key}:phase")  # 仅散射相位 target-specific
```

- waveform / receiver noise 种子**不含** target slot 或 source_key：
  目标顺序置换、输出槽位置换都不改变 `X_b`、`W_b`（逐比特不变）。
- phase 种子使用稳定 A01 `source_key`：车辆间相位独立；同一车辆跨 SNR 复用同一相位（配对比较）。
- `source_key` 只存在于 A 域 simulator 内部；不进入信道模型输出，也绝不进入 B 域任何模块。
- 实现约定：`synthesize_shared` 内部派生三类种子（见 `INTERFACE_SPEC.md` §3.1）。

## 3. Received power / SNR 定义（取代"每目标同 SNR"）

单基地雷达方程简化为两级：

```text
P_r,i / P_ref = (σ_i/σ0) · (R0/r_i)^4          （功率）
α_i           = sqrt(P_r,i/P_noise)            （幅度，相对单位噪声）
              = 10^(snr_ref/20) · (R0/r_i)² · sqrt(σ_i/σ0)
```

冻结参数：

| 参数 | V1 值 | 说明 |
|---|---|---|
| `snr_ref` | **不冻结**（校准后决定） | 控制变量：距离 `R0=100 m`、参考 RCS 下的逐 entry 接收 SNR(dB) |
| `R0` | 100 m | 参考距离（沿用旧 `noise_parameters` 的 reference_range_m=100） |
| `σ0` | 10 m² (10 dBsm) | 参考 RCS |
| `σ_i` | 固定 `σ0`（V1） | 车辆 RCS 先固定；TR 38.901 方向性 RCS / 对数正态起伏留 V1.1 |
| 噪声 `W` | 逐 entry 方差 1，`CN(0,1)`，每 BS 一份 | 与波形无关的等价噪声底 |

推论：同一帧内不同车辆接收功率因 `r_i` 天然不同（1/r⁴）；同一目标在三 BS 也不同。
禁止在检测/估计/跟踪任何 API 中传入逐目标 SNR 真值。

## 4. 单 BS 处理链（Range–Doppler 主路径 + 逐峰 AoA，P0-1 修订）

输入 `Y_b: [A,K,N]` 与已知 `X_b: [K,N]`。

```text
Z_b = Y_b / X_b                                      （去调制，噪声方差不变）
每个 Rx 元素 a 独立做：
  range:   Hann 窗(k) → IFFT(K→Nr=2K=512)         距离栅格 Δr = 412.17/512 = 0.805 m
  doppler: Hann 窗(n) → FFT(N→Nv=2N=512) 并 fftshift 速度栅格 Δv = 504.7/512 = 0.986 m/s
  → S[a,r,v]
非相干阵列合并：  P_RD[r,v] = Σ_a |S[a,r,v]|²         shape [512,512]（主路径只保留 2D RD 功率）
2D CA-CFAR on P_RD → 2D NMS → RD 峰列表 (r*,v*)
逐 RD 峰 AoA： S[:,r*,v*] → 矩形窗 → 64 点 FFT/Bartlett → 功率谱
              → 1D 峰值拾取（1D NMS + 抛物线插值）→ u_hat → θ_hat
```

- 主路径**不再构造** `[Nr,Nv,Na]`（约 1.68e7 单元）的全尺寸 3D 功率立方；
  AoA 只对 CFAR/NMS 后的少量 RD 峰（≤`max_candidates`）各做一次 64 点 FFT。
- 非相干阵列积累改善检测统计/处理性能；具体收益由 calibration 实测确定，
  本文不给出解析 SNR gain 数值。
- AoA 每 RD 峰允许输出多个显著峰（`aoa_max_peaks`）；V1 冻结为 **1（最强 AoA 峰）**，
  1D NMS 与上限保护已就位，多散点/多 AoA 峰留 V1.1（T5 记录该限制）。
- 不采用旧实现的 static-clutter mean subtraction：本场景没有静态杂波模型，减均值会削弱目标自身。
  （NIST `getRangeDoppler.m` 的 clutter removal 仅作参考，见 `EXTERNAL_REUSE_MATRIX.md`。）
- DBSCAN：V1 不启用（V1 车辆按单散点建模）。触发条件与升级路径见 `DECISIONS.md` D07。

## 5. Detection（2D Range-Doppler CA-CFAR + NMS + 逐峰 AoA）

```text
1) 2D CA-CFAR on P_RD[r,v]：训练/保护窗（range, doppler 两轴独立半径），边缘自适应计数
   threshold = noiseEst · α，α = Ktrain·(Pfa_cell^(-1/Ktrain) − 1)   （CA-CFAR 标准式）
2) 2D NMS：局部极大（3×3 邻域）+ 贪心抑制（抑制半径=保护半径），按功率降序保留
3) RD 峰值插值：log 功率 range/doppler 两轴抛物线插值
4) 逐峰 AoA：S[:,r*,v*] → 64 点 FFT → 1D NMS（u 轴，半径 ≈ 2/A 对应栅格数）
   → 1D 抛物线插值 → u_hat；V1 只取最强 AoA 峰（aoa_max_peaks=1）
5) 几何换算：斜距→水平距离 r_h = sqrt(r²−h²)；φ = boresight_b + asin(u)
   局部笛卡尔 p = s_b + r_h·[cos φ, sin φ]
6) 质量字段：peak_power / noise_floor / peak_to_noise_db / cfar_score / 训练单元数
```

- **目标数量** = 2D CFAR/NMS 后的 RD 峰数（再经 AoA 与跨 BS 关联成观测），任何阶段不得来自 GT。
- **虚警**：由 `Pfa_cell` 决定，`Pfa_cell = target_false_alarms_per_bs_frame / N_cells`，
  **`N_cells = Nr·Nv = 512·512 = 262 144`**（2D RD map；不再使用 3D 的 1.68e7 单元数）。
  默认目标虚警 ≈0.1–1/BS/帧；阈乘数由噪声-only CPI 标定（D08）。
- **漏检**：低 SNR、超分辨、视轴外、被强目标旁瓣掩盖时自然发生。
- **近邻目标**：距离差 <1.61 m、角度差 ≲6.35°、速度差 <1.97 m/s 时会被合并为 1 个检测；
  这是 V1 的**预期行为**，由 T5 定量描述，不做人为拆分。
  V1 每个 RD 峰只输出 1 个 AoA 峰，因此**同一距离-速度单元内的多目标会合并为一个检测**，
  该限制由 T5 明确记录（V1.1 可放开 `aoa_max_peaks`）。
- 每 BS 候选上限 `max_candidates=32`（沿用旧配置），超出按功率截断并计数。
- 输出 detection 字段（见 `INTERFACE_SPEC.md` §4）：`station_id, time_ns, r_m, vr_mps, u, bearing_rad,
  x_m, y_m, C_xy(2×2), peak_power, noise_floor, peak_to_noise_db, cfar_score, grid_index`。

### 5.1 单站测量协方差 `C_xy`（P0-3：calibration LUT 与传播）

`C_xy` 不是实现者临场选择的常数，而由离线 calibration 报告产生的保守 LUT 决定：

```text
C 域离线标定（允许使用 GT，仅用于统计残差）：
  1) 单目标合成场景，扫描 range / angle / snr_ref（含较低 SNR 档）
  2) 运行 detector（不使用 GT），统计 e_r = r_hat − r_gt, e_u = u_hat − u_gt
  3) 按 detector 可观察量 q = peak_to_noise_db 分箱（默认 [<6, 6–12, 12–20, >20] dB，可调）
     取稳健尺度：sigma_r(q) = 1.4826·MAD(e_r | q)，sigma_u(q) = 1.4826·MAD(e_u | q)
  4) 样本不足的箱取相邻箱上包络（保守）；写出 reports/f01e/covariance_calibration.json
B 域 inference（只用观测）：
  q = detection.peak_to_noise_db → (sigma_r(q), sigma_u(q))
  R_ru = diag(sigma_r², sigma_u²)
  C_xy = J · R_ru · Jᵀ + εI,  ε = 0.01 m²（数值下限）
```

Jacobian（`frontend/sensing/coords.py` 实现并单测）：

```text
ρ = sqrt(r² − h²),  φ = boresight_b + asin(u)
x = x_b + ρ·cos φ,   y = y_b + ρ·sin φ
∂ρ/∂r = r/ρ
∂(x,y)/∂r = (r/ρ)·[cos φ, sin φ]
∂(x,y)/∂u = ρ·[−sin φ, cos φ] / sqrt(1 − u²)
J = [[∂x/∂r, ∂x/∂u], [∂y/∂r, ∂y/∂u]]
```

- LUT 数值必须来自 calibration 报告，不得拍脑袋；数据不足时可退化为"上包络常数"，并在报告中声明。
- B 域禁止使用 GT 或 SNR truth 选择 σ；只允许观测 `q`。

## 6. 跨 BS 关联与轻量融合（P0-5：顺序 Hungarian grouping）

唯一冻结算法（可直接编码；不使用 GT 选择顺序或消解冲突）：

```text
输入：per-BS detections {D0,D1,D2}，每站先按 (peak_power 降序, grid_index 升序) 确定性排序
传感器顺序：固定 [BS0, BS1, BS2]（BS0 为参考站，理由见下）

Stage 1：BS0 ↔ BS1
  cost(i,j) = Mahalanobis²(p0_i, p1_j | S = C0_i + C1_j)，超 χ²(2,0.99)=9.21 记 BIG
  Hungarian 最小总代价匹配；未匹配检测各自成为 singleton group
  每个 group 的 state = 成员检测的逆协方差融合 (p_g, C_g)（singleton 即自身）

Stage 2：groups ↔ BS2
  cost(g,k) = Mahalanobis²(p_g, p2_k | S = C_g + C2_k)，同样门控 + Hungarian
  未匹配 BS2 检测成为 singleton group

输出：final groups；每组内每 BS 至多 1 个检测，n_bs∈{1,2,3}
  n_bs≥2 → C_f = (Σ C_b⁻¹)⁻¹，p_f = C_f Σ C_b⁻¹ p_b（逆协方差融合）
  n_bs=1 → 保留单站 observation（更大协方差）
```

- Hungarian 用 `scipy.optimize.linear_sum_assignment`；出格项设 BIG，保证不会强行匹配。
- **assignment 后必须显式 post-filter（Stage 1/Stage 2 各自执行）**：对每个被分配的 pair
  重新核验 `Mahalanobis² ≤ 9.21`；凡代价为 BIG 或超门限的 assignment 一律恢复为 unmatched
  （检测回到 singleton group），不得把 BIG 匹配带入 group 或融合。
- 固定 BS0→BS1→BS2 的理由：BS0/BS2 在道路同侧、BS1 在另一侧；BS0–BS2 直接配对的视差在
  道路纵向几何上更弱，先配 BS1（异侧站）再补 BS2 在观测几何上最稳。
  顺序只依赖站点几何与检测质量，不含 GT。
- 已知局限：Stage 1 的配对是局部最优，若 BS0–BS1 先把两个近邻目标配错，Stage 2 无法回溯；
  T4/T5 量化该现象，V1.1 可改为全三元枚举。
- 单 BS 观测允许更新既有航迹；**不能单独确认新航迹**（D10/D12）。
- 几何覆盖率支撑该策略：A01 统计 ≥2 BS 可见率 99.95%（全 3 BS 约 18%，典型为 2 BS）。
- 门限来自测量协方差（χ²(2,0.99)=9.21），不得来自 GT；标定只允许用无目标噪声 CPI。
- 关联正确率、ID switch 等作为 C 域指标（§9），不得回写感知状态。

## 7. CV-KF 跟踪（速度来源）

状态与模型：

```text
x = [x, y, vx, vy]ᵀ
F(Δt) = [[I, ΔtI],[0, I]],  Δt = 0.1 s
Q(Δt) = q_a · [[Δt³/3·I, Δt²/2·I],[Δt²/2·I, Δt·I]]     （连续白噪声加速度，q_a 标定）
H = [I, 0]（只测位置；V1 不融合径向速度）
R = C_fused（+ 生效的膨胀因子）
```

航迹状态机：

```text
birth:      首次观测建 tentative 航迹（初始速度 0，初始 P 膨胀 κ_birth=4）
confirm:    最近 3 帧内 ≥2 次更新，且其中 ≥1 次来自 n_bs≥2 的融合观测
            满足 → confirmed（分配 slot）；3 帧未满足 → tentative 过期删除
coast:      本帧无更新 → 只预测；detected=0，track_exists=1；最多连续 5 帧（0.5 s）
delete:     连续 misses>5 → 删除（slot 进入冷却）
velocity:   由 KF 状态给出；早期（<4 帧更新）速度先验为 0、协方差大，输出为滤波值
            不做 (vr1,vr2,vr3)→2D WLS；若 V1 验收速度 RMSE 不达标，V1.1 再启用
```

- 只输出 confirmed 且 alive 的航迹；tentative 不进入下游、不占 slot。
- 速度来源是 CV-KF 位置序列，径向多普勒只保留在 detection payload（诊断/V1.1）。
- **T6-B 消融模式（P0-4）**：仅在 1/2/3-BS 公平比较的诊断中允许把 `confirm_requires_nbs2`
  置为 false（"3 帧内 ≥2 次 hit"即可确认）；正式 mainline 恒为 true，
  消融报告须标注"ablation mode，不改变正式 tracker"。

## 8. 匿名航迹 → 固定 8 槽（推理时策略）

```text
track_key:        内部自增整数，永不复用；无 GT 输入
slot:             0..7；confirmed 时分配最小空闲 slot；alive 期间 slot identity 固定（sticky）；
                  删除后冷却 10 帧再回收
容量控制:         confirmed >8 时，已持 slot 者保持；slotless confirmed 在 slot 冷却结束后
                  按 (misses 升序, trace(P_xy) 升序, age 降序, track_key 升序) 补位
每帧输出:         对 slot 0..7：state_hat[4] float32（缺失填 0）、track_exists、detected
```

不做"slot 与车辆一一对应"的假设；对应关系只在 C 域离线标注。

## 9. 监督标签（C 域，感知全部结束后）

```text
对每个 prediction origin（A01 grid）：
  输入侧：20 帧历史每个 slot 的 state_hat/track_exists/detected（B 域产物，冻结）
  C 域：track slot ↔ GT 车辆 的离线匹配
        cost(slot, veh) = 历史公共帧上的平均位置距离（要求 ≥3 公共帧）
        Hungarian 匹配；匹配对距离 > d_gate=5 m 或公共帧 <3 → 不匹配
        匹配成功 → label = 该车未来 20 帧原始位置；label_valid 由 GT 存在性给出
        未匹配 slot → label_valid 全 False（输入保留，损失掩码）
        未被任何 slot 匹配的 GT → 不在输入中出现
  记录：matching cost、公共帧数、ID switch 计数（诊断，不回写 B 域）
```

标签构造不得修改 sensing state、track_key、slot 或 detector 输出（用输入文件 hash 核对）。

## 10. 三域隔离（摘要，详见 `GT_ISOLATION_SPEC.md`）

```text
A 模拟器/GT 域：NGSIM → scene_manifest → source_states → 共享回波生成 → 未来标签
B 感知/推理域：Y/X → detector → association → fusion → tracker → state_hat/masks
C 监督/评估域：B 冻结后，track↔GT 匹配、指标、校准分析
```

B 域模块不得 import A 域模块；GT（目标数、ID、真值位置/速度、未来轨迹）不得进入 B 域 API。