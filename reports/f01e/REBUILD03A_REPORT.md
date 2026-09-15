# REBUILD-03A：共享多目标感知核心（P0–P2）报告（R1 修订版）

- 分支：`sens-rebuild-03a`（基线 `sens-design-02` @ `d04403005230574e362f863bf234ee9c65c4cd68`）
- 范围：P0 脚手架、P1 shared echo + T1、P2 2D RD-CFAR + NMS + 逐峰 AoA + CFAR/协方差标定 + T2 + 单目标 sanity
- R1 修订：range geometry（水平距离 ρ / 斜距 r / height 对齐 A01）、T1 原始噪声独立性与 summary/checks 一致性、
  CFAR multiplier 0.19 写入正式 config、移除 detector 对 A 域 simulator 的 `__main__` 依赖、detector counters 语义、
  全量重跑 T1 / CFAR verify / covariance / T2 / sanity
- 本轮未进入 P3–P7；未生成正式缓存；未训练下游
- 运行环境：服务器 `YrM_TwYhB`，`/home/dell/YrM/envs/ICCT/bin/python`，CUDA float64/complex128

## 1. 结论总表（R1 后）

| 项 | 结果 | 证据 |
|---|---|---|
| P0 脚手架 | **PASS** | `configs/shared_frontend.json`、`frontend/sensing/*`、`frontend/fusion|tracking/__init__.py`、`reports/f01e/baseline_hashes.json`（26 项，missing=0） |
| P1 shared echo | **PASS** | `frontend/sensing/{waveform,simulator}.py`（R1 geometry 修复后） |
| T1 | **PASS** | `reports/f01e/p1_shared_echo/{summary,checks}.json`（`passed=true`，两份 checks 完全一致） |
| P2 detector | **PASS** | `frontend/sensing/{detector,coords}.py`；smoke 在 `code/07_shared_frontend/run_selfchecks.py` |
| CFAR calibration | **PASS** | `reports/f01e/cfar_calibration.json`（multiplier=0.19 已写入正式 config；final_verify 见下） |
| Covariance calibration | **PASS** | `reports/f01e/covariance_calibration.json`（R1 重跑） |
| T2 no-GT | **PASS** | `reports/f01e/t2_no_gt/{summary,checks}.json`（8/8，含 counter 与 simulator import 检查） |
| 单目标 sanity | **PASS** | `reports/f01e/p2_detector/{summary,checks}.json` + CSV（R1 重跑） |
| 修改 `data/f01d` | **NO** | 未写入 `data/**`；仅 `baseline_hash.py` 读取并记录 SHA256 |
| 读取 V_confirm/test | **NO** | 全部脚本只用 A01 几何 + 合成目标 |
| 下游训练 | **NO** | 无 GNN/QGNN/GPT-2 调用 |

## 2. R1 修订内容

1. **Range geometry（`frontend/sensing/simulator.py`）**
   - `rho = ||p−s||` 水平距离；`r = sqrt(rho² + h²)` 斜距；
   - delay、1/r⁴ 功率、range visibility 使用 `r`；`radial = (s−p)·v / r`；bearing/AoA 用水平几何；
   - `height_m` 为必填参数，来自 A01 geometry / config（`visibility.height_difference_m=5.0`），
     脚本启动时断言 config 与 A01 一致；删除未使用的 `H_M` 常量。
2. **T1 原始噪声独立性**：simulator 返回原始 `W`（每 BS 每帧一份）；T1 逐比特比较
   `X`、`W` 在目标顺序置换下不变，并逐比特验证 `Y == clean + W`（噪声只在求和后加一次）。
   `summary.json` 与 `checks.json` 使用同一 checks 对象，`passed` 一致，不再出现报告与 raw checks 不一致。
3. **CFAR multiplier 评审通过**：`configs/shared_frontend.json` 写入
   `cfar_threshold_multiplier=0.19`；covariance/sanity/T2 全部使用正式值；`snr_levels_db` 与
   `power.snr_ref_db` 仍为 `null`。
4. **B 域隔离**：`frontend/sensing/detector.py` 不再包含 `__main__`（原先 import A 域 simulator）；
   smoke 移到 `code/07_shared_frontend/run_selfchecks.py`；T2 静态审计把 `simulator` 列入 B 域禁止 import。
5. **Detector counters**：`candidates_before_cap` 为 cap 前真实候选数（NMS + 几何过滤后），
   `overflow = candidates_before_cap − returned_count`，可真实记录被 cap 丢弃的候选；T2 增加
   `counter_semantics` 检查。
6. **全量重跑**：T1、CFAR final verify、covariance calibration、T2、single-target sanity（见 §3）。

## 3. 关键数值（R1 后）

### T1（`reports/f01e/p1_shared_echo/summary.json`）

- `Y_shared == Σ Y_single`（clean）：最大相对误差 **3.76e-16**
- 原始接收噪声经验功率（3 BS）：1.00056 / 1.00071 / 1.00096（±5% 容差内）
- 目标顺序置换：`X` 与原始 `W` **逐比特不变**；`Y == clean + W` 逐比特成立
- phase 按 `source_key`：同 key 逐比特复现，不同 key 改变回波；同参数调用逐比特确定
- `Y` shape `[3,16,256,256]`，无目标维；32×32 crop 诊断（1.5 MB，`*.npz` 被仓库策略排除，未提交）

### CFAR noise-only 标定与最终 verify（`cfar_calibration.json`）

- `N_cells = 512×512 = 262144`；目标虚警 0.5/BS/帧；multiplier **0.19** 已按评审写入 config
- 原 sweep 的 Stage B（128 帧/BS）：0.19 → **0.500** 虚警/BS/帧
- **final_verify（最终代码/config，256 帧/BS，独立 episode 9102）：0.397 虚警/BS/帧**
  （per-BS 107/96/102），`within_tolerance=true`
- 说明：0.19 的 verify 结果在 0.40–0.50 之间波动属统计涨落（每 256 BS-帧约 100 次虚警，σ≈0.04）
- 正式 `snr_levels_db` 仍为 `null`

### 协方差 LUT（`covariance_calibration.json`，R1 重跑）

- 8 range × 5 bearing × 6 `snr_ref` × 6 realizations；3 BS 匹配 **2896** 条残差；检测率 0.658–0.675
- 与 R0 一致：本轮条件下 q 全部 >20 dB；LUT 为保守常数（低 q 箱按规则继承 pooled 包络）
  - `sigma_r = 2.57e-3 m`、`sigma_u = 1.40e-4`；median |e_r| = 2.13e-3 m、median |e_u| = 9.44e-5
- 记录为 P3 前的可选事项：如需低 q 区分度，扩大低 SNR/远距扫描（非阻断）

### 单目标 sanity（`p2_detector/`，R1 重跑）

- 35 个 (range, bearing) 组合 × 3 realizations，`snr_ref=20 dB`，检测 **104/105**（300 m 有 1 次漏检）
- 中位绝对误差：range **0.00224 m**、AoA u **4.01e-05**、径向速度 **0.00116 m/s**
- 最大位置误差 0.187 m；Cartesian 反算一致性 ≤1e-9；全部 checks PASS

## 4. R1 专项检查（工单第 7 项）

### 30 m range bias

- R0（修复前）：30 m 三组样本 median `e_r = −0.4222 m`（与 `sqrt(30²−5²)−30` 的理论偏差一致）
- R1（修复后）：30 m 15 个样本 median `e_r = +0.0026 m`；全部 range（30–300 m）median `e_r` 均在
  ±0.003 m 内（见 `p2_detector/summary.json:range_bias_by_range_m`）
- 结论：**稳定偏置已消除**，该偏置确由 simulator 使用水平距离生成、而评估用斜距导致

### Radial velocity gross outlier

- R0：存在两条约 −30.7 / −34.5 m/s 的离群（30 m，`peak_to_noise_db` 仅 4.4–4.8 dB，
  与低 SNR 下匹配到弱峰/旁瓣有关）
- R1：最差三条 |e_vr| = 0.0041 / 0.0037 / 0.0035 m/s，对应 `peak_to_noise_db` 23.1–25.1 dB；
  **未再出现 gross outlier**（`p2_detector/summary.json:worst_radial_velocity_rows`）
- 本轮如实记录：R0 的离群样本为 30 m、u=±0.5 的两帧（上表），R1 已不可复现

## 5. 实现要点（与冻结文档对齐）

- 共享回波：每 BS 每帧一份 `Y=[A,K,N]`，Σ targets → 加一份 `CN(0,1)`；`X_b` 每 BS 每帧共享
- 种子合同（D18）：`waveform/noise` 按 `(episode,frame,bs)`；`phase` 按 `(episode,frame,bs,source_key)`；
  `source_key` 仅 A 域 simulator 内部
- 检测（D07/D08）：逐 Rx 元素 RD → `P_RD=Σ_a|S|²` → 2D CA-CFAR（边缘自适应）→ 2D NMS →
  RD 抛物线插值 → 逐 RD 峰 64 点 AoA（1D NMS + 插值，`aoa_max_peaks=1`）→ 单站几何
- C_xy（D23）：LUT（只依赖 observed `peak_to_noise_db`）+ Jacobian
  `∂(x,y)/∂r=(r/ρ)[cosφ,sinφ]`、`∂(x,y)/∂u=ρ/√(1−u²)[−sinφ,cosφ]`，`+0.01·I`
- GT 隔离（D16）：B 域无 GT import/参数（AST + 签名 + import 哨兵 + 置换 + counters，8/8 通过）

## 6. 复现命令

```bash
cd /home/dell/YrM/ICCT
/home/dell/YrM/envs/ICCT/bin/python -u code/07_shared_frontend/baseline_hash.py
/home/dell/YrM/envs/ICCT/bin/python -u code/07_shared_frontend/run_selfchecks.py
/home/dell/YrM/envs/ICCT/bin/python -u code/07_shared_frontend/check_shared_echo.py
/home/dell/YrM/envs/ICCT/bin/python -u code/07_shared_frontend/calibrate_cfar.py --verify-only
/home/dell/YrM/envs/ICCT/bin/python -u code/07_shared_frontend/calibrate_covariance.py
/home/dell/YrM/envs/ICCT/bin/python -u code/07_shared_frontend/check_single_target.py
/home/dell/YrM/envs/ICCT/bin/python -u code/07_shared_frontend/check_no_gt.py
```

## 7. 停止状态

P2 已完成并验收；**本轮在 P2 停止**，未进入 P3 association/fusion、P4 tracker、P5 packing、
P6 SNR sweep、P7 正式缓存与任何下游训练。阻断项：无。