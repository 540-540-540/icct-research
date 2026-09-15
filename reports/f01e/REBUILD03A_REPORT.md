# REBUILD-03A：共享多目标感知核心（P0–P2）报告

- 分支：`sens-rebuild-03a`（基线 `sens-design-02` @ `d04403005230574e362f863bf234ee9c65c4cd68`）
- 范围：P0 脚手架、P1 shared echo + T1、P2 2D RD-CFAR + NMS + 逐峰 AoA + CFAR/协方差标定 + T2 + 单目标 sanity
- 本轮未进入 P3–P7；未生成正式缓存；未训练下游
- 运行环境：服务器 `YrM_TwYhB`，`/home/dell/YrM/envs/ICCT/bin/python`，CUDA float64/complex128

## 1. 结论总表

| 项 | 结果 | 证据 |
|---|---|---|
| P0 脚手架 | **PASS** | `configs/shared_frontend.json`、`frontend/sensing/*`、`frontend/fusion|tracking/__init__.py`、`reports/f01e/baseline_hashes.json`（26 个旧资产 hash，missing=0） |
| P1 shared echo | **PASS** | `frontend/sensing/{waveform,simulator}.py` |
| T1 | **PASS** | `reports/f01e/p1_shared_echo/{summary,checks}.json` |
| P2 detector | **PASS** | `frontend/sensing/{detector,coords}.py`，模块自检 + 单目标检测 |
| CFAR calibration | **PASS** | `reports/f01e/cfar_calibration.json` |
| Covariance calibration | **PASS** | `reports/f01e/covariance_calibration.json` |
| T2 no-GT | **PASS** | `reports/f01e/t2_no_gt/{summary,checks}.json`（7/7 checks） |
| 单目标 sanity | **PASS** | `reports/f01e/p2_detector/{summary,checks}.json` + `single_target_accuracy.csv` |
| 修改 `data/f01d` | **NO** | 本轮从未写入 `data/**`；只有 `baseline_hash.py` 读取用于记录 SHA256 |
| 读取 V_confirm/test | **NO** | 所有脚本只用 A01 几何 + 合成目标；未打开任何 split 缓存 |
| 下游训练 | **NO** | 无 GNN/QGNN/GPT-2 调用；GPU 只用于 simulator/detector |

## 2. 关键数值

### T1（`reports/f01e/p1_shared_echo/summary.json`）

- `Y_shared == Σ Y_single`（clean）：最大相对误差 **3.49e-16**
- 单份接收噪声经验功率（3 BS）：1.00056 / 1.00071 / 1.00096（目标 1.0，±5% 容差）
- 目标顺序置换：`X` 逐比特不变；噪声最大差 ≤1e-12（残差为 clean 求和次序的浮点舍入）
- phase 种子按 `source_key`：同 key 逐比特复现，不同 key 改变回波；同参数调用逐比特确定
- `Y` shape `[3,16,256,256]`，无目标维；32×32 crop 诊断文件已存（1.5 MB；
  `*.npz` 被仓库 `.gitignore` 策略排除，未提交 GitHub，可在服务器复现生成）

### CFAR noise-only 标定（`cfar_calibration.json`）

- `N_cells = 512×512 = 262 144`；目标虚警 0.5/BS/帧
- Stage A（512 帧/BS，局部极大计数量）与 Stage B（128 帧/BS，完整 detector 路径）双估计
- 选定 **multiplier = 0.19**，Stage B 实测 **0.5 虚警/BS/帧**（within_tolerance=true）
- 阈值曲线（Stage B，虚警/BS/帧）：0.16→15.1，0.18→1.60，0.19→0.50，0.20→0.16，≥0.24→0
- 说明：乘数只写入标定报告，`configs/shared_frontend.json` 的
  `cfar_threshold_multiplier` 保持 `null`，正式冻结留待后续评审（D08/D20 精神）

### 协方差 LUT（`covariance_calibration.json`）

- 扫描 8 range × 5 bearing × 6 `snr_ref` × 6 realizations；3 BS 共匹配 2880 条检测残差
- 分箱（`q = peak_to_noise_db`）后的保守 σ（已按“低 q ≥ 高 q、样本不足取包络”规则）：

| q 箱 | samples | sigma_r (m) | sigma_u |
|---|---:|---:|---:|
| <6 dB | 22 | 0.1319 | 1.619e-3 |
| 6–12 dB | 0 | 0.0421 | 1.384e-4 |
| 12–20 dB | 0 | 0.0421 | 1.384e-4 |
| >20 dB | 2858 | 0.0413 | 1.370e-4 |

- pooled robust σ：r 0.0421 m / u 1.384e-4；每 SNR 检测率 2/3（其余为几何 FoV 外，符合 A01 覆盖率）
- 说明：本轮条件下 q 几乎全部 >20 dB，LUT 实际近似保守常数；低 q 箱样本少，已用 pooled 包络。
  **P3 关联前若需要更细的低 q 区分度，可扩大低 SNR/远距扫描**（记录为下一轮可选事项，非阻断）

### 单目标 sanity（`p2_detector/`）

- 35 个 (range, bearing) 组合 × 3 realizations，`snr_ref=20 dB`，检测率 **105/105**
- 中位绝对误差：range **0.081 m**、AoA u **4.30e-05**、径向速度 **0.0038 m/s**
- 最大位置误差 0.428 m；Cartesian 反算一致性 ≤1e-9
- 验收阈值：range ≤0.8 m、u ≤0.02、vr ≤0.5 m/s，全部通过

## 3. 实现要点（与冻结文档对齐）

- **共享回波**：每 BS 每帧一份 `Y=[A,K,N]`，Σ targets → 加一份 `CN(0,1)` 噪声；
  `X_b` 每 BS 每帧共享（目标维不进入接收数据）
- **种子合同（D18）**：`waveform/noise` 按 `(episode,frame,bs)`；`phase` 按 `(episode,frame,bs,source_key)`；
  `source_key` 仅 A 域 simulator 内部；同帧跨 SNR 复用同波形/噪声/相位
- **检测（D07/D08）**：逐 Rx 元素 RD 处理 → `P_RD=Σ_a|S|²` → 2D CA-CFAR（边缘自适应计数）→
  2D NMS → RD 抛物线插值 → 逐 RD 峰 64 点 AoA（1D NMS + 插值，`aoa_max_peaks=1`）→ 单站几何
- **C_xy（D23）**：LUT（只依赖 observed `peak_to_noise_db`）+ Jacobian
  `∂(x,y)/∂r=(r/ρ)[cosφ,sinφ]`、`∂(x,y)/∂u=ρ/√(1−u²)[−sinφ,cosφ]`，`+0.01·I`
- **GT 隔离（D16）**：B 域模块无 GT import/参数（AST + 签名 + import 哨兵 + 置换测试全通过）

## 4. 过程记录：设计冲突 / 偏差 / 修正（均非静默改设计）

1. **配置 schema 缺 `cfar_threshold_multiplier`**：D08 要求标定阈乘数，但 INTERFACE_SPEC §4.1 的
   detector 块未列该键。处理：新增该键（默认 `null` = 乘数 1.0），标定值仅写入
   `cfar_calibration.json`，正式配置保持 null。**非阻断，记录待评审。**
2. **`frontend/` 不在 Syncthing 同步范围**：新 sensing 代码通过 `scp` 部署到服务器；
   本地仓库包含同一份文件用于 GitHub。**流程记录，供后续轮次沿用。**
3. **实现 bug 修正（P1/P2 内部）**：方位角 wrap、4D 广播形状、空目标集合、`source_keys` 传参、
   有限差分调试残留清理。均为实现修正，不改冻结设计。
4. **协方差标定检测率 2/3**：由于 A01 的 ±70° FoV，单目标相对部分 BS 不可见；已在报告中解释，
   不影响 LUT 统计有效性。
5. **CFAR 虚警曲线陡峭**：Gamma(16) 非相干积累下阈值对乘数极敏感；已用细化网格命中目标虚警，
   未改任何验收目标值。

## 5. 复现命令

```bash
cd /home/dell/YrM/ICCT
/home/dell/YrM/envs/ICCT/bin/python -u code/07_shared_frontend/baseline_hash.py
/home/dell/YrM/envs/ICCT/bin/python -u code/07_shared_frontend/check_shared_echo.py
/home/dell/YrM/envs/ICCT/bin/python -u code/07_shared_frontend/calibrate_cfar.py
/home/dell/YrM/envs/ICCT/bin/python -u code/07_shared_frontend/calibrate_covariance.py
/home/dell/YrM/envs/ICCT/bin/python -u code/07_shared_frontend/check_single_target.py
/home/dell/YrM/envs/ICCT/bin/python -u code/07_shared_frontend/check_no_gt.py
```

## 6. 停止状态

P2 已完成并验收；**本轮在 P2 停止**，未进入 P3 association/fusion、P4 tracker、P5 packing、
P6 SNR sweep、P7 正式缓存与任何下游训练。等待独立验收后再继续。