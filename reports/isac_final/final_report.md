# Automatum ISAC Route-B Final Execution 报告（V2-FROZEN）

- 基线：`module/isac` @ `eaadbe6d826f9f6e83baf3caf3c01bf75ac093d1`
- 本轮：最终执行与冻结（Phase A geometry → Phase B recalibration → Phase C 全量验收 → Phase D FREEZE）。
- 数据规则：geometry 搜索与 calibration 只用 **train** prediction-history unique states（train 49,765）；**val 只做验证**；**test 完全未使用**（`final_metrics.json → test_used = false`、冻结清单同）。
- 冻结结果：**config revision `AUTOMATUM-CONTROLLED-ISAC-V2-FROZEN`，status `FROZEN`**；目标带全部 PASS，五档严格单调，train/val 一致。

---

## 1. Phase A — scene 0 几何最小修正

### 1.1 修改内容

| 项 | 修改前 | 修改后 |
|---|---|---|
| scene 0 BS0 | x=26.200525, y=7.623030, boresight −167.383844° | 不变 |
| scene 0 BS1 | x=−30.432527, y=33.092568, boresight −75.415062° | 不变 |
| **scene 0 BS2** | x=−26.549336, y=−32.757553, **boresight 79.551839°** | 坐标不变，**boresight 64.551839°（−15°）** |
| scene 0 FOV | ±70° | 不变 |
| scene 0 BS 坐标 | — | 不变 |
| **scene 1** | — | **完全未修改**（汉字面验证见 selfcheck 2） |

原因（Final Audit）：scene 0 的 1-BS 状态（6.04%）全部只被 BS1 看到；这些状态位于 BS2 的 FOV 边缘（bearing −70.0°…−83.2°，p50 −76.3°）。BS2 boresight −15° 后全部进入 ±70° 视场；15° 是覆盖该区间的最小偏移（−10° 仍差 3.2°）。搜索按 Level 1（boresight）→ Level 2（FOV）→ Level 3（BS 坐标）顺序，Level 1 即达到目标，FOV 与坐标未被改动。

### 1.2 Top-5 geometry candidate（train scene 0，rank 规则见 `geometry_search.json`）

| # | boresight offsets (BS0/BS1/BS2) | FOV | 坐标移动 | 1-BS | P(≥2) | |Δboresight| 和 |
|---|---|---|---|---|---|---|
| **1（选中）** | **0° / 0° / −15°** | 70° | 无 | **0** | **100%** | 15° |
| 2 | +5° / 0° / −15° | 70° | 无 | 0 | 100% | 20° |
| 3 | 0° / +5° / −15° | 70° | 无 | 0 | 100% | 20° |
| 4 | 0° / −5° / −15° | 70° | 无 | 0 | 100% | 20° |
| 5 | −5° / 0° / −15° | 70° | 无 | 0 | 100% | 20° |

### 1.3 覆盖（SNR 无关，unique prediction-history states）

| split | scope | 0 BS | 1 BS | 2 BS | 3 BS | P(≥2) | 修改前 P(≥2) |
|---|---|---|---|---|---|---|---|
| train | overall | 0 | **0** | 19,153 | 30,612 | **100%** | 97.04% |
| train | scene 0 | 0 | **0** | 9,048 | 15,304 | **100%** | 93.96% |
| train | scene 1 | 0 | 0 | 10,105 | 15,308 | 100% | 100%（未改） |
| val | overall | 0 | **0** | 2,238 | 2,541 | **100%** | 95.84% |
| val | scene 0 | 0 | **0** | 1,039 | 1,425 | **100%** | 91.92% |
| val | scene 1 | 0 | 0 | 1,199 | 1,116 | 100% | 100%（未改） |

condition ratio 中位数（几何量）：train overall 0.408（scene 0 0.408 / scene 1 0.351），val overall 0.383 → 两 BS 不再存在共线病态；2-BS 状态的 SNR→误差梯度不再被恒定几何误差污染。

## 2. Phase B — SNR → sensing error 重新标定

- 噪声模型保持不变：`sigma_q(gamma) = sqrt(floor_q² + (a_q·10^(−gamma/20))²)`，`k_q = 1`；floors 保持 SMOKE-01 值（range 0.0018 m、bearing 1.36e-4 rad、vr 0.0023 m/s）。
- 搜索：coarse grid `a_range 0.10–0.40`（7）× `a_bearing 0.003–0.012`（7）× `a_vr 0.15–0.40`（6）= 294 点，再用 coordinate refine（每通道 ×{0.85,0.925,1.0,1.075,1.15}，2 轮），共 324 次评估；objective 只用 target-band loss + 单调性罚项，soft reference 仅作 tie-break（不使用旧 `100·exp(−s)`）。
- 搜索结果 **coarse best = (0.35, 0.012, 0.30)**，refine 后 **selected a = (a_range=0.35 m, a_bearing=0.0111 rad, a_vr=0.30 m/s)**；band loss = 0，单调性违背 = 0，soft loss 0.054。
- 搜索使用向量化 replica `fast_eval.py`；selfcheck 与真实 frontend 抽样逐状态比对（max Δposition 1.5e-13 m，max Δvelocity 2.1e-8 m/s，仅 2×2 求解器浮点差异；n_bs 完全一致）。
- 候选表：`calibration_candidates.csv`（top-20 + 旧 Candidate B 参考）；完整搜索日志 `calibration_search.json`。

### 目标带对照（train，真实 frontend 验收值）

| SNR | position RMSE | band | velocity RMSE | band | soft ref (pos/vel) |
|---|---|---|---|---|---|
| +10 | **0.1308 m** | 0.10–0.30 ✓ | **0.1590 m/s** | 0.10–0.30 ✓ | 0.15 / 0.175 |
| 0 | **0.4137 m** | 0.30–0.70 ✓ | **0.5028 m/s** | 0.30–0.80 ✓ | 0.45 / 0.50 |
| −10 | **1.3090 m** | 0.80–1.50 ✓ | **1.5908 m/s** | 1.00–2.00 ✓ | 1.15 / 1.50 |

五档严格单调（+10→+5→0→−5→−10）：position 0.131 → 0.233 → 0.414 → 0.736 → 1.309 m；velocity 0.159 → 0.283 → 0.503 → 0.894 → 1.591 m/s。

## 3. Phase C — train/val 全量正式验收

### 3.1 overall（unique prediction-history states；train n=49,765，val n=4,779；无 1-BS，全部 ≥2 BS）

| SNR | Position MAE / RMSE / median / P90 / P95 / P99 (m) | Velocity MAE / RMSE / median / P90 / P95 / P99 (m/s) | vx RMSE | vy RMSE |
|---|---|---|---|---|
| train +10 | 0.108 / **0.131** / 0.092 / 0.202 / 0.250 / 0.367 | 0.132 / **0.159** / 0.114 / 0.240 / 0.299 / 0.450 | 0.099 | 0.085 |
| train +5 | 0.192 / 0.233 / 0.163 / 0.359 / 0.445 / 0.652 | 0.235 / 0.283 / 0.202 / 0.427 / 0.532 / 0.800 | 0.177 | 0.151 |
| train 0 | 0.341 / 0.414 / 0.290 / 0.639 / 0.792 / 1.160 | 0.418 / 0.503 / 0.359 / 0.760 / 0.947 / 1.424 | 0.314 | 0.269 |
| train −5 | 0.607 / 0.736 / 0.516 / 1.136 / 1.409 / 2.064 | 0.743 / 0.894 / 0.638 / 1.352 / 1.684 / 2.534 | 0.559 | 0.478 |
| train −10 | 1.080 / **1.309** / 0.917 / 2.022 / 2.510 / 3.668 | 1.321 / **1.591** / 1.135 / 2.404 / 2.992 / 4.519 | 0.994 | 0.851 |
| val +10 | 0.116 / 0.143 / 0.096 / 0.224 / 0.276 / 0.416 | 0.142 / 0.175 / 0.119 / 0.263 / 0.337 / 0.519 | 0.112 | 0.090 |
| val +5 | 0.207 / 0.255 / 0.171 / 0.398 / 0.491 / 0.740 | 0.253 / 0.311 / 0.211 / 0.468 / 0.599 / 0.923 | 0.198 | 0.160 |
| val 0 | 0.368 / 0.453 / 0.305 / 0.707 / 0.873 / 1.317 | 0.450 / 0.553 / 0.376 / 0.832 / 1.067 / 1.641 | 0.353 | 0.284 |
| val −5 | 0.655 / 0.806 / 0.542 / 1.258 / 1.555 / 2.343 | 0.800 / 0.983 / 0.668 / 1.473 / 1.898 / 2.918 | 0.628 | 0.506 |
| val −10 | 1.165 / **1.434** / 0.963 / 2.239 / 2.767 / 4.166 | 1.423 / **1.748** / 1.186 / 2.635 / 3.374 / 5.193 | 1.117 | 0.899 |

### 3.2 分 scene（position / velocity RMSE）

| split/scope | +10 | +5 | 0 | −5 | −10 |
|---|---|---|---|---|---|
| train scene 0 | 0.126 / 0.150 | 0.223 / 0.266 | 0.397 / 0.473 | 0.706 / 0.841 | 1.256 / 1.496 |
| train scene 1 | 0.136 / 0.168 | 0.241 / 0.298 | 0.429 / 0.530 | 0.763 / 0.943 | 1.358 / 1.677 |
| val scene 0 | 0.131 / 0.155 | 0.232 / 0.275 | 0.413 / 0.489 | 0.733 / 0.869 | 1.305 / 1.545 |
| val scene 1 | 0.156 / 0.194 | 0.277 / 0.345 | 0.493 / 0.613 | 0.877 / 1.091 | 1.560 / 1.941 |

### 3.3 2-BS vs 3-BS（train，position / velocity RMSE）

| subset | n | +10 | 0 | −10 |
|---|---|---|---|---|
| 2 BS | 19,153 | 0.175 / 0.201 | 0.553 / 0.634 | 1.750 / 2.005 |
| 3 BS | 30,612 | 0.093 / 0.126 | 0.295 / 0.400 | 0.932 / 1.264 |

2-BS 与 3-BS 都保持完整 SNR 梯度；差异来自几何融合增益，不再是恒定几何盲差（对比 Final Audit：旧 1-BS 速度误差恒定 2.26 m/s）。

### 3.4 速度区间（train，position / velocity RMSE）

| speed bin | n | +10 | 0 | −10 | velocity/speed @−10 |
|---|---|---|---|---|---|
| 0–2 m/s | 6,615 | 0.101 / 0.122 | 0.318 / 0.386 | 1.007 / 1.222 | 低速不报百分比，只报绝对误差 |
| 2–5 | 3,771 | 0.102 / 0.127 | 0.321 / 0.402 | 1.017 / 1.273 | 28.1% |
| 5–10 | 9,176 | 0.109 / 0.137 | 0.345 / 0.434 | 1.092 / 1.373 | 13.9% |
| 10–20 | 20,059 | 0.143 / 0.171 | 0.453 / 0.542 | 1.435 / 1.715 | 7.9% |
| ≥20 | 10,144 | 0.149 / 0.182 | 0.470 / 0.575 | 1.487 / 1.820 | 5.6% |

低速端（<2 m/s，13.3% 状态）在 −10 dB 的绝对速度误差 RMSE 1.22 m/s、中位 0.98 m/s；不使用不稳定的速度百分比。

### 3.5 Poor-but-usable 稳定性（train，normal speed ≥2 m/s）

| SNR | 帧间位移误差 median / P95 (m) | 二阶差分 median / P95 (m) | 方向角 median / P95 (°) | 方向翻转率 | 速度模长比 median | 速度异常率 |
|---|---|---|---|---|---|---|
| +10 | 0.133 / 0.365 | 0.218 / 0.615 | 0.34 / 1.55 | 0 | 1.0001 | 0 |
| 0 | 0.421 / 1.153 | 0.714 / 1.966 | 1.08 / 4.89 | 0 | 1.0005 | 0 |
| −10 | 1.334 / 3.646 | 2.287 / 6.243 | **3.43 / 15.4** | **0.0023%** | 1.0043 | 0.46% |

验收要求（§27）：方向翻转率 <1% ✓（实测 0.0023%）；方向角中位 <10° ✓（实测 3.43°）。轨迹图人工检查见 §3.6。

### 3.6 轨迹图（同一车辆、同一 20-frame 窗口，GT + 三个 SNR）

- `plots/trajectories_train_scene0.png`、`plots/trajectories_train_scene1.png`
- `plots/trajectories_val_scene0.png`、`plots/trajectories_val_scene1.png`

四联图：GT（黑实线）/ +10 dB / 0 dB / −10 dB（红虚线），+10 几乎贴合 GT，0 dB 可见偏差，−10 dB 明显抖动但运动方向与轨迹趋势保持。

## 4. Phase D — FREEZE

`configs/automatum_controlled_isac.json`：

- `revision = AUTOMATUM-CONTROLLED-ISAC-V2-FROZEN`，`status = FROZEN`
- 冻结内容（详见 `freeze_manifest.json`）：Route B 链路与假设、scene 0/1 几何、visibility（5–150 m、±70°、Δh=5 m）、SNR 档位与语义、sigma 公式（k=1）、`a = (0.35, 0.0111, 0.30)`、floors、velocity fusion（弱先验 σ=200 m/s + 3D LOS 方向）、帧间噪声独立、`state_dim=4`、`vehicle_id` 仅内部索引。
- `frontend/controlled_isac/state_quality.py` 顶部注释已标记 **"LEGACY descriptive score; not used for calibration or evaluation"**；正式指标只用 position/velocity RMSE 与 Good/Medium/Poor-but-usable 语义。
- 冻结清单含各证据文件的 SHA256；**本轮不生成 train/val/test 全量 sensing cache**（等独立审核后另开 production-cache 工单）。

## 5. Self-check 与确定性

- selfcheck（`experiments/automatum_controlled_isac_final/selfcheck.py`）：**20/20 PASS**（baseline、scene 1 未改、几何/标定只用 train、val 只验证、test 未用、dt、vx/vy、SNR 档位、seed 不含 SNR、同 base noise、无 smoothing、无 detector、无直接状态加噪、误差来自 range/bearing/vr、state=[x,y,vx,vy]、无特征泄漏、五档单调、确定性、旧审计文件未改）。
- 确定性：geometry 搜索、calibration、final acceptance 各自连续两次运行，全部输出文件 SHA256 逐位一致。
- 声明：**test 未使用；samples.npz / split / canonical 数据未修改；未生成 downstream cache；frontend 数值路径仅改动 `state_quality.py` 的注释（无数值变化）**。

## 6. §37 交付回答

| 项 | 值 |
|---|---|
| branch / commit | `module/isac`（见提交） |
| scene 0 geometry | 仅 BS2 boresight 79.551839° → 64.551839°；坐标/FOV 不变 |
| train 0/1/2/3-BS | 0 / 0 / 19,153 / 30,612（P≥2 = 100%） |
| val 0/1/2/3-BS | 0 / 0 / 2,238 / 2,541（P≥2 = 100%） |
| sigma 公式 | `sqrt(floor² + (a·10^(−γ/20))²)`，k=1 |
| a_range / a_bearing / a_vr | 0.35 m / 0.0111 rad / 0.30 m/s |
| k_q ≠ 1 | 否（全部 k=1） |
| +10/+5/0/−5/−10 position RMSE (train) | 0.131 / 0.233 / 0.414 / 0.736 / 1.309 m |
| +10/+5/0/−5/−10 velocity RMSE (train) | 0.159 / 0.283 / 0.503 / 0.894 / 1.591 m/s |
| +10/0/−10 position RMSE (val) | 0.143 / 0.453 / 1.434 m |
| +10/0/−10 velocity RMSE (val) | 0.175 / 0.553 / 1.748 m/s |
| P95/P99 tail | train −10：pos P95 2.51 / P99 3.67 m；vel P95 2.99 / P99 4.52 m/s |
| 方向角 / 翻转 | −10 dB median 3.43°，翻转率 0.0023% |
| target bands | **6/6 PASS** |
| Route-B FREEZE | **YES** |
| config revision/status | `AUTOMATUM-CONTROLLED-ISAC-V2-FROZEN` / `FROZEN` |
| test used / samples modified / frontend architecture changed | NO / NO / NO（仅注释） |
| report / metrics / plots | `reports/isac_final/final_report.md`；`reports/isac_final/{geometry_search,calibration_search,final_metrics,freeze_manifest}.json`、`{geometry_final_audit,calibration_candidates,final_metrics,final_metrics_by_bs,final_speed_bins,final_stability}.csv`；`reports/isac_final/plots/` |

## 7. 复现命令

```bash
python experiments/automatum_controlled_isac_final/optimize_geometry.py
python experiments/automatum_controlled_isac_final/calibrate_final.py
python experiments/automatum_controlled_isac_final/evaluate_final.py
python experiments/automatum_controlled_isac_final/run_final.py freeze
python experiments/automatum_controlled_isac_final/selfcheck.py
```