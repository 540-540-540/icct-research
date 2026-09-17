# Automatum Route B — Controlled ISAC Calibration V1 报告

- 基线：`module/isac` @ `92e8c88ebea67181c6d3c188e5f5ef554a498873`（本轮开始前工作树干净；旧 SMOKE-01B WIP 已 stash，见附录 A）
- 性质：**Calibration V1**。只使用 train 做参数搜索，val 只验证，**test 完全未使用**；不生成全量 sensing cache，不改写 samples.npz，未 freeze 任何参数。
- 正式路线：**Route B（受控多基站 ISAC 状态感知）**。论文假设：目标检测与身份关联均已正确完成；SNR 只通过 `range / bearing / radial-velocity` 量测误差影响最终 `[x_hat, y_hat, vx_hat, vy_hat]`。主线不模拟 detector、CFAR、漏检、假警、anonymous track。
- 主要交付：`configs/automatum_controlled_isac.json`；`frontend/controlled_isac/automatum_measurement.py`、`automatum_frontend.py`、`state_quality.py`；`experiments/automatum_controlled_isac_calib_v1/{run,calibrate,evaluate,selfcheck,stability}.py`。

---

## 1. 本轮链路与数学约定

```text
Automatum split trajectories.csv
→ scene_id + canonical frame → 该帧全部车辆 [x,y,vx,vy]（复用 automatum_scene_source，dt=3/29.97，世界系速度不再旋转）
→ 每个几何可见 BS 计算真实 r / bearing / vr（复用 frontend/controlled_isac/measurement.py 的已验证公式：
   3D range r=sqrt(rho²+Δh²)，bearing 相对 boresight，径向速度 dot(station-position, velocity)/r 朝 BS 为正）
→ 按 SNR 加独立高斯量测噪声：[r_hat, bearing_hat, vr_hat] → [x_hat_bs, y_hat_bs]
→ covariance-weighted position fusion（复用 cross_bs_association.fuse_positions 的数学，不含 anonymous association）
→ 多 BS radial-velocity fusion（复用 velocity_fusion.recover_velocity；3D LOS 方向作为最小兼容扩展）
→ vehicle_id + [x_hat, y_hat, vx_hat, vy_hat]
```

- 每个 GT `vehicle_id` 始终对应一个 sensing 状态（无检测/关联失败）；`vehicle_id` 只是索引，不是模型特征。
- 无 Kalman、无 temporal smoothing、无 per-target 调参、无直接状态加噪。
- 单帧即输出当前估计；SNR 误差不会被时间平滑洗掉。

## 2. 噪声模型与 SNR 规则

```text
sigma_q(gamma) = sqrt( floor_q² + (a_q · 10^(-gamma/20))² ),  q ∈ {range, bearing, radial_velocity}
```

- 基础标准正态 `z_q` 由 `(scene_id, frame, bs_id, vehicle_id, channel)` 的确定性 seed 生成，**SNR 不进入 seed**；同一 `(scene, frame, BS, vehicle)` 的 5 个 SNR 档复用完全相同 realization，只改误差强度。
- `floor_q` 来源：SMOKE-01 `floor_probe.json` 高 SNR 单 BS estimator floor（60 dB 档）：
  `floor_range = 0.0018 m`，`floor_bearing = 1.36e-4 rad (0.0078°)`，`floor_vr = 0.0023 m/s`。
  这是估计器地板量级的参考，**不是硬件测量值**；Route B 是可解释的受控仿真模型。
- V1 不加 range/speed/vehicle/scene 依赖（距离影响由几何自然产生）。
- 三个 BS 的噪声独立，同一 BS 不同车辆的噪声独立，不同帧独立。

## 3. 几何覆盖审计（§16）

对 SMOKE-01 相同 train/val 帧内的每个真实目标状态，统计几何可见 BS 数（SNR 无关）：

| split | scope | n | 1 BS | 2 BS | 3 BS | P(≥1) | P(≥2) | P(=3) |
|---|---|---|---|---|---|---|---|---|
| train | overall | 425 | 30 | 216 | 179 | 100% | 92.9% | 42.1% |
| train | scene 0 | 270 | 30 | 121 | 119 | 100% | **88.9%** | 44.1% |
| train | scene 1 | 155 | 0 | 95 | 60 | 100% | 100% | 38.7% |
| train | prediction cohort | 274 | 23 | 124 | 127 | 100% | 91.6% | 46.4% |
| val | overall | 111 | 10 | 60 | 41 | 100% | 91.0% | 36.9% |
| val | scene 0 | 60 | 10 | 23 | 27 | 100% | **83.3%** | 45.0% |
| val | scene 1 | 51 | 0 | 37 | 14 | 100% | 100% | 27.5% |
| val | prediction cohort | 92 | 9 | 52 | 31 | 100% | 90.2% | 33.7% |

**结论 / 是否需要调整 geometry：需要。**
- 所有目标至少 1 个 BS 可见（无 0-BS 目标），因此每个车辆都能输出状态；
- 但 scene 0 有 10–17% 的目标状态只有 1 个 BS 可见，其二维速度不可观测（rank=1，只能给出径向分量 + 弱先验的退化估计）。该部分仍输出状态并标记 `rank / condition_ratio / n_bs`，**没有用 GT velocity 补**；
- scene 1 的 ≥2BS 覆盖为 100%。建议下一轮调整 scene 0 的 BS 位置 / boresight / FOV（本轮按工单要求不改 geometry，不 freeze）。

## 4. Calibration V1 方法

- 搜索目标（train，`n_bs≥2` 子集）：使 provisional state quality 满足 `Q(+10)≈90%`、`Q(-10)≈40–50%`，quality 随 SNR 单调，同时保持物理误差合理。
- 确定性 coarse grid（180 组）：`a_range ∈ {0.03…0.40}`、`a_bearing ∈ {0.001…0.016}`、`a_vr ∈ {0.03…0.40}`；
- 之后做 2 轮 coordinate refinement（每通道 ×{0.7, 0.85, 1.0, 1.15, 1.4}）；最优解未被 refinement 移动，说明 coarse 网格已落在局部最优。
- 目标函数：`|Q(-10)-45| + |Q(+10)-90| + 20×单调性违背数`；推荐解 score = 1.55，单调性违背 = 0。
- **推荐参数（NOT FROZEN）**：
  `a_range = 0.10 m`，`a_bearing = 0.004 rad`，`a_radial_velocity = 0.16 m/s`
  对应 `sigma(-10 dB) = 0.316 m / 0.725° / 0.506 m/s`；`sigma(+10 dB) = 0.032 m / 0.072° / 0.051 m/s`。
- 由推荐解派生 4 组候选：A = ×0.5（偏轻）、B = ×1（推荐）、C = ×2（偏重）、D = ×3（很重）。

## 5. 候选参数与主指标（train overall，`n_bs≥2`）

| 候选 | a_range / a_bearing / a_vr | SNR | position RMSE / MAE / med / P95 (m) | velocity RMSE / MAE / med / P95 (m/s) | vx RMSE | vy RMSE | quality mean/med |
|---|---|---|---|---|---|---|---|
| A_light | 0.05 / 0.002 / 0.08 | +10 | 0.023 / 0.019 / 0.017 / 0.045 | 0.048 / 0.039 / 0.033 / 0.094 | 0.037 | 0.030 | 95.6 / 96.2 |
| A_light | | 0 | 0.072 / 0.060 / 0.052 / 0.142 | 0.150 / 0.123 / 0.104 / 0.298 | 0.118 | 0.093 | 86.9 / 88.4 |
| A_light | | −10 | 0.226 / 0.189 / 0.163 / 0.447 | 0.475 / 0.388 / 0.330 / 0.939 | 0.373 | 0.294 | 65.5 / 67.7 |
| **B_recommended** | **0.10 / 0.004 / 0.16** | **+10** | **0.045 / 0.038 / 0.033 / 0.090** | **0.095 / 0.078 / 0.066 / 0.188** | 0.075 | 0.059 | **91.4 / 92.5** |
| **B_recommended** | | **0** | **0.143 / 0.120 / 0.104 / 0.283** | **0.300 / 0.245 / 0.208 / 0.594** | 0.236 | 0.186 | **76.0 / 78.1** |
| **B_recommended** | | **−10** | **0.453 / 0.379 / 0.323 / 0.894** | **0.949 / 0.776 / 0.658 / 1.876** | 0.745 | 0.588 | **45.1 / 45.8** |
| C_heavy | 0.20 / 0.008 / 0.32 | +10 | 0.091 / 0.076 / 0.066 / 0.179 | 0.190 / 0.155 / 0.131 / 0.376 | 0.149 | 0.118 | 83.9 / 85.5 |
| C_heavy | | 0 | 0.286 / 0.240 / 0.206 / 0.566 | 0.600 / 0.491 / 0.417 / 1.188 | 0.471 | 0.372 | 59.1 / 61.0 |
| C_heavy | | −10 | 0.905 / 0.757 / 0.651 / 1.784 | 1.898 / 1.552 / 1.309 / 3.743 | 1.491 | 1.175 | 23.5 / 21.3 |
| D_very_heavy | 0.30 / 0.012 / 0.48 | +10 | 0.136 / 0.114 / 0.098 / 0.268 | 0.285 / 0.233 / 0.197 / 0.564 | 0.224 | 0.176 | 77.1 / 79.1 |
| D_very_heavy | | −10 | 1.359 / 1.137 / 0.972 / 2.670 | 2.847 / 2.328 / 1.952 / 5.600 | 2.236 | 1.762 | 13.4 / 9.9 |

（完整 5 档 SNR × overall/scene0/scene1 × all/ge2/eq3 × train/val ⇒ `candidate_metrics.csv`。）

### 推荐候选 B 的完整 SNR 梯度

| SNR | quality (train / val) | pos RMSE train / val | vel RMSE train / val |
|---|---|---|---|
| +10 | 91.4 / 91.2 | 0.045 / 0.051 | 0.095 / 0.100 |
| +5 | 85.5 / 85.1 | 0.081 / 0.090 | 0.169 / 0.178 |
| 0 | 76.0 / 75.5 | 0.143 / 0.160 | 0.300 / 0.316 |
| −5 | 62.4 / 61.8 | 0.255 / 0.284 | 0.534 / 0.562 |
| −10 | 45.1 / 44.5 | 0.453 / 0.505 | 0.949 / 1.001 |

- 五档单调，train/val 一致（val 仅验证，未参与搜索）；
- 全部为 `n_bs≥2` 子集；三 BS 子集更优（train +10：pos 0.030 / vel 0.070，Q=93.4；−10：pos 0.299 / vel 0.704，Q=52.9）。

### 分 scene（推荐候选 B，含 1-BS 目标的 `all` 口径）

| split/scope | +10 pos / vel | 0 pos / vel | −10 pos / vel | Q(+10) / Q(−10) |
|---|---|---|---|---|
| train scene 0 | 0.053 / 0.782 | 0.168 / 0.818 | 0.531 / 1.123 | 83.5 / 43.2 |
| train scene 1 | 0.053 / 0.109 | 0.167 / 0.345 | 0.527 / 1.092 | 90.4 / 41.7 |
| val scene 0 | 0.076 / 0.884 | 0.240 / 0.906 | 0.757 / 1.122 | 81.0 / 43.7 |
| val scene 1 | 0.059 / 0.123 | 0.186 / 0.388 | 0.588 / 1.228 | 89.4 / 38.7 |

- scene 0 的 `all` 口径速度指标被 1-BS 退化目标拉平（+10 仍有 0.78 m/s）：这是几何覆盖问题，不是 SNR 模型问题；`ge2` 口径下 scene 0 的 vel RMSE 为 0.095→0.949（train）。
- scene 1 无 1-BS 目标，`all` 口径即干净梯度。

## 6. State quality 候选（未 freeze）

```text
s = sqrt( (Δp / p_ref)² + (Δv / v_ref)² ),  p_ref = 1.0 m, v_ref = 1.0 m/s
quality = 100 · exp(-s)        （逐状态计算后按目标/帧平均）
```

- 无坐标原点奇异性，不使用 `|x_hat−x|/|x|` 类分量比值；
- `state_quality_mapping.csv` 给出 quality ↔ 物理误差对照（例如 Q=90% ⇒ 纯位置误差 0.105 m 或纯速度误差 0.105 m/s 或 (0.091 m, 0.053 m/s)；Q=45% 附近 ⇒ 组合误差 ≈0.8 单位）；
- 该定义与 `p_ref / v_ref` 均为 provisional candidate，最终百分比定义待 freeze；
- 主指标始终是 position RMSE [m] 与 velocity RMSE [m/s]（见 §5）。

## 7. 稳定性与可用性（推荐候选 B）

`stability.json`（基于逐状态输出，无 temporal smoothing）：

| SNR | 速度方向翻转比例 | 帧间位移误差 median / P95 / max (m) | 二阶差分误差 P95 / max (m) |
|---|---|---|---|
| +10 | 0.93% | 0.047 / 0.131 / 0.290 | 0.208 / 0.353 |
| 0 | 1.49% | 0.148 / 0.414 / 0.914 | 0.680 / 1.115 |
| −10 | 2.05% | 0.468 / 1.305 / 2.892 | 2.183 / 3.531 |

- −10 dB 下 2.9% 以下的帧间估计抖动属独立量测噪声的固有表现（每帧独立、不随时间平滑）；
- 速度方向翻转 ≤2.1%，无大面积方向翻转；轨迹整体方向与运动趋势保持；
- 轨迹图（同一批车辆、同一时间段，20 帧 ≈ 2 s）：
  `plots/trajectories_scene0_good_medium_poor.png`、`trajectories_scene1_good_medium_poor.png`
  （+10 dB sensing 与 GT 几乎重合，0 dB 出现可见小偏差，−10 dB 明显抖动但趋势清晰）。

## 8. PASS / FAIL / WARN

### PASS（对应工单 §33）

1. 每个 GT vehicle 始终有对应 sensing state（selfcheck 3/4；几何审计 0 个 0-BS 目标）；
2. 无检测/关联失败：输出车辆集合与 GT 完全一致，无假目标（selfcheck 5）；
3. `−10 → +10 dB` 最终状态误差清晰单调：pos RMSE 0.453→0.045 m，vel RMSE 0.949→0.095 m/s（ge2，train；val 0.505→0.051 / 1.001→0.100）；
4. 误差来自 `range/bearing/vr` 三条量测通道而非直接状态加噪（selfcheck 7/10/12；种子与 SNR 解耦）；
5. 3-BS fusion 正常：covariance-weighted position fusion（selfcheck 13）+ radial-velocity fusion（rank/condition，selfcheck 14）；
6. 无 Kalman、无 temporal smoothing、无 detector（selfcheck 6/17，代码路径检查）；
7. low-SNR 轨迹明显更差但仍可辨认、可用（§7 稳定性指标 + 轨迹图）；
8. 两个 scene 趋势一致（scene 0/1 均单调，scene 0 的 1-BS 部分单独披露）；
9. val 趋势与 train 一致（§5 表）；
10. 给出 4 组候选且未 freeze 任何一组。

### FAIL

- 无。

### WARN

1. **Geometry**：scene 0 有 10–17% 目标状态只有 1 个可见 BS，二维速度退化（rank=1）。本轮按要求不调整 geometry、不用 GT 补；建议下一轮微调 scene 0 的 BS 位置/boresight/FOV，并在正式 freeze 前把 `n_bs≥2` 覆盖率纳入验收。
2. **绝对 sigma 不是硬件标定值**：Route B 是受控模型；`a_q` 由本轮的 quality 语义目标标定得到，与 SMOKE-01 的共享回波物理链（处理增益 ~+44.6 dB、误差 cm 量级）不是同一数量级。报告明确区分：Route B 保留误差方向与物理变量，不声称绝对 sigma 真实。
3. **quality 定义未 freeze**：`p_ref=1 m / v_ref=1 m/s` 与 `100·exp(-s)` 是候选；换参考尺度会整体平移 quality 曲线，物理 RMSE 不受影响。
4. **速度融合的 3D LOS 修正**：`velocity_fusion.recover_velocity` 增加了可选 `member["direction"]` 扩展（旧调用行为不变，selfcheck 14 覆盖）；该修正在 Route B 中保证与 3D 径向速度约定自洽。

## 9. 推荐与 NOT FROZEN 声明

- **推荐 Candidate B_recommended**：`a_range=0.10 m, a_bearing=0.004 rad, a_vr=0.16 m/s`。
  理由：`Q(+10)=91.4%`、`Q(−10)=45.1%` 命中目标语义；五档单调且 train/val 一致；+10 状态平稳，−10 误差显著但轨迹可用；三通道量级（0.32 m / 0.73° / 0.51 m/s @ −10 dB）物理可解释。
- 备选：A_light（若后续认为 −10 应更易用，Q=65.5%）；C_heavy（若希望 −10 更差，Q=23.5%）；D 不建议（+10 也明显退化）。
- **本轮所有参数（a、floors、prior、quality 参考）均 NOT FROZEN**，等待审核后再进入 Production V1。

## 10. 复现命令

```bash
python experiments/automatum_controlled_isac_calib_v1/selfcheck.py   # 17/17 PASS
python experiments/automatum_controlled_isac_calib_v1/run.py         # ~150 s；两次运行全部输出逐位一致
python experiments/automatum_controlled_isac_calib_v1/stability.py
```

## 11. 工单必答项

| 问题 | 回答 |
|---|---|
| 是否修改正式数据 | **否** |
| 是否修改 split | **否** |
| 是否修改 samples.npz | **否**（几何/cohort 审计只读） |
| 是否使用 test | **否** |
| 是否运行 detector | **否** |
| 是否使用 temporal smoothing | **否** |

---

## 附录 A：旧 SMOKE-01B WIP 处理记录

按工单要求未删除、未混入本轮提交，已用 `git stash push -u -m "aborted SMOKE-01B before Route-B"` 保存（`stash@{0}`）。stash 内容清单：

```text
experiments/automatum_isac_smoke_01b/{common.py, diagnose.py, dynamic_range.py, processing_gain.py}
reports/isac_smoke/automatum_isac_smoke_01b/{metrics.json, coverage_by_snr.csv,
  common_targets_single_bs.csv, oracle_fusion_{2bs,3bs,targets}.csv,
  dynamic_range.{csv,json}, processing_gain.json,
  plots/{snr_vs_p_ge_2bs,distance_vs_peak_margin,dynamic_range_histogram}.png}
```

可复用结论（如需追溯）：`dynamic_range.json` 实测逐帧最强/最弱目标功率比 ≤55.9 dB（几何 1/R⁴ 理论上限 59.1 dB），`processing_gain.json` 独立验证 RD 处理增益 44.63 dB；SMOKE-01 已提交的 `floor_probe.json` 提供了本轮 floors。恢复方式：`git stash pop`（建议新开分支或工单后再操作）。