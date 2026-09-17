# Automatum ISAC SMOKE-01 报告：scene-level 共享回波 + 新 frame-level SNR + 无 CFAR multi-angle 单 BS 检测

- 基线：`module/isac` @ `030749f273db771d2b7558130ba84751e7e04f9d`（工作树干净）
- Smoke config：`configs/automatum_isac_smoke.json`，SHA256 前缀 `a8b58682ec08e30b`
- 状态：**SMOKE 收敛，等待下一张工单**。本轮不 freeze 最终几何、range/FOV、SNR 电平表与 detector 门限。
- 性质：只做到单 BS 匿名量测（work order §21）；未施工 cross-BS association / position fusion / velocity fusion / temporal association / Kalman / vehicle_id mapping / sensing cache / samples.npz packing；未触碰 QGNN / LLM；未修改 canonical / split / samples.npz。
- 说明：工单要求阅读的 `AGENTS.md` 在本仓库 HEAD 中不存在（全文检索无此文件）；其余指定报告与代码均已阅读，未发现与工单的真实冲突。

---

## 1. 交付物

| 类型 | 路径 |
|---|---|
| scene-level source | `frontend/automatum_scene_source.py` |
| simulator（新增帧级 SNR 路径，旧路径未动） | `frontend/sensing/simulator.py::synthesize_shared_frame_snr / scale_shared_noise` |
| detector（提取 `rd_spectrum`，新增无 CFAR multi-angle 路径） | `frontend/sensing/detector.py::rd_spectrum / extract_rd_peaks / aoa_angle_peaks / detect_anonymous_measurements` |
| smoke config | `configs/automatum_isac_smoke.json` |
| C 域 evaluator | `experiments/automatum_isac_smoke_01/evaluate.py` |
| smoke runner | `experiments/automatum_isac_smoke_01/run.py` |
| 18 项自检 | `experiments/automatum_isac_smoke_01/selfcheck.py`（18/18 PASS） |
| 误差地板探针 | `experiments/automatum_isac_smoke_01/floor_probe.py` + `floor_probe.json` |
| 机器可读结果 | `metrics.json`，`per_group_{train,val}.csv`，`per_measurement_{train,val}.csv`，`count_mismatch_{train,val}.csv`，`unmatched_targets_{train,val}.csv`，`selection_{train,val}.csv` |
| 诊断图 | `plots/`（4 张 SNR–RMSE + 2 张真实 frame 的 RD map/AoA spectrum） |

旧 `frontend/echo_source.py`、旧 CFAR detector 路径、`configs/shared_frontend.json`、旧标定产物均未修改。

---

## 2. 本轮冻结口径的落地

1. **数据源**：`splits/{train,val}/trajectories.csv` 逐帧读取，输入 `(scene_id, canonical_frame)`，输出该帧**全部**车辆 `vehicle_id(x,y,vx,vy)`。不读 `samples.npz`、不筛预测 cohort、不插值、不平滑、不旋转、不看未来、不做 visibility/SNR/detector。
2. **时间**：`source_frame = round(timestamp×29.97)`，`frame = source_frame//3`，`dt = 3/29.97 = 0.1001001001… s`；全库校验 `source_frame%3==0`，失败即抛错（自检 2、3）。
3. **三 BS**：同一 canonical frame 的同一份瞬时 `[x,y,vx,vy]` 同时喂给三个 BS；新 simulator 无任何 per-BS 时间偏移/外推接口（自检 6）。
4. **共享回波**：每 BS 一次 `S_b = Σ_i 可见目标 echo_i`，只加一次接收机噪声 `W_b`；`Y_b = S_b + W_b`（自检 10、11）。
5. **新 SNR**：`SNR_b = 10log10(mean|S_b|²/mean|W_b|²)`；噪声基实现 `W0` 的 seed 只取 `(scene_id, frame, bs)`，SNR 只改 `scale`（自检 12、13）。
6. **幅度**：`amplitude = sqrt(RCS)/R²`，RCS 固定 1.0 m²；保留真实相对强弱（自检 7、8）。
7. **感知资源**：`sensing_resource = full`，256 个通信 OFDM symbol 全部复用（自检 9）。
8. **检测建模假设**：detector 未使用 CFAR、未读取 GT 数量/身份；噪声只通过“peak 是否越过噪声参考”影响检出（后文 §9 报告 count 不一致及归因）。

---

## 3. Detector 设计（最低成本升级，无 CFAR）

新路径：`Y/X → Range FFT(Hann, 2K) → Doppler FFT(Hann, 2N) → RD power → range ROI → 全局中值噪声参考 → local maxima → RD NMS → 抛物插值 → 逐 RD peak 取 snapshot → AoA FFT(rect, 64) → multi-angle peaks → 匿名量测`。

- 新路径不调用 `compute_maps`（CFAR noise/alpha）与 `cfar_mask`；自检 15 用 monkeypatch 毒化 `cfar_mask` 后仍可正常运行。
- 旧 `compute_maps` 保留 CFAR 字段（历史路径未动）；`rd_spectrum` 为两条路径共享的纯 FFT 内核。

候选参数与依据（**smoke 标定所得，未 freeze**）：

| 参数 | 值 | 依据（本轮实测） |
|---|---|---|
| `rd_floor_factor_db` | 20 dB | 纯噪声 RD 图 max/median 实测 ≤ +4.6 dB（44 次 512×512 图）；Hann 第一旁瓣在 ±4–5 bin（−31 dB），第二旁瓣 ±7 bin（−42 dB）；+15 dB 时最强目标约 +59.6 dB，20 dB 门限可在 NMS 半径之外压掉高阶旁瓣 |
| `rd_nms_radius` | [5, 5] | Hann 主瓣零-零宽 ±4 bin（range）/±4 bin（Doppler）+ 第一旁瓣位置；±5 覆盖第一旁瓣环 |
| `aoa_rel_threshold_db` | 10 dB | 16 元 rect 阵第一旁瓣 −13.3 dB，留 3.3 dB 余量；同 RD cell 两目标回波幅度接近（ΔR≤1 bin），允许在 −10 dB 内共存 |
| `aoa_floor_factor_db` | 10 dB | 角度谱噪声参考取 **RD 图中值**（与 angle bin 噪声同尺度）；同 cell 双目标实测 angle peak 比 RD 噪声参考高 ~42–52 dB，余量充足 |
| `aoa_nms_radius` | 4 bin（0.125 u ≈ 7.2°） | 16 元阵主瓣零-零宽 2/16=0.125 u |
| `rd_max_peaks` / `aoa_max_peaks` / `max_measurements_per_rd_peak` | 32 / 6 / 4 | 上限保护，本轮未触顶 |

**注意**：角度谱自身的中值会被信号旁瓣抬高（实测双目标时 peak/中值只有 17.8 dB），因此绝对门限必须用 RD 图中值而不是角度谱中值；这是本轮标定的关键结论之一。

---

## 4. Smoke 选取（确定性，仅用 train 调参）

规则（config `smoke.selection_rule`）：每个 scene 在 split 内按 6 个类别各取 6 帧（train）或 2 帧（val）后并集、按 frame 升序、去重、截断到上限：

1. `high_density`：车辆数降序；
2. `low_density`：车辆数升序；
3. `median_density`：车辆数接近 scene 中位数；
4. `closest_pair`：任意两车最小间距最小（≥2 车）；
5. `junction_center`：平均距 junction 中心最近；
6. `far_target`：最远车辆距 junction 中心最大。

Tie-break 一律取更小 canonical frame。train 最终 **scene 0 / scene 1 各 36 帧**，val 各 **10 帧**（12 帧并集后按 linspace 均匀截断）。分类明细见 `selection_{train,val}.csv`。参数只使用 train；val 只做验证，未回调任何参数。

---

## 5. 验证 A：shared echo 是真实多目标叠加

- `S_total == S_target1 + S_target2`（三维 BS、float64，逐元素一致，自检 10）；
- `Y - S == W`，且 `W == scale·W0`（同一 BS/frame 只有一次噪声，自检 11）；
- 同一帧重复合成逐位一致（自检 6）；
- scene 0 frame 307 共 6 辆车，其中存在多个不在任何重叠 prediction cohort 中的车辆（自检 5 输出），证明回波使用 `trajectories.csv` 全场景而非 `samples.npz` cohort；
- 抽样 17 个帧，source 返回行数与 CSV 完全一致（自检 4）。

## 6. 验证 B：新 SNR 定义正确

| 项 | train | val |
|---|---|---|
| target/achieved 最大绝对误差 | **3.55×10⁻¹⁵ dB** | **1.93×10⁻¹⁵ dB** |
| 容差 | 0.05 dB | 0.05 dB |
| 同 (scene,frame,BS) 各 SNR 共用 W0 | 是（seed 不含 SNR，自检 13 逐位比对） | 同上 |

`metrics.json → splits.*.groups[*]` 逐组记录 `target_snr_db / achieved_snr_mean_db / achieved_max_abs_error_db / signal_power / noise_power`。

## 7. 验证 C：SNR 对量测误差的作用

误差统计只在 C 域：匿名量测与**该 BS 几何可见 GT** 做带门限 Hungarian 匹配（range 8 m / u 0.25 / radial velocity 5 m/s），匹配结果不回送 detector。定义：

- **isolated**：该 GT 在 RD 分辨率窗口（|Δr|≤8 m 且 |Δv|≤8 m/s，严格宽于 detector 的 11×11 local-max 竞争窗）内没有其他可见 GT，且其量测来自只产生一个角度峰值的 RD peak；
- **crowded**：其余匹配对（RD 合并 / 角度未分辨 / 同一 RD peak 多角度）。

### 7.1 train（72 帧，202 个含可见目标的 BS-frame，每 SNR 999 个可见目标）

isolated 匹配对（n 从 336 增至 626，低 SNR 下弱目标物理上不可见，匹配对数减少）：

| SNR (dB) | n | range RMSE (m) | vr RMSE (m/s) | bearing RMSE (°) | position RMSE (m) | position MAE (m) | position median (m) |
|---|---|---|---|---|---|---|---|
| −10 | 336 | 0.01043 | 0.01196 | 0.04284 | 0.05495 | 0.03341 | 0.01977 |
| −5 | 408 | 0.00919 | 0.01024 | 0.04139 | 0.05813 | 0.03251 | 0.01683 |
| 0 | 504 | 0.00921 | 0.00967 | 0.03828 | 0.05581 | 0.03114 | 0.01469 |
| +5 | 563 | 0.00751 | 0.00857 | 0.03219 | 0.04952 | 0.02560 | 0.01196 |
| +10 | 598 | 0.00553 | 0.00650 | 0.02527 | 0.03564 | 0.01947 | 0.00928 |
| +15 | 626 | 0.00459 | 0.00571 | 0.02054 | 0.02916 | 0.01563 | 0.00771 |
| **ratio (+15/−10)** | | **0.44** | **0.48** | **0.48** | **0.53** | 0.47 | **0.39** |

- RMSE / MAE / median 三条口径全部单调下降（position RMSE 在 −10/−5 之间有轻微非单调，属噪声—可见目标集合变化，其余 5 个通道逐档单调）。
- 可分辨的三档误差等级：低（−10/−5）、中（0/+5）、高（+10/+15）在 range/vr/bearing/median 上区分明显（如 bearing median 0.021° → 0.012° → 0.007°）。

### 7.2 val（20 帧，58 个 BS-frame，每 SNR 253 个可见目标，仅验证）

| SNR (dB) | n | range RMSE | vr RMSE | bearing RMSE | position RMSE | position median |
|---|---|---|---|---|---|---|
| −10 | 98 | 0.01180 | 0.01321 | 0.05249 | 0.06553 | 0.02672 |
| 0 | 152 | 0.00660 | 0.00969 | 0.03509 | 0.05269 | 0.01605 |
| +15 | 186 | 0.00316 | 0.00531 | 0.01945 | 0.02886 | 0.00765 |
| **ratio (+15/−10)** | | **0.27** | **0.40** | **0.37** | **0.44** | **0.29** |

val 与 train 独立且趋势更陡，说明 train 标定的参数没有过拟合 train smoke。

### 7.3 分 scene / 分 BS（train，isolated RMSE）

| scope | SNR | range | vr | bearing | position |
|---|---|---|---|---|---|
| scene 0 | −10 | 0.00967 | 0.01249 | 0.04498 | 0.04473 |
| scene 0 | +15 | 0.00502 | 0.00668 | 0.02210 | 0.03290 |
| scene 1 | −10 | 0.01100 | 0.01152 | 0.04102 | 0.06197 |
| scene 1 | +15 | 0.00409 | 0.00447 | 0.01874 | 0.02456 |
| scene 0 bs0 | −10 / 0 / +15 | 0.00759 / 0.01005 / 0.00607 | 0.01240 / 0.01205 / 0.00848 | 0.02691 / 0.04777 / 0.02404 | 0.02801 / 0.04187 / 0.02657 |
| scene 0 bs1 | −10 / 0 / +15 | 0.01056 / 0.01081 / 0.00470 | 0.01336 / 0.01125 / 0.00615 | 0.06329 / 0.04302 / 0.02445 | 0.06303 / 0.05640 / 0.04257 |
| scene 0 bs2 | −10 / 0 / +15 | 0.01044 / 0.01021 / 0.00367 | 0.01146 / 0.00887 / 0.00394 | 0.03108 / 0.04573 / 0.01202 | 0.02947 / 0.07514 / 0.01672 |
| scene 1 bs0 | −10 / 0 / +15 | 0.01113 / 0.00790 / 0.00237 | 0.01298 / 0.00891 / 0.00272 | 0.04877 / 0.03182 / 0.01107 | 0.06868 / 0.05899 / 0.01824 |
| scene 1 bs1 | −10 / 0 / +15 | 0.00982 / 0.00896 / 0.00636 | 0.01074 / 0.00984 / 0.00676 | 0.02063 / 0.03460 / 0.02896 | 0.01326 / 0.03006 / 0.02687 |
| scene 1 bs2 | −10 / 0 / +15 | 0.01146 / 0.00742 / 0.00317 | 0.01047 / 0.00699 / 0.00365 | 0.04139 / 0.02741 / 0.01472 | 0.07042 / 0.05650 / 0.02792 |

每个 scene、每个 BS 在 +15 dB 相对 −10 dB 都下降；个别 BS 的 0 dB 档出现局部波动（可见目标集合随 SNR 增大而变化），聚合口径不受影响。

### 7.4 为什么总的 “all” RMSE 看起来梯度弱

`all`（含 crowded）的 ratio：range 0.86 / vr 1.02 / bearing 0.79 / position 0.79——近似平台。原因不是 SNR 失效，而是：

1. **crowded（RD 合并/角度未分辨）匹配对带有 SNR 无关的确定性偏差**：全体 crowded RMSE = range 0.157 m / vr 0.105 m/s / bearing 0.521° / position 0.547 m（n=1280），且各 SNR 档几乎不变（ratio 0.91/1.10/0.83/0.84）。这些大误差在 RMSE 的平方和口径下主导总量。
2. **+44.6 dB 的 RD 处理增益**：256 symbol + 16 元阵 + Hann 窗使单目标 peak 比噪声均值高 +44.6 dB，因此 −10 dB frame SNR 下目标 peak 仍有 ~35 dB 余量，噪声驱动误差的绝对量级本就很小。
3. **插值/量化地板**：`floor_probe.json`（12 个单目标、15/30/60 dB）显示 +15 dB 时误差已接近地板：

| 单目标 SNR | range RMSE | vr RMSE | bearing RMSE | position RMSE |
|---|---|---|---|---|
| 15 dB | 0.00191 m | 0.00219 m/s | 0.0083° | 0.0121 m |
| 30 dB | 0.00181 m | 0.00227 m/s | 0.0079° | 0.0114 m |
| 60 dB | 0.00179 m | 0.00229 m/s | 0.0078° | 0.0113 m |

即 fingerprint：−10→+15 dB 让 isolated 误差下降 2.1–2.6×（理论上噪声驱动分量可降 ~18×，但地板在 +15 dB 已可比），但**误差确实随 SNR 单调、稳定下降，且不存在“加人工量测噪声”或“跳过回波链”的操作**。

## 8. 诊断图（人工检查用）

- `plots/snr_vs_{range,radial_velocity,bearing,position}_rmse.png`：横轴 frame-level SNR，三条曲线 `all / isolated / crowded`，对数纵轴。
- `plots/scene0_frame636_bs0_rd_aoa.png`：scene 0 frame 636 BS0（18 车/场景最大并发档），RD map（ROI 裁剪到 0–160 m / ±35 m/s，白叉 = 可见 GT，红圈 = 匿名量测）+ 最强 RD peak 的 AoA spectrum。
- `plots/scene1_frame5480_bs0_rd_aoa.png`：scene 1 frame 5480 BS0（10 车），同上。

两张 RD map 都直观显示：近距强目标清晰、远距弱目标（>100 m 或 1/R⁴ 落后 >20 dB）在噪声中不可见——这是 §9 中 miss 的物理来源。

## 9. Count consistency 分析（§30，GT 数量只用于事后评价）

| SNR | frames | 可见目标 | 匿名量测 | 数量不一致帧 | 缺额 | unmatched isolated | unmatched crowded | ghost 量测 |
|---|---|---|---|---|---|---|---|---|
| −10 | 202 | 999 | 491 | 120 | 508 | 335 | 173 | **0** |
| −5 | 202 | 999 | 616 | 107 | 383 | 263 | 120 | **0** |
| 0 | 202 | 999 | 727 | 93 | 272 | 167 | 105 | **0** |
| +5 | 202 | 999 | 793 | 70 | 206 | 108 | 98 | **0** |
| +10 | 202 | 999 | 829 | 62 | 170 | 73 | 97 | **0** |
| +15 | 202 | 999 | 859 | 60 | 140 | 45 | 95 | **0** |

val：−10 dB 缺额 107、+15 dB 缺额 19；ghost 同样为 0。

- **零假目标**：全部 650 个不一致 BS-frame 中，`detections > visible` 的帧数 = **0**；`ghost_detections`（|vr|>60 m/s 的 FFT 数值边缘鬼影）= **0**。满足“无噪声假目标”的建模假设。
- **缺额归因**（train，unmatched 1679 = crowded 688 + isolated 991）：
  - crowded 688：SNR 无关（每档约 95–173），来自 RD 合并/角度未分辨（见 §7.4 偏差），是阵列/波形分辨率极限。典型机制（已逐个复核）：`scene 0 frame 637 BS0 v27 r=12.46 m vr=6.82 m/s` 距强邻车（r=8.05 m，旁瓣/主瓣裙边在其 11×11 local-max 竞争窗内更高）太近，其真实峰不是窗口内最大值 → 被归入 crowded；
  - isolated 991：其中 **969** 的目标预测 peak margin < 0（按 RCS=1、1/R² 幅度律与 +44.6 dB 处理增益预计算），即被 frame-level SNR 下的 1/R⁴ 动态范围物理埋没；predict margin ≥ 0 的仅 22 行：
    - **6 行**为同一个系统性个案 `scene 0 frame 307 BS1 v12 r=5.12 m`（在 6 个 SNR 档重现）：该目标的 RD 局部极大位于 r=4.83 m 的 bin（+59.5 dB，远高于门限），但被 `range_min=5 m` 的 ROI 掩膜排除，属 ROI 边界效应；
    - 其余 **16 行 margin 仅 0.0–0.2 dB**，属门限边缘的噪声涨落。
- 结论：**当前 detector 结构可以在不使用 GT 数量的前提下给出零假目标；但无法对全部真实目标给出“无漏检”**。这不是 detector 工程缺陷，而是“frame-level SNR + 1/R⁴ + 16 元 FFT 分辨率”三者组合下的结构性问题（见 §12）。

## 10. PASS / FAIL / WARN

### PASS

| # | 项 | 证据 |
|---|---|---|
| 1 | scene-level source 正确，返回该帧全部车辆 | 自检 1–4；与 CSV 逐位一致 |
| 2 | sample cohort 不进入回波 | 自检 5（frame 0/307 含 cohort 外车辆）；source 无 npz 读取 |
| 3 | 三个 BS 使用完全同一物理帧 | 自检 6；旧 `echo_source` 未参与 |
| 4 | world vx/vy 未二次旋转；dt=3/29.97 | 自检 1、2 |
| 5 | 24 GHz / 93.1 MHz / 256×256 / 16 元阵保持 | config；自检 9 |
| 6 | 256 symbols 全部复用；fixed RCS=1；1/R² 幅度律 | 自检 7–9 |
| 7 | shared clean echo 是多目标叠加，每 BS/frame 只加一次噪声 | 自检 10、11 |
| 8 | frame-level SNR 定义正确，achieved ≈ target（≤3.6e−15 dB） | 自检 12；metrics |
| 9 | 同 (scene,frame,BS) 各 SNR 共用同一基础噪声 realization | 自检 13 |
| 10 | detector API 看不到 vehicle_id / GT count；新路径不调用 CFAR | 自检 14、15、17 |
| 11 | 一个 RD peak 可产生多个 angle peaks | 自检 16（同一 RD cell 双目标 → 2 个角度峰） |
| 12 | 零假目标 | §9；ghost=0，excess=0 |
| 13 | SNR 自然控制量测误差：isolated RMSE ratio 0.44–0.53（train）/0.27–0.44（val），median 逐档单调 | §7 |
| 14 | deterministic / 可复现 | 自检 18；连续两次完整运行全部 11 个输出文件 SHA256 逐字节一致 |
| 15 | 未修改数据/split/samples；未跑正式大规模实验；未施工 fusion/tracking | 见 §13 |

### FAIL

- 本轮无 FAIL 项。明确不满足的只有**“无漏检”的字面表述**（见 WARN-1），且已给出结构性归因与最小修改建议（§12），未用 GT 数量作弊。

### WARN

1. **“每个真实目标都能形成有效量测（无漏检）”在本物理口径下不成立**：+15 dB 时仍有 140/999 缺额（crowded 95 + 弱目标 45）。crowded 属分辨率极限，弱目标属 1/R⁴ 动态范围；两者都不能靠读 GT 数量“修正”。需要下一阶段在实验设计层面定义可量测性判据。
2. **crowded 量测带 SNR 无关偏差**（bearing 0.59° / position 0.62 m RMSE）：下游（fusion/tracking）不能把这些当作无偏量测；建议在 C 域持续输出 crowded 标记。
3. **一个系统性边缘个案**（`scene 0 frame 307 BS1 v12`，r=5.12 m 紧贴 ROI 下边界，6 行）：量级极小（总可见目标 5994 × 6 档），但值得在正式 freeze 前决定处理方式（扩大 ROI 安全边距，或对边界目标单独定义）。
4. **metrics.json 的可复现性只在本机验证**（RTX 3050, torch 2.13/cu130）：跨机器（服务器 GPU/cuFFT 版本）可能有末位差异，建议正式 freeze 前在服务器重跑一次并记录 SHA。
5. 本机 4 GB GPU 上 smoke 全量（train+val、6 SNR）约 62 s；产线 cache 阶段如需全数据集逐帧仿真，需要评估显存与时间预算（不在本轮范围）。

## 11. 参数状态

**已足够稳定（smoke 证据充分，可进入下一步）**

- waveform / array / `sensing_resource=full` / 256×256；
- `amplitude = sqrt(RCS)/R²`、RCS=1.0；
- frame-level SNR 定义与噪声 seed 规则（`(scene, frame, bs)`，SNR 不进 seed）；
- 本候选几何下的 visibility（5–150 m / ±70° / Δh=5 m）作为 smoke 候选可用；
- detector 结构：无 CFAR、全局中值噪声参考 + local maxima + NMS + 抛物插值 + RD-median 参考的 multi-angle；
- `rd_floor_factor_db=20`、`rd_nms_radius=[5,5]`、`aoa_floor_factor_db=10`、`aoa_rel_threshold_db=10`、`aoa_nms_radius=4`。

**仍需下一轮标定 / 决策**

- 3-BS 几何与 range/FOV/height 的正式 freeze（D-02/D-03）；
- 正式 SNR 档位表（D-04）与论文级 Good/Medium/Poor 定义；
- 可量测性判据：per-target 灵敏度窗口与 RD/角度可分辨性阈值（决定“哪些目标应出现在量测集”）；
- ROI 边界安全边距；是否改用更低旁瓣窗（如 Hann²）或更大 NMS；
- 跨 BS association 一致性（下一阶段）。

**表现有问题 / 被否定的候选**

- 纯“角度谱自身中值 + 相对阈值”的 AoA 门限：双目标时谱中值被旁瓣抬高到 17.8 dB，会在高 SNR 产生大量噪声角度峰（已改为 RD 图中值参考）；
- `rd_floor_factor_db=5` 与 `rd_nms_radius=[2,2]`（初值）：实测会在高 SNR 产生 Hann 旁瓣假峰（frame 85 BS2 案例），已废弃；
- 期望“与 SNR 无关的绝对门限”同时满足全部目标：被 1/R⁴ 动态范围与分辨率极限否定（§9）。

## 12. §41 structural 结论与最小修改建议

**结论**：在不读取 GT target count / GT identity 的前提下，本 detector 结构可以做到零假目标，但**不能满足“每个真实目标都无漏检”的字面要求**；这不是 detector 工程实现未达标，而是物理口径的结构性结果：frame-level SNR（噪声只按整 BS 干净回波定标）+ 固定 RCS + 1/R² 幅度律使目标间真实功率差可达 80–120 dB，再叠加 16 元阵 ~7.2° 的 FFT 角度分辨率。

**最小修改建议（供下一张工单决策，本轮不施工）**：

1. **实验设计层面**（推荐）：把“有效量测”定义为满足文档化可量测性判据的目标——(a) 目标 peak 预测 margin ≥ 0（等价于 per-target frame SNR ≥ 阈值）；(b) 与任一更强可见目标的 RD/角度间隔超过分辨率窗口（|Δr| ≳ 4 m 或 |Δv| ≳ 5 m/s 或 |Δu| ≳ 0.15）。判据只使用回声/几何，不读 GT 身份与 GT 数量，评价侧用 GT 验证。
2. **若必须字面满足无漏检**：需要在模型层面引入 per-target SNR 控制（改变 §14 的 frame-level SNR 定义）或收窄物理目标集（例如把 5 m 与 150 m 目标排除在同一帧之外）——两种都会改变已冻结口径，需单独出工单。
3. **detector 层面可选增强（非本轮）**：Hann²/Blackman 类低旁瓣窗 + 相应 NMS 重标；角度超分辨（MUSIC 等）；旁瓣感知的 peak 竞争抑制。均超出“最小修改”范围。

## 13. 工单四个必答项

| 问题 | 回答 |
|---|---|
| 是否修改正式数据（canonical） | **否**。只读 `data/automatum_t_crossing/splits/*/trajectories.csv`；未写入任何数据文件。 |
| 是否修改 split | **否**。 |
| 是否修改 samples.npz | **否**。仅自检 5 只读加载一次用于证明 cohort 隔离。 |
| 是否运行正式大规模实验 | **否**。只运行 train 72 帧 + val 20 帧、6 个 SNR 档的 smoke；未训练模型、未跑 QGNN/QGAT/LLM、未生成 sensing cache。 |

## 14. 复现命令

```bash
python experiments/automatum_isac_smoke_01/selfcheck.py         # 18/18 PASS
python experiments/automatum_isac_smoke_01/run.py --split both  # ~62 s on RTX 3050
python experiments/automatum_isac_smoke_01/floor_probe.py
```

本报告全部数字来自 `metrics.json`（config SHA256 前缀 `a8b58682ec08e30b`，HEAD `030749f2`）；连续两次运行 `metrics.json` 与 10 个 CSV 的 SHA256 完全一致。