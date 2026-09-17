# Automatum → ISAC 正式适配前最终审计（AUDIT ONLY）

- 基线：`module/isac` @ `d017d49d688c0cc92294674b764d3813fa21b327`；canonical `a41ac4e2…3442`（校验通过）；split `d75994cb…/c9964adb…/6fb3117e…`（校验通过）。
- 性质：**只读审计**。未修改数据、split、样本、configs 或任何 ISAC 科研代码；未训练；未运行正式感知实验。
- 机器可读结果：`reports/isac_adaptation/automatum_isac_adaptation_audit.json`
- 审计脚本（只读，可复现）：`tools/isac_adaptation/audit_automatum_isac_adaptation.py`
- 数据口径：canonical vx/vy 已是**世界系**（车体系→psi 旋转已完成），ISAC 侧严禁再次旋转；真实 dt = 3/29.97 = 0.1001001001 s。

---

## 1. sample 车辆 vs 真实场景车辆（最高优先级）

对 train/val/test 的 **12,394 个正式 sample**，以 `trajectories.csv` 重建每帧真实在场车辆，逐 sample 核对 cohort（40/40 存在的 2–8 辆）与物理场景：

| | Train | Val | Test | 合计 |
|---|---|---|---|---|
| samples | 10,366 | 893 | 1,135 | 12,394 |
| **40 帧内完全没有额外车辆** | 3.93% | 2.91% | 1.59% | **3.64%** |
| **history 内至少 1 辆额外车辆** | 92.93% | 95.63% | 90.57% | **92.91%** |
| history 额外车辆 P50 / P90 / P95 / max | 2 / 4 / 4 / 6 | 3 / 4 / 5 / 6 | 2 / 3 / 3 / 4 | 2 / 4 / 4 / 6 |
| 40 帧内实际总并发 P50 / P90 / P95 / max | 7 / 9 / 10 / 14 | 6 / 11 / 12 / 12 | 8 / 9 / 10 / 10 | — |
| 出现实际并发 >8 的 sample 数 | 1,082 | 201 | 231 | **1,514（12.2%）** |
| 全部 sample 实际并发最大值 | 16 | 12 | 10 | 16 |
| future 额外车辆 P50 / P90 / P95 / max | 2 / 4 / 5 / 8 | 2 / 5 / 5 / 7 | 2 / 3 / 4 / 4 | — |

分 scene（samples / zero-extra / history≥1 extra / history 额外 P50-P90-P95-max / >8 并发数 / 场景最大并发）：

| split | scene 0 | scene 1 |
|---|---|---|
| train | 4,430 / 2.01% / 95.35% / 2-4-4-6 / 971 / 16 | 5,936 / 5.36% / 91.12% / 2-3-4-6 / 111 / 10 |
| val | 418 / 1.44% / 95.69% / 2-4-4-6 / 91 / 12 | 475 / 4.21% / 95.58% / 3-5-5-6 / 110 / 10 |
| test | 532 / 0.00% / 93.42% / 2-3-3-4 / 231 / 10 | 603 / 2.99% / 88.06% / 1-2-3-4 / 0 / 6 |

**结论：B。**
samples.npz 的 `vehicle_ids` 只是“完整 40/40 存在的预测 cohort”，**不是**物理场景全部车辆：96.4% 的窗口在 40 帧内存在额外真实车辆，92.9% 的 history 内即有额外车辆（中位 2 辆），12.2% 的 sample 窗口出现 >8 辆真实并发（最大 16）。若 ISAC 共享回波只使用 N 辆 cohort，会系统性删除临时出现/离场/进入车辆，破坏“真实共享回波”。**共享回波必须从 `trajectories.csv` 按 scene + canonical frame 重建完整场景**；`samples.npz` 只定义预测目标 slot 与标签。

---

## 2. 两个 scene 的正式场景信息

| | scene 0 Gaimersheim Stadtweg | scene 1 St2214 Dünzlau Umgehung |
|---|---|---|
| recording | `T-Crossing--GaimersheimStadtweg_e2e6-…` | `T-Crossing-St2214-DuenzlauUmgehung_1b9b-…` |
| canonical x 范围 | [−58.09, 76.13] m | [−29.38, 63.82] m |
| canonical y 范围 | [−49.24, 69.35] m | [−83.28, 83.76] m |
| vx 范围 | [−23.73, 18.62] m/s | [−18.71, 18.23] m/s |
| vy 范围 | [−15.72, 20.25] m/s | [−27.27, 25.76] m/s |
| speed P50 / P90 / P95 / P99 / max | 12.77 / 19.31 / 20.46 / 22.95 / 30.39 m/s | 17.95 / 23.91 / 25.14 / 27.64 / 30.66 m/s |
| 每帧车辆数 P50 / P90 / P95 / max | 6 / 10 / 12 / 18（空帧 19） | 4 / 6 / 7 / 10（空帧 988） |
| 轨迹点总数 | 41,944 | 39,506 |
| canonical 时间范围 | 0.0 – 650.651 s | 8.509 – 1125.626 s |
| staticWorld.xodr | `data/automatum_t_crossing/raw/extracted/T-Crossing--GaimersheimStadtweg_e2e6-…/staticWorld.xodr` | `data/automatum_t_crossing/raw/extracted/T-Crossing-St2214-DuenzlauUmgehung_1b9b-…/staticWorld.xodr` |
| junction 中心 | (−21.05, −2.95) m（审计 §14 冻结值） | (4.47, 26.76) m |

---

## 3. 3-BS 几何重新审计（CANDIDATE，未冻结）

旧 `reports/f01a/geometry.json` 是 Lankershim 直线走廊几何（stations (−64.2,−106.9)/(114.2,0)/(−64.2,106.9)、boresight 0/180/0°），**不能**用于 Automatum。本轮按“每条进近臂 1 个固定 BS + 非共线”原则，从 `staticWorld.xodr` 的三臂结构与路口中心重新生成：

**scene 0（三角形面积 1,815 m²，最小内角 55.9°）**

| BS | x, y (m) | boresight（指向路口中心） | 所属臂 |
|---|---|---|---|
| BS0 | (26.20, 7.62) | −167.4° | road 4（主干道 SE 侧） |
| BS1 | (−30.43, 33.09) | −75.4° | road 4（主干道 NW 侧） |
| BS2 | (−26.55, −32.76) | 79.6° | road 5（支路 S） |

**scene 1（面积 1,798 m²，最小内角 43.7°）**

| BS | x, y (m) | boresight | 所属臂 |
|---|---|---|---|
| BS0 | (42.98, 39.27) | −162.0° | road 4（主干道 NE） |
| BS1 | (26.94, −20.71) | 115.3° | road 4（主干道 SW） |
| BS2 | (−13.38, 52.72) | −55.5° | road 5（支路 N） |

车辆到最近 BS 距离（canonical 全部状态点，平面距离）：
scene 0 P50 21.8 / P90 41.0 / P95 53.2 / max 68.8 m；scene 1 P50 21.4 / P90 57.3 / P95 68.7 / max 81.9 m。

覆盖矩阵（range_min = 5 m，Δh = 5 m；boresight 指向路口中心；range_max × FOV half）：

| range_max / FOV | scene 0 ≥1 / ≥2 / 3 BS | scene 1 ≥1 / ≥2 / 3 BS |
|---|---|---|
| 100 m / 60° | 89.6% / 80.8% / 50.9% | 82.7% / 80.0% / 42.6% |
| 100 m / 70° | 89.6% / 87.2% / 54.3% | 82.7% / 80.0% / 48.1% |
| 100 m / 80° | 96.1% / 88.8% / 56.9% | 82.7% / 80.0% / 51.9% |
| 120 m / 70° | 97.6% / 88.2% / 54.3% | 92.7% / 91.7% / 48.1% |
| **150 m / 70°** | **100% / 88.2% / 54.3%** | **100% / 100% / 48.1%** |
| 150 m / 60° | 100% / 81.8% / 50.9% | 100% / 100% / 42.6% |

可见 BS 数量分布（推荐组合 150 m / 70°）：scene 0 每点可见 0/1/2/3 BS = 0 / 9.4% / 33.9% / 54.3%（均值 2.40）；scene 1 = 0 / 1.0% / 43.7% / 48.1%（均值 2.33）。

**推荐“待冻结候选”：**
`range_min = 5 m`，`range_max = 150 m`，`FOV half = 70°`，`station_height = 6.5 m`，`vehicle_height = 1.5 m`（高度差 5 m，兼容现有 `H_M=5.0` 与 simulator 常量）——标记 **CANDIDATE，不是本轮 freeze**。
理由：150 m 使两个路口的 ≥1-BS 覆盖都达到 100%（120 m 时 scene 1 仍有 7.3% 点无覆盖），≥2-BS 覆盖 88.2%/100%，再增大 range 不再提升；FOV 70° 维持现有值；几何按道路结构确定，未使用 val/test 指标优化。

---

## 4. 当前 ISAC 代码与新数据的逐文件不兼容点

9 个已知疑点全部核实（行号为当前 HEAD）：

| # | 疑点 | 结论 | 证据 |
|---|---|---|---|
| 1 | echo_source 绑定旧数据 | **确认** | `frontend/echo_source.py:27-29` 直接读 `data/f01_source/source_states.npy`、`reports/f01a/episodes.jsonl`、`reports/f01a/geometry.json` |
| 2 | 100 ms / dt=0.1 / deadline%100 | **确认** | `echo_source.py:69` `elapsed%100`、`:46` `age_ns>=100_000_000`、`:97` deadline 步进 100 ms；`ofdm_echo.py` `Waveform.dt=0.1`；`cv_kf.py:21-23` 硬断言 dt=0.1；`shared_frontend.json` tracker dt 0.1；`temporal_association.py:14` dt=0.1 |
| 3 | simulator 旧 SNR/可见性 | **确认** | `simulator.py:50` `snr_ref_db`；`:94` `10^(snr/20)·(100/radii)²·sqrt(σ/10)`；`:91` 可见性硬编码 `10–300 m`、`±70°`；`:60-61` 要求 height_m=5.0 |
| 4 | shared_frontend.json 旧引用 | **确认** | `:25` `geometry_source=reports/f01a/geometry.json`；`:35-39` `snr_ref_db/reference_range_m=100`；`:62` `covariance_lut=reports/f01e/snr_rebuild_06/…`；`:93-96` `calibration=reports/f01e/snr_rebuild_06/…` |
| 5 | detector 标定绑定旧链 | **确认** | `detector.py:194` 硬编码 `10–300 m`、`±70°`；`:204` LUT 以 **post-FFT** `peak_to_noise_db` 为键；`shared_frontend.json:85-92` `snr_levels_db=[-5..20]` 绑定 F01-E 标定产物 |
| 6 | fusion 只用位置 | **确认** | `fusion/association.py:133-141` 输出仅 `x_m,y_m,C_xy,bs_mask,n_bs,quality_db`；detector 已给出 `vr_mps`（`detector.py:214`）但 fusion 未使用 |
| 7 | tracker 初始速度 0、靠位置估速 | **确认** | `cv_kf.py:60-63` birth `state=[x,y,0,0]`、`P_vel=100·I`；`:123-144` 仅位置量测更新 |
| 8 | velocity_fusion 可复用 | **确认可用** | `controlled_isac/velocity_fusion.py`：多 BS 径向速度 → vx,vy MAP（`-u_b·v` 约定、rank/条件数输出）；先验 `sigma=30 m/s` 为 NGSIM 尺度，需重标 |
| 9 | measurement 只能辅助 | **确认** | `controlled_isac/measurement.py:128-148` 真值极坐标 + 人工噪声 `make_measurement`；`:24` 绑定 `reports/f01a/geometry.json`；供受控探针，不能作为正式 sensing 链 |

其它已核实：`frontend/ofdm_echo.py::noise_parameters` 同样是 100 m reference 的 per-target SNR 定义（`:67-86`）；`frontend/f01e_dataset.py` 把实验控制绑在 per-SNR F01-E npz 上（`ALLOWED_SNR_DB`）；调用上述模块的入口主要是 `code/07_shared_frontend/*`（probe/calibration）与 `experiments/qgat_* / qgnn_* / snr_downstream_smoke_01`（经 `frontend/symbol_dataset.py`）。

---

## 5. 新数据 → ISAC 数据入口：方案 A vs B

| 维度 | A：sample-level（samples.npz/history 生成回波） | B：scene-level（trajectories.csv 全场景逐帧缓存，再按 start_frame 切窗） |
|---|---|---|
| 未来信息泄漏 | 用 history 则无；但每样本独立仿真容易误用 future | 每帧仅用该帧在场状态，因果；无泄漏 |
| 人为删除临时车辆 | **会**（cohort 96.4% 窗口含额外车辆） | **不会**（物理在场车辆全部贡献） |
| 真实共享回波 | 差：回波只含 cohort | 好：每 BS 每帧一份共享回波 |
| 同一物理帧一致性 | 不一致（重叠样本各自仿真，目标集/噪声不同） | 一致（(scene,frame) 唯一回波，可缓存） |
| 计算量 | 表面低（12,394×20 帧×3 BS），但高冗余 | 前置高（train 约 1.4 万帧×3 BS），每帧只算一次 |
| 缓存复用 | 无 | 高（任意窗口/stride/split/消融复用） |
| 与 samples.npz 对接 | 直接 | 按 `start_frame` 切片；cohort 只定义 slot/标签 |

**推荐：方案 B（scene-level 缓存）**。cohort 的 N≤8 约束在 B 中自然成为“预测 slot 数”，而回波始终包含该帧全部真实车辆——这也直接回答了 §1 的结论 B，并避免同一物理帧在不同 sample 中得到不同回波。

## 6. 匿名 track ↔ sample vehicle_id 的三域对接

| 域 | 允许内容 |
|---|---|
| **A 域（仿真真值）** | simulator 内部持有位置/速度/vehicle_id；只输出共享回波与审计元数据 |
| **B 域（正式 sensing）** | detector/fusion/tracker 只收回波/匿名量测；输出匿名 `track_key`/`slot` 状态；禁止 vehicle_id/UUID/future/GT identity |
| **C 域（离线标签与评价）** | 允许使用 vehicle_id 做标签切片与评分；**不得**把 id/GT/评分回送给 B 域（不得进入门限、标定、关联决策） |

推荐对齐方法（仅 C 域）：按 split、scene 离线运行；逐帧用 Hungarian 将匿名 track 位置（有速度时加速度项）与 GT 状态匹配（Mahalanobis + 卡方门限）；每条 track 以多帧多数投票确定 `track_key → vehicle_id`，输出带置信度（投票边际）的映射表；tracker 的复用时隙（slot cooldown）要求用 `track_key` 而非 `slot` 做身份对齐。

## 7. SNR 新定义迁移检查

新定义：`Y_b = S_b + W_b`；`SNR_b = 10log10(mean|S_b|² / mean|W_b|²)`，位置在 FFT 前、检测前、融合前；禁止 100 m reference SNR、post-FFT/effective/per-target SNR 作为实验主控制量。

**与新定义冲突的字段/函数/配置/报告：**

| 位置 | 冲突 |
|---|---|
| `configs/shared_frontend.json` power 块 | `snr_ref_db`、`reference_range_m=100`、`reference_rcs_m2/fixed_rcs_m2/noise_variance` 为 per-target reference SNR 控制 |
| `configs/shared_frontend.json` `snr_levels_db`、`detector.covariance_lut`、`calibration` | 电平与 LUT 均按旧定义 + post-FFT 峰值 SNR 标定（f01e/snr_rebuild_06） |
| `frontend/ofdm_echo.py::noise_parameters` | 由 100 m 参考增益推噪声方差（旧定义） |
| `frontend/sensing/simulator.py` | `snr_ref_db` 幅度律 `10^(snr/20)(100/radii)²`（per-target） |
| `frontend/echo_source.py` | `observation(snr_db=20)`、pressure 仅 20 dB（旧适配器） |
| `frontend/f01e_dataset.py` | `ALLOWED_SNR_DB` 把控制量绑死在 F01-E per-SNR npz |
| `reports/f01e/snr_rebuild_06/*`、`reports/sensing_audit/*` | 旧定义下的标定/审计产物，不能继续作为正式标定来源 |

**后续修改清单（本轮不改代码）**：simulator 改为“给定 SNR_b 目标 → 对叠加干净回波整体缩放 → 加一次接收机噪声”，并输出每 BS/帧 SNR_b；config 删除 100 m reference 与 RCS→SNR 链路，仅保留 noise_variance + 电平；detector/coords 的 LUT 在新定义下重新标定（LUT 仍可按峰值信噪比索引，但标定必须以 SNR_b 为控制量）；报告与标定产物全部重建。

---

## 8. 文件级迁移表（下一轮施工入口）

| 文件/模块 | 当前作用 | Automatum 后是否保留 | 需要修改什么 | 优先级 |
|---|---|---|---|---|
| `frontend/echo_source.py` | F01-B 因果源适配器（旧 npy/jsonl/geometry + 100 ms 网格） | 重构/替换 | 换为 Automatum scene 适配器（trajectories.csv + canonical frame + dt=3/29.97），保留 A 域接口 | **P0** |
| `frontend/sensing/simulator.py` | 正式共享回波（旧 SNR 律、10–300 m、70°、5 m） | 小改 | 可见性/高度配置化；SNR 改为 SNR_b 帧级定义；保留共享回波+单次噪声 | **P0** |
| `frontend/sensing/waveform.py` | 波形/阵列常量 + `load_geometry()` 读 f01a | 小改 | `load_geometry` 改读 Automatum 候选几何；波形不变 | **P0** |
| `frontend/tracking/cv_kf.py` | 8-slot 匿名 CV 跟踪（断言 dt=0.1） | 小改 | dt=3/29.97（配置）；重标 `q_a` 与 birth inflation；接入径向速度融合（可选） | **P0** |
| `configs/shared_frontend.json` | 单一生产配置（旧几何/SNR/标定引用） | 重构 | 替换 geometry/power/calibration；新增 dataset/scene 块（路径、SHA、dt、帧相位） | **P0** |
| `frontend/ofdm_echo.py` | OFDM 参考回波 + 旧 SNR 定义 | 小改 | dt 参数化；SNR 语义迁移（波形常量保留） | P1 |
| `frontend/sensing/detector.py` | CFAR/NMS/AoA（硬门限 + post-FFT LUT） | 小改 | 范围/FOV 配置化；LUT 按新 SNR 重标；保留 `vr_mps` | P1 |
| `frontend/controlled_isac/measurement.py` | 真值+人工噪声受控量测 | 保留（辅助） | 重绑 Automatum 几何/范围/高度；seed 键改为 (scene, frame, BS, vehicle)；不进正式链 | P1 |
| `frontend/controlled_isac/velocity_fusion.py` | 多 BS 径向速度 → vx,vy | 复用（进正式链） | 速度先验 30 m/s 按 Automatum 尺度重标（P99≈27.6 m/s） | P1 |
| `frontend/controlled_isac/temporal_association.py` | 帧间 slot 关联（dt=0.1） | 小改 | dt 配置化 | P1 |
| `frontend/fusion/association.py` | 3-BS 位置融合（无径向速度） | 保留（可选扩展） | 位置融合可复用；速度由 velocity_fusion 提供，是否把速度并入融合状态为 P2 选项 | P2 |
| `frontend/sensing/coords.py` | 极/直角坐标 + LUT 协方差（H_M=5） | 保留 | 高度配置化；LUT 重标在 detector 侧 | P2 |
| `frontend/controlled_isac/cross_bs_association.py` | 受控量测的跨 BS 关联 | 保留（辅助） | 仅 smoke test；正式关联仍用 fusion | P2 |
| `frontend/symbol_dataset.py` | F01-E 符号级预测输入（QGAT/QGNN 训练） | 替换为新 loader | 基于 samples.npz + sensing cache 新建输入构建；不复用 F01-E packing | P2 |
| `frontend/f01e_dataset.py` | F01-E 冻结加载器（per-SNR npz） | 退役（历史） | 由新 split/sample loader 取代 | P3 |
| `code/07_shared_frontend/*` | 历史 probe/标定入口（调用上述模块） | 实验辅助 | 新适配器与新 SNR 定义就绪后再回跑 | P3 |
| `experiments/*`（qgnn/qgat、snr smoke） | F01-E 预测训练/评测入口 | 暂不参演 | 新 loader 与 sensing cache 完成前不得在 Automatum 上运行 | P3 |

计数：**P0 = 5，P1 = 5，P2 = 4，P3 = 3**。

## 9. 待决事项（open decisions）

| ID | 问题 | 建议 |
|---|---|---|
| D-01 | sample-level vs scene-level 回波 | **scene-level（B）**，待确认 |
| D-02 | 3-BS 坐标是否冻结 | 先做 B 路线 smoke，再冻结候选 |
| D-03 | range/FOV/height 候选 | range 5–150 m、FOV half 70°、Δh 5 m（6.5 m 站 / 1.5 m 车），CANDIDATE |
| D-04 | SNR 控制定义与电平表 | 采用 SNR_b；重推 `snr_levels_db` 并重标 CFAR/LUT |
| D-05 | velocity_fusion 先验 σ | 由 30 m/s 改为 Automatum 尺度（P90 速度差量级 ~15 m/s），结合几何条件数确定 |
| D-06 | 匿名 track ↔ vehicle_id 对齐 | C 域 Hungarian + 多数投票，输出置信度映射 |
| D-07 | 临时车辆是否进入共享回波 | 进入（B 路线）；cohort 仅定义预测 slot/标签 |

## 10. 本轮保证

- 未修改任何正式数据（canonical/split/samples 均未触碰，SHA 校验一致）；
- 未修改任何 ISAC 科研代码或配置；
- 未运行训练或正式 ISAC 实验；
- 仅新增 2 个审计文件与 1 个只读审计脚本。