# Automatum T-Crossing Canonical 数据预处理报告（fixed stride 3 / 9.99 Hz 返工版）

Canonical dataset: `data/automatum_t_crossing/processed/trajectories_10hz.csv`（文件名为项目名义名，真实采样率见下）
Scenes: 2（scene 0 = Gaimersheim Stadtweg / Recording A；scene 1 = St2214 Dünzlau Umgehung / Recording B）
Vehicles: 683（scene 0: 299，scene 1: 384）
Rows: 81,450（scene 0: 41,944；scene 1: 39,506）
Frequency: 真实无插值采样率 **9.99 Hz**（29.97 / 3）；项目名义任务率 ≈ 10 Hz
Columns: `scene_id, vehicle_id, timestamp, x, y, vx, vy`（共 7 列）
Timestamp rule: **真实源时间** `source_frame / 29.97`（不再伪装成 0.1 s 网格）
Velocity coordinate: **世界系**（`vx_w = cos(psi)·vx_body − sin(psi)·vy_body`；`vy_w = sin(psi)·vx_body + cos(psi)·vy_body`，psi 为原始连续角）
Validation: **PASS**（`reports/data_preprocessing/automatum_canonical_validation.json`，36/36 checks）
SHA256（canonical 三件套，重复 build 完全一致）:

| 文件 | 行数 | 大小 | SHA256 |
|---|---|---|---|
| `trajectories_10hz.csv` | 81,450 | 5,262,358 B | `a41ac4e24a8f21039dde60064868871b21093289a661ca9bef28ee7d1aab3442` |
| `vehicle_id_mapping.csv` | 683 | 59,400 B | `5301f2403a34e3be718919a3eb1277b03335f8ec3c424c02ed220e386223c157` |
| `scene_metadata.json` | 2 scenes | 4,295 B | `f70c74ccf0f02f89b0e9f5148a2d6abe16bbc60e178a41146eb70051131d1827` |

（stats JSON 自身 SHA256 = `a4ad4d5d9d7169076ab275fa60a60e920e1aa14bd20af068a9fec11312bdb859`；所有路径均为相对/文件名，跨机器重建字节一致。）

---

## 0. 本轮修改范围声明

上一版 commit `96f63e05dc8ccffea6146af31734749ebcc578ea` 主体验收通过，但 nearest-0.1s-grid 规则造成 250 个相邻 canonical 点实际只跨 2 个原始帧（scene 0: 124、scene 1: 126），标签时间与真实时间错配，位置差分检查出现最大 ~7.8 m/s 尖峰。

本轮**只修改重采样与 timestamp 规则**；车辆集合、scene 划分、7 列 schema、x/y、车体系→世界系旋转、不插值/不平滑/不做 map matching/不筛 N≤8/不划 split/不生成样本/不设 BS 等全部冻结项均未改。未修改任何原始文件；`data/` 下正式数据未提交 Git。

---

## 1. 采样规则（新）与修改理由

### 1.1 新规则

1. 每个 scene 使用**统一原始全局 frame phase**，只保留 `source_frame % 3 == 0` 的原始状态（0, 3, 6, 9, …）；
2. `source_frame = round(time × 29.97)`，并独立验证 `|time − source_frame / 29.97| < 1e-9`（实测最大 2.3×10⁻¹³ s，纯浮点尾差）；
3. **timestamp 直接写真实源时间** `frame / 29.97`（保留 raw time），不再把 `0.1001001…` 改写为 `0.1`；
4. 每辆车只保留其轨迹中落在该 scene 统一相位上的帧——**不是**每辆车从自己的第一帧起每 3 帧取一次（否则不同车辆会失去真实同步关系）；
5. 无插值、无平滑：x/y 与 vx/vy 全部是真实原始帧状态。

### 1.2 为什么从 nearest-0.1s-grid 改为 fixed stride=3（工单 §18 要求说明）

当前 ICCT 只使用 **20 + 20 个采样点**的短时预测窗口。相比“长时绝对时间对齐”，更重要的是 **局部物理时间间隔严格恒定**以及 **x/y/vx/vy 与 timestamp 三者对应同一次真实采样**：

- nearest-grid 法会让相邻 canonical 点的真实源间隔在 2–3 帧间跳变（0.0667 s 与 0.1001 s 被统一标成 0.1 s），造成位置差分与速度的局部错配（上一版 max 7.8 m/s）；
- fixed stride=3 使每个场景的所有车辆共享同一全局帧相位，Δt 恒为 3/29.97 s，位置/速度/时间严格对齐；
- 代价是名义 10 Hz 变为真实 9.99 Hz（0.1% 偏差），对 40 点窗口仅 ~4 ms 物理差异，任务接受。

### 1.3 真实频率

| 项 | 值 |
|---|---|
| 源频率 | 29.97 Hz（Δt = 0.0333667 s） |
| stride | 3（scene 全局相位） |
| **canonical 真实采样率** | **29.97 / 3 = 9.99 Hz** |
| canonical Δt | 3 / 29.97 = **0.1001001001… s** |
| 项目名义任务率 | ≈ 10 Hz（文档与文件名沿用） |

---

## 2. 执行环境与输入

- 执行环境：本地仓库副本 `E:\NJUPT\ICCT会议`（Syncthing 与服务器 `/home/dell/YrM/ICCT` 同步源码）。正式数据应保留在服务器；服务器侧已用同一脚本重建，SHA256 与本表一致（见 §10）。
- 审计依据：`f03e230d094abb0a3f268c2b7df9db7c6c675552`。
- 原始 recording 目录（服务器解压位置 `data/automatum_t_crossing/raw/extracted/`）：

| scene | 目录 | dynamicWorld.json SHA256 | staticWorld.xodr SHA256 |
|---|---|---|---|
| 0 | `T-Crossing--GaimersheimStadtweg_e2e6-e2e6f4bb-4668-4654-ac7e-bcd90c9df4c2` | `899037a7be1f61dd75915fbdeab32e8e2c01c8d2b4be4e814775216896f2cd19` | `afee5cbdad64d2098cd64e95d2df845ab940a433aee995beb258b604dbdeef55` |
| 1 | `T-Crossing-St2214-DuenzlauUmgehung_1b9b-1b9bf4b8-9fa6-4d23-abd2-2b715d087e8f` | `72c5bfbbcfc7ed59a258f543f8c74fad787ee0bdaaa1236fc354564d618fcce2` | `641d193b5cd502406a495ab71ec8738ffac7376a0d8fe12f642b1199649f300e` |

原始 ZIP SHA256 = `ea2c528fafdb306ea20686065b15c39ae2332342594a0f8031477bdd8f91f19f`（本地与服务器一致）。

---

## 3. 统计（工单 §14）

| 指标 | scene 0（Recording A） | scene 1（Recording B） | 合计 |
|---|---|---|---|
| raw points | 125,860 | 118,517 | 244,377 |
| canonical points | **41,944** | **39,506** | **81,450** |
| vehicles | 299 | 384 | 683 |
| timestamp range | 0.0 – 650.6506506506507 s | 8.508508508508509 – 1125.6256256256256 s | — |
| canonical x range | [−57.97, 76.13] m | [−29.23, 63.82] m | — |
| canonical y range | [−49.15, 69.28] m | [−83.48, 84.08] m | — |
| Δt = 3/29.97 s（median / P95 / P99 / min / max） | 0.100100100100107 / 0.100100100100121 / 同 / 0.100100100100008 / 0.100100100100121 | 0.100100100100121 / 同 / 0.100100100100235 / 0.100100100100008 / 0.100100100100235 | — |
| source stride 分布 | `{3: 41,645}` | `{3: 39,122}` | `{3: 80,767}` |
| 轨迹持续时间分布 <2 / 2–4 / 4–6 / 6–10 / 10–20 / >20 s | 0 / 10 / 5 / 149 / 87 / 48 | 0 / 2 / 12 / 268 / 71 / 31 | 0 / 12 / 17 / 417 / 158 / 79 |
| 帧回推误差 max | 1.14×10⁻¹³ s | 2.27×10⁻¹³ s | — |

行数与上一版（81,554）相差 −104 行（−0.13%），来自固定相位裁掉轨迹两端的零散帧，属于预期变化。

---

## 4. 位置—速度一致性验证（工单 §11，单一物理口径）

10 Hz canonical 数据上以真实 Δt = 3/29.97 s 做位置差分，与官方世界系速度比较（全部 80,767 对）：

| scene | MAE vx | RMSE vx | MAE vy | RMSE vy | P95 (vx/vy) | P99 (vx/vy) | max (vx/vy) |
|---|---|---|---|---|---|---|---|
| 0 | **0.0330** | 0.0482 | **0.0310** | 0.0461 | 0.105 / 0.100 | 0.166 / 0.170 | 0.324 / 0.368 m/s |
| 1 | **0.0344** | 0.0506 | **0.0390** | 0.0555 | 0.113 / 0.121 | 0.179 / 0.184 | 0.330 / 0.493 m/s |
| 合计 | 0.0337 | 0.0494 | 0.0348 | 0.0509 | 0.109 / 0.111 | 0.172 / 0.177 | **0.330 / 0.493 m/s** |

方向误差（运动点）：median **0.069°**，P95 0.93°，P99 2.18°，max 6.95°。

判定：

- **上一版 5–8 m/s 尖峰已完全消失**（max 从 7.84 → 0.49 m/s，降为物理量级 a·Δt/2）；
- 无 x/y 交换、无符号错误、无 psi 旋转方向错误、无 90°/180° 偏转（方向中位 0.07°），bias ≈ 0；
- 结论：世界系 vx/vy 可直接作为 ICCT Ground Truth 速度。

---

## 5. 速度旋转与模长验证（工单 §12）

逐点独立复核（validator 重新读取 raw + CSV）：

- x/y 与旋转后 vx/vy 与所选真实源帧逐点一致（容差 2×10⁻⁶ m / 2×10⁻⁶ m/s，0 违规）；
- 旋转前后速度模长一致：两 scene 最大差 **7.1×10⁻¹⁵ m/s**（纯浮点误差）→ PASS。

---

## 6. 场景全局同步性检查（工单 §13）

`(scene_id, timestamp)` → source_frame 必须唯一：

- 实际检查了两个 scene 的 **全部 16,655 个 timestamp**（远超“随机抽 100 个”的要求）；
- 违规数：**0**（同一 scene+timestamp 下所有车辆来自同一原始帧）。

---

## 7. vehicle_id 稳定性（工单 §8）

上一版规则 `(scene_id, 首次 canonical timestamp, UUID)` 依赖重采样结果；本轮改用更稳定的确定性规则：

```
final rule = (scene_id, raw first timestamp, original UUID) → vehicle_id = 1..683
```

与上一版 `vehicle_id_mapping.csv`（SHA256 `3d211ffd…bbcb`）对比：

| 对比 | 变化车辆数 | 说明 |
|---|---|---|
| 上一版 mapping vs 新 mapping | **4** | 两对相邻交换（329↔330、428↔429），位移均为 1，全部在 scene 1 |
| 其中：采样相位变化（旧规则下） | 6 | 首次 canonical 采样相位改变使部分车辆排序移动（±1 帧 = ±33 ms 量级） |
| 其中：改为新规则的修正 | −2（抵回） | 新规则与重采样无关，把 2 辆修正回稳定位置 |
| stats 内 `changed_vs_legacy_first_canonical_rule` | 2 | 同一份新采样数据上，旧规则与新规则的差异车辆数 |

结论：变化极小（4/683，且只是相邻对调），新规则在未来任何重采样变化下 **不会漂移**。未为了保持旧 SHA 而人为冻结错误映射。

---

## 8. 独立验证（工单 §17，36/36 PASS）

`validate_automatum_canonical.py` 直接重新读取 raw JSON / CSV / mapping / metadata / stats，独立实现复核：

| # | 检查 | 结果 |
|---|---|---|
| 1 | schema 恰为 7 列且顺序正确 | PASS |
| 2 | scene_id 只有 0/1；两 scene 车辆零重叠 | PASS |
| 3 | vehicle_id 连续 1..683 | PASS |
| 4 | `(scene_id, vehicle_id, timestamp)` 无重复 | PASS |
| 5 | NaN / Inf = 0；x/y/vx/vy 全有限 | PASS |
| 6 | 每车 timestamp 严格递增 | PASS |
| 7 | Δt 恒 = 3/29.97 s（`physical_dt_constant`） | PASS |
| 8 | `fixed_stride_3`：全部 80,767 对 stride=3 | PASS |
| 9 | `no_stride_2` / `no_stride_4` | PASS / PASS |
| 10 | `true_timestamp_preserved`：timestamp = 真实源时间 | PASS |
| 11 | `all_frames_on_global_phase`：frame % 3 == 0 | PASS |
| 12 | `scene_global_phase_consistent`：16,655 个 timestamp 全部唯一 source frame | PASS |
| 13 | `no_source_frame_reuse` | PASS |
| 14 | `positions_copied_from_source`：x/y 等于真实源帧 | PASS |
| 15 | `rotation_reproduces_world_velocity` | PASS |
| 16 | `speed_norm_preserved` | PASS |
| 17 | `velocity_fd_no_grid_label_artifact`：max < 1 m/s | PASS（0.493） |
| 18 | `velocity_fd_consistency`：MAE/方向误差达标 | PASS |
| 19 | 无位置瞬移（≤45 m/s）、速度物理合理（≤35 m/s） | PASS |
| 20 | `vehicle_id_rule_reproduced`：映射可由可得规则复算 | PASS |
| 21 | `mapping_traceable_to_raw`：683 行、UUID 唯一、与 raw 属性一致 | PASS |
| 22 | 原始文件未修改（raw SHA256） | PASS |
| 23 | 文件按 scene→timestamp→vehicle 排序 | PASS |
| 24 | scene_metadata 与主表一致（行数/车辆/时间范围/9.99 Hz/stride） | PASS |
| 25 | `prediction_window_supply_match`（≥40 点统计一致） | PASS |
| 26 | 三个输出 SHA256 与 stats 记录一致 | PASS |

验证输出：`reports/data_preprocessing/automatum_canonical_validation.json`。

### 可复现性（工单 §19）

- build 无随机数、无时间戳、无人工编辑输入；连续两次运行三件套 SHA256 完全一致；
- 文件统一 LF 换行；timestamp 用最短浮点表示（round-trip），x/y/vx/vy 固定 6 位小数；
- 服务器重建验证：同一脚本在服务器环境（numpy 2.4.6）重建，三件套 SHA256 与本报告一致。

---

## 9. 2s→2s 任务统计口径（工单 §15，仅统计，不生成样本）

正式 sample builder 统一定义：**history = 20 samples，future = 20 samples，共 40 个 canonical 状态点**（不使用“严格 4.000 s”或“41 点”定义）。9.99 Hz 下 40 点覆盖约 3.996 s，与名义 4 s 差异 ~0.1%，任务接受。

| 指标 | scene 0 | scene 1 | 合计 |
|---|---|---|---|
| 具备 ≥40 canonical 点的车辆 | **289** | **382** | **671 / 683（98.2%）** |

未计入的 12 辆车（10 + 2）轨迹较短，仍完整保留在 canonical 数据中。

---

## 10. 数据落地与服务器一致

- Canonical 数据与 mapping/metadata 位于服务器 `data/automatum_t_crossing/processed/`（该目录受 `.gitignore` 与 `.stignore` 双重排除，不进 Git、不走 Syncthing，属项目“数据留在服务器”设计）。
- 服务器使用同一脚本从 raw ZIP 一键重建：解压 → build → validate；三件套 SHA256 与本地完全一致（`a41ac4e2…`、`5301f240…`、`f70c74cc…`）。
- 复现命令：

```bash
cd /home/dell/YrM/ICCT
/home/dell/YrM/envs/ICCT/bin/python tools/data_preprocessing/build_automatum_canonical.py \
    --raw-dir data/automatum_t_crossing/raw \
    --out-dir data/automatum_t_crossing/processed \
    --stats-out reports/data_preprocessing/automatum_canonical_stats.json
/home/dell/YrM/envs/ICCT/bin/python tools/data_preprocessing/validate_automatum_canonical.py
```

---

## 11. 明确未做的事项（冻结项）

- 未划 Train/Val/Test；未生成 41 点或 40 点任何样本；未做 N≤8 筛选、ROI、局部归一化；
- 未布置 BS、未生成 ISAC 量测；未构造图、未改 GNN/QGNN/ISAC 代码；
- 未补 lane/road/TTC 字段、未做 map matching；
- 未删除/加工 NGSIM；未提交 `data/` 下任何数据文件。

---

## 12. 验收标准对照（工单 §21）

| # | 标准 | 结论 | 证据 |
|---|---|---|---|
| 1 | source frame stride 恒 = 3 | PASS | 80,767 对全部为 3 |
| 2 | 无 stride=2 | PASS | 0 对 |
| 3 | 无 stride=4 | PASS | 0 对 |
| 4 | 同一 scene/time 共享同一 source frame | PASS | 16,655 个 timestamp，0 违规 |
| 5 | timestamp 使用真实 source time | PASS | 与 raw time 逐点一致（容差 1e-12 s） |
| 6 | Δt 恒 = 3/29.97 s | PASS | min/max 差 2.3e-13 s |
| 7 | 无插值 | PASS | 每点来自真实帧，x/y 逐点一致 |
| 8 | x/y 仍为真实源帧值 | PASS | 容差 2e-6 m，0 违规 |
| 9 | world vx/vy 旋转规则不变且正确 | PASS | 独立旋转复核 0 违规，模长保持 |
| 10 | 上一版 5–8 m/s 尖峰消失 | PASS | max 0.493 m/s |
| 11 | 683 辆车全部保留 | PASS | scene 0/1 = 299/384 |
| 12 | 两 scene 独立 | PASS | 车辆集合零重叠，无拼接 |
| 13 | 无 NaN / Inf / duplicate | PASS | 0 / 0 / 0 |
| 14 | deterministic build | PASS | 两次 build SHA 一致 |
| 15 | SHA256 可复现 | PASS | 三件套 SHA 已记录，跨机器一致 |
| 16 | 未提前生成 split/windows/BS/model 数据 | PASS | 见 §11 |

最终判定：**PASS**。

---

## 13. 遗留说明

1. 文件名沿用 `trajectories_10hz.csv` 以避免下游路径变更，但真实采样率为 **9.99 Hz**；所有文档与 metadata 均已写明。
2. 官方 vx/vy 绝对精度仍无外部真值参考（视觉跟踪数据；内部一致性已验证到 0.03–0.05 m/s）。
3. 若未来需要严格 0.1 s 物理间隔（而非 0.1001 s），只能通过插值实现；当前任务明确禁止插值，故不采用。