# Automatum Data T-Crossing 数据集一次性完整审计

- 性质：**AUDIT ONLY**。未修改 ZIP 与官方文件；未降采样、未清洗、未生成 canonical CSV、未重编号、未划分 split、未生成 2s→2s 样本、未设置 BS、未改动 ISAC/GNN/QGNN 代码。
- 审计脚本：`tools/data_audit/audit_automatum_t_crossing.py`（只读；本机整轮运行约 25 s）
- 机器可读结果：`reports/data_audit/automatum_t_crossing_stats.json`
- 逐车统计：`reports/data_audit/automatum_vehicle_stats.csv`（683 行，一行一辆车）
- 异常明细：`reports/data_audit/automatum_anomalies.csv`（**0 条异常**，仅表头）
- 图：`reports/data_audit/figures/automatum/`（11 张）
- 复现命令：

```bash
python tools/data_audit/audit_automatum_t_crossing.py \
    --zip data/automatum_t_crossing/raw/automatum_data_crossing.zip \
    --extracted data/automatum_t_crossing/raw/extracted \
    --out-dir reports/data_audit \
    --fig-dir reports/data_audit/figures/automatum
```

---

## 0. 执行环境与实际路径（与工单预设的差异）

工单预设数据位于 `/home/dell/YrM/ICCT/data/automatum_t_crossing/`。本轮审计实际在 Syncthing 同步的本地仓库副本上执行，实际路径为：

```
E:\NJUPT\ICCT会议\data\automatum_t_crossing\raw\automatum_data_crossing.zip
E:\NJUPT\ICCT会议\data\automatum_t_crossing\raw\extracted\<两个 recording 目录>\
```

数据同一性以 ZIP SHA256 为准（见 §2），服务器与本地副本可据此核对为同一下载件。两个 recording 的真实名称、文件与时序均以实际解压文件为准，未采用 Hugging Face / README 的 schema 描述。

---

## 1. 原始 ZIP 校验

| 项 | 值 |
|---|---|
| 文件 | `automatum_data_crossing.zip` |
| 大小 | 33,479,907 bytes（31.93 MiB） |
| SHA256 | `ea2c528fafdb306ea20686065b15c39ae2332342594a0f8031477bdd8f91f19f` |
| ZIP 完整性 | `testzip() == None`，18 个成员全部 CRC 通过，可正常解压 |
| 解压位置 | `data/automatum_t_crossing/raw/extracted/`（保持官方目录结构） |
| 解压后总大小 | 109,410,458 bytes（104.3 MiB） |

解压目录树（两个 recording 各一份）：

```
T-Crossing--GaimersheimStadtweg_e2e6-e2e6f4bb-4668-4654-ac7e-bcd90c9df4c2/
    dynamicWorld.json      51,283,310 B      staticWorld.xodr  18,679 B
    T-Crossing--GaimersheimStadtweg_e2e6-...html  4,823,478 B（Bokeh 交互可视化）
    img/kpis.json 314 B;  img/*_map.jpg 83,178 B;  img/*_trajectories.jpg 101,742 B;
    img/*_centerImg_thumb.jpg 14,777 B
T-Crossing-St2214-DuenzlauUmgehung_1b9b-1b9bf4b8-9fa6-4d23-abd2-2b715d087e8f/
    dynamicWorld.json      48,353,942 B      staticWorld.xodr  22,085 B
    T-Crossing-St2214-DuenzlauUmgehung_1b9b-...html  4,558,768 B
    img/kpis.json 315 B;  img/*_map.jpg 56,736 B;  img/*_trajectories.jpg 70,396 B;
    img/*_centerImg_thumb.jpg 22,738 B
```

Recording 命名（本报告沿用，不再混用）：

| 标签 | 真实名称 | 简称 |
|---|---|---|
| **Recording A** | `T-Crossing--GaimersheimStadtweg_e2e6-e2e6f4bb-4668-4654-ac7e-bcd90c9df4c2` | Gaimersheim Stadtweg |
| **Recording B** | `T-Crossing-St2214-DuenzlauUmgehung_1b9b-1b9bf4b8-9fa6-4d23-abd2-2b715d087e8f` | St2214 Dünzlau Umgehung |

注意：工单 §16 “约 18.8 min 的较大 recording” 指 **Recording B**。

厂商 `img/kpis.json` 与本轮独立计算结果交叉验证一致（A：最长 109.4 km/h、最大加速度 4.7 m/s²；B：110.4 km/h、5.8 m/s²，均与 §8 实测吻合）。

---

## 2. 实际数据结构（以文件为准）

`dynamicWorld.json` 顶层：`Contact / License / Release(=3.0) / UTM-ReferencePoint / WGS84-ReferencePoint / UUID / RecordingName / Web / videoInfo / objects / Object_counts / DrivenDistanceInMeter / MedianDrivenDistanceInMeter`。

- `videoInfo = {fps: 29.97, frame_count: 19501 (A) / 33736 (B)}`；时间戳网格与 frame_count 一致。
- `objects` 为**逐车轨迹数组**，不是逐帧快照；A 299 辆、B 384 辆。
- 每辆车的字段（实际 schema）：

| 字段 | 含义 | 实测 |
|---|---|---|
| `UUID` | 车辆全局标识 | 683 辆内及跨 recording 均唯一（见 §9） |
| `length`, `width`, `objType` | 尺寸与类型 | car / van / truck |
| `time` | 秒，recording 起点为 0，逐车从出现到消失 | 1/29.97 s 严格均匀网格 |
| `x_vec`, `y_vec` | **米制局部世界坐标**（与 staticWorld.xodr 同系） | 与 xodr 车道几何 99.8%–100% 对齐 |
| `psi_vec` | 航向角（弧度，**未折叠**，可见 −3.03…+4.55） | 与位置差分方向一致（p95 0.02 rad） |
| `vx_vec`, `vy_vec` | **车体系**纵向/横向速度（关键，见 §7） | 与位置差分 MAE 0.04 m/s |
| `ax_vec`, `ay_vec` | 车体系加速度 | 最大 4.69 / 5.80 m/s² |
| `jerk_x_vec`, `jerk_y_vec`, `curvature_vec` | 加加速度、曲率 | 完整 |
| `lane_id_vec`, `road_id_vec`, `road_type_list`, `object_relation_dict_list`, `dist_along_road_segment`, `lane_change_flag_vec`, `ttc_dict_vec`, `tth_dict_vec`, `lat_dict_vec`, `long_dist_dict_vec`, `distance_left_lane_marking`, `distance_right_lane_marking` | **存在但全部为空** | **0/299、0/384 辆车有数据** |

结论（证据等级 CONFIRMED）：

1. 每个 object 内部所有状态向量长度与 `time` **完全一致**（0 个不一致）；
2. **不存在单独的 timestamp vector**，也不存在“逐帧文件”结构，时间就在每辆车的 `time` 里；
3. 实际采样间隔恒为 `1/29.97 = 0.0333667000333… s`，**不是 30.00 Hz**，也**没有任何抖动/缺帧**（见 §5）；
4. 数据集**不提供车道/道路 ID、TTC/TTH、车道变更标志**（schema 字段为空），若要车道级标签必须由项目自行用 xodr 投影生成——这属于项目预处理，不是数据错误。

---

## 3. 两个 recording 基本统计

| 指标 | Recording A | Recording B |
|---|---|---|
| 名称 | Gaimersheim Stadtweg | St2214 Dünzlau Umgehung |
| 时长 | **650.65 s（10.84 min）** | **1125.63 s（18.76 min）** |
| 总帧数 | 19,501 | 33,736 |
| 实际平均采样率 | 29.97 Hz（精确均匀） | 29.97 Hz（精确均匀） |
| 轨迹点总数 | 125,860 | 118,517 |
| 车辆数 | **299**（car 265 / truck 31 / van 3） | **384**（car 358 / truck 16 / van 10） |
| 轨迹点数 P50 / max | 291 / 2,129 | 233 / 1,406 |
| 存在时间（轨迹时长）中位 / 均值 | 9.68 s / 14.01 s | 7.74 s / 10.26 s |
| 轨迹长度中位 | 166.1 m（总行驶里程 47.9 km） | 162.7 m（总行驶里程 60.9 km） |
| 活动区 x 范围 | −58.3 … +76.7 m | −29.4 … +64.0 m |
| 活动区 y 范围 | −49.3 … +69.7 m | −83.5 … +84.3 m |
| 活动区凸包面积 | 7,678 m² | 7,813 m² |
| 速度范围（vx/vy） | 0.001 … 30.39 m/s（109.4 km/h） | 0.001 … 30.66 m/s（110.4 km/h） |
| 加速度范围 | 0.001 … 4.69 m/s² | 0.0003 … 5.80 m/s² |
| jerk max | 2.34 m/s³ | 2.87 m/s³ |
| 地图坐标参考 | UTM 32U 675222.84, 5406668.51 / WGS84 48.788309, 11.385632 | UTM 32U 670421.18, 5405184.06 / WGS84 48.776300, 11.319700 |

两个路口相距约 **5.0 km**（UTM 差 ΔE 4,802 m、ΔN 1,484 m），是同一地区（Ingolstadt 附近）的两个 T 型路口。

### 轨迹持续时间分布（工单 §4）

| 区间 | Recording A | Recording B |
|---|---|---|
| <2 s | 0 | 0 |
| 2–4 s | 10 | 2 |
| 4–6 s | 5 | 11 |
| 6–10 s | 145 | 269 |
| 10–20 s | 91 | 71 |
| >20 s | 48 | 31 |
| **≥4 s（完整 2s+2s 的基本条件）** | **289 / 299（96.7%）** | **382 / 384（99.5%）** |

每辆车明细见 `automatum_vehicle_stats.csv`；分布图见 `figures/automatum/track_duration_distribution.png`。

---

## 4. 时间与轨迹连续性审计

对两个 recording 的 683 辆车逐辆检查（证据等级 CONFIRMED）：

| 检查项 | A | B |
|---|---|---|
| timestamp 严格递增 | 299/299 通过 | 384/384 通过 |
| 重复时间点 | 0 | 0 |
| 时间倒退 | 0 | 0 |
| Δt 分布 | 全部 = 0.0333667000 s（std ≈ 3–5×10⁻¹⁴，仅浮点尾差） | 同左 |
| 正常采样间隔比例 | **100%**（125,561 个间隔） | **100%**（118,133 个间隔） |
| 缺帧 / gap | **0** | **0** |
| 轨迹中断后重现 | **0** | **0** |
| 跨 recording 检查 | 两个 recording 时间各自从 0 起算，无共享时间轴，不存在跨文件时间矛盾 |  |

结论：**不需要插值、不需要补点、不需要排序**。这是与 NGSIM（文件序 228 个负步、220 个 200 ms/≥1 s 假跳变）最本质的差别。

---

## 5. 位置质量审计

- x/y **不是 UTM 原值**，是米制局部笛卡尔坐标，局部系与 `staticWorld.xodr` 一致（§10 定量验证）；UTM/WGS84 参考点在 metadata 中给出，可无损换算。
- NaN / Inf：**0**。
- 位置突跳：以逐帧位移/Δt 检验，**无任何超过 45 m/s 的步进**；最大帧间位移 A/B 均对应 ≤30.7 m/s（合法高速）。
- 位移速度（distance(t, t+1)/Δt）分布（与官方 vx/vy 已互验，两者 MAE 0.04 m/s，下表为全部帧样本）：

| 分位 | A (m/s) | B (m/s) |
|---|---|---|
| P50 | 12.77 | 17.95 |
| P95 | 20.46 | 25.14 |
| P99 | 22.95 | 27.63 |
| P99.9 | 30.32 | 29.56 |
| max | 30.39 | 30.66 |

- 最异常的 20 条轨迹片段：本数据集**没有**真正的“异常片段”可列（位置突跳 0、非有限值 0）。`automatum_anomalies.csv` 因此只有表头；`automatum_vehicle_stats.csv` 中 `fd_speed_*`、`max_step_*` 即用于人工复核的最坏样本清单——按这些列排序可复现“最差 top-20”，全部落在物理合理范围。
- 明显漂移/瞬移：0。

---

## 6. vx/vy 可信度（本轮关键结论）

**核心发现：`vx_vec/vy_vec` 是车体系（纵向/横向）速度，`x/y` 是世界坐标，`psi` 是航向。** 世界系速度需旋转：`v_world = R(psi) · (vx, vy)`。

独立验证（两 recording 全部 243,694 个帧间样本）：

| 检验 | A | B |
|---|---|---|
| |FD 速度 − hypot(vx,vy) 中点值| MAE | **0.0377 m/s** | **0.0406 m/s** |
| RMSE | 0.0549 m/s | 0.0584 m/s |
| 偏差（bias） | +0.0065 m/s | +0.0065 m/s |
| P95 / P99 / max | 0.121 / 0.184 / 0.445 m/s | 0.129 / 0.192 / 0.524 m/s |
| 旋转 psi 后与 FD 世界速度的 MAE | **0.0514 m/s** | **0.0603 m/s** |
| 若错误地当世界系直接相减的 MAE | 14.11 m/s | 22.78 m/s |
| psi 与 FD 方向（仅运动点） | 均值 0.0040 rad，P95 0.0194 rad | 均值 0.0045 rad，P95 0.0223 rad |

**结论（证据等级 CONFIRMED）：**

1. **官方 vx/vy 足够可靠，可直接作为 ICCT Ground Truth**。残余 0.04–0.06 m/s 的量级正好等于“瞬时速度 vs 33 ms 弦平均”的二阶差分（≈ a·Δt/2），不是噪声也不是平滑失真；bias 近似 0。
2. **不存在实质性平滑或时间对齐差异**：若做过重平滑，FD 高频分量会与官方速度分离；实际两者逐帧一致到厘米级。
3. **不需要重新计算速度**。后续预处理直接搬运 vx/vy（以及 ax/ay、jerk、curvature），并强制记录“车体系”语义；只有需要世界系速度时才做 `R(psi)` 旋转。这符合“不为统一而重造速度”的原则。
4. `psi` 可靠（运动点 P95 偏差 0.02 rad），但**未折叠**（−π…π 之外连续），下游使用时按连续角处理、不要盲目 wrap 差分。

图：`figures/automatum/position_velocity_consistency.png`。

---

## 7. 物理合理性

| 指标 | A median / P95 / P99 / max | B median / P95 / P99 / max |
|---|---|---|
| speed (m/s) | 12.77 / 20.46 / 22.95 / 30.39 | 17.95 / 25.14 / 27.63 / 30.66 |
| speed (km/h) | 46.0 / 73.6 / 82.6 / 109.4 | 64.6 / 90.5 / 99.5 / **110.4** |
| accel (m/s²) | 0.38 / 2.32 / 3.16 / 4.69 | 0.79 / 2.83 / 3.77 / 5.80 |
| jerk (m/s³) | 0.21 / 1.02 / 1.55 / 2.34 | 0.26 / 1.28 / 1.83 / 2.87 |

- 离谱高速（FD > 45 m/s）：**0 起**。
- 离谱加速度（>8 m/s²）：**0 起**（工单预设的“5.8 m/s² 急刹”是真实刹车，不是脏数据）。
- 明显跳点：0。heading 突变（运动状态下单帧 >0.5 rad）：**0 起**。
- 最高速 109–110 km/h 与 xodr 限速 100 km/h + 10% 容差一致，且与厂商 KPI 完全吻合——数据集内部自洽。
- 慢速/静止段（如 A 的 object #149 在路口区内低速蠕行/等待约 50 s）位置抖动在厘米级，`psi` 保持合理；这类怠速不是脏数据。

---

## 8. 车辆 ID 审计

- 每个 recording 内 `UUID` **唯一**（299/299、384/384）；**两个 recording 之间 0 重叠**（683 个 UUID 全局唯一）。
- 车辆顺序与 UUID 无固定规律，不应依赖 `objects` 数组下标。
- 建议的追溯键（本轮**只提规则，未重编号**）：

```
trace_key          = (recording_name, UUID)      # 等价 (recording_id, UUID)
final_vehicle_id   = 稳定整数/哈希，由 (recording_name, UUID) 唯一映射生成
scene_id           = recording_name 或 recording_uuid（两个路口天然不同域）
```

若未来两个路口合并使用，按上式生成即可，不会出现 NGSIM 那种“同一 ID 跨时段 936 次复用、端点跳跃 P50 314 m”的问题。

---

## 9. 地图适配性（staticWorld.xodr）

两个 xodr 均为 **1 个 T 型路口 + 6 条 road**（2 条外部道路 + 4 条路口内连接道），lane 级几何完整（driving/border，含 `paramPoly3` 曲率与变宽车道），RHT（右侧通行）。

| 项 | A | B |
|---|---|---|
| 外部道路 | road 4（193.6 m，主干）、road 5（61.8 m，支路） | road 4（232.1 m，主干）、road 5（71.3 m，支路） |
| 连接道 | 4 条，40–89 m（渠化路口，转弯路径较长） | 4 条，62–123 m |
| 车道定义 | lane 级（driving/border + 变宽），主干道含双向车道与转向/渠化车道 | 同左 |
| 车辆轨迹落在车道面内（残差 ≤0.5 m）比例 | **99.82%** | **100.0%** |
| 残差 P50 / P100 | −1.66 m（车道内）/ +0.76 m | −1.43 m / +0.43 m |
| 轨迹缩放、平移、朝向与地图一致性 | 通过（叠加图目视 + 定量 KD-tree 残差） | 通过 |

结论：**坐标一致，轨迹基本全部位于可行驶车道上**；0.18% 的 A 残差点集中在转弯/路口区内，是车道中心线近似误差而非坐标错误。

图：`figures/automatum/map_tracks_recording_A.png`、`map_tracks_recording_B.png`（全场景 + 路口 50 m 放大）。

---

## 10. 同时存在车辆数量（N_max = 8 适配性）

| 指标 | A | B |
|---|---|---|
| mean | **6.45** | **3.51** |
| median | 6 | 4 |
| P90 / P95 / P99 / max | 10 / 12 / 15 / **18** | 6 / 7 / 8 / **10** |
| N=0 占比 | 0.29% | 9.55% |
| N=1 占比 | 1.15% | 10.15% |
| 2 ≤ N ≤ 4 占比 | **21.2%** | **47.0%** |
| 5 ≤ N ≤ 8 占比 | **59.8%** | **32.4%** |
| N > 8 占比 | 17.6% | **0.9%** |

判断（回答工单 §11）：**两个 recording 都天然容易形成 2–8 辆车的局部多车场景**。

- B 几乎全程 ≤8（99.1% 帧），是“天然 N≤8”的场景，可直接整场景使用；
- A 全局同时存在最多 18 辆，但这是**整个 135×120 m 场景**的计数；ICCT 的 N≤8 约束最终落实在“局部多车 scene builder”（例如以目标车为中心、20–30 m 邻域选取 ≤8 辆），A 的 P95=12 说明裁剪后仍可轻松满足；A 的场景密度反而提供更丰富的邻近交互（见 §11）。
- 两 recording 的密度差异（A 高、B 低）对 GNN/QGNN 是**有益分布多样性**，不是缺陷。

图：`figures/automatum/concurrent_vehicle_count_A.png` / `_B.png`。

---

## 11. 多车交互强度（是否真的适合 GNN/QGNN）

统计口径：逐帧在场景内所有并存车辆之间计算（vehicle-frame 计数）。

| 指标 | A | B |
|---|---|---|
| 有邻车（≤10 m）的 vehicle-frame 占比 | **34.0%** | 19.1% |
| 有邻车（≤20 m）占比 | **65.8%** | 45.7% |
| 有邻车（≤30 m）占比 | 80.9% | 67.5% |
| 最近邻距离 P25 / P50 / P90 | 7.9 / **14.1** / 42.7 m | 12.3 / **21.9** / 55.7 m |
| 10 m 内邻车数 mean / P95 / max | 0.49 / 2 / 5 | 0.22 / 1 / 4 |
| 20 m 内邻车数 mean / P95 / max | 1.34 / 4 / 7 | 0.66 / 2 / 5 |
| 30 m 内邻车数 mean / P95 / max | 2.21 / 6 / 10 | 1.18 / 3 / 6 |
| 30 m 内成对 heading 差 P50 / P75 | 0.84 rad / 3.10 rad | 0.57 rad / 2.75 rad |
| 30 m 内配对：同向<30° / 交叉 30–150° / 对向>150° | 61,388 / 26,690 / 50,480 | 32,063 / 17,420 / 18,261 |
| 存在交叉型配对的帧占比 | **52.0%** | 26.9% |
| 路口“忙帧”数（60 m 内运动车辆） | 19,199 | 28,426 |
| 其中“主干道 + 支路同时出现”的帧占比 | **15.3%** | **29.6%** |

代表性时间段（脚本自动选出，见 JSON `interaction.representative_windows`）：

| Recording | 时段 | 同时车辆（峰值/均值） | 构成 |
|---|---|---|---|
| A | 62.2–66.2 s | 18 / 16.1 | 15 car + 3 truck |
| A | 319.3–323.3 s | 15 / 14.2 | 16 car |
| B | 982.4–986.4 s | 10 / 9.4 | 12 car |
| B | 547.2–551.3 s | 10 / 9.0 | 10 car + 1 van |

结论：**不是大量互不相关的单车轨迹**。两个路口都存在稳定的跟驰（同向对数最多）、对向通过、以及转弯/汇入/让行产生的交叉型配对；A 有半数以上帧存在交叉型配对，B 近三分之一帧存在交叉配对且有 29.6% 的路口忙帧同时有主干道+支路车辆（转弯/让行交互的核心情形）。对 GNN/QGNN 的邻接关系与交互建模而言，**存在真实且可统计的交互结构**。

图：`figures/automatum/nearest_neighbour_distance.png`。

---

## 12. 29.97 Hz → 10 Hz 可行性

前提：源时间戳是**精确均匀**的 29.97 Hz 网格（0.0333667 s），而非 30.00 Hz。因此“每隔 3 帧取一次”并不精确等于 0.1 s。

两种方法实测（在真实时间戳上离线模拟，未生成 10 Hz 数据）：

| 指标 | 方法 A：每 3 帧取一帧 | 方法 B：取最接近 0.1 s 网格的样本 |
|---|---|---|
| 采样间隔 | 恒 0.10010010 s | 步长 3 为主，偶发 2（A: 20 次；B: 34 次），平均 0.1 s |
| 对 0.1 s 网格的最大时间误差 | **0.651 s（A）/ 1.127 s（B）**，随时间线性漂移 | **0.0167 s = ±0.5 帧**，无累计漂移 |
| 结束时漂移 | +0.651 s（650 s 后）/ +1.127 s（1126 s 后） | −0.0161 s（A）/ −0.0077 s（B） |
| 有效平均频率 | 9.9900 Hz | 10.00025 Hz（A）/ 10.00007 Hz（B） |
| 与对方方法的位置差（A / B） | P95 0.303 / 0.253 m，max 1.08 / 0.64 m | — |
| 与对方方法的速度差（A / B） | P95 0.031 / 0.033 m/s，max 0.16 / 0.11 m/s | — |

说明：单条轨迹内部的漂移远小于上表（轨迹中位仅 ~8–14 s），上表是整段录制的累计效应；A 方法在 4 s 窗口内部的时间偏差仅 ~4 ms，因此**对窗口内训练精度影响很小**，但方法 B 在“与绝对时间对齐、跨窗口拼接、ISAC 帧时序对齐”上严格占优。

**推荐：方法 B（按真实时间戳选择最接近 0.1 s 网格的源样本）**。数据集时间戳严格均匀，使方法 B 无需插值即可把误差保持在半帧以内；本轮未执行转换。

机器可读结果：`stats.json → recordings.*.downsample_10hz`。

---

## 13. 固定任务 2 s history → 2 s future 的窗口供给（10 Hz 估算）

窗口定义（写入脚本与 JSON，避免歧义）：**41 个连续 10 Hz 采样（4.0 s，含当前帧），起点对齐 0.1 s 网格，要求整窗落在该车轨迹内**。以下为索引级模拟统计，**未生成任何训练样本**。

| 指标 | A | B | 合计 |
|---|---|---|---|
| 单车有效窗口（每车可开窗数之和） | 30,018 | 24,095 | **54,113** |
| 处于“≥2 车同时可开窗”场景的窗口数 | 29,711 | 21,988 | 51,699 |
| **处于 2–8 车同时可开窗场景的窗口数** | **24,853** | **21,988** | **46,841** |
| 处于 >8 车场景的窗口数 | 4,858 | 0 | 4,858 |
| 多车锚点数（≥2 车）/ 其中 2–8 车 | 6,027 / 5,544 | 6,950 / 6,950 | 12,977 / 12,494 |
| 同一锚点并发开窗数 P50 / P95 / max | 4 / 9 / 13 | 2 / 5 / 7 | — |

解读：

- B：**全部多车窗口（21,988）天然满足 N≤8**，连裁剪都不需要；
- A：4,858 个窗口所在锚点全局车辆数 >8，scene builder 需按局部邻域（如 25 m）裁剪到 ≤8——这是预处理步骤，不是数据缺陷；
- 合计约 **4.7 万个 2–8 车 4 s 窗口**，对 GNN/QGNN 训练规模充足，且可用场景/时间双维度划分 train/val/test。

图：`figures/automatum/multi_vehicle_window_supply.png`。

---

## 14. 3-BS ISAC 几何适配性（仅可行性示意，未冻结坐标）

从 xodr 提取路口中心与 3 条进近臂（主干道双向 + 支路），候选 BS 放置于各臂路侧（自路口沿臂 30 m、横向 8 m）：

| 指标 | A | B |
|---|---|---|
| 候选 BS 数 | 3（主干道 NW、主干道 SE、支路 S） | 3（主干道 NE、主干道 SW、支路 N） |
| 三角形面积 | 1,815 m² | 1,798 m² |
| 三角形角度 | 62° / 62° / 56° | 88° / 44° / 48° |
| 非共线性 | **成立**（最小角 56°） | **成立**（最小角 44°） |
| 车辆到最近 BS 距离 | P50 21.8 m，P95 53.3 m，max 69.3 m | P50 21.4 m，P95 68.7 m，max 82.1 m |
| 车辆点落在 100 m 覆盖内比例 | **100%** | **100%** |

结论（回答工单 §15）：

1. T 型路口与 3 个固定路侧 BS **天然匹配**（每臂一个，无需在直线路段硬凑）；
2. 三臂张角给出良好非共线几何（最小内角 44°–56°，不存在“近直线阵列”导致的方位模糊）；
3. 三个 BS 可同时覆盖主要交互区域（路口 + 三条进近），全部轨迹点在 100 m 内；
4. 车辆–BS 距离范围约 21–82 m（中位 ~21 m），正适合 ISAC sensing simulation 的近中距场景；
5. 路口尺度（活动区 ~135–170 m，凸包 ~7,700–7,800 m²）既不局促也不过散；
6. 未发现道路过大/过小问题。注意：这是 **geometry feasibility illustration**，不是正式实验配置。

图：`figures/automatum/bs_layout_recording_A.png` / `_B.png`。

---

## 15. 两个路口是否应该一起使用

| 维度 | 方案 A：只用 B（18.8 min） | 方案 B：A + B 一起（29.6 min / 683 车） |
|---|---|---|
| 单车 4 s 窗口 | 24,095 | **54,113** |
| 2–8 车窗口 | 21,988 | **46,841** |
| 同时车辆均值 | 3.51（几乎全程 ≤8） | 6.45 / 3.51（两种密度） |
| 交互强度 | 交叉配对帧 26.9% | 52.0% / 26.9%，两种强度 |
| 类型构成 | car 93.2% / truck 4.2% / van 2.6% | 加 A 的 10.4% truck，卡车/厢式更丰富 |
| 速度分布 | 中位 64.6 km/h | 46.0 / 64.6 km/h，双峰分布 |
| GNN/QGNN 泛化 | 单场景内时间划分，易过拟合单一路口布局 | **跨路口域泛化可验证**（2 个路口天然域分离） |
| Train/Val/Test 划分 | 只能按时间切 | 可按路口（leave-one-intersection-out）+ 按时间双策略 |
| 3-BS 配置 | 1 套 | 2 套独立（两路口坐标相距 5 km，互不影响，复杂度线性增加） |
| 训练时间 | 基线 ×1 | ×1.8（数据量 1.8×） |

判断：**推荐方案 B（两个一起使用），但把 recording B 作为主训练/评估场景、recording A 作为交互增强与跨域验证场景**。理由不是“数据更多”，而是：A 与 B 的密度、车型、速度与侧路交互比例显著不同（A 更密更慢卡车更多，B 更快更稀疏），这正是 GNN/QGNN 公平比较所需的分布多样性；683 个车辆 UUID 全局唯一使合并零冲突；两路口各自 3-BS 互不干扰。单用 B 的代价是失去域泛化验证与高密度交互样本（A 贡献 53.1% 的 2–8 车窗口）。

---

## 16. 清洗工作量判断

**A 类：真正的数据清洗（修脏）——本数据集为 0 项。**

| 常见脏数据项 | 实测 |
|---|---|
| 缺失值 / NaN / Inf | 0 |
| 断轨 / 缺帧 | 0 |
| 时间错误（倒退、重复、抖动） | 0 |
| 跳点 / 瞬移 | 0 |
| 重复轨迹 / 重复行 | 0 |
| ID 冲突 | 0（UUID 全局唯一） |
| 文件序错位 | 0（已严格排序） |

**B 类：项目预处理（项目约定，不是修脏）：**

1. 29.97 Hz → 10 Hz（推荐方法 B，见 §12）；
2. `vehicle_id` 由 `(recording_name, UUID)` 生成；`scene_id = recording`；
3. 车辆筛选（如需剔除极短轨迹，本数据集 ≥4 s 占比已达 96.7%/99.5%）；
4. 2 s→2 s 窗口生成（约 5.4 万窗口）；
5. N≤8 局部 scene builder（主要对 A 需要）；
6. Train/Val/Test 划分（按路口与时间）；
7. 局部坐标转换（当前已是局部米制坐标，仅需记录 UTM 参考点）；
8. 若需要车道/道路标签：用 xodr 投影生成（数据集不提供，见 §2）。

把 B 类称为“脏数据修复”是不准确的。

---

## 17. 与 NGSIM Lankershim 的项目层面对比

详见 `reports/data_audit/automatum_vs_ngsim.md`（含逐项证据与数字）。摘要：

| 维度 | NGSIM Lankershim | Automatum T-Crossing | 胜方 |
|---|---|---|---|
| 数据时长 | 2 时段共 2,167 s（含 100.6 s 重叠） | 1,776 s，无重叠 | NGSIM（长 22%） |
| 车辆数（轨迹） | 2,442 条 | 683 条 | NGSIM（多） |
| 原始频率 | 10 Hz | 29.97 Hz（精确均匀） | NGSIM（原生 10 Hz） |
| 坐标单位 | 英尺，Local 跨时段有平移 | 米，与 xodr 同系 | Automatum |
| vx/vy | 仅标量 v_Vel/v_Acc | vx/vy（车体系）+ ax/ay + jerk + curvature | Automatum |
| ID 问题 | 936 个 ID 跨时段复用 | UUID 全局唯一 | Automatum |
| 重复轨迹 | 4 条边界重复 | 0 | Automatum |
| 时间排序 | 228 负步 + 440 个 200 ms/≥1 s 假跳变 | 0，严格单调 | Automatum |
| 地图质量 | 无地图文件 | lane 级 xodr，对齐 99.8–100% | Automatum |
| 清洗复杂度 | 中（排序、会话键、单位、去重） | **无** | Automatum |
| 预处理复杂度 | 低（原生 10 Hz） | 中（29.97→10 Hz，一次） | NGSIM（略） |
| 多车交互 | 极密（63–86 辆/帧） | 天然 2–8 辆，交互可统计 | 取决于目标：ICCT 选 Automatum |
| 3-BS 几何 | 500 m 直线干道，近共线 | T 型三臂，非共线三角 | Automatum |
| 2s→2s 适配 | 好 | 好（均匀网格，窗口供给明确） | 平 |
| N≤8 场景适配 | 需从 60+ 车中重度裁剪 | **天然满足（B）/ 轻裁剪（A）** | Automatum |
| 训练规模 | 1.6 M 状态点（4 s 窗口估算 ~1.5 M） | 244 k 状态点 / 54 k 窗口 | NGSIM（点 6.6×、窗口 ~28×，但 ICCT 不需要） |

---

## 18. 迁移结论

- **Q1：Automatum 原始轨迹质量是否足够？→ PASS**
  683 辆轨迹 0 缺失、0 断轨、0 时间错误、0 跳点、0 重复、0 ID 冲突；位置/速度/航向内部一致；与地图对齐 99.8%–100%；与厂商 KPI 交叉吻合。
  （唯一“缺口”是 aux 字段为空——但那是元数据缺失，不是轨迹质量问题，且可用 xodr 自行投影。）

- **Q2：是否显著比当前 NGSIM 更省数据清洗工作？→ YES**
  NGSIM 需要行序修复、会话键、单位换算、边界重复处理与坐标平移核对；Automatum 无需任何修复动作，直接进入预处理。

- **Q3：是否适合作为 ICCT（3 fixed BS + multi-vehicle ISAC + N_max=8 + 2s→2s + GNN/QGNN）的 Ground Truth？→ SUITABLE**
  证据：① 3-BS 非共线几何天然成立（最小角 44°–56°，100% 车辆点 <100 m）；② 2–8 车 4 s 窗口合计 46,841 个；③ 交互结构真实（交叉配对、主支路同时接近、跟驰/对向/转弯）；④ 官方 vx/vy 可直接作为 GT（需按车体系使用）。需遵守的约定（非阻塞）：vx/vy 车体系语义、A 需局部裁剪到 N≤8、车道标签需自行投影。

- **Q4：是否建议正式从 NGSIM 迁移到 Automatum？→ RECOMMEND MIGRATION**
  在“省清洗 + 地图对齐 + 稳定 UUID + 可控 N≤8 + 3-BS 几何 + 真实交互”六个决定性维度上 Automatum 全面优于 NGSIM；仅“原始时长为 82%、训练规模 1/27”两项落后，而 ICCT 目标（最多 8 车、2s→2s、训练时间可控）并不需要 NGSIM 的规模。建议：正式迁移 Automatum 为 ICCT 主 Ground Truth；NGSIM 暂时保留归档，不再作为主线（不删除现有数据，本轮亦未清理任何服务器数据）。

---

## 19. 本轮输出与 Git 状态

| 类别 | 文件 |
|---|---|
| 主报告 | `reports/data_audit/automatum_t_crossing_audit.md`（本文件） |
| 统计 JSON | `reports/data_audit/automatum_t_crossing_stats.json` |
| 逐车 CSV | `reports/data_audit/automatum_vehicle_stats.csv`（683 行） |
| 异常 CSV | `reports/data_audit/automatum_anomalies.csv`（0 条异常） |
| 对比报告 | `reports/data_audit/automatum_vs_ngsim.md` |
| 审计脚本 | `tools/data_audit/audit_automatum_t_crossing.py` |
| 审计图 | `reports/data_audit/figures/automatum/`：`map_tracks_recording_A/B.png`、`track_duration_distribution.png`、`concurrent_vehicle_count_A/B.png`、`speed_distribution.png`、`position_velocity_consistency.png`、`nearest_neighbour_distance.png`、`multi_vehicle_window_supply.png`、`bs_layout_recording_A/B.png` |

Git：分支 `module/isac`；只提交审计代码、报告、统计与审计图；`data/automatum_t_crossing/` 处于忽略状态（仓库 `.gitignore` 第 `data/` 条），ZIP、dynamicWorld.json、xodr、HTML、JPG、npz 等大文件不入库。

## 20. 证据等级与剩余不确定项

- CONFIRMED：§1–§16 全部定量结论（可脚本复现）。
- STRONG EVIDENCE：vx/vy 车体系语义（物理与数值双重一致，无厂商文档佐证）。
- UNCERTAIN：绝对定位精度（视觉跟踪数据，无独立 GPS/激光参考）；厂商 aux 字段为何为空；两路口录制日期与设备无元数据。