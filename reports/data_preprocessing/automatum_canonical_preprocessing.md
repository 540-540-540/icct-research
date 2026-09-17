# Automatum T-Crossing Canonical 数据正式预处理报告

Canonical dataset: `data/automatum_t_crossing/processed/trajectories_10hz.csv`
Scenes: 2（scene 0 = Gaimersheim Stadtweg / Recording A；scene 1 = St2214 Dünzlau Umgehung / Recording B）
Vehicles: 683（scene 0: 299，scene 1: 384）
Rows: 81,554（scene 0: 41,995；scene 1: 39,559）
Frequency: 10 Hz（Δt = 0.1 s；源 29.97 Hz → 最近 0.1 s 网格采样，无插值）
Columns: `scene_id, vehicle_id, timestamp, x, y, vx, vy`（共 7 列）
Timestamp rule: 写入 **target 0.1 s 网格时间**；每个点来自真实原始帧，最大映射误差 ≤ 16.6834 ms
Velocity coordinate: **世界系**（`vx_w = cos(psi)·vx_body − sin(psi)·vy_body`；`vy_w = sin(psi)·vx_body + cos(psi)·vy_body`，psi 使用原始连续角）
Validation: **PASS**（`reports/data_preprocessing/automatum_canonical_validation.json`，30/30 checks 通过）
SHA256（canonical 三件套，重复 build 完全一致）:

| 文件 | 行数 | 大小 | SHA256 |
|---|---|---|---|
| `trajectories_10hz.csv` | 81,554 | 4,265,176 B | `2a4b38fa3ffafd6cf4ca28463532c7fb95cadf5bfc0bded40b2da2c3760b203e` |
| `vehicle_id_mapping.csv` | 683 | 59,400 B | `3d211ffd3c9029a3e0fe435a8e96ef3555ba36387665fbca9d39802569b2bbcb` |
| `scene_metadata.json` | 2 scenes | 3,703 B | `708f85c497bf4f2afb2cbb0026952b07979d89abc3be9d33b9aa35930cbf72bc` |

---

## 0. 工单遵守声明

- 输入只读：未修改 ZIP、`dynamicWorld.json`、`staticWorld.xodr`（原文件 SHA256 已记录并在验证中复核）。
- 未划分 Train/Val/Test；未生成 2s→2s 样本；未做 N≤8 筛选/ROI；未布置 BS；未生成 ISAC 量测；未构造 GNN/QGNN 图；未改任何模型；未动 NGSIM 数据。
- 两个 recording 保持为两个独立 scene，未拼接时间线。
- 原始 aux 字段（lane/road/TTC 等）为空，本轮**未自行补造**，也未做 map matching。

---

## 1. 执行环境与输入

- 实际执行环境：Syncthing 同步的本地仓库副本 `E:\NJUPT\ICCT会议`（与服务器 `/home/dell/YrM/ICCT` 同一份数据；服务器可直接用命令复现）。
- 审计依据：`f03e230d094abb0a3f268c2b7df9db7c6c675552`（`reports/data_audit/automatum_t_crossing_audit.md`，结论：29.97 Hz 均匀网格、vx/vy 为车体系、UUID 全局唯一）。
- 原始 recording 目录（实际解压位置 `data/automatum_t_crossing/raw/extracted/`）：

| scene | 目录 | dynamicWorld.json SHA256 | staticWorld.xodr SHA256 |
|---|---|---|---|
| 0 | `T-Crossing--GaimersheimStadtweg_e2e6-e2e6f4bb-4668-4654-ac7e-bcd90c9df4c2` | `899037a7be1f61dd75915fbdeab32e8e2c01c8d2b4be4e814775216896f2cd19` | `afee5cbdad64d2098cd64e95d2df845ab940a433aee995beb258b604dbdeef55` |
| 1 | `T-Crossing-St2214-DuenzlauUmgehung_1b9b-1b9bf4b8-9fa6-4d23-abd2-2b715d087e8f` | `72c5bfbbcfc7ed59a258f543f8c74fad787ee0bdaaa1236fc354564d618fcce2` | `641d193b5cd502406a495ab71ec8738ffac7376a0d8fe12f642b1199649f300e` |

（`scene_metadata.json → raw_file_sha256` 与上表一致，验证脚本会重新计算并复核。）

---

## 2. 预处理方法（全部规则已冻结在 build 脚本）

1. **10 Hz 网格**：每个 scene 独立建立 `0.0, 0.1, 0.2, …, ⌊t_last/0.1·0.1` 网格。
2. **时间网格最近邻采样**（禁止每 3 帧固定抽取）：对每个 target 时间取最近的真实原始帧；
   仅当 `|source_time − target_time| ≤ 0.5/29.97 = 0.016683 s` 时选中（实际由网格最近邻天然保证）。
3. **无插值**：canonical 点全部复制真实观测帧的 `x/y` 与 `vx/vy`（车体系），不生成任何新状态。
4. **写 target 时间**：最终 `timestamp` 为 0.1 s 网格值，而不是 source_time。
5. **源帧不可复用**：相邻两个 canonical 时间点必须来自不同原始帧；实测源步长分布为
   scene 0 `{2: 124, 3: 41,572}`、scene 1 `{2: 126, 3: 39,049}`（步长 2 的出现是 29.97 与 10 非整数比的自然舍入，平均步长 2.997 帧，已验证相邻源帧索引严格递增）。
6. **车体系 → 世界系速度**：使用原始连续 `psi`（不 wrap），
   `vx = cos(psi)·vx_body − sin(psi)·vy_body`，`vy = sin(psi)·vx_body + cos(psi)·vy_body`。
7. **坐标**：`x/y` 保持官方局部米制世界坐标，不平移原点、不归一化、不缩放，与 `staticWorld.xodr` 同系。
8. **vehicle_id**：按 `(scene_id, 首次 canonical timestamp, 原始 UUID)` 排序后赋连续整数 `1..683`，无随机性、可重复生成。
9. **物理排序**：文件按 `scene_id → timestamp → vehicle_id` 升序。
10. **无筛选**：683 辆车（car/van/truck）全部保留；轨迹 <4 s 的车辆同样保留（canonical GT 原则：完整保存真实数据）。

脚本：

```
python tools/data_preprocessing/build_automatum_canonical.py \
    --raw-dir data/automatum_t_crossing/raw \
    --out-dir data/automatum_t_crossing/processed \
    --stats-out reports/data_preprocessing/automatum_canonical_stats.json

python tools/data_preprocessing/validate_automatum_canonical.py
```

---

## 3. 输出文件

| 文件 | 说明 | 是否入 Git |
|---|---|---|
| `data/automatum_t_crossing/processed/trajectories_10hz.csv` | canonical 主表（7 列） | 否（`data/` ignored） |
| `data/automatum_t_crossing/processed/vehicle_id_mapping.csv` | 追溯表：scene_id, vehicle_id, original_uuid, obj_type, length, width, original_first_time, original_last_time | 否 |
| `data/automatum_t_crossing/processed/scene_metadata.json` | scene 元数据 + 原始文件 SHA256 | 否 |
| `reports/data_preprocessing/automatum_canonical_stats.json` | 全部统计与输出 SHA256 | 是 |
| `reports/data_preprocessing/automatum_canonical_validation.json` | 独立验证结果（30 checks） | 是 |
| `tools/data_preprocessing/build_automatum_canonical.py` | 确定性构建脚本 | 是 |
| `tools/data_preprocessing/validate_automatum_canonical.py` | 独立验证脚本 | 是 |

---

## 4. 统计（工单 §22）

| 指标 | scene 0（Recording A） | scene 1（Recording B） | 合计 |
|---|---|---|---|
| raw points | 125,860 | 118,517 | 244,377 |
| canonical 10 Hz points | **41,995** | **39,559** | **81,554** |
| raw / canonical vehicles | 299 / 299 | 384 / 384 | 683 |
| raw duration | 650.651 s | 1,125.626 s | 1,776.277 s |
| canonical timestamp range | [0.0, 650.6] s | [8.5, 1125.6] s（首车出现前无行） | — |
| canonical x range | [−57.97, 76.13] m | [−29.23, 63.82] m | — |
| canonical y range | [−49.15, 69.28] m | [−83.48, 84.08] m | — |
| 时间采样误差 median / P95 / P99 / max | 8.041 / 15.849 / 16.517 / **16.6834 ms** | 8.609 / 15.883 / 16.517 / **16.6834 ms** | 8.342 / 15.849 / 16.517 / 16.6834 ms |
| 轨迹持续时间分布 <2 / 2–4 / 4–6 / 6–10 / 10–20 / >20 s | 0 / 10 / 5 / 145 / 91 / 48 | 0 / 2 / 12 / 268 / 71 / 31 | 0 / 12 / 17 / 413 / 162 / 79 |
| canonical 数据量 | — | — | 4,265,176 B（主表） |

时间误差上界 = half source frame = **16.683350 ms**，实测 max = **16.683350 ms**（未超界）。

---

## 5. vx/vy 世界系转换的独立验证（工单 §10–§11）

10 Hz canonical 数据上的位置差分与官方速度对比（两种口径都报告，避免“挑好看的”指标）：

| 口径 | scene | MAE vx | RMSE vx | MAE vy | RMSE vy | P95 (vx/vy) | P99 (vx/vy) | max (vx/vy) |
|---|---|---|---|---|---|---|---|---|
| **真实源间隔**（Δp / 实际源 Δt，旋转正确性判据） | 0 | **0.0330** | 0.0482 | **0.0310** | 0.0461 | 0.105 / 0.100 | 0.167 / 0.169 | 0.33 / 0.37 m/s |
|  同左 | 1 | **0.0344** | 0.0506 | **0.0390** | 0.0555 | 0.113 / 0.121 | 0.178 / 0.184 | 0.33 / 0.49 m/s |
| 0.1 s 标签间隔（Δp / 0.1 s，工单 §10 字面口径） | 0 | 0.0437 | 0.188 | 0.0401 | 0.166 | 0.110 / 0.105 | 0.179 / 0.186 | 5.75 / 4.77 m/s |
|  同左 | 1 | 0.0447 | 0.180 | 0.0575 | 0.286 | 0.118 / 0.129 | 0.195 / 0.200 | 5.14 / 7.84 m/s |

方向误差（运动点，相对位置差分方向）：

| scene | median | P95 | P99 | max |
|---|---|---|---|---|
| 0 | **0.067°** | 0.90° | 2.26° | 6.95° |
| 1 | **0.071°** | 0.95° | 2.12° | 6.54° |

判定：

1. **无系统性错误**：若存在 x/y 交换、正负号错误、psi 旋转方向错误或 90°/180° 偏转，方向误差中位数会落在数十度到 180°；实测中位 0.07°、P95 <1°，bias 近 0（vx −0.0004～+0.0001 m/s）。
2. **真实源间隔口径下误差完全符合物理预期**：瞬时速度 vs 区间弦平均的差 ≈ a·Δt/2，最大加速度 5.8 m/s² 对应 ~0.3 m/s，实测 max 0.33–0.49 m/s。
3. **0.1 s 标签口径的 max 尖峰已定位**：来自 250 对“步长 2”标签点（源帧间隔 0.0667 s 却按 0.1 s 计），仅出现于高速直线段，属最近邻网格法的已知有界效应，不是旋转错误；MAE/P95 与真实口径同量级（0.04–0.06 / ~0.12 m/s）。
4. **速度模长保持**：旋转前后模长差 max = 7.1×10⁻¹⁵ m/s（纯浮点误差）→ PASS（工单 §11）。
5. 结论：世界系 vx/vy **可直接作为 ICCT Ground Truth 速度**；无需重新计算速度，无需再平滑。

---

## 6. 独立验证（工单 §23–§26）

`validate_automatum_canonical.py` 直接重新读取 raw JSON、CSV、mapping、metadata 与 stats，用独立实现复核（不复用 build 内部函数），**30/30 通过**：

| # | 检查 | 结果 | 关键数值 |
|---|---|---|---|
| 1 | schema 恰为 7 列且顺序正确 | PASS | header 完全匹配 |
| 2 | scene_id 只有 0/1 | PASS | {0, 1} |
| 3 | vehicle_id 连续 1..683 | PASS | 683 个、min 1、max 683 |
| 4 | 两 scene 车辆不混用 | PASS | scene0 299 / scene1 384，重叠 0 |
| 5 | `(scene_id, vehicle_id, timestamp)` 无重复 | PASS | 81,554 行全唯一 |
| 6 | timestamp 严格 0.1 s 网格 | PASS | max 网格偏差 0 |
| 7 | 每车 timestamp 严格递增 | PASS | 0 违规 |
| 8 | 每车相邻 Δt 均为 0.1 s | PASS | 0 例异常间隔 |
| 9 | NaN / Inf = 0 | PASS | 0 / 0 |
| 10 | x/y/vx/vy 全有限 | PASS | 全有限 |
| 11 | 无位置瞬移 | PASS | max 帧间速度 30.66 m/s ≤ 45 |
| 12 | 速度物理合理 | PASS | max |v| = 30.66 m/s ≤ 35 |
| 13 | world vx/vy 与位置差分方向一致 | PASS | 方向中位 0.07°，P95 0.95° |
| 14 | 旋转前后模长一致 | PASS | max 差 7.1e-15 m/s |
| 15 | 相邻 target 不复用同一 source frame | PASS | 源帧索引严格递增 |
| 16 | canonical 点 x/y 等于真实源帧 | PASS | 容差 2e-6 m，0 违规 |
| 17 | mapping 可追溯（683 行、UUID 唯一、与原始 first/last time/尺寸一致） | PASS | 0 mismatch |
| 18 | 原始文件未被修改 | PASS | 两 scene 的 dynamicWorld.json / xodr SHA256 与 build 记录一致 |
| 19 | 文件按 scene→timestamp→vehicle 排序 | PASS | 单调 |
| 20 | scene_metadata 与主表一致 | PASS | 行数/车辆数/时间范围一致 |
| 21 | ≥4 s 统计与 stats 一致 | PASS | 289 / 382 |
| 22 | 三个输出的 SHA256 与 stats 记录一致 | PASS | 全部匹配 |

验证输出：`reports/data_preprocessing/automatum_canonical_validation.json`；`verdict = PASS`。

### 可复现性（工单 §24）

- build 脚本无随机数、无时间戳、无人工编辑输入；重复运行两次，三个输出 SHA256 **完全一致**（见开头表）。
- 文件写入统一使用 LF 换行与固定小数位（timestamp `%.1f`，x/y/vx/vy `%.6f`），跨平台一致。

---

## 7. 2 s history → 2 s future 统计（工单 §28，仅统计，不生成样本）

正式 sample builder 的窗口定义（下一轮冻结）：10 Hz 下 history = 2 s = 20 个输入时间步，future = 2 s = 20 个目标时间步，共 **40 个状态时间步**。本轮只统计轨迹供给：

| 指标 | scene 0 | scene 1 | 合计 |
|---|---|---|---|
| 具备至少 4.0 s canonical 轨迹的车辆 | **289** | **382** | **671 / 683（98.2%）** |
| 具备 ≥40 个 canonical 状态点的车辆 | 289 | 382 | 671 |

未被计入的 12 辆车（10 + 2）轨迹 2–4 s，仍完整保留在 canonical 数据中；是否可开窗由下一轮 sample builder 判定。

---

## 8. 明确未做的事项

- 未做 Train/Val/Test 划分（含随机/按车/按时间的一切 split）。
- 未生成任何 prediction window 或张量。
- 未做 N≤8 筛选、ROI 定义、局部坐标归一化、目标车为原点的表示。
- 未布置 3 个 BS，未生成 range/AoA/Doppler。
- 未构造图结构、未改 GNN/QGNN/ISAC 代码。
- 未删除或继续加工 NGSIM（raw/processed/reports 原样保留）。
- 未补 lane/road/TTC 字段，未做 map matching。

---

## 9. 验收标准对照（工单 §32）

| # | 标准 | 结论 | 证据 |
|---|---|---|---|
| A | 原始 Automatum 文件未修改 | PASS | validator 复核两组 raw SHA256 |
| B | 两 scene 独立保留 | PASS | scene_id 0/1，车辆集合零重叠，无合并时间线 |
| C | 683 辆车辆全部可追溯 | PASS | `vehicle_id_mapping.csv` 683 行，UUID 唯一且对应 raw |
| D | 10 Hz 时间网格正确 | PASS | 全部 timestamp 在 0.1 s 网格，逐车 Δt=0.1 s |
| E | 最大时间映射误差 ≤16.684 ms | PASS | max 16.6834 ms |
| F | 无人工轨迹插值 | PASS | 每点来自真实源帧（x/y 与源帧逐点一致，容差 2e-6 m） |
| G | vehicle_id 全局唯一连续 | PASS | 1..683 连续、两 scene 不重复 |
| H | x/y 保持官方米制世界坐标 | PASS | 无平移/缩放/归一化，与 xodr 同系 |
| I | vx/vy 正确转为世界系 | PASS | 公式 + 独立旋转复核 |
| J | 速度旋转通过独立验证 | PASS | 方向中位 0.07°，MAE ≤0.04 m/s（真实口径） |
| K | 无 NaN / Inf | PASS | 0 / 0 |
| L | 无重复状态 | PASS | 主键 81,554 行全唯一 |
| M | 构建完全可复现 | PASS | 两次 build SHA256 一致 |
| N | 输出 SHA256 固定 | PASS | 三件套 SHA256 已记录 |
| O | 未提前做 split / windows / BS / 模型数据 | PASS | 见 §8 |

最终判定：**PASS**。

---

## 10. 遗留不确定项（不影响本轮验收）

1. 官方 `vx/vy` 绝对精度仍无外部真值参考（视觉跟踪数据；内部一致性已验证到 0.04 m/s）。
2. 步长 2 的 250 个标签点使 §10 字面口径的 max 偏差达 7.8 m/s；若下一轮 sample builder 需要严格的 0.1 s 物理间隔，可在**不插值**的前提下把“步长 2”点按最近真实帧重排（例如允许步长只取 3/4），这属于表示层决策，本轮未做。
3. `scene_metadata.json` 的 y/x 范围来自 canonical 行样本，与审计报告中的全 raw 范围存在 ≤0.2 m 的采样差异（正常）。