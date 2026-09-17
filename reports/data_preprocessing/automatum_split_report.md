# Automatum T-Crossing Train / Val / Test 时间划分报告（8:1:1, 40-frame gap）

- 性质：**SPLIT ONLY**。未修改 canonical GT、vehicle_id、x/y/vx/vy、采样率；未插值/平滑；未做 N≤8、ROI、局部坐标、BS、ISAC、图结构；未生成任何 20+20 样本、sample index、Dataset/DataLoader。
- 母数据：`data/automatum_t_crossing/processed/trajectories_10hz.csv`，SHA256 **验证通过**
- 划分脚本：`tools/data_preprocessing/build_automatum_splits.py`
- 验证脚本：`tools/data_preprocessing/validate_automatum_splits.py`
- 机器可读统计：`reports/data_preprocessing/automatum_split_stats.json`
- Manifest（runtime + Git 镜像，字节一致）：`data/automatum_t_crossing/splits/split_manifest.json` / `reports/data_preprocessing/automatum_split_manifest.json`
- 验证结果：`reports/data_preprocessing/automatum_split_validation.json`（**19/19 PASS**）
- 复现：`python tools/data_preprocessing/build_automatum_splits.py && python tools/data_preprocessing/validate_automatum_splits.py`
- 目录迁移（后续工单）：三份 split CSV 已迁移为 `splits/{train,val,test}/trajectories.csv`，**字节与 SHA256 未变**；`split_manifest.json` 的 output 路径已同步更新，边界、gap 与比例均未改动。

---

## 1. 输入冻结检查

| 项 | 值 |
|---|---|
| canonical 路径 | `data/automatum_t_crossing/processed/trajectories_10hz.csv` |
| canonical SHA256（build 前/后均校验） | `a41ac4e24a8f21039dde60064868871b21093289a661ca9bef28ee7d1aab3442` |
| 冻结基线 commit | `c9721eaf1c36509744cbf4d34b2145d0eeaf1b7a` |
| 行数 | 81,450（与冻结统计一致） |
| 采样 | source 29.97 Hz，stride 3，canonical 9.99 Hz，Δt = 3/29.97 = 0.1001001001 s |

canonical SHA 不等于冻结值则直接 FAIL 并拒绝生成（脚本内置硬校验）。

## 2. 划分设计（写入脚本与 manifest 的确定性规则）

1. **两个 scene 各自独立**沿自身连续 canonical 时间轴切分，然后同名 split 合并（scene 0 + scene 1）。
2. 比例按 **canonical frame slots（时间）** 计算，不按 CSV 行数。
3. canonical frame 恢复规则：`source_frame = round(timestamp × 29.97)`，必须 `% 3 == 0`，`canonical_frame = source_frame // 3`（反推误差实测 < 1e-9 s，仅浮点尾差）。
4. 有效区间取每 scene 真正有数据的 `first_active_frame … last_active_frame`；`F = last − first + 1`，`usable = F − 2G`。
5. `G = 40` canonical frame slots（≈ 40/9.99 = 4.004 s）；两个 gap 内的数据保留在 canonical，不进入任何 split。
6. nominal 切点：`train = round(0.8·usable)`，`val = round(0.1·usable)`，`test = usable − train − val`；由此得 `nominal_gap1_start = first + train`，`nominal_gap2_start = nominal_gap1_start + G + val`。
7. **自然边界搜索**：`gap1`、`gap2` 起点分别在 nominal ±30 s = ±300 canonical frames 内联合搜索；候选对必须使 **每个 scene 的 train/val/test 时间比例与 0.80/0.10/0.10 的偏差 ≤ 2 percentage points**（[0.78,0.82]/[0.08,0.12]/[0.08,0.12]），否则 `NO_VALID_NATURAL_PAIR` 并停止（本轮两个 scene 均有大量可行解）。
8. **候选排序（确定性 lexicographic，无人工挑选）**：
   1. 两个 boundary 的 cross-gap vehicles 总数最少；
   2. 两个 gap 的 gap_rows 总数最少；
   3. mean_active_vehicles_in_gap 总量最少；
   4. max_active_vehicles_in_gap 总量最少；
   5. 与 nominal 的总偏移帧数最小；
   6. 仍相同则选更早的 (gap1, gap2)。
9. Split 区间（闭区间 frame slots）：Train `[first, g1−1]`、Gap1 `[g1, g1+39]`、Val `[g1+40, g2−1]`、Gap2 `[g2, g2+39]`、Test `[g2+40, last]`。
10. 输出 CSV **逐行原样复制** canonical 行文本（纯子集），排序保持 `scene_id → timestamp → vehicle_id`；同一 `(scene_id, timestamp)` 绝不跨 split。

## 3. 理论切点与搜索空间

| 项 | scene 0 | scene 1 |
|---|---|---|
| active range（frame / s） | [0, 6500] / [0.0, 650.6506506506507] | [85, 11245] / [8.508508508508509, 1125.6256256256256] |
| F（active slots） | 6,501 | 11,161 |
| usable slots（F − 80） | 6,421 | 11,081 |
| nominal g1（frame / s） | 5,137 / 514.2142142142143 | 8,950 / 895.8958958958959 |
| nominal g2（frame / s） | 5,819 / 582.4824824824825 | 10,098 / 1010.8108108108108 |
| 搜索范围（±300 frames） | 601 × 601 候选对 | 601 × 601 候选对 |
| 满足 ±2pp 约束的候选对 | **49,537** | **147,187** |

## 4. 最终选中边界与胜出原因

| 项 | scene 0 | scene 1 |
|---|---|---|
| selected g1（frame / s） | **5,198 / 520.3203203203203** | **9,050 / 905.9059059059059** |
| g1 shift（vs nominal） | +61 frames（+6.11 s） | +100 frames（+10.01 s） |
| g1 cross-gap vehicles | **2** | **0** |
| g1 gap rows | 169 | 113 |
| g1 mean / max active vehicles | 4.225 / 6 | 2.825 / 5 |
| selected g2（frame / s） | **5,872 / 587.7877877877878** | **10,098 / 1010.8108108108108** |
| g2 shift（vs nominal） | +53 frames（+5.31 s） | 0 frames（0 s） |
| g2 cross-gap vehicles | **5** | **0** |
| g2 gap rows | 257 | **0**（该 40-frame 窗口内无任何车辆数据） |
| g2 mean / max active vehicles | 6.425 / 7 | 0.0 / 0 |
| 胜出 pair 排序键 | cross 7、rows 426、mean 10.65、max 13、offset 114 frames | cross 0、rows 113、mean 2.825、max 5、offset 100 frames |

胜出原因：在所有满足比例约束的候选中，这两组是排序规则下的**字典序最小值**。scene 0 找不到 cross-gap 更少的可行解（最少 7 辆跨边界），故在 cross 相同约束下选择 gap rows 最少者；scene 1 存在 **0 辆跨边界**的可行解，并按 rows → mean → max → offset 取最小，`gap2` 恰好落在无车时段（0 行被丢弃）。

## 5. Split 时间长度与比例

| split | scene 0 slots / duration | scene 1 slots / duration | 每 scene 占比（slots/usable） |
|---|---|---|---|
| Train | 5,198 / 520.320 s | 8,965 / 897.397 s | 0.809531 / 0.809043 |
| Val | 634 / 63.463 s | 1,008 / 100.901 s | 0.098739 / 0.090967 |
| Test | 589 / 58.959 s | 1,108 / 110.911 s | 0.091730 / 0.099991 |

所有绝对值偏差 ≤ 2 percentage points（最大偏差：scene 0 Train +0.95pp、scene 0 Test −0.83pp、scene 1 Val −0.90pp）。

## 6. 行数 / 车辆数（row 与 time 比例不同是正常的）

| split | rows | 占 canonical 行数 | unique vehicles |
|---|---|---|---|
| train.csv | **66,652** | 81.83% | 565 |
| val.csv | **7,141** | 8.77% | 70 |
| test.csv | **7,118** | 8.74% | 55 |
| gap（不进任何 split） | 539 | 0.66% | — |
| 合计 | 81,450 | 100% | — |

Train 行数占比（81.83%）略高于时间占比（80.95%/80.90%）是因为 Train 时段车辆密度更高——按工单要求，主划分标准是时间比例，不因行数比例返工。

## 7. Gap 丢弃数据量

| scene | gap1 rows | gap2 rows | 合计 |
|---|---|---|---|
| 0 | 169 | 257 | **426** |
| 1 | 113 | 0 | **113** |
| 合计 | 282 | 257 | **539**（占 0.66%） |

## 8. Vehicle ID 跨集合情况（允许，如实披露）

| 交集 | 车辆数 | 原因 |
|---|---|---|
| train ∩ val | **2** | scene 0 有 2 辆车跨越 gap1（gap 前有数据且 gap 后仍有数据） |
| val ∩ test | **5** | scene 0 有 5 辆车跨越 gap2 |
| train ∩ test | **0** | — |

这些车辆的 Val/Test 段未被删除；后续 sample builder 必须按窗口所属时间块取数据。`vehicle_id` 后续不得作为模型输入特征。

## 9. 40-point 潜在窗口统计（仅统计，不生成样本/索引文件）

定义：某车在**同一 split 内**拥有 40 个连续 canonical frame，则从一个起点形成一个潜在 40-point（20 history + 20 future）窗口。

| split | 潜在单车窗口数 | 有窗口的车辆数 |
|---|---|---|
| Train | **44,661** | 557 |
| Val | **4,417** | 69 |
| Test | **5,024** | 50 |

（分 scene 明细见 `automatum_split_stats.json → window_supply.per_scene`。）

## 10. 多车 40-point anchor 分布（仅统计）

对每个合法 40-point 起始 frame，统计“接下来 40 帧内完整存在的车辆数”（按 scene 分别作为 anchor 计；下表为两 scene 合并）：

| split | anchor 总数 | ≥2 车 anchors | 2–8 车 anchors | >8 车 anchors |
|---|---|---|---|---|
| Train | 12,625 | 10,793 | 10,366 | 427 |
| Val | 1,202 | 968 | 893 | 75 |
| Test | 1,387 | 1,135 | 1,135 | 0 |

本轮**未**进行任何选车、未生成样本或图结构。

## 11. 输出文件与 SHA256

| 文件 | rows | size | SHA256 |
|---|---|---|---|
| `train.csv` | 66,652 | 4,308,924 B | `d75994cb999d10ac5c05fab00f95a11d98b8e8cc61c475b87373ba6b51a82a11` |
| `val.csv` | 7,141 | 460,201 B | `c9964adb9b73e828754912381f5d93cfd8adf215bad4ba85db766d4166ab894d` |
| `test.csv` | 7,118 | 458,816 B | `6fb3117e95368e2c5b66cca5a28cb45a1ff91691d4afea4fb9b8c1a17eb75371` |
| `split_manifest.json`（runtime） | — | 6,792 B | `21bee09b8ccea23ed1d234bbad938e0de9a076681a250e138eaaf7f8d3d25f46` |
| `split_manifest.json`（Git 镜像） | — | 6,792 B | 同上（字节一致） |

重复 build 两次，上述 SHA256 完全一致（确定性）。

## 12. 独立验证（19/19 PASS）

`validate_automatum_splits.py` 独立读取 canonical / 三份 CSV / 两份 manifest / stats，用独立实现复核并通过：

| # | 检查 | 结果 |
|---|---|---|
| 1 | parent canonical SHA256 = 冻结值 | PASS |
| 2 | 三份 split schema 严格 7 列且与 canonical 表头一致 | PASS |
| 3 | 所有 split rows 均存在于 canonical | PASS |
| 4 | split 数值逐字段等于 canonical（整行文本相等） | PASS |
| 5 | 各 split 内无重复行 | PASS |
| 6–8 | train∩val、train∩test、val∩test 行交集 = 0 | PASS |
| 9 | gap rows 不出现在任何 split | PASS |
| 10 | canonical 每行恰好属于 train/val/test/gap 之一 | PASS |
| 11 | 三份 CSV + gap = 81,450 行 | PASS |
| 12 | 同一 `(scene_id, timestamp)` 不跨 split | PASS（16,655 个 timestamp，0 违规） |
| 13 | 两 scene 各自按连续时间块划分，五个区间首尾相接且覆盖 active range | PASS |
| 14 | 每个 boundary gap 恰为 40 frame slots | PASS |
| 15 | selected 边界在 ±300 frames 搜索范围内 | PASS |
| 16 | 每 scene 时间比例偏差 ≤ 2pp | PASS |
| 17 | 三份 CSV 排序为 scene→timestamp→vehicle | PASS |
| 18 | 边界选择规则独立复算与 manifest 完全一致（含 feasible pair 数 49,537 / 147,187） | PASS |
| 19 | manifest rows/size/SHA 与文件一致；runtime 与 Git 镜像字节一致；canonical 未变 | PASS |

## 13. 本轮未做的事项（范围冻结）

- 未生成任何 20+20 样本、sample index、`*.npz`、Dataset/DataLoader、N≤8 选车结果或图结构；
- 40-point window 与多车 anchor 仅做统计，不形成任何新数据文件；
- 后续实验的正式 split 数据源即为 `train.csv / val.csv / test.csv`。