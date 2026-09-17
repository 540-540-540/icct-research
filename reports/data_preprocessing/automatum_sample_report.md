# Automatum T-Crossing 20+20 多车预测样本构建报告

- 性质：**SAMPLE BUILD ONLY**。未修改 canonical GT、scene_id、vehicle_id、9.99 Hz 采样、Train/Val/Test 边界、40-frame gap；未做坐标变换/图/ISAC/模型预处理；未做数据增强。
- 母数据：`splits/{train,val,test}/trajectories.csv`，三份 SHA256 与旧 split CSV **字节一致（SHA 未变）**：

| split | trajectories SHA256 |
|---|---|
| train | `d75994cb999d10ac5c05fab00f95a11d98b8e8cc61c475b87373ba6b51a82a11` |
| val | `c9964adb9b73e828754912381f5d93cfd8adf215bad4ba85db766d4166ab894d` |
| test | `6fb3117e95368e2c5b66cca5a28cb45a1ff91691d4afea4fb9b8c1a17eb75371` |

- 构建脚本：`tools/data_preprocessing/build_automatum_samples.py`
- 验证脚本：`tools/data_preprocessing/validate_automatum_samples.py`
- 统计：`reports/data_preprocessing/automatum_sample_stats.json`
- 验证结果：`reports/data_preprocessing/automatum_sample_validation.json`（**37/37 PASS**）
- 样本 manifest Git 镜像：`reports/data_preprocessing/sample_manifests/{train,val,test}_sample_manifest.json`（与 runtime 逐字节一致）

## 1. 目录结构（已完成迁移）

```
data/automatum_t_crossing/splits/
├── train/
│   ├── trajectories.csv        (= 旧 train.csv，字节一致)
│   ├── samples.npz
│   └── sample_manifest.json
├── val/  (同上)
├── test/ (同上)
└── split_manifest.json         (outputs.path 已更新为新路径，边界未变)
```

旧平铺 `train.csv / val.csv / test.csv` 在验证字节一致后已移除，不保留两套正式 split CSV。

## 2. 样本定义（冻结）

| 项 | 值 |
|---|---|
| history | 20 canonical frames |
| future | 20 canonical frames |
| total | 40 consecutive canonical frames |
| canonical fps | 9.99 Hz（名义 2s → 2s，代码以 20+20 frames 为准） |
| window stride | **1**（逐帧，完整保留所有合法窗口） |
| 车辆数 N | **2 ≤ N ≤ 8**（8 为整样本车辆总数上限） |
| eligible 规则 | 车辆在 40 帧中 **40/40 都存在** |
| N > 8 | **整窗拒绝**（禁止选最近 8 辆/随机 8 辆/截前 8 辆） |
| 样本内排序 | `vehicle_id` 升序 |
| 状态 | `[x, y, vx, vy]`，state_dim=4，直接来自 trajectories.csv，无任何变换 |
| padding | 固定 8 车槽：state=0，`vehicle_id=-1`，`mask=0`，`num_vehicles=N` |
| 数据类型 | history/future float32；vehicle_ids int32；mask bool；num_vehicles/scene_id uint8；start_frame int32；start_timestamp float64 |

每车辆的真实 40 帧不允许中间补零（temporal padding 禁止）；padding 只出现在第 N+1…8 个车辆槽。

## 3. 样本数量与 N 分布

| | Train | Val | Test |
|---|---|---|---|
| **samples** | **10,366** | **893** | **1,135** |
| N=2 | 2,560 | 225 | 282 |
| N=3 | 2,455 | 186 | 266 |
| N=4 | 2,547 | 160 | 143 |
| N=5 | 1,690 | 189 | 92 |
| N=6 | 641 | 78 | 142 |
| N=7 | 283 | 25 | 154 |
| N=8 | 190 | 30 | 56 |
| **N>8 rejected** | **427** | **75** | **0** |
| 其中 scene 0 / scene 1 | 4,430 / 5,936 | 418 / 475 | 532 / 603 |

N 分布总和 = 样本数（校验通过）。

## 4. 与上一轮 anchor 统计的核对（§14）

| split | 上一轮 2–8 anchors | 本轮 samples | 上一轮 >8 anchors | 本轮 rejected | 结论 |
|---|---|---|---|---|---|
| Train | 10,366 | **10,366** | 427 | **427** | 一致 |
| Val | 893 | **893** | 75 | **75** | 一致 |
| Test | 1,135 | **1,135** | 0 | **0** | 一致 |

口径完全一致：`frame` = canonical frame（source_frame/3），连续性 = 同一 `(scene_id, vehicle)` 在 `[f, f+39]` 每帧都有行，边界 = 各 split frame range，scene 分开统计。

## 5. NPZ 内容与 SHA256

每个 `samples.npz` 包含：

| key | shape | dtype |
|---|---|---|
| history | [S, 20, 8, 4] | float32 |
| future | [S, 20, 8, 4] | float32 |
| vehicle_ids | [S, 8] | int32 |
| vehicle_mask | [S, 8] | bool |
| num_vehicles | [S] | uint8 |
| scene_id | [S] | uint8 |
| start_frame | [S] | int32 |
| start_timestamp | [S] | float64 |

| split | NPZ size | NPZ SHA256 |
|---|---|---|
| train | 53,635,566 B | `1b86a6453a53779b9b647f8fa04bf3b40c56bba0dfc26f0d19f6de895a58e0c9` |
| val | 4,622,264 B | `51dc713ba5873ba438267e5c46f97902e4e744cbd3069bf3e289a485d058ff31` |
| test | 5,874,372 B | `dec95982e48686e03e0ba059fb3a977f095b881342ed8ce9416c4c0ec8424413` |

可复现性：npz 写入固定 ZipInfo 时间戳（1980-01-01）并使用 ZIP_STORED，连续两次 build 三份 NPZ SHA256 完全一致；跨机器重建同样一致。

## 6. 精度（float32 仅 casting，无额外变换）

- 所有真实状态与 CSV float64 原值逐点比对：**存储值 == `np.float32(csv_value)` 精确成立**（0 例外）；
- **float32 cast max abs error = 3.814e-06**（161,457 个 float 值的最大值，符合 float32 正常精度）；
- 无中心化、相对坐标、归一化、旋转、加噪。

## 7. 样本边界与连续性

- 每个样本 40 帧严格为 `f … f+39`（来自 `start_frame` 与 CSV frame 回推），无跨 split、无跨 scene；
- `start_timestamp` 等于该 frame 在 CSV 中的真实时间（float64 精确相等）；
- 样本顺序为 `(scene_id, start_frame)` 递增（确定性）。

## 8. 独立验证（37/37 PASS）

`validate_automatum_samples.py` 独立重新解析 trajectories.csv、samples.npz、manifest（不调用 build 内部函数），覆盖：

- 输入：三份轨迹 SHA 与冻结值一致；NPZ SHA/大小与 manifest 一致；runtime 与 Git manifest 镜像字节一致；
- 数据结构：dtype/shape 全对；
- anchor 对账：每个 2≤N≤8 anchor 恰好出现一次、N>8 全部缺席、N 分布与 per-scene 计数与 manifest 一致；
- 逐样本 QA（18 项）：N 范围、mask 和、ids 前 N 真实/后补 -1、ids 唯一升序、padding 全 0、40 帧连续、**所有真实状态与 CSV float32 精确回查一致**、history=0..19、future=20..39、scene_id/start_frame/start_timestamp 正确、不跨 split/scene；
- 精度：仅 float32 casting 误差。

## 9. 本轮未做的事项

- 未做数据增强、相对坐标、邻接矩阵/edge feature、GNN/QGNN 输入、ISAC 误差、Dataset/DataLoader；
- 未改动 canonical / split 边界 / gap / 8:1:1 比例；
- `data/` 下 trajectories.csv、samples.npz、sample_manifest.json 保持 Git ignored。