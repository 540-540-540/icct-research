# ICCT canonical 轨迹母数据清洗报告

- 性质：正式数据清洗（**不改模型、不划分 Train/Val/Test、不生成 GNN/QGNN 样本、不做 ISAC 感知**）。
- 依据：source-session 审计 commit `cdf3a28`（`reports/data_cleaning/source_session_audit.md`）。
- 原始数据只读：`/home/dell/YrM/ICCT/data/Lankershim_Vehicle_Trajectories.csv`
  - SHA256 `ec10b87ac065f41cdef143ce8e75f4c77d45e056c0094573d27beeb85447d180`（清洗前后一致，未修改）
- 机器可读统计：`reports/data_cleaning/canonical_cleaning_stats.json`（构建）、`reports/data_cleaning/canonical_validation_stats.json`（验证）
- 结论：**PASS**（全部 16 项校验通过；见 §6）

---

## 1. 输出

| 文件（服务器） | 大小 | 行数 | SHA256 |
|---|---|---|---|
| `data/processed/trajectories_clean.csv` | 103,010,868 B | 1,605,279（含表头） | `a546e44f560e993b5ce8ca0818a309ef180346c7595a71262a39f8296b53b153` |
| `data/processed/vehicle_id_mapping.csv` | 26,590 B | 2,443（含表头） | `ad15d01aa32d88f978f345a070789012fbed6508c9eaa65fe0deebbecc76258c` |

最终主表列**只有 6 列**：`vehicle_id,timestamp,x,y,vx,vy`

- `timestamp` = (Global_Time − 1118935680200 ms) / 1000，单位 s，单一时间轴；范围 **0.000 – 2066.8 s**
- `x, y` = Global_X/Global_Y × 0.3048，单位 m，**不平移原点、不归一化、不缩放**
- `vx, vy` = 对最终 x/y 的因果差分（m/s）：
  - 第 1 帧（首帧）：前向差分 `(x[1]−x[0])/0.1`（**保留首帧，不删除任何真实轨迹点**）
  - 第 i>0 帧：后向差分 `(x[i]−x[i−1])/0.1`
  - 不平滑、不插值
- 主表排序：`timestamp → vehicle_id`（同刻车辆相邻，便于后续多车场景构造）
- 轨迹完整性检查按 `vehicle_id → timestamp`（§6）

`vehicle_id_mapping.csv`：`vehicle_id,source_session_id,original_vehicle_id`，用于追溯（不参与模型训练）。普通车辆 1 行；4 辆跨 session 重复车各 2 行。

---

## 2. Session 整理（沿用审计结论）

| | Session 0 | Session 1 |
|---|---|---|
| 帧时钟 offset | 1,118,935,679,900 | 1,118,936,699,900 |
| Global_Time 范围 | 1,118,935,680,200 – 1,118,936,800,600 | 1,118,936,700,000 – 1,118,937,747,000 |
| 真实边界 | — | 1,118,936,700,000（= 08:45:00.0 PDT） |
| 源轨迹数 | 1,211 | 1,231 |

- 车辆在清洗阶段的主键为 `(source_session_id, original_vehicle_id)`；不跨 session 使用 `Vehicle_ID`。
- 两 session 位于同一真实时间轴（`timestamp` 连续），重叠区不按时间删除：审计确认两段为互补车队（仅 4 辆重复）。

---

## 3. 4 辆重复车辆合并

合并规则：真实 session 边界 `1118936700000 ms`（= timestamp 1019.8 s）之前取 Session 0 行，边界及之后取 Session 1 行；同 timestamp 只保留一条；不平均位置、不平滑。

| canonical vehicle_id | 来源 | 合并前行数 | 保留 S0 行(边界前) | 保留 S1 行(边界后) | 合并删除行 |
|---|---|---|---|---|---|
| 1208 | S0 #1418 + S1 #2 | 978 + 965 | 11 | 965 | 967 |
| 1209 | S0 #1420 + S1 #6 | 35 + 35 | 1 | 35 | 34 |
| 1210 | S0 #1421 + S1 #5 | 927 + 926 | 1 | 926 | 926 |
| 1211 | S0 #1422 + S1 #7 | 115 + 114 | 1 | 114 | 114 |
| 合计 | | | | | **2,041** |

### 拼接点实际数值（不修复，如实列出）

| 车辆 | 时间差 | Δx (m) | Δy (m) | 位置跳变 | 前段速度 | 后段速度 | 跳变隐含速度 |
|---|---|---|---|---|---|---|---|
| 1208 (1418/2) | 0.1 s | −0.3557 | +0.1661 | **0.393 m** | 5.19 m/s | 6.47 m/s | 3.93 m/s |
| 1209 (1420/6) | 0.1 s | −0.4621 | +0.7654 | **0.894 m** | n/a（仅 1 个前段点） | 1.95 m/s | 8.94 m/s |
| 1210 (1421/5) | 0.1 s | −0.2451 | +0.1515 | **0.288 m** | n/a（仅 1 个前段点） | 1.53 m/s | 2.88 m/s |
| 1211 (1422/7) | 0.1 s | +0.3328 | −0.0055 | **0.333 m** | n/a（仅 1 个前段点） | 0.00 m/s | 3.33 m/s |

说明：4 个拼接点时间间隔正常（0.1 s），位置跳变均为**亚米级**（0.29–0.89 m），与审计得到的跨 session 跟踪偏差同量级（重复轨迹中位误差 0.25–0.90 m），属于两个 session 独立标定的正常差异；**不是**异常断裂。对 3 辆只有 1 个边界前 S0 点的车，首帧速度由规定的前向差分给出（= 上表“跳变隐含速度”）；按工单要求不修复、不插值，仅在报告披露。

---

## 4. vehicle_id 重编号（可复现）

1. 完成 4 辆重复车合并（2,442 条源轨迹 → 2,438 辆 canonical 车辆）；
2. 按每辆车**首 timestamp** 升序；
3. 首 timestamp 相同则按 `(source_session_id, original_vehicle_id)` 升序；
4. 依序分配 `vehicle_id = 1..2438`（连续、全数据集唯一）。

---

## 5. 行账本：每少一类都解释

| 环节 | 行数 | 说明 |
|---|---|---|
| 原始 CSV 数据行 | **1,607,319** | 未修改，SHA256 与审计一致 |
| 按 `(session, original_vehicle_id, Global_Time)` 排序 | 1,607,319 | 排序**不删行**（修复 228 个负步/671 个非 100 ms 文件序异常） |
| 4 辆重复车合并删除 | **−2,041** | 均为 S0 在边界及之后、与 S1 同 timestamp 的重复点（§3 表） |
| 首帧"速度不可计算"删除 | **0** | 规则已更新：首帧保留，速度用第 1→2 帧前向差分 |
| **最终** | **1,605,278** | 2,438 辆车，全部为真实轨迹点 |

校验公式：1,607,319 − 2,041 − 0 = 1,605,278 ✓（构建与验证脚本各自独立复算一致）

---

## 6. 验证结果（`canonical_validation_stats.json`，overall PASS）

| # | 校验 | 结果 |
|---|---|---|
| 1 | `vehicle_id` 从 1 连续编号 | 1 – 2,438，无缺号 ✓ |
| 2 | 重复 `(vehicle_id, timestamp)` | 0 ✓ |
| 3 | 每辆车 timestamp 严格递增 | 非正步 0 ✓ |
| 4 | 时间间隔 0.1 s | 1,602,840 步全部 0.1 s，最大偏差 3.6e-13 ✓ |
| 5 | NaN | 0（全部列） ✓ |
| 6 | Inf | 0（全部列） ✓ |
| 7 | 明显位置跳跃 | 轨迹内最大步长 8.688 m；>3 m 96 个、>5 m 6 个、**>10 m 0 个** ✓（原始 Global 坐标跳变，按"不平滑"规则保留） |
| 8 | vx/vy 有限 | 全部有限 ✓ |
| 9 | 速度物理合理 | P50 3.624 m/s，P99 18.888 m/s，max 86.880 m/s（113 行 >30 m/s，原始位置跳变伪影，未平滑/未删点） ✓ |
| 10 | 与原始 `v_Vel` 一致性 | 匹配 1,605,278/1,605,278 = **100%**；中位 3.6332 vs 3.6242 m/s；中位绝对差 0.0112 m/s；P90 绝对差 0.772 m/s；Pearson **0.9873** ✓（`v_Vel` 仅 sanity check） |
| 11 | 4 辆重复车只剩 4 个 vehicle_id | 1208/1209/1210/1211，各 2 条来源 ✓ |
| 12 | 两 session 互补车辆全部保留 | canonical 2,438 = 2,442 − 4；每辆车的行数与原始逐车复算一致 ✓ |
| 13 | 总数 | 2,438 辆 / 1,605,278 行，列恰为 6 列 ✓ |
| 14 | 删除行解释 | 仅合并删除 2,041 行；首帧 0 删除 ✓ |
| 15 | 首帧速度规则 | 2,438 辆全部满足：首帧 = 前向差分，其余 = 后向差分 ✓ |
| 16 | 主表排序 | 严格 `timestamp → vehicle_id` ✓ |

---

## 7. 明确没有做的事

未做轨迹平滑 / Savitzky-Golay / 插值 / 删除急刹·停车·拥堵·转弯 / ROI 筛选 / 最多 8 车筛选 / 2 s 历史未来窗口 / Train-Val-Test 划分 / ISAC 感知生成 / GNN-QGNN 图构造；未修改任何模型、checkpoint、正式实验报告、split 或前端代码。

---

## 8. 复现命令

```bash
cd /home/dell/YrM/ICCT
/home/dell/YrM/envs/ICCT/bin/python tools/data_cleaning/build_canonical_trajectories.py \
    --csv data/Lankershim_Vehicle_Trajectories.csv \
    --output data/processed/trajectories_clean.csv \
    --mapping data/processed/vehicle_id_mapping.csv \
    --stats reports/data_cleaning/canonical_cleaning_stats.json
/home/dell/YrM/envs/ICCT/bin/python tools/data_cleaning/validate_canonical_trajectories.py \
    --clean data/processed/trajectories_clean.csv \
    --mapping data/processed/vehicle_id_mapping.csv \
    --csv data/Lankershim_Vehicle_Trajectories.csv \
    --output reports/data_cleaning/canonical_validation_stats.json
```

脚本确定性、无随机数；重复运行产生相同输出与相同 SHA256。

---

## 9. 验收清单

| 验收条件 | 结果 |
|---|---|
| 1 原始 CSV 完全未修改 | PASS（SHA256 一致） |
| 2 两个 session 正确整理 | PASS |
| 3 4 辆重复车合并 | PASS（删 2,041 行，逐对可追溯） |
| 4 新 vehicle_id 唯一连续 | PASS（1–2,438） |
| 5 timestamp 统一为秒 | PASS |
| 6 x/y 统一为米（Global） | PASS |
| 7 vx/vy 为 m/s | PASS |
| 8 每辆车时间严格递增 | PASS |
| 9 无重复 (vehicle_id,timestamp) | PASS |
| 10 无 NaN/Inf | PASS |
| 11 无未解释的大规模删数据 | PASS（仅 2,041 行合并删除） |
| 12 可由脚本完全重新生成 | PASS（见 §8） |
| 13 旧加工数据已盘点移出 active data | PASS（见 `legacy_data_inventory.md`） |
| 14 不影响 checkpoint 与正式报告 | PASS（未触碰 `results/`、`reports/`、`checkpoints/`、模型代码） |