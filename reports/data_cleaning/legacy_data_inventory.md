# 服务器旧加工数据盘点与归档报告

- 性质：**移动，不删除**；不触碰模型 checkpoint 与正式实验结果。
- 执行工具：`tools/data_cleaning/archive_legacy_data.py`（默认 dry-run，`--execute` 执行）
- 清单：`reports/data_cleaning/legacy_data_manifest.csv`（3,111 行：B 类 3,094 + C 类 17，含逐文件 SHA256）
- 归档目录：`/home/dell/YrM/ICCT/archive/legacy_data_precanonical/`（保持原相对目录结构）

## 分类规则（冻结，按路径前缀，首个命中生效）

**A 保留（active data path）**

| 路径 | 理由 |
|---|---|
| `data/Lankershim_Vehicle_Trajectories.csv` | 原始数据（只读母本） |
| `data/processed/**` | 本轮 canonical 干净母数据 + 追溯表 |
| `data/f01_source/**` | 当前 ISAC 前端输入（A01 SourceEpisodes；`frontend/echo_source.py`、`configs/qgnn_bottleneck_diagnostic.json` 在用） |
| `data/f01e/**` | 当前冻结共享前端缓存（BDX-01/冻结轮次在用） |
| `data/prompt_bank/**` | 小型静态提示词库 |

**B 移入 legacy archive（旧派生物，可重新生成，当前代码不再引用）**

**C 保留待人工确认（本轮不移动）**

## 盘点与执行结果

| 类别 | 文件数 | 大小 |
|---|---|---|
| A 保留 | 3,423 | 551,904,171 B（526.3 MiB） |
| B 移入 archive | **3,094** | **2,997,936,299 B（2.79 GiB）** |
| C 待确认 | 17 | 144,173,183 B（137.5 MiB） |

### B 类明细（已移动）

| 原路径前缀 | 文件数 | 大小 (MiB) | 归档理由 |
|---|---|---|---|
| `data/dataset_lankershim_clean_v1/` | 449 | 2,317.9 | 旧 clean 数据集；`reports/exposure_manifest.json` 已将其标注为 legacy split |
| `data/f01_echo_audit/` | 12 | 204.0 | F01-B 回波审计缓存；可重新生成；仅历史 config 引用 |
| `data/f01_frontend/` | 185 | 157.8 | 旧 F01-C 生产前端 detections/tracks；已被冻结 f01e 取代 |
| `data/f01_frontend_baseline/` | 135 | 91.8 | 旧 baseline 前端输出；可重新生成 |
| `data/f01d/` | 2,265 | 31.4 | F01-D 中间缓存；可由 `scripts/run_f01d_gpu.py` 从 raw 重建 |
| `data/multitarget_lankershim_h20_p40_matched_v1.npz` | 1 | 30.4 | 旧长时域 QGNN 缓存；当前 pipeline 无引用 |
| `data/multitarget_lankershim_v1.npz` | 1 | 24.6 | 旧多目标 QGNN 缓存；当前 pipeline 无引用 |
| `data/f01e_dryrun/` | 46 | 1.1 | F01-E dry-run 缓存；仅历史 freeze 脚本引用 |

归档后 active `data/` 仅剩：原始 CSV、`processed/`（canonical）、`f01_source/`、`f01e/`、`prompt_bank/`（`du -sh data` = 536 MiB）。

### C 类明细（未移动，等待人工确认）

| 路径 | 大小 | 说明 |
|---|---|---|
| `.codex-work/isac-redesign-probe-01/windows_train.pkl` | 43.1 MB | ISAC 重设计探针的中间窗口缓存；是否保留由人工决定 |
| `.codex-work/isac-redesign-probe-01/` 其余 2 个 npz/pkl | 3.5 MB / 3.7 MB | 同上 |
| `checkpoints/qgat_candidate/smoke/**`（7 个 .pt） | ~43 MB | 模型 checkpoint（smoke），**明确保护，永不移动** |
| `checkpoints/qgat_candidate/trial/**`（7 个 .pt） | ~43 MB | 模型 checkpoint（trial），同上 |

## 注意事项

1. 只执行了 `move`，没有 `rm`；空目录仅在移动后按 B 前缀范围清理，非空目录一律保留。
2. `results/`、`reports/`、`models/`、`prediction/`、`frontend/`、`experiments/`、`checkpoints/` 未被扫描为 B，正式实验结果与 checkpoint 未受影响。
3. 历史文档/清单中仍有指向旧路径的引用（如 `configs/f01b.json` 的 `data/f01_echo_audit`、`reports/f01c/*production_manifest.json` 的 `data/f01_frontend`）。这些属于历史阶段记录；如需复跑相应历史流程，可从 archive 原路径恢复。
4. `data/f01_source/` 与 `data/f01e/` 保留在本轮 active path：前者是当前 ISAC 前端输入，后者是被当前实验冻结依赖的前端缓存；待后续正式 pipeline 完全切到 canonical 后再评估归档。
5. 清单 SHA256 在移动前计算（B）或移动后即时计算（C），可用于完整性复核。

## 复现命令

```bash
cd /home/dell/YrM/ICCT
/home/dell/YrM/envs/ICCT/bin/python tools/data_cleaning/archive_legacy_data.py            # dry-run 盘点
/home/dell/YrM/envs/ICCT/bin/python tools/data_cleaning/archive_legacy_data.py --execute  # 执行归档并写 manifest
```