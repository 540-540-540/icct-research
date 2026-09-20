# ICCT：三基站 ISAC 车辆跟踪与轨迹预测

静态导航文档。研究进展保存在 Git 分支与提交历史中；实验事实保存在 `reports/` 与 `results/` 中。

> **师兄审阅入口（2026-09-20）**：本分支是冻结在服务器提交 `bdd4aa0` 的完整研究快照。请先阅读 [`QGNN_RESEARCH_HANDOFF.md`](QGNN_RESEARCH_HANDOFF.md)。它说明当前真正需要解决的研究问题、tag1 组会后全部探索、证据边界、运行协议影响和建议重点审阅的代码。

## 仓库结构

| 路径 | 内容 |
|---|---|
| `frontend/` | 感知 / ISAC 前端实现、共享接口、数据集加载器与配套运行组件 |
| `prediction/` | QGAT/QGNN 运行时：经典与量子图核心、共享 GPT-2/LoRA 时序预测器、训练工具（仅供 import 路径使用；不由 ISAC 流程运行） |
| `experiments/` | QGAT/QGNN 实验入口及其配置 |
| `scripts/` | Stage-1 训练/评测/协议入口脚本与环境检查 |
| `code/` | 共享研究代码：远程依赖快照、PennyLane QGNN 核心、量子关联、诊断与绘图工具 |
| `configs/` | 冻结的前端配置（`shared_frontend.json`）与实验配置 |
| `docs/` | 静态参考：会议模板（`docs/ICCT2026官方模板/`）与参考资料 |
| `reports/` | 机器可读实验证据（JSON/CSV/图）与完整性清单 |
| `results/` | 小型机器可读开发证据；大型训练输出保留在服务器 |

## 运行环境

- 服务器专用环境：`/home/dell/YrM/envs/ICCT`（`source .../bin/activate`）。
- 依赖与环境说明：`ENVIRONMENT.md`、`requirements.txt`、`requirements-lock.txt`。
- 所有正式运行均在服务器 `/home/dell/YrM/ICCT` 上执行。

## 大型资产（不进入 Git）

原始数据集（`data/`）、模型权重（`models/`）、训练 checkpoint（`checkpoints/`）以及大型生成数组/权重（`*.npy`、`*.npz`、`*.pt`、`*.pth`、`*.ckpt`、`*.bin`）有意排除在 Git 之外，保留在服务器（或本地镜像）。F01-E 冻结数据集的完整性证据保存在 `reports/f01e/final_freeze_11/`。

## 历史快照 (Git Tags)

历史阶段资产由不可变 Git tags 永久存档保存在仓库历史中，当前 `main` 工作树仅维护当前最新有效的工作流水线：

| Tag | 提交 Commit | 历史含义 |
|---|---|---|
| `stage0-senior-original` | `ca1682a47e9129fbaf50dab4e2a6abc574b60098` | 2026-09-11 17:00 +08:00 交接前的师兄原始项目快照 |
| `stage1-meeting-freeze-20260915` | `8adc24c5791daaa6a27c986c9aeb1915e1c5e7c7` | 2026-09-15 组会前第一版完整稳定项目快照 |

## 本交接分支的冻结状态

- **冻结边界**：`stage1-meeting-freeze-20260915` 之后，至服务器提交 `bdd4aa0`（2026-09-20 10:47 +08:00）。
- **当前数据主线**：SinD public Changchun + Xi'an，20 帧历史预测 20 帧未来；正式历史输入来自冻结的 3-BS controlled-ISAC sensing cache。
- **QGNN 状态**：仍处于开放研究与方案设计阶段。既有路线均作为证据和设计素材，不构成封闭候选集合。
- **最新新增方案**：TRC-QGNN 与 TO-JQGNN 已完成工程预检；名称中的 `finalists` 只是开发期内部目录名，不表示完成最终选型。
- **正式训练状态**：两种最新方案及其 matched classical controls 尚未启动 15,802-window 正式训练；预测 test 仍关闭。
- **大型资产**：原始数据、sensing cache、checkpoint 和完整运行日志仍在服务器 `/home/dell/YrM/ICCT`，不进入 Git。

tag1 之后的完整研究尝试索引见 [`docs/qgnn/POST_STAGE1_ATTEMPT_LEDGER.md`](docs/qgnn/POST_STAGE1_ATTEMPT_LEDGER.md)。
