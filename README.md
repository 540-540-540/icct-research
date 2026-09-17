# ICCT：三基站 ISAC 车辆跟踪与轨迹预测

静态导航文档。研究进展保存在 Git 分支与提交历史中；实验事实保存在 `reports/` 与 `results/` 中。

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

## 历史

| Tag | 含义 |
|---|---|
| `stage0-senior-original` | 2026-09-11 17:00 +08:00 交接前的师兄原始项目快照 |
| `stage1-meeting-freeze-20260915` | 2026-09-15 组会前的第一版稳定项目 |

历史开发分支已完整包含在当前开发历史中。

## 当前开发模块

ISAC（共享前端）重构在 `module/isac` 开发分支上继续，尚未正式 freeze。F01-E 加载器（`frontend.f01e_dataset.F01EDataset`）与 `data/f01e/` 数据契约属于历史冻结实验链，本身不代表最终感知方案；当前实现以 `module/isac` 上的代码为准。QGNN/LLM 模块将在 ISAC freeze 之后跟进。