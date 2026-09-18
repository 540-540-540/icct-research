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

## 历史快照 (Git Tags)

历史阶段资产由不可变 Git tags 永久存档保存在仓库历史中，当前 `main` 工作树仅维护当前最新有效的工作流水线：

| Tag | 提交 Commit | 历史含义 |
|---|---|---|
| `stage0-senior-original` | `ca1682a47e9129fbaf50dab4e2a6abc574b60098` | 2026-09-11 17:00 +08:00 交接前的师兄原始项目快照 |
| `stage1-meeting-freeze-20260915` | `8adc24c5791daaa6a27c986c9aeb1915e1c5e7c7` | 2026-09-15 组会前第一版完整稳定项目快照 |

## 当前阶段状态 (Pre-Model Data Ready)

当前 `main` 分支作为最新稳定 baseline，已完成进入模型阶段前的全部数据与感知基建闭环：

- **ISAC Route-B**: `FROZEN` (`AUTOMATUM-CONTROLLED-ISAC-V2-FROZEN`，3-BS 确定性纯物理链路)
- **Production ISAC Sensing Cache**: `COMPLETE` (`data/automatum_t_crossing/isac/{train,val,test}/sensing_cache.npz`)
- **Model-Agnostic Prediction Dataset**: `COMPLETE` (`frontend/automatum_prediction_dataset.py`，物理时钟 $\Delta t = \frac{3}{29.97}\text{ s}$)
- **Current-Stage Cleanup**: `COMPLETE`（当前工作树仅保留权威工具、配置与终审报告，移除过时中间文件与非当前有效历史工作文件）
- **Formal Classical GNN Baseline**: `PENDING (NOT SELECTED)`
- **Formal QGNN 主模型**: `PENDING (NOT SELECTED)`