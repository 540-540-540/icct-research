# GNN 同配置补充对照：两卡训练一个模型

用户选择：两卡共同训练一个 GNN，种子 2026，总批次大小 16。正式训练由用户在服务器终端手动启动。

```bash
/home/dell/YrM/envs/ICCT/bin/python -u /home/dell/YrM/ICCT/experiments/qgat_relation/console.py --gnn --start
```

中断后在同一命令末尾增加 `--resume`。不加 `--start` 时只监视，不启动。面板含实时进度、损失、验证结果、剩余时间及预计结束时刻；Ctrl+C 停止该面板启动的训练并保留断点。

## 固定条件

- 原有强基线 `GNNGraph`：2 层、宽度 128、4 个注意力头、45 米邻域、既有 7 维物理边特征，结构与历史 GNN 相同。
- 相同 GPT-2、LoRA、轨迹头和输入；时间模块初值与原 D / 新 D 的种子 2026 初值一致。
- 全量 5549 个 train origins、400 个 V_select origins；5/10/15/20 dB 调度、样本顺序和选点规则相同。
- 固定 40 轮，patience 41；每轮 347 次优化，末批 13 个样本，总计 13880 次优化。
- 全局 batch=16。每卡 micro-batch=1，两卡同时处理两个样本；共享训练器每次接收这两个样本的输出，按实际样本数累积梯度。未额外除以卡数。
- 图学习率 0.0003，其他参数学习率与权重衰减沿用既有 GNN 训练器。
- 训练使用 cuda:0 和 cuda:1 的 DataParallel；模型参数、优化器和检查点统一保存一份。验证保持单卡、micro-batch=1，沿用原 D 比较的计算方式。
- CPU 初始化和 GPU0 训练随机种子保持原规则；GPU1 使用固定偏移 1000000 的独立随机流。检查点同时保存和恢复两卡随机状态。
- 仅 train 优化、V_select 选点，不访问 V_confirm 或 test。来源合同关联已完成的 D 组来源，并记录新增 GNN 入口。

两卡使用方式是实现层变化，不能据此假定速度一定比单卡快；本轮用于匹配训练预算后的预测效果比较。

## 启动前检查

双卡与单卡的预测最大误差 1.53e-5；关闭 dropout 后，完整 batch16 / 尾批13 的梯度最大误差分别为 7.15e-7 / 4.77e-7。两卡实际 dropout 前向在恢复随机状态后逐值一致。共享时间模块初值通过验证。

隔离小样本检查使用 17 条 train、2 条 V_select，完成完整批次和尾批、2 次更新及四档 SNR 验证，确认 GPU0/GPU1 均参与前向；检查点参数有限、名称保持原样、两卡随机状态均已保存。完成后的恢复没有增加优化步。该项不冒充中途精确恢复测试。

检查记录：`checks.json`、`smoke_checks.json`。隔离临时权重验收后清理，正式训练尚未由代理启动。

正式结果目录：`/home/dell/YrM/ICCT/results/qgat_relation_gnn/seed2026/gnn/`。

状态和日志目录：`/home/dell/YrM/ICCT/reports/qgat_relation_gnn/seed2026/`。
