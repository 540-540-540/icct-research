# GNN / 新 QGAT：从第 40 轮续训至总计 100 轮

本轮由用户亲自在服务器终端启动。两个模型分别占用一张 GPU，并行续训。原 QGAT 不参加本轮，旧实验文件与断点保留。

## 手动启动

第一次启动本轮延长训练：

```bash
/home/dell/YrM/envs/ICCT/bin/python -u /home/dell/YrM/ICCT/experiments/qgat_relation/console.py --extend100 --start
```

首次启动会在新目录继承两份第 40 轮最新断点，然后开始第 41 轮。这里不加 `--resume`；`--resume` 用于恢复已经创建的这轮延长训练。

中断后恢复：

```bash
/home/dell/YrM/envs/ICCT/bin/python -u /home/dell/YrM/ICCT/experiments/qgat_relation/console.py --extend100 --start --resume
```

仅查看进度：

```bash
/home/dell/YrM/envs/ICCT/bin/python -u /home/dell/YrM/ICCT/experiments/qgat_relation/console.py --extend100
```

面板每 1 秒刷新，显示完成轮次、更新次数、损失、最近验证、最佳 J 和预计剩余时间。剩余时间使用本次续训的新耗时估算。Ctrl+C 停止该面板启动的两组训练，保留最近一次完整更新；只读面板的 Ctrl+C 只关闭显示。

## 训练配置

| 项目 | GNN | 新 QGAT（关系旋转版） |
|---|---|---|
| 物理显卡 | GPU0 | GPU1 |
| 单次前向样本数 | 16 | 8 |
| 完整总 batch 的累积次数 | 1 | 2 |
| 有效总 batch | 16 | 16 |
| 验证 batch | 32 | 128 |
| 起点 | 第 40 轮 latest | 第 40 轮 latest |
| 总目标 | 100 轮 | 100 轮 |
| 新增轮次 / 更新次数 | 60 / 20820 | 60 / 20820 |

共同条件：seed 2026，5549 个 train origins、400 个 V_select origins，5/10/15/20 dB 调度，原模型结构、学习率、权重衰减和梯度裁剪规则。图学习率 0.0003；固定总 100 轮，patience 101；每轮 347 次更新、尾批 13，总计 34700 次更新。第 41 轮继续使用原调度的 `epoch=40`，后 60 轮每个样本在每档 SNR 下各训练 15 次。

保持 FP32，量子计算保持 complex128；关闭 AMP、TF32 和编译优化，CPU 数学线程数 1。每次更新后原子保存检查点。每个子进程只看见自己分配的物理 GPU，因此其内部设备名均为 `cuda:0`；面板显示物理卡号。

配置文件：`configs/qgat_extend100.json`。入口：`experiments/qgat_relation/run_extend100.py`。

## 断点继承与来源

GNN 来源：`results/qgat_relation_gnn_fast/seed2026/gnn/`。

新 QGAT 来源：`results/qgat_relation/seed2026/relation_D/`。

两者都从第 40 轮的 `latest.pt` 继续，完整继承模型权重、AdamW 状态、13880 次更新、40 条训练历史及随机状态。原有最佳 `best.pt` 同时保留：GNN 最佳在第 40 轮，新 QGAT 最佳在第 39 轮；若后续未产生更好的 J，仍能返回原有最佳模型。源断点不修改。

新副本写入总 100 轮和本次运行配置，保留原始来源记录。恢复时核对来源与配置，缺失 latest 或 best 时拒绝启动，防止意外从头训练。

GNN 从原双卡改为单卡，继承原 GPU0 的随机流；原 GPU1 随机流留在原断点中。新 QGAT 继承原 GPU1 随机流，并采用已验证的 micro-batch 8。物理分批方式变化会改变后续 Dropout 抽样，本轮保留学习条件与完整优化器状态，不宣称逐位复现旧执行方式。

## 输出与验证

新结果目录：`/home/dell/YrM/ICCT/results/qgat_extend100/seed2026/`，下分 `gnn/` 和 `relation_D/`。

日志与状态：`/home/dell/YrM/ICCT/reports/qgat_extend100/seed2026/`。

最终分别报告共同第 100 轮和各自最佳 J 检查点；仅使用 train 优化、V_select 选点，不读取 V_confirm 或 test。此次仍是单种子充分训练对照，不预设 QGAT 优于 GNN。

启动前检查记录在 `checks.json`、`console_checks.json` 和 `readiness.json`。隔离检查从源状态各增加 2 次更新，验证完整 AdamW 与历史继承、单卡运行、实际验证批次及已完成分支的恢复；其中子集验证指标不作为正式第 41 轮结果。
