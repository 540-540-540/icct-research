# GNN 提速配置：重新从头训练 40 轮

本轮由用户亲自在服务器终端启动。复用已测的运行配置：两张 GPU 共同训练一个 GNN，每卡 8 个样本，总 batch 16，验证 batch 32。终端每 1 秒刷新。

## 手动启动

首次启动使用下面的命令，不加 `--resume`：

```bash
/home/dell/YrM/envs/ICCT/bin/python -u /home/dell/YrM/ICCT/experiments/qgat_relation/console.py --gnn-fast --start
```

中断后，恢复本轮训练：

```bash
/home/dell/YrM/envs/ICCT/bin/python -u /home/dell/YrM/ICCT/experiments/qgat_relation/console.py --gnn-fast --start --resume
```

仅查看进度，不启动：

```bash
/home/dell/YrM/envs/ICCT/bin/python -u /home/dell/YrM/ICCT/experiments/qgat_relation/console.py --gnn-fast
```

Ctrl+C 会停止该面板启动的训练并保留最近一次完整更新的断点；只读面板的 Ctrl+C 仅关闭面板。

## 本轮固定条件

- 从种子 2026 的原始初始化开始，不加载上一轮 GNN 的训练权重或优化器状态；使用相同的固定 GPT-2 预训练基础权重。
- GNN、GPT-2、LoRA、输入适配器和轨迹头的结构保持原样，时间模块初值与原 D / 新 D 相同。
- 全量 5549 个 train origins，400 个 V_select origins；保持 5/10/15/20 dB 调度和样本顺序。只用 train 优化、V_select 选点。
- 固定 40 轮，patience 41；每轮 347 次更新，完整 epoch 的尾批为 13，总计 13880 次更新。
- 总 batch 16；一次前向处理 16 个样本，DataParallel 分到两卡各 8 个，正常批次不再分段累积。尾批按实际样本数归一化。
- 验证使用 GPU0，batch 32，完整覆盖 400 origins × 4 档 SNR；按原 best-J 规则选点。
- 图学习率 0.0003，其他学习率与权重衰减沿用原 GNN 优化器，梯度裁剪上限 1，每次更新后原子保存断点。
- FP32，关闭 AMP、TF32 和编译优化，CPU 数学线程数 1；GPU1 的 Dropout 随机种子偏移 1000000，两卡随机状态同时保存和恢复。

配置来源为 `configs/qgat_runtime_profile.json` 中的共享训练条件和 GNN 项，新入口在启动时核对已验证的配置。物理批次调整已通过数值检查和三组随机流短程对照；完整 40 轮的效果由本轮结果检验。

## 结果与历史记录

本轮结果：`/home/dell/YrM/ICCT/results/qgat_relation_gnn_fast/seed2026/gnn/`。

本轮日志和状态：`/home/dell/YrM/ICCT/reports/qgat_relation_gnn_fast/seed2026/`。

旧 `run_gnn.py`、旧 `qgat_relation_gnn` 报告与断点保留原样；旧 `--gnn` 模式仍指向旧轮次。新模式 `--gnn-fast` 使用独立目录，从头训练与恢复训练都有重复启动和来源合同检查。

启动前验证记录为同目录下 `checks.json`：只在隔离临时目录执行 17 条训练样本、33 条验证样本的小样本检查，正式训练的更新次数为 0。此次准备不代替用户启动正式训练。
