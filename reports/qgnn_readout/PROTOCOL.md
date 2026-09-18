# QGNN 增加 X 读出的最小配对试跑

日期：2026-09-14。用户已授权 Codex 完成检查并启动训练，接受最多 150 轮及 patience=20；确认两组正式开始更新后，Codex 停止工作，训练在服务器后台独立继续。本轮授权覆盖此前要求用户手动启动的安排。

目标：判断新增 X 读出能否改善原 QGNN 的预测误差。仅比较两个 QGNN 分支，seed=2026；不新增 QGAT 训练。强 GNN 的既有结果保留为历史参照。

| 项目 | 两组共同规则 |
|---|---|
| 模型 | 原 `InheritedQGNNGraph`；增加 X 读出的 `XReadoutQGNNGraph` |
| 初始化 | 从相同共享参数初始化联合训练；不载入历史训练检查点；两组复用相同冻结 GPT-2 基座 |
| 数据 | 全部 5,549 个 train 起点；全部 400 个 V_select 起点 |
| SNR | 5/10/15/20 dB，训练按既有平衡轮换；每轮验证全部四档 |
| 批量 | batch=16，micro_batch=16，validation_micro_batch=16；尾批保留 |
| 训练预算 | 最多 150 轮；连续 20 轮 J 未严格降低就早停；min_delta=0，沿用共享训练器 |
| 选模 | 最低 V_select J=ADE+0.5×FDE；从同一个最佳检查点分别报告 ADE、FDE |
| 优化器 | 复用现有 AdamW 分组；图模块 lr=0.0003；LoRA lr=0.00003；其他可训练时间模块 lr=0.0001；量子旋转参数无权重衰减 |
| 资源 | 原版独占物理 GPU0，X 读出版独占物理 GPU1；各单进程、单 GPU；FP32 模型、复数双精度量子态，关闭 TF32 |
| 保存 | 每次参数更新保存 latest；改善时保存 best；逐轮验证记录；显式恢复并校验来源与配置；拒绝缺少检查点时静默重新训练 |

两个图层仍各用 6 个量子比特、3 层电路深度。原 12 个 Z/ZZ 读出及 LayerNorm 保留；每层新增 6 个 X 读出，分别通过零初始化的 4×6 矩阵影响注意力分数和门控。总共增加 96 个参数，不改变量子旋转参数数目。原版可训练参数为 825,014 个，X 读出版为 825,110 个。

检查包括完整共享参数及初始化随机流一致、量子读出与梯度、优化器覆盖、真实 GPU 两次更新、16+1 尾批、四档验证覆盖、首次验证前中断恢复、最佳权重恢复，以及完成后恢复不多训练。小规模检查位于独立临时目录，不作为预测收益证据。

生产服务器为 `/home/dell/YrM/ICCT`，环境为 `/home/dell/YrM/envs/ICCT`。配置：`configs/qgnn_readout.json`；入口：`experiments/qgnn_readout/run.py`。

- 训练检查点及逐轮指标：`results/qgnn_readout/seed2026/{original,x_readout}/`。
- 运行状态与日志：`reports/qgnn_readout/seed2026/coordinator.json`、`jobs/`、`logs/`。
- 成功完成后自动生成：`reports/qgnn_readout/seed2026/summary.json` 和 `RESULT.md`。
- 启动记录：`reports/qgnn_readout/LAUNCH.json`。

两组比较仅为单种子 V_select 开发证据；不读取 V_confirm/test。达到上限不自动等于充分收敛；结果审阅须结合曲线、早停状态和 ADE/FDE 的分别变化。历史强 GNN 前 40 轮的物理批量与本次不同，不将其当作本次完全匹配的第三实验组。
