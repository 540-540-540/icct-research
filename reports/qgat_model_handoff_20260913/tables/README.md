# 现有结果导出说明

本目录仅整理包内 `project/` 的既有 JSON 数值，没有训练、重新预测或新增统计检验。具体源文件和逐项校验见 `export_checks.json`，每张 CSV 也保留源路径列。

| 文件 | 来源与定义 |
|---|---|
| `history_100epochs_merged.csv` | `project/results/qgat_extend100/seed2026/{gnn,relation_D}/history.json`；两模型各100轮，合计200行；前40轮与各自父运行历史完全一致。 |
| `checkpoints_40_100_best_final.csv` | 各运行既有 `history.json`、`summary.json` 与对应 `validation_epoch_*.json`；40轮 GNN/原D/新QGAT和100轮 GNN/新QGAT的各自最佳J、最终轮，共10行。 |
| `metrics_100_best_final_by_snr.csv` | 100轮最佳/最终检查点对应验证 JSON 的 `by_snr` 原值；2模型×2取点×4 SNR，共16行。 |
| `scene_differences_100_best_J_paired.csv` | GNN第97轮与新QGAT第85轮既有 `per_scene` 按 `(origin,snr_db)` 一一配对，共1600行。 |
| `scene_differences_100_final_paired.csv` | 两模型第100轮既有 `per_scene` 以相同身份配对，共1600行。 |
| `../figures/training_history_100epochs.{pdf,png}` | 上述两条完整100轮历史，2×2显示 loss/ADE/FDE/J。 |
| `../figures/training_history_continuation_41_100.{pdf,png}` | 同一历史的第41–100轮局部放大，未平滑或删去波动。 |

`loss` 是原训练入口记录的该轮平均训练目标；`ADE/FDE/J` 是 `V_select` 的既有验证指标，`J = ADE + 0.5×FDE`。训练 loss 与验证 J 的数据和平均方式不同，定义见 `project/prediction/temporal.py::masked_trajectory_loss` 和 `project/prediction/training.py::{scene_metrics,evaluate}`。距离指标单位为米。`elapsed_seconds` 保留原累计值，包含40→41前后的不同执行设置，不能直接作为结构速度比较。

差值统一为 **新QGAT（relation_D）减 GNN**，负值表示该项新QGAT更低；逐场景 `scene_J` 仅由已有 ADE/FDE 做上述算术组合。两份配对表各有400个 origin，每个 origin 在4个SNR条件下重复出现，**1600行不是1600个独立样本**。best取点分别由同一 `V_select` 选择，best/final也不是独立重复实验。

图中竖线位于40.5，表示40→41轮执行设置切换；曲线为全部原始轮值，J面板标明最佳轮97/85。CSV采用UTF-8 BOM并保留浮点值；未计算置信区间、p值或显著性结论。
图表已用CPU绘制并核对全部曲线点与源JSON一致，详见 `../figures/render_checks.json`。两张最终PNG已逐张查看，标题、图例、坐标和最佳轮标注清晰且无截断；PNG为300 dpi，PDF为矢量并嵌入字体。
