# RAJ QGNN 完整实验证据包

本包从远程权威项目 `/home/js_cn/sensing/icct-research-raj-qgnn` 提取，日期为 2026-09-21。

## 目录

- `results/raj_qgnn/`：正式汇总结果的完整目录，包括 `exact_two_seed_aggregate.json`、两种子跨 SNR 结果和冻结测试集结果。
- `runs/seed2026_learning_curve/`：seed 2026 的 1000、2000、4096 样本量子/等参数经典训练原始结果，以及学习曲线和跨 SNR 汇总。
- `runs/seed2027_exact/`：seed 2027 的 4096 样本量子/等参数经典训练原始结果、跨 SNR、冻结测试集和两种子汇总。
- `source_snapshot/raj_qgnn/`：当前 RAJ QGNN 模型源码。
- `source_snapshot/scripts/`：训练、结构检查、配对分析、跨 SNR 与冻结测试集评估脚本。
- `source_snapshot/raj_qgnn_tokens.json`：模型 token 配置。
- `source_snapshot/RAJ_QGNN.md` 与 `RAJ_QGNN_RESULTS_20260920.md`：方案和结论说明。
- `SHA256SUMS.txt`：包内文件的 SHA-256 校验值。

## 每个训练目录包含

- `best.pt`：最佳验证检查点。
- `best_validation_rows.json`：逐场景验证结果及交互风险字段。
- `config.json`：训练配置。
- `training.json`：逐轮训练和验证记录。
- `summary.json`：最佳指标、参数量、初始化哈希与训练索引哈希。
- `heartbeat.json`：训练完成时的运行状态。
- `console.log` 或同级 `quantum.log` / `matched_classical.log`：实时训练日志。

为节省服务器空间，训练完成后的 `last.pt` 已在实验固化时删除；每个实验的 `best.pt` 均完整保留。

## 冻结测试集核心结果

- 全部 2086 个测试场景，两种子平均 J 改善：3.161%。
- 输入预定义的最高风险 10% 场景，两种子平均 J 改善：4.276%。
- 量子图核心 126,535 参数；等参数经典图核心 125,697 参数，差异 0.66%。
