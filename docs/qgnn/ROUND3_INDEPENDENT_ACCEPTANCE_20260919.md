# 第三轮独立验收补充 — 2026-09-19

本文件是对既有 Round3 实验的独立复核，不新增候选架构、不训练外部 router。主定义见 `ROUND3_SCENE_ADAPTIVE_QGNN_FREEZE_20260919.md`，精确原始证据见 `reports/qgnn/round3_final_acceptance.json`。

## 已实际执行的复核

使用服务器正式 Python，从每组 `best_validation_rows.json` 重新计算三版 adaptive Q/C 与 cap16 冻结参考 Q/C，共八次已完成训练的整体 ADE/FDE/J。检查每组均为 4096 train、1880 val、12 epoch、1536 step、0 dB、seed2026、B32；并核对训练顺序、共享下游初始参数和码本哈希。每版 Q/C 的记录源码哈希一致。逐场景键、best epoch 与训练日志也分别核对，而不是只接受 summary 中的结论。

重新从完整 1880 行 history diagnostic 构建原有分层，核验冻结 strata projection 的源 SHA256、每个 history 字段与原记录一致，且投影没有 deltaJ/oracle 标签。final-history 的高动态组为140窗口，recent5为115窗口。所有14组指标及五条 full-train 判据逐项重新计算，与三版汇总一致。

从服务器真实 `best.pt` 重建选中的 v2 `phase_feedback` 两组模型，再次推理全部1880个 validation窗口。Q重新推理逐窗口误差为0；C最大逐窗口差约1.07e-6m、整体指标差低于1e-9，属于此设备计算的数值容差范围。不能将“复算通过”误读成“量子精度胜出”。

## 对照没有被结构性削弱

不关闭controller，不关闭任何量子线路，仅将新增调制参数设为中性值，并复制共同的原核心参数。N=1、3、8、20时，新 Q 和新 C 都分别精确恢复对应 R2 核心输出。该构造证实模型族包含原有核心函数，不等于有限步优化一定找到不差于原核心的参数。

因此新模型训练后更差，不能仅归因为“加入模块后原核心表达能力必然丢失”。目前可继续检验的是初始化、优化路径及有限曝光下反馈是否提供独立收益；这些尚不是被证明的失败原因。

## 不得越过的解释边界

三版既有 small pilot 都未通过原先 full-train 门槛。v2是在已测试 adaptive 版本中按整体 validation J选出的研究入口，不是相对冻结 R2 的已验收性能升级。有限预算、单seed、重复窗口及验证集上的模型选择，不支持严格的总体显著性声明。

删除反馈后J只轻微恶化是 post-hoc 分布外依赖检查，不能替代从头训练的无反馈消融。K3与ΔJ相关也不能推出“强化K3就会降低误差”。

有限shots检查只对16个train窗口的中间反馈采样，最终读出仍精确。报告中的毫米/厘米级预测漂移不是ADE/FDE提升，不是完整硬件噪声评测。真实设备需独立前缀制备、测量估计和重放；不是免费读取同一未坍缩态。

## 代码与数据保护

独立验收脚本为 `scripts/check_qgnn_round3_final.py`。脚本只实例化validation loader；数据哈希检查也仅读取train/val，没有访问test。R2核心、训练入口、冻结架构和码本与 `591030c` 比较未改变；训练/验证数据、感知缓存和GPT-2权重哈希与R2已存证据一致。

本次复核运行使用独立进程，PID、命令、GPU和日志在 `reports/qgnn/round3_final_acceptance_job/job.json`。最终报告同时区分 `status=PASS`（工程与证据核对）和 `performance_status=FAILED_PREDECLARED_FULL_TRAIN_GATE`（性能门槛未通过）。后续同结构初始化校准若存在，应单独报告，不能覆盖三版既有负结果。
