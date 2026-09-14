# ICCT：三基站ISAC车辆追踪与预测

当前执行依据：[研究方案书](研究方案书.md)与[ROADMAP](ROADMAP.md)顶部的 **2026-09-14 主候选与资源决策**。下方旧版协议与验收记录保留用于追溯。

## 当前状态（2026-09-14）

当前研究目标是：在保留强GNN对照和可比实验条件的前提下，研究如何扩大师兄双层QGNN在ADE、FDE上的优势，并使改善能够稳定复现。优先定位限制收益的关系表示、消息门控及后期泛化问题，再选择少量有机制依据的改动进行验证；不预设改动必然提升。QGAT继续暂停训练投入。

用户已确定：**当前主方案依然是师兄双层 QGNN；三轮改进均不替换原方案；QGAT 暂停训练资源投入。** 主方案为 `experiments/qgnn_inherited/graph.py` 的 `InheritedQGNNGraph`，将师兄两层量子关系/消息模块接入当前逐帧输入和公共 GPT-2/LoRA 时间预测器；不是恢复旧16比特A02路线，也不是完整复现师兄原GRU、未来词和信任融合架构。结构与复跑条件见[QGNN协议](reports/qgnn_inherited100/PROTOCOL.md)。

QGNN单种子100轮已完成：按最佳J检查点同时报告的ADE/FDE为0.547521/1.095579，较现有GNN改善0.742%/2.323%；共同第100轮排序反转。现阶段为seed2026、V_select开发证据，尚无这条QGNN候选的多种子或独立确认结论，见[比较记录](reports/qgnn_inherited100/comparison.json)。强GNN继续保留为主要对照。

围绕主方案的三轮改进均已完成，当前均不替换原方案。第一次X读出最佳ADE/FDE为0.546512/1.103938，较同轮原Q退步0.690%/0.777%。第二次行驶方向关系编码最佳ADE/FDE为0.551578/1.114769，较同协议原Q退步1.623%/1.766%。第三次可学习有向上下文残差最佳ADE/FDE为0.542487/1.096786；相对本轮从头训练原Q，ADE改善0.052%、FDE退步0.125%，未通过两项同时改善的开发门槛。见[X读出结果](reports/qgnn_readout/seed2026/RESULT.md)、[行驶方向结果](reports/qgnn_motionframe/seed2026/RESULT.md)和[有向上下文状态](reports/qgnn_directed_context/STATUS.md)。三轮均只是seed2026、V_select开发证据，未读取V_confirm/test。

因此当前基线保持不变：继续采用师兄双层QGNN作为主方案、强GNN作为主要对照；三项改进仅作为已完成的开发记录。后续计划尚未确定，也未授权新训练。此前[案例诊断](reports/qgnn_cases/REPORT.md)未支持简单删除后方或远距离邻居，门控强度与邻居间差异须区分。

QGAT现有代码、配置、日志、检查点和报告保留，不新增试跑、重训、加训或自动恢复。最新relation_D/relation_value两组均已完成100轮、34700次更新。完整决策见[修订记录](amendment_log.md)。

## 历史A04适配状态（2026-09-12）

A04图核心适配与共同时间模型验收已通过（passed，70项）：正式量子后端为PennyLane，GNN复用师兄Plain图核心；两侧使用同规格GPT-2、LoRA和预测头，展示名仅QGNN/GNN。前端仍为A03-symbol-v4-cuda-stable-solver，Q数学仍为A02，F01-D已通过。当前汇总70项通过：量子48项、接口16项、源G图4组、主入口2项；两模型均通过真实train缓存前后向。F03未执行，正式训练由用户在服务器终端手动启动。

交接入口：[F02新任务交接](docs/F02_新任务交接.md)。原始数据、前端沿革、两次数值修复、数据接口和F02 A–H验收边界均在交接文件中。运行状态以服务器reports/f01d/run_status.json及最终报告为准。

旧F01-B和C06是叠加回波/匿名前端的历史结果，保留[历史F01-C报告](reports/f01c/F01C_REPORT.md)，其20dB与覆盖/误检门槛不用于A03。A01源划分与共同几何继续复用，V_select历史开发曝光仍保留。

## 实际运行位置
- 专用环境：`/home/dell/YrM/envs/ICCT`；激活：`source /home/dell/YrM/envs/ICCT/bin/activate`。依赖、复现与验证见[环境说明](ENVIRONMENT.md)。
- 服务器项目：/home/dell/YrM/ICCT，SSH别名YrM_TwYhB。所有实践在服务器执行。
- 原始数据：data/Lankershim_Vehicle_Trajectories.csv。
- 新预处理：frontend/scene_manifest.py，配置configs/f01a.json；共同几何frontend/station_geometry.py。
- 当前修订：[主候选决策及历史修订记录](amendment_log.md)。
- 源侧衍生数据：data/f01_source/，不能直接作为模型估计输入。
- GPT-2：models/gpt2/，已下载并通过完整性与CPU前向检查。
- code/保留历史实现；本轮GNN明确复用其中的真实源类，正式运行接口在prediction/，不直接运行旧实验脚本。

## 历史资料保留范围
[接手任务书](docs/GPT-6-pro研究交接与深度调研任务书.md)、[历史实验版本](docs/历史实验版本索引.md)保留用于来源、曝光和旧失败证据追溯。随着新流程替代旧内容，逐项清理确认无用途的文档；本轮更新过时README入口，没有批量删除历史证据。

## 同步边界
本地E:/NJUPT/ICCT会议用于文档访问。Syncthing选择性同步代码、文档及配置；data/models/results/archive不自动同步，reports/frontend等新增目录是否同步由现有.stignore决定。服务器产物以服务器文件为准。

## F01-D完整缓存（v4全量验收通过）
全量生成、打包和验收已通过，产物位于`data/f01d/inputs/`、`labels/`、`metadata/`、`sequences/`，统一loader为`frontend/symbol_dataset.py`。标准化只用train四档SNR的有效缓存状态项同权拟合。本节记录前端验收状态，后续预测实验见顶部当前状态；测试缓存仅固定前端生产和封存，不用于开发选择。

## GPU前端实施（F01-D已完成）
GPU生产使用`frontend/generate_symbol_gpu.py`与`frontend/symbol_level_gpu.py`，双RTX4090独立批处理；完整切窗使用`frontend/pack_symbol_dataset.py pack`，统一loader及标准化使用`frontend/symbol_dataset.py normalize`。旧CPU全量generate入口禁用；CPU只负责源读取、路由和写盘，标准化数值也使用torch CUDA。GPU独立验收PASS，见`reports/symbol_gpu/validation.json`；episode0/95整合试跑见`reports/f01d/gpu_smoke_generation.json`。F01-D全量完成情况见最终验收报告。

## 用户手动运行
正式入口为`python scripts/run_f01d_gpu.py`，由用户在服务器项目根目录、专用ICCT环境中手动执行。入口依次运行生成→pack→normalize→validate，进度显示在前台，并写入`reports/f01d/run_status.json`与`reports/f01d/F01D_REPORT.md`。F01-D已由用户启动并完成；后续正式实验仍由用户启动。



## 历史A04模型与F02（不作为当前主候选入口）
展示名仅为 **QGNN** 和 **GNN**。`prediction/quantum.py`采用PennyLane执行A02电路，用Snapshot一次演化取得浅/深态；`quantum_torch_reference.py`仅保留为数学校核参考。GNN适配师兄Plain图核心：2层4头、128维DenseEdgeGraphAttention、7维边及45m邻域+self；每帧6维状态/标记经MLP映射128维，图输出接共同TrajectoryPredictor(128)。两侧使用同规格GPT-2、LoRA、预测头、恒速外推和损失。师兄原GRU、时间embedding和decoder不纳入本轮主模型，因此不称完整原Plain复现或继承历史成绩。

当前图核心适配验收passed，汇总70项，见[阶段报告](reports/F02_REPORT.md)。统一入口`prediction/model.py`的`build_model`只接受QGNN/GNN，配置为`configs/prediction.json`。历史量子校核证据保留，错误整套G版本的接口通过不替代当前验收。下一阶段F03重新测当前两模型成本；F04使用共同小头，F05恢复共同3+2+15日程。用户取消新增连接与表示宽度隔离实验的决定保持有效。
