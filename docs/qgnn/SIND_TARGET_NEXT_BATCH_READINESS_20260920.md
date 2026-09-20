# SinD 下一批实验准备状态

## 本轮验收

依据 reports/qgnn/sind_target_self_repair/completed_audit.json：
五组完成，五组 best/last 完整 val 回放通过；test 未访问。
TCN/LSTM/LLM 的最佳在第20轮，固定预算结果成立，尚不能声称收敛。
新数据 TCN/LSTM 接近，LLM 优于 CV 但明显落后于经典 Self。
独立 GPT6Pro 任务设计书为任务重设计提供另一条研究线，未在本文件中假定其最终建议已经产出。

## 可重叠推进的准备工作

### 2秒交互消融与量子/经典比较

固定同一个新数据 Self 检查点与逐目标监督。以 TCN 为当前强 Self 基准，
其最佳检查点为 results/qgnn/self_repair_v1/target_tcn_ego_v1_0db_seed2026/best.pt。
这不是把 TCN 宣布为稳定胜者，也不代表已验证 LLM 论文叙事。

最小受控矩阵：
1. 冻结 TCN 原样回放，无需再训练；
2. 冻结 TCN + own-only residual；
3. 冻结 TCN + strong pairwise residual；
4. 冻结 TCN + 增加自身计算容量的 pairwise residual；
5. 冻结 TCN + pair+triplet residual；
6. 冻结 TCN + Raj quantum residual；
7. 冻结 TCN + 与 Raj 匹配的 classical residual。

own-only 与 pairwise 使用相近额外容量，pairwise 大容量控制用于 triplet，
量子与匹配经典共享融合与输出接口。所有组构图用 vehicle_mask，监督用
target_mask，沿用 origin 权重和完整 val 口径。量子模拟的 micro-batch
应以实测为准，通过梯度累积保持匹配经典对照相同有效 batch 和更新次数。

这些训练在依赖的 Self 和各自接口固定后彼此独立，可并行计算；
解释时先判断 pair 是否优于 own-only，再判断高阶与量子实现的独立贡献。
不能拿较弱 Self 上的改善，替代强 Self 上的邻车信息价值。

当前正式 RajResidualModel 强绑定 LLM，现有 classical 输出为双token，
quantum 已输出融合64维。需要新增独立冻结 Self 包装器和训练入口；
不能复用 Self-only checkpoint 保存逻辑而遗漏 core 参数。
这批完整启动器尚不存在，本文件不提供未经验证的启动命令。

### 4秒候选时域

现有已修复数据状态池足以构建共同目标，无需重算历史 ISAC：
train 296,838 targets / 15,788 origins，val 28,638 targets / 1,880 origins。
相对当前2秒目标保留率分别为92.80%、91.76%，train有14个起点因split尾部
不能覆盖60帧。以上为只读可用性计数，4秒新数据产物尚未构建。

直接沿用每个目标原来的历史和邻车集合，只扩充目标未来到40帧，
重算共同集合 origin 权重。2/3/4秒共享同一目标队列，标签可取前20/30/40步。
模型需显式参数化未来长度，默认20；禁止全局将20替换40。
4秒模型实现与原2秒代码隔离，避免破坏已有检查点的源码恢复合同。

已有2秒模型可在共同验证目标上重评作诊断；正式时域归因要求2秒和4秒
对照在同一共同训练集合上训练。4秒不是已经冻结的论文主任务。

## 启动边界与必要验证

预计超过约5分钟的训练由用户启动，交付一条总启动命令。
继续保持 train/val 范围，不访问 test。

在长训练前完成：中心监督/完整上下文、零残差复现 Self、冻结参数不变、
邻居与padding行为、有限有效梯度、完整交互检查点/断点一致性、
两种时域共同目标与split边界、当前数据上的量子吞吐短测。

## 当前数据量子短测

已在空闲RTX4090上，用固定随机 train 目标图、完整上下文、中心读出标量完成
一次前向+反向短测，无优化器、无权重更新、未访问test。外层150秒硬限时。
B8/B16/B32分别2.960/5.024/5.450秒，B32峰值已分配显存6.315GiB；
146/146参数tensor均有非零有限梯度。批内车辆数分组会影响时间，不能把单批
比例视为严格线性规律。

按当前B32粗略外推，现有核心处理319855目标的单轮前向反向约15.1小时，
20轮约302.6小时。仅为现有核心短测的线性外推，不是最终训练器ETA：
尚未包括输出包装器、数据加载、验证、并行争用，也没有进行只计算中心
receiver的优化。显存充足不能消除该模拟算力瓶颈。

因此全量量子命令必须在减少无用receiver计算、保持中心结果/梯度等价并
重新测速后再定预算；不能照搬Self模型20轮的运行时间预期。
原始证据：reports/qgnn/sind_target_self_repair/quantum_current_train_throughput.json。
可复现短测：scripts/profile_sind_target_quantum.py，使用外部timeout 150秒。

只计算中心的优化边界已核对：每个j分支/车辆数分组共享一次整图量子态，
并不是为8个receiver各执行一次完整QNode。当前重复工作主要是每个receiver
的观测量读出及后续MLP/fusion。中心专用读出可省去其余receiver的这部分工作，
但整图电路仍须保留，不能预先声称整体8倍加速。独立receiver mask必须在
canonical_pack排序时同步映射，不能将target_mask当vehicle_mask或改变门顺序。
