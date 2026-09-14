# 最新模型与F02验收

版本：1.4（A04），2026-09-12。**当前状态：通过，F02汇总70项。** 展示名称仅QGNN和GNN。[汇总证据](theory_validation.json)

## 当前比较合同

两边都保留相同规格的GPT-2、LoRA、时间位置处理、预测头、估计起点CV项和损失；本轮替换的是图分支。主入口为服务器 `/home/dell/YrM/ICCT/prediction/model.py` 的 `build_model('QGNN'或'GNN')`，配置见[configs/prediction.json](../configs/prediction.json)。

| 名称 | 图分支 | 时间接口 |
|---|---|---|
| QGNN | `prediction/quantum.py`：PennyLane标准门，A02默认3层24维读出；Snapshot在同次整图演化中获取浅层和深层 | 4维状态+2标记+24维图特征，经适配器进入共同GPT-2及预测头 |
| GNN | `prediction/classical.py:GNNGraph`：师兄原2层4头128维图注意力核心、7维边、45m邻接和自环；6维共同节点输入经MLP6→128→128 | 4维状态+2标记+128维图特征，经适配器进入相同规格GPT-2及预测头 |

GNN直接复用 `code/00_remote_shared_dependencies/target_interaction_graph.py` 的 DenseEdgeGraphAttention 和 build_edge_features。保留原图层的注意力、边处理、归一化、FFN、dropout与零初始化门控。原GRU、132维节点投影、未来时间embedding和独立decoder不接入当前主模型，避免在共同LLM之外叠加另一套时序预测器。

这是师兄图核心对当前共同接口的适配，不是原完整Plain模型的逐点复现；不继承原完整模型的历史预测成绩。源代码是正式依赖，不能当无用历史文件删除。

此前把完整无LLM预测器设为主基线，是对用户意图的误解，已经纠正；对应包装已删除。共同LLM规则一直适合本轮比较目标。主模型名称不附加实现前缀。

## 已完成验收

- **当前F02 A–H共70项通过。** 48项量子数学、16项数值接口/辅助旧图回归、4组当前G图核心、2组正式模型入口。旧A02图适配器的辅助检查不代表它仍是主基线。
- **PennyLane正式后端切换另有41项对照通过。** 覆盖N3/N8、批量、非零参数、mask、旧Q兼容、层状态、外部角度、history checkpoint、输入和参数梯度；最大状态/读出误差4.63e-14，梯度3.75e-13。[证据](f02/pennylane_promotion.json)
- **G图核心4组通过。** 与独立原图层同参数计算，输出、输入和50组参数梯度误差均为0；7维边、45m边界、自环、排列、缺失槽污染、padding及梯度通过。[证据](f02/plain_baseline.json)
- **两边共同LLM入口已通过真实train数据和标签侧车loss反向。** 相同12层/12头/768维、相同LoRA可训练参数294912；输出均为`[1,20,8,2]`，没有GRU旁路。标签只在forward之外进入损失。[证据](f02/primary_models.json)

量子使用float64/complex128，输出与梯度门槛分别1e-10/1e-8；GPT-2使用float32。每个图批次只进行一次批量量子态演化，不为浅深读出或不同车辆重跑整图。既有23组404项[PennyLane交叉结果](f02/pennylane_crosscheck.json)保留；旧Torch实现仅在`quantum_torch_reference.py`中作为独立参考。

## 后续

当前未训练，优化器更新0次。新增连接与表示宽度隔离实验已取消。F03需按当前两个模型重新实测成本，然后沿共同LLM方案推进有界先导与正式训练，正式实验仍由用户在服务器终端启动。

服务器项目根目录复核F02的命令为：

```bash
/home/dell/YrM/envs/ICCT/bin/python -u scripts/run_f02.py
```

必要检查已经通过，无需重复执行才能继续。图模块、内部派生特征及必要投影构成两分支的差别；不能将系统差异直接称作量子硬件优势。
