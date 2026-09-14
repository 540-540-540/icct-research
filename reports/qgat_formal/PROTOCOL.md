# A07 新QGAT路线正式实验协议

日期：2026-09-13。授权：用户明确“开始吧”，启动新路线正式实验。用户随后明确要求停在亲手启动阶段。当前状态：六个正式训练任务及六项V_confirm评估均已完成，结束于2026-09-13 03:55:02+0800，总耗时2.696小时。结果审查已完成，见`reports/qgat_audit/AUDIT.md`；未发现阻断性训练错误，但收敛充分性未解决，QGAT跨车诊断增益较弱，后续结构变更及加训均未启动；不新增训练、不延长本批日程、不更换结构、不读取test。

## 1. 模型与比较范围

QGAT图模块沿用A06首轮试跑结构，不在本批新增结构搜索：6→3角度encoder、Q/K/V独立且跨节点共享的U3参数、5-qubit等价模型、8维读出经Linear(8,128)+LayerNorm。实际计算使用PennyLane单比特Q/K/V子线路与精确张量收缩；完整五比特线路等价验收已有输出6.66e−16、梯度1.42e−14的最大误差证据。该实现可在经典设备上高效精确计算，不作量子硬件加速或量子优势主张。

GNN使用既有图模块及共同GPT-2/LoRA接口。两图均输出128维，连接同规格时间模型、预测头、估计起点恒速项及损失。三seed为2023、2024、2025；每seed两侧共享同一完整可变时间模块初始权重，包含本车/标记投影、图投影、LoRA和预测头。基础GPT-2保持冻结；各图分支独立初始化。不能继承A06已训练权重当作本次从头训练。

经典baseline与QGAT的图拓扑、容量和参数化不同，结果衡量本协议下模型整体表现，不能将差异单独归于量子门。三seed提供重复训练证据，不保证优势或充分覆盖训练不确定性。

## 2. 数据与权限

使用既有划分的全部train合法起点，沿用共同前端、标准化、合法mask、槽位及监督侧车。本次Dataset实查：train为5549个起点，其中5439个含至少一个origin_eligible目标；V_select为400个起点，400个含可预测目标；V_confirm为352个起点，346个含可预测目标。四SNR时间戳/mask一致、输入白名单及标签shape/有限性检查通过；此处是数据检查，不是模型评价。不剔除不含可预测目标的起点，损失及评价按合法mask分母处理并记账。不得通过预测结果筛选起点或重新划分集合。

训练SNR为5/10/15/20dB：按同seed同manifest固定平衡轮换，每起点每epoch仅一个SNR，连续四epoch见遍四档。Q/G同seed采用相同起点顺序和条件日程。

每epoch在完整V_select四档评价，按J选择best和早停。六任务的best全部锁定后，统一在完整V_confirm四档最终评价；V_confirm不用于本批候选、学习率、停止规则或checkpoint调整。V_select/V_confirm既有开发曝光如实保留，不宣称新的盲测。当前启动入口不读取test；之后在best锁定后另行完成最终test评价，不能据测试结果调参。

训练继续使用现有场景有效目标步归一化ADE+0.5FDE；评价先逐目标有效步平均、再场景平均、再每SNR有效场景平均，四档等权宏平均，J=ADE_macro+0.5FDE_macro。FDE仅统计第20个预测步标签有效且origin_eligible的目标，缺末步对象的场景用独立分母并记账；Q/G分母一致。无监督批跳过优化器并保留记录；数值失败不得静默删样本。

## 3. 训练合同

- 两模型×三seed，共六次独立训练，全部直接联合训练最多20epoch，V_select J早停耐心4。
- 不执行旧路线的本车warmup、图adapt或F04门禁；旧16-qubit路线永久停止，旧F05不得启动。
- 有效场景batch16、micro_batch1；AdamW，图学习率3e−4、LoRA3e−5、其余可训练参数1e−4；全局梯度裁剪范数1。
- 矩阵weight_decay0.01；theta、bias、normalization参数weight_decay0。基础GPT-2冻结，其输入反向保留。
- 配置、来源和初始化规则在训练前固定。到上限仍改善时如实报告训练充分性限制，不选择性加训或少报种子。

## 4. 调度与恢复

六任务使用两GPU动态队列；每GPU同时只运行一个任务，每任务独立新进程。完成即调度下一任务，不共享运行中的模型或优化器。

每optimizer步对应进度保存latest断点，包含配置、模型可变权重、optimizer、随机状态与进度；best按完整V_select J保存。恢复必须验证配置、代码、数据、基础GPT-2及所需来源一致，随机状态恢复到当前任务设备，禁止静默覆盖不同实验。断点机制须与全批无监督跳过规则一致，不能把跳过批记为已执行optimizer步。

全部六个best封存后才作本批统一V_confirm评价。逐任务保存训练日志、实际耗时、结束原因及选定epoch；运行失败按失败记录并恢复，不把未完成任务计为完整六次。自动化保持PAUSED，本次实验启动不会恢复旧监视或旧训练。

## 5. 历史启动准备与交付范围

历史交付时最终主审及入口验收已通过，状态为READY，等待用户亲自在服务器终端启动。该阶段现已结束；正式六任务和六项确认评估已完成。手动命令参数已核对如下；最终READY验收已通过，助手不代启。用户启动并确认实际运行后回填运行时间与结果目录；训练完成后交付六任务best、每epoch指标、成本、恢复与来源记录及固定V_confirm报告。最终test评价另行执行，本协议不预填结果或预判优势。

## 6. 历史用户手动启动命令（已完成批次，非当前执行指令）

以下保留当时交付的服务器终端命令，供复现追溯；当前批次已经完成，不据此新增训练或恢复已完成任务。默认配置为`configs/qgat_formal.json`。

```bash
cd /home/dell/YrM/ICCT
nohup env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/dell/YrM/envs/ICCT/bin/python -u experiments/qgat_candidate/run_formal.py --devices cuda:0 cuda:1 >> reports/qgat_formal/launcher.log 2>&1 < /dev/null &
```

中断后恢复使用同一命令并增加`--resume`，保留原结果目录及配置，恢复前通过来源一致性校验：

```bash
cd /home/dell/YrM/ICCT
nohup env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/dell/YrM/envs/ICCT/bin/python -u experiments/qgat_candidate/run_formal.py --devices cuda:0 cuda:1 --resume >> reports/qgat_formal/launcher.log 2>&1 < /dev/null &
```

查看启动日志：

```bash
tail -f /home/dell/YrM/ICCT/reports/qgat_formal/launcher.log
```

`--devices`后使用空格分隔两张卡。此处保留历史命令文本；当前结果审查不执行启动或恢复。

## 7. 历史启动前验收状态

主线程最终复核：check-only返回ready=true、existing_run=false、optimizer_steps_executed=0、test_opened=false；train5549、V_select400，三seed2023/2024/2025、QGAT/GNN、四SNR、max20/patience4及双GPU配置一致。

五项新增CPU检查通过：公平初始化、theta衰减规则、四SNR全覆盖、早停行为、六份best统一封存及变更拒绝。两模型隔离smoke均完成，smoke仅作实现验证，不计入正式训练步或正式成绩。最终runner已记录NumPy版本并封存GNN实际来源文件。

交付时未发现正式训练或旧F04进程，GPU无计算进程。上述为启动前交付快照，不代表当前运行状态；正式批次现已完成。自动化保持PAUSED，第6节命令仅作历史记录。

## 8. 当前结果与审查边界（2026-09-13）

六个正式训练任务及六项V_confirm评估全部完成，结束时间2026-09-13 03:55:02+0800，总耗时2.696小时。确认集三seed均值如下：

| 模型 | ADE | FDE | J |
|---|---:|---:|---:|
| QGAT | 0.7454015 | 1.5822786 | 1.5365408 |
| GNN | 0.6560988 | 1.3720508 | 1.3421242 |

本批QGAT的三项均值均高于GNN；这是既定日程下的结果，不代表两模型已充分收敛，也不构成量子优势证据。结果审查已完成，见`reports/qgat_audit/AUDIT.md`；未发现阻断性训练错误，但收敛充分性未解决，QGAT跨车诊断增益较弱，后续结构变更及加训均未启动；未授权更换结构或延长训练，无新增训练，test未读取。旧16-qubit路线仍停止，自动化仍为PAUSED。原协议和历史手动启动记录保留，不将历史READY解释为当前尚未启动。
