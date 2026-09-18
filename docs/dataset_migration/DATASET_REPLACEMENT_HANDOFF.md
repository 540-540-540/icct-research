# ICCT 新数据集迁移任务交接说明

更新时间：2026-09-18
用途：交给独立对话执行“替换为更高交互密度车辆轨迹数据集”的完整迁移。
状态：任务合同，不代表数据集已经选定。

## 1. 总目标

为 ICCT 项目寻找并迁移到一个比 Automatum 更难、车辆交互更密集、更适合检验 GNN/QGNN 交互建模价值的数据集。

迁移完成后的链路必须仍然是：

新真实轨迹数据集
→ 数据清洗/统一坐标/采样
→ 固定 3 个 ISAC BS 场景
→ 多车辆共享感知
→ Route-B 类冻结 ISAC sensing
→ 每车历史状态估计 [x_hat,y_hat,vx_hat,vy_hat]
→ model-agnostic prediction dataset
→ 后续 Classical GNN / QGNN / GPT-2+LoRA

本对话只负责“新数据集 + ISAC + model-facing prediction data”完整迁移，不负责最终 QGNN 设计。

## 2. 数据集选择要求

先严格比较候选，再选一个；不要先拍脑袋建分支。

优先候选：
- INTERACTION Dataset
- SinD / SinD v2.0
- inD
- Waymo Open Motion Dataset（高成本候选）

最低筛选要求：
1. 真实多车强交互场景，不只是同框车辆很多；
2. intersection / merging / roundabout / negotiation 等交互明确；
3. 有足够多 2–8 车可用窗口，尤其 N>=4；
4. 轨迹具备连续世界坐标与可靠时间戳，最好直接有速度或可稳定计算速度；
5. 采样率最好约 10 Hz，或可无损/低风险重采样到约10 Hz；
6. 许可和下载方式可用于科研复现；
7. 能够重新构建 3 个固定 BS 的受控 ISAC 感知输入；
8. 不要求依赖 HD map / traffic light / intent 才能定义基本预测任务；
9. 预处理工程成本不能高到阻塞主线。

选择时必须特别统计：
- N 分布；
- 20m / 30m 内邻居/车辆对数量；
- closing pairs；
- CPA 风险；
- 交互事件密度；
- 可形成 20-history / 20-future 窗口的数量；
- 场景/路口数量；
- train/val/test 能否按场景或轨迹组合理切分，避免严重窗口泄漏。

## 3. 分支规则

数据集确定后，再从当前干净冻结基底创建新分支。

分支名必须直接体现数据集名称，例如：
- interaction
- sind
- ind
- waymo-motion

不要在 qgnn 分支直接做数据集迁移。

创建前先确认：
- 当前仓库状态；
- 当前远端；
- 起始 commit；
- 不携带其他未提交实验文件进入新分支。

迁移过程中的代码、文档、配置、审计报告全部提交到该数据集专用分支。

## 4. 工作环境

本地仓库：
E:\NJUPT\ICCT会议

服务器仓库：
/home/dell/YrM/ICCT

服务器：
- Ubuntu
- 2 × RTX 4090 24GB
- Python env: /home/dell/YrM/envs/ICCT

本地与服务器通过 Syncthing 同步。

原则：
- 服务器作为主要执行/审计位置；
- 避免本地和服务器同时编辑同一文件；
- Git 操作前先检查同步与工作树状态。

## 5. 强制使用 Remote Desktop Commander

新对话必须使用 ChatGPT 的 Remote Desktop Commander 插件连接用户电脑，并通过用户电脑上的 Git SSH 连接服务器。

Windows Git SSH：
C:\Program Files\Git\usr\bin\ssh.exe

SSH Host alias：
YrM_TwYhB

服务器工作目录：
/home/dell/YrM/ICCT

优先通过 Remote Desktop Commander：
- 检查本地仓库
- SSH 服务器
- 下载/解压/审计数据
- 编写和运行 preprocessing
- 运行 ISAC 生成与验证
- 检查 GPU/磁盘/环境
- git commit
- 必要时由本地已认证 Git push

不要只给用户 shell 命令让用户手动执行，除非是大规模正式长训练/长生成且用户明确希望终端可见。

## 6. 数据处理阶段

完成：
1. 原始文件完整审计；
2. 坐标系/单位/采样率/轨迹字段说明；
3. 缺失值、重复帧、异常跳变、轨迹长度审计；
4. 场景选择规则；
5. 车辆类型选择规则；
6. 统一状态 [x,y,vx,vy]；
7. 合理 train/val/test 划分；
8. 20 history + 20 future 窗口生成；
9. 2<=N<=8 的定义；
10. N>8 如何处理必须明确，不能为了方便静默丢车造成身份偏差；
11. 不允许未来泄漏；
12. 生成完整数据统计与交互密度报告。

如新数据集更适合调整 Automatum 的某些冻结规则，可以提出，但必须明确：
- 为什么调整；
- 对任务定义有什么影响；
- 是否仍能与旧项目公平对照。

## 7. ISAC 迁移阶段

目标不是重新追求 SOTA ISAC，而是复用当前“上游可信、稳定、不限制下游”的原则。

必须完成：
1. 针对新场景重新布置 3 个固定 BS；
2. 重新定义/验证 BS 坐标、boresight、覆盖范围；
3. 多车辆共享观测；
4. 生成每车 sensing state [x_hat,y_hat,vx_hat,vy_hat]；
5. 正式 SNR 仍优先保持 [-10,-5,0,5,10] dB；
6. 检查误差随 SNR 单调/合理变化；
7. 避免把 GT future 泄漏进 sensing；
8. 生成 frozen sensing cache；
9. 输出每档 SNR 的 position / velocity error；
10. 写完整物理合理性与数值稳定性报告。

注意：
当前 Automatum Route-B 属于受控 sensing 链路，不含 detection/association。
若新数据集迁移继续采用这一受控设定，要明确保持一致。
若要重新加入 detection/association，必须先说明为什么，不要偷偷改变科研问题。

## 8. Model-facing prediction dataset

最终必须形成与现有模型尽量兼容的接口：

history_state: [20,8,4]
future_state: [20,8,4]
vehicle_mask: [8]
history_timestamp: [20]
scene_id: metadata only
vehicle_ids: metadata only

正式 history：
来自新数据集上的 frozen ISAC sensing cache。

正式 future：
来自新数据集 GT。

vehicle ID / scene ID 默认不能作为模型 feature。

## 9. 验证要求

至少验证：
- shape / dtype / finite
- padding strictly zero
- timestamp 正确
- train/val/test 无明显轨迹或时间泄漏
- sensing cache 与窗口严格对齐
- 所有 SNR 可加载
- 每档 SNR 误差合理
- interaction statistics 显著强于 Automatum，或者明确报告未达到换数据集目的

重点与 Automatum 对比：
- N>=4 占比
- close-pair 数量
- closing-pair 数量
- CPA-risk 场景占比
- 有效交互窗口数量
- 交互事件多样性

如果新数据集并没有明显提高交互密度，不要因为已经投入工作就强行迁移，允许 no-go。

## 10. 完成标准

只有以下全部完成，才认为数据集迁移成功：
- 数据集正式选定并有证据；
- 专用 Git 分支已创建；
- 原始数据审计完成；
- 清洗/预处理完成；
- train/val/test 冻结；
- 3-BS ISAC 完成；
- sensing cache 完成；
- prediction dataset 完成；
- 全部 validation checks 通过；
- 交互密度相对 Automatum 的对比报告完成；
- 文档说明如何从原始数据复现到最终模型输入；
- 代码/配置/报告已 commit/push。

最后输出：
1. 新分支名和冻结 commit；
2. 新数据集统计；
3. 新 ISAC 指标；
4. 新 prediction dataset 接口；
5. 相对 Automatum 的交互复杂度对比；
6. 尚未完成或风险项；
7. 给 QGNN 主线对话的 handoff 文档。

## 11. 重要边界

不要：
- 主动设计最终 QGNN；
- 调 QGNN 指标；
- 用 test 选择数据处理超参；
- 为追求“更难”而故意制造标签噪声；
- 为了车辆更多就保留明显无关车辆；
- 在没有统计证据前宣称新数据集一定更适合 QGNN。

允许最终结论是：
“候选数据集没有比 Automatum 提供更有效的交互学习空间，因此不迁移。”

