# SinD 逐目标邻域与 Self 坐标修复：执行协议

## 版本与范围

本轮从服务器 /home/dell/YrM/ICCT 的 qgnn@df435d9 开始修复。
旧 SinD splits/isac、旧模型检查点与结果全部保留。新计算只使用 train/val。
当前仍为 20 点历史、20 点未来、0 dB、seed 2026。4 秒和 Interaction 训练
依赖本轮 Self 验收，本启动脚本不包含它们。

用户约定：预计超过约 5 分钟的训练由用户手动启动。短时 preflight 只验证
数据、目标计分、梯度、设备与恢复功能，不作为性能结果。

## 数据合同

target_views_v1 固定沿用旧 train/val 接受窗口的起点，不重新按效果选样。
在每个起点，完整历史车辆构成候选上下文；完整历史与未来的车辆构成
监督目标。每辆监督目标单独作为 slot 0，按自身的历史关系选择至多 7 个
邻居。上下文是否存在不依赖其未来标签是否完整。

每个 (scene_id, start_frame, target_vehicle_id) 恰好一条样本。
vehicle_mask 标记上下文，target_mask 只标记 slot 0；邻居的 future_state
保持零，不能重复作为该样本的监督。Self 阶段只执行 target_mask 中的车辆；
Interaction 阶段应使用 vehicle_mask 执行图计算、target_mask 计算损失。

选邻居沿用原构造器的最后历史 GT 状态规则，不将其声称为已验证的纯在线
因果感知筛选。原始 SinD 平滑状态及完整标签条件仍需单独审计。
历史状态按唯一 scene/frame/vehicle 键储存；复用原 0 dB 缓存，仅补缺键。
新目录 data/sind/target_views_v1/{train,val} 与旧数据分开，拒绝 test。

主指标：同一 origin 的所有目标平均后，再对 origins 等权平均。
同时报告 target_ADE/target_FDE。均匀目标采样的训练权重为
总目标数 / (origin 数 × 该 origin 目标数)，使期望训练目标仍为 origin-macro。
主检查点按 ADE + 0.5 FDE 选一个，ADE/FDE 始终分别记录。
小型 subset preflight 的不完整 origins 只作工程检查。

## Self 接口

--self-frame legacy 保留旧 checkpoint 的计算与参数键。
--self-frame ego_v1 使用自身历史确定参考方向，并给 Self 提供该参考系中的
自身位置变化/速度；本车坐标残差旋回全局坐标后再加 CV。邻车信息不进入
Self。未来运动 token 辅助目标维持原语义，token 刻度文件保持冻结。

原始共享 FinalMotionGPT2、已有经典 Self 架构及量子核心不改动。
现有 Interaction 残差仍按全局坐标解释，不在本轮训练。
完全无运动且没有可确定方向的理想静止输入，Self 方向修正保持零；
实际有历史运动方向的低速/停车车辆仍可使用历史参考方向。

## 两个独立问题的对照

1. legacy / repaired LLM：
   旧 15,802 / 1,880 窗口、batch 32、20 epoch、同一优化条件；
   只验证 Self 坐标接口修复，不用它与新目标集直接比较绝对分数。
2. target / LLM、LSTM、Transformer、TCN：
   同一逐目标数据、batch 256、20 epoch、同一目标顺序与学习率方案；
   全部从头训练，保留 CV 初始参考，不用旧裁剪目标模型的训练结果替代新控制。

新目标集合比旧图内目标更多，20 个完整数据遍历的更新次数将变化。
本轮只作明确协议下的验证比较，不声称收敛、多 seed 稳定性或量子优势。
若某模型在最后一轮仍改善，后续给所有对照公平的追加训练机会。

## 用户启动

在服务器终端执行：

~~~bash
cd /home/dell/YrM/ICCT
bash scripts/run_sind_self_repair.sh
~~~

默认五组并行：GPU 0 同时运行旧数据 LLM 和新逐目标 LSTM；
GPU 1 同时运行新逐目标 LLM、Transformer、TCN。五组没有前后依赖。
显存充足允许同时驻留，但共享算力后单组速度可能下降，实际总耗时以日志为准。
原两队列方式保留为可选参数 sequential；只改变调度，不改变数据、batch、
epoch、seed、优化器或检查点选择规则。

正在运行的旧两队列不会因修改脚本自动切换。在原终端按 Ctrl-C，等待当前
batch 保存以及训练进程退出，再执行：

~~~bash
bash scripts/run_sind_self_repair.sh resume parallel
~~~

已有 last.pt 的实验继续恢复；尚未开始的实验从头启动。旧 LLM 仍在 GPU 0，
TCN 仍在 GPU 1。若提示启动器仍在运行或保存，等待保存结束后再执行同一命令。
输出分开保存在 results/qgnn/self_repair_v1/。新启动拒绝覆盖已有运行。
中断后检查日志，可用 bash scripts/run_sind_self_repair.sh resume 恢复同一协议。
恢复同时核对源码与数据 manifest；不能跨接口或更换数据后续训。

## 验收后再推进

读取全部曲线和配对 checkpoint 回放，分别检查 ADE/FDE、选中 epoch 与 CV
改善；不能只看程序 COMPLETED。随后在相同正常 Self 上比较额外本车残差、
强 pairwise、pair+triplet，再与匹配经典核心比较新 Raj。
4 秒任务使用共同有效目标并重新检查窗口边界，不能只改一个长度参数。

## 本轮工程验收

- train: 15,802 origins / 319,855 targets; val: 1,880 origins / 31,211 targets.
- 两个split的全部目标键、历史/未来索引、时间戳和边界验收通过；各128旧感知键重算逐位一致。
- Self接口31项检查通过；旧best/last检查点小批输出逐位一致。
- 四个模型均通过短时训练；小型Transformer中断恢复与连续8次更新的参数及全部epoch记录逐位一致。
- TERM中断能保存初始恢复点；运行目录锁阻止第二个写入进程。每次恢复获得新的本次时间预算，累计时间仍记录。
- 全新目标集CV全量val: ADE 0.784092 m / FDE 1.629495 m (origin-macro); target-macro 0.755176 / 1.543123 m。
- 验收记录: reports/qgnn/sind_target_self_repair/preflight.json；短时训练不是性能证据，用户曾启动旧 LLM 和新 TCN，现按用户要求暂停并保存断点，等待五组并行恢复。
- 启动器为每个独立进程组设置中断处理并持有运行锁；Ctrl-C后等待当前batch保存完成，再运行resume命令。

- 并行启动器通过替身进程验收：五组同时在运行、GPU 分配及训练参数、两组恢复/三组新建、重复启动锁、五组 TERM 保存与最终锁释放；未自动启动正式训练。
