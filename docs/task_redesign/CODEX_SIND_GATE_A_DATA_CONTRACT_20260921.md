# SinD-IC4 Gate A 数据合同

日期：2026-09-21
状态：`IMPLEMENTED / PREFLIGHT PASSED / TEST SEALED`

## 1. 正式任务

- 数据：SinD public Changchun + Xi'an 官方 `Veh_smoothed_tracks.csv`。
- 输入：约 2 s（20 帧）官方 canonical state 经冻结 3-BS、0 dB sensing 后的 `[x_hat,y_hat,vx_hat,vy_hat]`。
- 输出：约 4 s（40 帧）target canonical `x,y`，缺失位置只通过 `future_mask` 屏蔽。
- 样本键：`(scene_id, t0, target_vehicle_id)`；target 固定为 slot 0。
- 构造顺序：history qualification -> freeze sample key/context -> attach future label/mask。

未来数据只承担监督标签。target membership、neighbor membership、IC membership、N<=8 ranking 和坐标变换均不读取未来轨迹或未来完整性。

## 2. 数据划分与封闭测试

沿用既有 train/validation/sealed-test 的 chronological physical-vehicle split。20->40 后将 validation 起点再向后移动 20 帧，使 train->validation 以及 validation->test 的 guard 都达到 60 帧。physical vehicle overlap 为 train/val=0、train/test=0、val/test=0。

构建器只读取 test 的 `scene_id/vehicle_id/frame_id` 做物理实体交叉审计；没有加载 test 坐标、标签，没有生成 test NPZ。训练、阈值、模型选择、checkpoint 和 Gate A 判决都只使用 train/validation。

## 3. 正式 evaluation views

- Full2 / IC2 使用前 20 个 future steps；Full4 / IC4 使用前 40 个 future steps。
- IC2 与 IC4 共用同一个 history-side membership（`k>=1`）。Full/IC 是同一训练 population 的分层，不另建训练集。
- ADE 使用 horizon 内可见 mask；另报告 complete cohort。
- FDE2/FDE4 只在固定第 20/40 帧 endpoint 存在时计算；不以“最后可见点”替代。
- 汇总以 prediction origin 为 macro unit；checkpoint 统一最小化 validation IC4 `J=ADE+0.5*FDE4`。

## 4. 冻结交互定义

只用最后一帧 history state；半径 50 m，并沿用预登记 closing/CPA/following 条件：closing>=0.5 m/s、0<TCPA<=4.004 s、DCPA<=5 m、following heading<=30 deg、headway<=2 s、lateral<=2.5 m、minimum speed=0.5 m/s。阈值未依据结果调整。

N<=8 Graph 保留 target+前 7 个 history-ranked neighbors；AllGraph 使用半径内全部 history-qualified neighbors（本 cache 最大 33 nodes）。两者共享同一 target population。

## 5. Manifest 规模

| split | targets | origins | IC | k>=2 | 4s endpoint | N>8 |
|---|---:|---:|---:|---:|---:|---:|
| train | 36,643 | 1,892 | 12,142 | 3,814 | 31,778 | 29,597 |
| validation | 3,855 | 251 | 1,149 | 273 | 3,285 | 2,657 |

train/validation cache SHA256 分别为 `fa61b328...96baab` 与 `c6444eaf...89dd4`；完整值保存在 manifest。新增 sensing state 与既有 cache 使用同一个 config hash `f898e43f...f89c2`。

## 6. 模型公平合同

五个 classical controls 共用 target TCN、宽度、ego-local transform、CV anchor、40-step decoder、masked loss、优化器、训练样本、checkpoint 规则与 evaluation：Strong Self、Own-only capacity、Context Pooling、Graph N<=8、All-neighbor Graph。参数量依次为 244,578、302,178、302,690、319,203、319,203。

## 7. 适用范围

本 benchmark 可称为“对 SinD 官方 canonical trajectory states 的 strict history-only task construction”。官方发布的 smoothed states 被项目正式接受为共同输入；本项目没有从 raw video 重建在线 detector/tracker/filter，因此不声称完成了 raw-video strict causal state reconstruction。

权威产物：

- `configs/gate_a_sind_ic4.json`
- `data/task_redesign/sind_gate_a_ic4_v1/manifest.json`
- `reports/task_redesign/sind_gate_a_preflight.json`
