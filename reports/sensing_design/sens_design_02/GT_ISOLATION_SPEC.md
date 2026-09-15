# GT_ISOLATION_SPEC.md — 三域隔离规范（V1 冻结草案）

依据：SENS-AUDIT-01 已确认当前 F01-D 的泄漏形态为 L3（检测前已知目标数量/身份/对应/分离）。
本规范的目标是让新前端在结构与执行上都不可能把 GT 泄漏进推理域。

## 1. 三域定义

```text
A. Simulator / GT Domain（模拟器/真值域）
   数据：NGSIM CSV、source_states.npy、episodes.jsonl（source_keys/splits/origins）
   代码：frontend/scene_manifest.py、frontend/echo_source.py（SourceEpisodes/at_time）、
         frontend/sensing/simulator.py（回波生成）
   允许：读 GT；生成共享回波；生成未来标签原料
   禁止：输出 target list / ID / 真值状态给 B 域函数

B. Sensing / Inference Domain（感知/推理域）
   数据：Y/X（共享回波）、per-BS detections、fused observations、tracks、state_hat/masks
   代码：frontend/sensing/detector.py、frontend/sensing/coords.py、
         frontend/fusion/association.py、frontend/tracking/cv_kf.py、frontend/symbol_dataset.py
   允许：只读观测与配置（波形/几何/功率/搜索参数）
   禁止：import A 域模块；任何 target 数量/ID/真值参数

C. Supervision / Evaluation Domain（监督/评估域）
   数据：labels、track↔GT 匹配、指标、校准报告
   代码：frontend/pack_shared_dataset.py（label 构造）、frontend/check_shared_dataset.py、
         scripts/calibrate_f01e_snr.py
   允许：在 B 域完全结束后读取 A 域 GT
   禁止：把匹配结果/GT 写回 B 域产物（输入 npz、sequences、track_key、slot）
```

## 2. 信息流矩阵（GT 逐项审计）

| GT 信息 | A 域用途 | 是否进入 B 域 | 隔离要求 |
|---|---|---|---|
| 目标数量 `N_t` | 生成回波求和项 | 否 | detector 只从 CFAR 峰计数；API 无 count 参数 |
| 车辆 ID / source_key | phase 种子派生（仅散射相位）、episode 路由、C 域匹配 | 否 | `seed_phase` 派生只发生在 simulator 内部；waveform/noise 种子不含它；track_key 独立自增 |
| 真值位置 `x,y` | 信道距离相位、可见性门控 | 否 | B 域只见 Y/X/几何常量 |
| 真值速度 `vx,vy` | 多普勒相位、径向速度 | 否 | 同上 |
| slot（A01 source_keys 下标） | 仅 C 域标签匹配路由 | 否 | A01 slot 不参与种子派生（P0-2）；新 slot 由 tracker 分配，不出现在 B 域 |
| 未来轨迹 | C 域标签 | 否 | labels 与 inputs 分离；loader 白名单 |
| 场景边界/几何 | A01 一次性场景常量（非逐目标） | 是（L1，声明） | `xy_bounds`/`v_bounds`/站点坐标视为公开几何先验 |
| SNR/功率 | 由配置 `snr_ref` + 距离/RCS 决定 | 否 | 不传逐目标 SNR；检测质量字段仅来自观测 |

## 3. 机制级隔离措施

1. **模块边界（静态）**：B 域模块不得 import `frontend.echo_source`、`frontend.scene_manifest`
   或 `data/f01_source`。`frontend/sensing/simulator.py` 是唯一读 GT 的感知相关文件，且**不属于 B 域**：
   它由 A 域入口调用，产出 `Y/X` 后即退出。
2. **API 合同（动态）**：`detector.process_baseline(Y_b, X_b, station_id, waveform, array, config)`、
   `fuse_frame(detections_by_bs, config)`、`CvKalmanTracker.step(observations, time_ns)` 的签名中
   没有任何 target/GT 参数；T2 用 AST 检查签名与调用点。
3. **文件不可变**：pack 完成后对 `data/f01e/sequences/**` 与 `inputs/**` 计算 SHA256 清单；
   C 域脚本（labels/metadata 匹配）运行前后复核哈希不变；任何写回视为失败。
4. **import 哨兵（T2）**：在 B 域单测中注入 poisoned 模块（如把 `frontend.echo_source` 替换为
   抛异常的 stub）后运行 detector/fusion/tracker，必须无异常；若某实现偷偷 import 会立即失败。
5. **数据脱敏**：给 detector 的观测字典只含 `Y/X/station_id/time_ns`；不给 `audit`、source 行、
   真值列表（旧 `echo_source.observation` 的 `audit` 字段永不进入新链路）。
6. **可视化/日志**：B 域日志只允许记录计数与观测统计；轨迹图/评测图属于 C 域。

## 4. 阶段时序

```text
[用户启动]
A: scene_manifest（一次性，不变）→ episodes/source_states
A+B 流式：for episode/frame/SNR:
    simulator: at_time(deadline) → Y/X → （B）detector → association → tracker
    每帧结束即丢弃 Y（可选保存少量 diagnostics echo）
B 结束：sequences + inputs 打包并封存（写 hash 清单）
C: labels（历史 track↔GT 匹配、未来轨迹）+ metrics + calibration 报告
```

C 域只允许在 B 域封存后启动（入口 `pack`/`validate` 阶段；与 generate/sense/track 分离进程阶段）。

## 5. 验收挂钩

| 检查 | 手段 | 通过条件 |
|---|---|---|
| GT 不进入 detector | T2 + import 哨兵 + 签名 AST | 无 GT 参数、无 A 域 import |
| target count 非 GT | 关闭 CFAR/降 SNR 实验 | 检测数随 SNR 变化，而非恒等于 GT 数 |
| detected 非 GT | T7/T3 | 存在 detected=0 且 track_exists=1；低 SNR detect 下降 |
| labels 不回写 | pack 前后 hash | 哈希不变 |
| 旧路径隔离 | 文件清单 | `data/f01d`、旧入口、旧 config 零改动 |

## 6. 与旧 F01-D 的已知差异（对照）

旧路径：`at_time` 目标清单直接展平为待估 batch（L3）→ `detected=True` 硬编码 → 无检测/关联。
新路径：GT 只出现在 simulator 内部（L0）；detector/association/tracker 无 GT 通道（无 L2/L3）；
C 域匹配只服务监督与指标。