# SENS-AUDIT-01 调用链（从生成 train/V_select sensing cache 的真实入口反向追踪）

执行时间：2026-09-15（北京时间）。服务器：`YrM_TwYhB` = `/home/dell/YrM/ICCT`。
所有源码 hash 与运行产物 hash 见 `checks.json:source_hashes` 与 `source_snapshot/`。
本地仓库基准提交：`8adc24c5791daaa6a27c986c9aeb1915e1c5e7c7`。

## 0. 判定缓存身份

当前 train/V_select sensing cache 指：

```text
data/f01d/inputs/{train,V_select}_snr_{5,10,15,20}.npz
data/f01d/labels/{train,V_select}.npz
data/f01d/metadata/{train,V_select}.json
data/f01d/sequences/snr_*/episode_*.npz        （pack 的上游）
```

 `reports/f01d/run_status.json`：`status=complete`、`structural_validation_passed=true`、
`backend=torch_cuda_complex128`、四档 `search_failures` 全 0、`prediction_training_started=false`。
`reports/f01d/generation.json:frontend_hashes.configs/symbol_frontend.json = 789168e8…`，
与本次静态审计的配置文件 hash 一致（`checks.json` 通过）。因此：**F01-D 缓存是本审计对象，
其生产入口为 `scripts/run_f01d_gpu.py`。**

## 1. 完整调用链

```text
data/Lankershim_Vehicle_Trajectories.csv                    （NGSIM 原始轨迹，模拟器侧真值）
  │ frontend/scene_manifest.py::main                        （A01，CPU，一次性构建）
  ├─ reports/f01a/episodes.jsonl                             （每 episode：split、start_ms、
  │                                                            source_keys ≤8 个真实车辆、prediction_grid_ms）
  └─ data/f01_source/source_states.npy                       （真实 [x,y,vx,vy] + 因果速度，模拟器/标签侧）

scripts/run_f01d_gpu.py                                      （正式入口，run_f01d_gpu.py:35 四阶段）
  │
  ├─[stage generate] python -m frontend.generate_symbol_gpu --devices 0 1 --batch-size 16
  │   frontend/generate_symbol_gpu.py::generate → run_device
  │     frontend/echo_source.py::SourceEpisodes             （只读装载 episodes/source_states/geometry）
  │     for 每个 episode 的 199 个 deadline:
  │       source.at_time(ep, deadline)                       （echo_source.py:35-50）
  │         → states[K,4]（真值状态）, slots[K]（真实目标 ID 槽位）, used[K]
  │       │   ★ 目标数量 = 列表长度；目标 ID = slot；真值状态直接返回；无检测步骤
  │       ↓
  │       把所有 (frame, slot) 对展平成 batch 元素（generate_symbol_gpu.py:34）
  │       for snr in [5,10,15,20]（generate_symbol_gpu.py:35）:
  │         seed_i = SHA256("2026:{episode}:{frame+1}:{slot}:101")  （:10-12，与 SNR 无关）
  │         observation = symbol_level_gpu.synthesize_batch(states[:, :2], states[:, 2:],
  │                                  stations, snr_db=snr, seeds)    （symbol_level_gpu.py:50-94）
  │           for 每个目标 i（:76-85）:
  │             X_i   = 随机 QPSK (3,K,N)            ← 每个目标一份独立发射符号
  │             φ_i   = 随机散射相位
  │             n_i   = 单位方差复高斯噪声
  │             Y_i   = 10^(snr/20) · e^{jφ_i} · delay_i · doppler_i · X_i + n_i
  │           ★ 目标之间没有求和；Y 的第 0 维就是目标维；每个目标独立信号+独立噪声
  │         B = divide_symbols(Y, X)                           （:98-103，逐目标除以自身 X）
  │         result = symbol_level_gpu.estimate_batch(B, stations, xy_bounds, v_bounds, search)
  │                                                            （:263-337）
  │           - 每目标粗估计：IFFT(距离) + FFT(多普勒)，取观测谱峰（:288）
  │           - 由三站粗距离 + 站几何解算初始位置（:296-301，只用观测）
  │           - 在初始位置 ±3 m 网格搜索 + 0.025 m 细化（:302-311）
  │           - 失败标记 = 优化器不收敛或网格触边（:312）
  │         a['state_hat'][frame, slot] = values               （:52）
  │         a['track_exists'][frame, slot] = True              （:52）
  │         a['detected'][frame, slot] = True                  （:52，硬编码，非检测器输出）
  │       写 data/f01d/sequences/snr_{s}/episode_{NNN}.npz + diagnostics JSON
  │
  ├─[stage pack] python -m frontend.pack_symbol_dataset pack
  │     frontend/pack_symbol_dataset.py::pack（:177-220）
  │       pack_history(sequence, origin)              （:137-151，20 帧因果窗口，纯缓存变换）
  │       build_labels(source, index, origins)        （:154-174，未来 20 帧原始 xy + label_valid）
  │       写 inputs/{split}_snr_{s}.npz（仅 state_hat,track_exists,detected,timestamp）
  │          labels/{split}.npz、metadata/{split}.json（仅路由身份）
  │
  ├─[stage normalize] python -m frontend.symbol_dataset normalize
  │     frontend/symbol_dataset.py::fit_normalization （:29-42，只用 train 四档有效项）
  │
  ─[stage validate] python -m frontend.check_symbol_dataset
        frontend/check_symbol_dataset.py             （结构/因果/loader 契约验收）

frontend/symbol_dataset.py::SharedPredictionInputs   （:44-60）
  → state_hat / standardized_state / track_exists / detected / timestamp / origin_eligible
  → 下游 GNN / QGNN / GPT-2 预测模型（本轮未触及）
```

## 2. 每个环节是否已知目标 ID / 真值

| 环节 | 文件::函数:行 | 已知目标数量 | 已知目标 ID | 已知真值 | 真值进入推理算法 |
|---|---|---:|---:|---:|---:|
| A01 场景构建 | `scene_manifest.py::main:57-220` | 是（≤8） | 是（source_key） | 是 | 否（模拟器/标签侧） |
| deadline 目标枚举 | `echo_source.py::at_time:35-50` | 是 | 是（slot） | 是 | 是（生成与路由） |
| 回波合成 | `symbol_level_gpu.py::synthesize_batch:50-94` | 是（batch 维） | 否 | 是（位置/速度） | 仅用于生成信道相位 |
| 检测 | 不存在 | — | — | — | — |
| 参数估计 | `symbol_level_gpu.py::estimate_batch:263-337` | 否（只处理单目标信号） | 否 | 否 | 否 |
| 结果落位 | `generate_symbol_gpu.py::run_device:52` | 是 | 是（oracle slot） | 否 | 是（对应关系由真值给定） |
| 打包 | `pack_symbol_dataset.py::pack_history:137-151` | 是 | 否 | 否 | 否（纯缓存变换） |
| 模型输入 | `symbol_dataset.py::SharedPredictionInputs:44-60` | 否 | 否（白名单拒绝 ID 字段） | 否 | 否 |

## 3. 对照：仓库内存在但未被当前缓存使用的旧多目标路径

```text
frontend/echo_source.py::observation:66-93
  → frontend/ofdm_echo.py::echo_cube:161-175
      - clean_echo:138-143 对全部目标回波求和（cube += …）
      - :164 "Sum all complex target echoes, then add one shared circular CN field"
      - 幅度模型含雷达方程 (100/r)^2（:134）、FoV/距离可见性（:133）
  → frontend/detector.py::detect（CA-CFAR，:1,:64,:170）→ frontend/jpda.py → frontend/tracker.py
  → frontend/run_frontend.py（F01-B/C 旧流水线）
```

该路径确实实现“共享多目标回波 + 统一接收噪声”，但其产物是
`data/f01_frontend*/`、`reports/f01c/` 等旧缓存，**不是当前 train/V_select sensing cache**。
当前缓存的生产链（`generate_symbol_gpu.py`）没有 import 该路径；对
`generate_symbol_gpu.py / symbol_level_gpu.py / pack_symbol_dataset.py / symbol_dataset.py /
echo_source.py` 的 import 与关键字审计（`detector/jpda/cfar/CFAR`）仅命中
`echo_source.py:88-89` 的一句注释，无实际调用。