# SENS-AUDIT-01：多目标 ISAC 回波与检测链路现状审计报告

- 审计日期：2026-09-15（北京时间）
- 服务器与运行目录：`YrM_TwYhB` = `/home/dell/YrM/ICCT`（本审计在该服务器读取运行时源码与缓存）
- 本地仓库基准提交：`8adc24c5791daaa6a27c986c9aeb1915e1c5e7c7`
- 审计对象：生成当前 `data/f01d/inputs/{train,V_select}_snr_{5,10,15,20}.npz` 的上游 sensing frontend
- 性质：只读审计 + 小规模 SNR probe；**未重写前端、未改正式 SNR 配置、未改缓存、未改 GNN/QGNN/GPT-2、未启动训练、未使用 V_confirm/test**

---

## 0. 四项判定（首页结论）

```text
A. Shared multi-target echo:                                 FAIL
B. Blind/unknown-target detection before estimation:         FAIL
C. No target-level ground-truth leakage into inference:      FAIL
D. SNR materially affects detection/estimation in current
   official range (5-20 dB):                                 FAIL
```

**Architecture verdict（主分类）**：**3. Known-target independent parameter estimation**
（已知目标独立参数估计；检测/分离/关联入口等价于 oracle 目标门控，数值估计器本身不接触真值，
因此不是类别 4 的 oracle 估计，也不存在类别 2 的多目标叠加被理想分离）。

---

## 1. 四项判定的证据

### A. Shared multi-target echo：FAIL

**代码事实**

- `frontend/symbol_level_gpu.py::synthesize_batch:50-94`：batch 维即目标维。对每个目标
  （循环 `:76-85`）独立生成：

  ```text
  X_i   = QPSK 随机符号 (3,K,N)     ← 每目标一份独立"发射"波形
  φ_i   = 随机散射相位 (3,)
  n_i   = (randn+j·randn)/√2        ← 每目标独立的单位方差接收噪声
  Y_i   = 10^(snr/20)·e^{jφ_i}·delay_i·doppler_i·X_i + n_i   （:93-94）
  ```

  函数内没有对目标维求和；返回 `Y:[B,3,K,N]`，每个 batch 元素是一辆车的孤立回波。
- `frontend/generate_symbol_gpu.py::run_device:31-34`：`at_time` 返回的全部 (frame, slot)
  被展平成 batch 元素，随后 `:47` 逐元素合成、`:48` 逐元素除以自身 `X`、`:49` 逐元素估计。
- 噪声在每个目标各自的信号上生成（`:82-85`），不存在"多目标叠加后统一加噪"的步骤。

**运行/产物证据**

- `data/f01d/sequences/*` 的四档估计目标状态计数完全相同（356171/档），且
  `search_failures` 全 0（`reports/f01d/run_status.json`）——与"目标各自独立"一致。
- probe 实测：同一目标跨 9 档 SNR 的 `noise_rms` 完全相同（最大差 2.2e-16），说明噪声属于该目标本身，
  与信号幅度解耦（`tables/probe_snr_response.csv`）。

**对照（不推翻结论）**：仓库内确有真正"共享多目标回波"的旧实现
`frontend/ofdm_echo.py::clean_echo:138-143`（`cube +=` 目标求和）与
`echo_cube:161-175`（"Sum all complex target echoes, then add one shared circular CN field"，
含雷达方程 `(100/r)^2`、FoV/距离可见性）。但该实现只被 `frontend/run_frontend.py`（F01-B/C 旧流水线）
使用，**不在当前 F01-D 缓存的生产链上**；`generate_symbol_gpu.py / symbol_level_gpu.py /
pack_symbol_dataset.py / symbol_dataset.py` 对其无 import（`checks.json` 与调用链审计）。

### B. Blind/unknown-target detection before estimation：FAIL

**代码事实**

- 当前链路不存在任何 detector：`frontend/detector.py`（CA-CFAR，`:1,:64,:170`）与 `frontend/jpda.py`
  只被旧 `run_frontend.py` 引用，F01-D 生产链不 import。
- 估计前的目标集合来自 `frontend/echo_source.py::at_time:35-50`：直接返回当前时刻存在的目标列表
  （`states`, `slots`），即**目标数量 = 列表长度、目标身份 = slot、真假由真值行决定**。
- `generate_symbol_gpu.py:34` 将 (frame, slot) 直接展平为待估目标；不存在"从接收数据判断目标数"的步骤。
- `detected` 是硬编码赋值（见 D 与 Q4），不是检测输出。

**实验事实**：probe 把 SNR 降到 -20 dB，80 个目标全部产出估计、`search_failure=0`
（`tables/probe_snr_aggregate.csv`），说明输出集合完全由 oracle 目标列表决定，与可检测性无关。

### C. No target-level ground-truth leakage into inference：FAIL

**代码事实**

| 信息 | 提供阶段 | 是否进入推理路径 | 分类 |
|---|---|---|---|
| 目标数量、身份 slot、真值状态 | `echo_source.at_time` → `generate_symbol_gpu.py:31-34` | 是（目标枚举与结果落位） | L3 + L0 |
| 真值状态用于回波生成 | `synthesize_batch:70-94` | 仅生成信道相位 | L0（合法） |
| 每目标独立信号（分离本身） | `synthesize_batch` + `estimate_batch` 调用 | 是（跳过检测/分离） | L3 |
| 输出与目标的对应关系 | `generate_symbol_gpu.py:52` 按 oracle slot 写回 | 是（跳过跨站关联） | L3 |
| 搜索初始化 | `estimate_batch:288-311` | 由观测粗估计（FFT+GN）给出，**非真值** | 无 L2 |
| 全局搜索域 | `configs/symbol_frontend.json:28-48`（训练位置极值+5 m） | 是 | L1 |
| 未来真值标签 | `pack_symbol_dataset.build_labels:154-174` → `labels/*.npz` | 否（loader 白名单：`symbol_dataset.read_inputs:16-27`） | L0 |

**关键澄清**：数值估计器 `estimate_batch` 签名（`:263`）只接收
"已分离的单目标 B、站几何、全局边界、搜索配置"，文档字符串明确
`without source state or target identity`（`:264`）。因此**不存在等价于 L2 的"真值锁定搜索中心"**：
搜索中心来自观测粗估（`:296-302`），窗口 `±3 m / ±3 m/s`、细化步长 0.025。
泄漏点在于**检测前目标级信息（数量/身份/对应/分离）已由 oracle 提供**，属于 L3。

### D. SNR materially affects detection/estimation in current official range：FAIL

**实验事实（缓存级，train/V_select × 4 档，全量只读统计）**

| 指标 | 5 dB | 10 dB | 15 dB | 20 dB |
|---|---:|---:|---:|---:|
| 与 20 dB 逐状态完全相同的比例（train） | 86.41% | 93.52% | 97.67% | 100% |
| 与 20 dB 逐状态完全相同的比例（V_select） | 85.30% | 92.53% | 97.26% | 100% |
| 位置 RMSE（train，m） | 0.01063 | 0.01051 | 0.01048 | 0.01047 |
| 位置 RMSE（V_select，m） | 0.01063 | 0.01050 | 0.01048 | 0.01048 |
| 最大单分量差（m，train / V_select） | 0.0291 / 0.0318 | 0.0301 / 0.0267 | 0.0286 / 0.0255 | — |
| mask 变化数 | 0 | 0 | 0 | 0 |
| `detected` 变化数 | 0 | 0 | 0 | 0 |

差分 p95 = 0.025 m / 0.025 m/s，**恰好等于细搜索步长**；错误上限 ≈ 1 个细格。
（`tables/cache_snr_diff.csv`、`tables/cache_error_vs_snr.csv`、`tables/detected_mask_audit.csv`）

**实验事实（配对 SNR probe，-20…20 dB，80 目标）**

| SNR | 位置误差中位数 (m) | p95 (m) | 与 20 dB 相同比例 | RD 峰值/底噪 (dB) | search_failure |
|---:|---:|---:|---:|---:|---:|
| -20 | 0.0309 | 0.0633 | 3.8% | 27.6 | 0 |
| -10 | 0.0112 | 0.0255 | 30.0% | 37.6 | 0 |
| 0 | 0.0096 | 0.0181 | 65.0% | 47.6 | 0 |
| 5 | 0.0096 | 0.0152 | 82.5% | 52.4 | 0 |
| 10 | 0.0094 | 0.0147 | 92.5% | 57.2 | 0 |
| 15 | 0.0094 | 0.0147 | 96.3% | 62.1 | 0 |
| 20 | 0.0094 | 0.0149 | 100% | 66.9 | 0 |

- 实测每 entry 接收 SNR 与标称值一致（偏差 ≤0.002 dB）。
- RD 峰值/底噪 ≈ 每 entry SNR + 47.6 dB，符合 256×256 相干积累的理论增益 10log10(K·N)=48.16 dB。
- 官方 5-20 dB 全部处于误差平台区（误差 ≈ 搜索格子分辨率）；-5 dB 以下才开始恶化。
- 结论：SNR 确实注入到了接收信号，但**在官方区间内不改变检测（不存在检测）与估计（误差被量化格支配）**。

**注意**：由于当前没有真正的 detection 步骤，上述 -20…20 dB 曲线**不能称为检测概率曲线**，
只能称为"估计响应曲线"。

---

## 2. 六问逐条回答（代码事实 / 实验事实 / 推断）

### Q1. 一个基站的一份接收信号里，是否真的同时叠加多辆车回波？

**否。** 当前缓存不是共享多目标回波。

- 文件/函数：`frontend/symbol_level_gpu.py::synthesize_batch:50-94`。
- 张量形状：`Y:[B,3,K,N]`，`B` 是目标维（每 batch 元素一个目标）；`X:[B,3,K,N]` 每目标独立。
- 目标维求和：**不存在**（代码中没有对 `B` 维的 `sum/stack` 合并）。
- 噪声：加在每目标信号上（`:82-85`、`:94`），不是加在多目标叠加之后。
- 检测器输入：当前无检测器；估计器输入是 **逐目标** 的 `B=Y/X`（`:98-103`）。
- （旧路径 `ofdm_echo.echo_cube:161-175` 才有目标求和 + 单份共享噪声，但不在本缓存链路。）

### Q2. 是否按 vehicle/slot/target 逐个独立生成感知信号并直接估计？

**是，完全逐目标独立生成 + 独立估计。**

- `generate_symbol_gpu.py:27-34`：199 帧 × 当前存在目标的 (frame, slot) 对，全部展平为 batch。
- `:33`：每目标每帧独立种子 `measurement_seed = SHA256("2026:{episode}:{frame}:{slot}:101")`。
- `:47-49`：逐目标 `synthesize_batch → divide_symbols → estimate_batch`（batch 内元素互不相干）。
- 每目标拥有独立 noise generator（`:77-85`）与独立接收信号（`Y_i`），甚至独立 `X_i`。
- 结果按 slot 直接写回（`:52`）；"group 成最多 8 槽"只是**落位**，不是重新聚类/关联。

完整调用链与逐步"是否已知目标 ID"标注见 `call_chain.md`。

### Q3. 检测/估计前是否提前知道真实目标数量、ID、对应关系与真值？

逐项：

1. **目标数量**：已知。`at_time` 返回列表长度即当前目标数；无"从接收数据判目标数"步骤。
2. **目标 ID**：已知。slot 来自 `episode['source_keys']`（真实车辆），用于 RNG 种子派生和输出落位；
   估计器本身不接收 ID（`estimate_batch:263-264`）。
3. **已知对应**：是。单站/三站结果直接继承 oracle slot；不存在多站关联（JPDA 只在旧路径）。
4. **真值辅助搜索**：否。搜索初值来自观测粗估计（`estimate_batch:288-302`），不是真实 `x,y,vx,vy`；
   没有真值初始器、proposal、search bound 或候选剪枝。

**真值泄漏分类**：回波生成 L0（合法）；场景边界 L1（已声明）；
**目标级身份/数量/对应为 L3**；未发现 L2。明细见 `truth_leakage_audit.csv`。

### Q4. 当前到底有没有真正的 Detection 步骤？

**没有。** 判断依据是算法行为，不是变量名：

1. `detected` 如何产生？`generate_symbol_gpu.py:41` 初始化为 False，`:52` 对每个 oracle 存在的
   目标赋值 `True`；`run_symbol_frontend.py:37`（CPU 旧适配器）同样硬编码 `True`。
2. `detected=False` 的条件？仅当该 (frame, slot) 上 oracle 目标列表中没有这辆车（或整帧无车）。
   运行异常会中止 episode 并记录 failure，**不会**产生 `detected=False`。
3. 低 SNR 是否可能漏检？不会。没有阈值；probe 到 -20 dB 仍全部产出、`search_failure=0`；
   四档 SNR 的 `estimated_target_states` 全等于 356171。
4. 是否存在 false alarm？不存在。全部缓存中 `detected & ~track_exists = 0`，
   `exists & ~detected = 0`（816 份 train/V_select 序列全检，`tables/sequence_outputs.csv`）。
5. 估计目标数与 GT 数是否可能不一致？不会。输出数 = oracle 存在目标数，逐 episode 与
   `generation.json` 完全一致（`checks.json` 第 3 项，通过）。
6. 是否"一辆真车一个输出"？是，且恒等于真车数。

### Q5. 当前 SNR 实际在哪里注入？

完整路径：

```text
config snr_db=5/10/15/20（configs/symbol_frontend.json:50-57）
  → generate_symbol_gpu.py:35 逐档循环
  → symbol_level_gpu.py:93  clean = 10^(snr/20)·e^{jφ}·delay·doppler·X     （幅度）
  → symbol_level_gpu.py:82-85 noise = (randn+j·randn)/√2, E|n|²=1          （噪声）
  → :94  Y = clean + noise（每目标独立）
  → :98-103 B = Y/X（|X|=1，噪声方差不变）
  → 估计器（无检测器）→ [x̂,ŷ,v̂x,v̂y]
```

- SNR 定义：每个复数接收 OFDM 符号 entry 的 `E|aX|²/E|noise|²`，`a=10^(snr/20)`，噪声方差 1。
- 噪声加在**单目标信号**上（不存在多目标混合）；三个 BS 使用同一受控 SNR（代码事实）。
- 不存在：路损、RCS/反射率、遮挡/FoV 门控、多径/衰落、目标间互扰。
- 存在：隐式相干积累增益（256×256 FFT+导向）；搜索量化误差地板（0.025 m / 0.025 m/s）。
- 配对性：同目标跨 SNR 使用同一种子，**噪声逐比特相同**（实测 2.2e-16），只有幅度变化。

结构化审计表：`snr_path_audit.json`。

### Q6. 为什么 5-20 dB 几乎不影响当前结果？

直接原因（全部有实测支撑）：

1. **没有检测环节**：`detected` 是 oracle 存在性赋值，没有阈值可跨；`detected` 变化数为 0。
2. **逐目标独立估计**：不存在多目标共享观测下的噪声竞争/互相干扰，SNR 只作用于该目标自身的粗估与打分。
3. **相干积累增益约 48 dB**：`K×N=65536`，5 dB/entry → 峰值处等效约 53 dB；probe 实测
   RD 峰值/底噪 = SNR + 47.6 dB（`figures/probe_rd_peak_vs_snr.png`）。
4. **误差被搜索格支配**：细步长 0.025；缓存实测最大分量差 0.022-0.032，p95=0.025；
   位置误差中位数 0.0094-0.0096 m 在 0…20 dB 几乎不变（`figures/probe_error_vs_snr.png`）。
5. **配对种子**：噪声实现跨 SNR 相同，差异只来自信噪比导致的**个别格子翻转**
   （85.3%/86.4% 完全相同；差异 ≤1 个细格）。
6. 平台区下限约在 -5…0 dB；官方 5-20 dB 完全位于平台内（扩展 probe，`tables/probe_snr_aggregate.csv`）。

D6.1/D6.2/D6.3 要求的分项数据分别在
`tables/probe_snr_response.csv`（信号/噪声范数、实测 SNR、RD 峰/底）、
`tables/cache_error_vs_snr.csv` + `tables/detected_mask_audit.csv`、
`tables/probe_snr_aggregate.csv` 中。

---

## 3. 代码事实 / 实验事实 / 推断 / 下一轮建议（分类汇总）

**代码事实**

- 生产入口 `scripts/run_f01d_gpu.py`（阶段列表 `:35`；失败聚合 `:13`），缓存 config hash 789168e8… 一致。
- 逐目标独立 `X/相位/噪声` 的合成（`symbol_level_gpu.py:76-94`）；无目标求和；无检测器；
  `detected=True` 硬编码（`generate_symbol_gpu.py:52`）。
- 估计器输入不含真值与 ID（`:263-264`）；搜索初值来自观测（`:288-302`）。
- 旧共享回波与 CFAR/JPDA 路径存在但未参与本缓存（`ofdm_echo.py:161-175`、`detector.py`）。

**实验事实**

- 16 个审计源码文件 hash 与服务器运行时一致（`checks.json:source_hashes` 全 true）。
- 缓存文件 SHA256 与历史记录一致、mtime 仍为 2026-09-12（未被审计改动）。
- train 701822 / V_select 57958 个有效目标状态；5 vs 20 dB 相同比例 86.41% / 85.30%；
  差异 ≤ 0.032，p95 = 细步长；mask 与 detected 变化 0。
- 位置 RMSE 8 组（2 split × 4 SNR）全部在 0.01047-0.01063 m；速度 RMSE 0.0085-0.0089 m/s。
- 配对 probe（80 目标 × 9 档）：SNR 注入准确；-20 dB 仍 0 失败；0…20 dB 误差平台。

**推断（非直接代码事实）**

- 现有 5-20 dB "无影响"不是数值实现 bug，而是前端架构（oracle 分离 + 无检测 + 细格误差地板）的
  必然结果；即使把 SNR 降到 0 dB，输出仍几乎不变。
- BDX-01 早期观测（85.3% 相同、detected 变化 0）与本次独立复核一致，可确认其不是模型侧现象。
- 以该上游训练出的 GNN/QGNN 无法体现"低 SNR 感知退化"，因此其鲁棒性结论只对"理想分离"成立。

**下一轮建议（本轮未执行，仅建议）**

1. 实现共享多目标接收：每 BS 每帧单一 `X`，`Y=Σ_i echo_i + 一次统一噪声`（可复用 `ofdm_echo` 的
   雷达方程/可见性模型，但更新到符号级波形）。
2. 在共享观测上引入真正的检测（CFAR/峰值/未知目标数）与跨站关联（可参考旧 `detector.py`+`jpda.py`，
   但不直接复活旧流水线）。
3. 移除检测前的 oracle 目标枚举/身份/对应：目标数由检测结果估计，关联由数据决定。
4. 补全物理层：路损、RCS、遮挡/多径、每 BS 独立 SNR，让 SNR 真实作用于检测概率与估计误差。
5. 明确误差地板与分辨率指标；若要观察 SNR 效应，应先降低量化地板或报告"格内偏差"。
6. 重做数据接口验收：确保 `detected` 语义与新检测器一致，并重新定义下游 mask 约定。

---

## 4. 验收标准核对

| AC | 内容 | 状态 | 证据 |
|---|---|---|---|
| AC1 | 从真实入口追踪完整调用链 | 通过 | `call_chain.md`；入口 `scripts/run_f01d_gpu.py:35` |
| AC2 | 明确同一 BS 接收是否多目标叠加 | 通过 | 判定 A 与 Q1：**否**（逐目标独立） |
| AC3 | 明确数量/ID/对应在哪个阶段变已知 | 通过 | `truth_leakage_audit.csv`；`echo_source.at_time:35-50` |
| AC4 | 明确 `detected` 语义与生成条件 | 通过 | Q4；`generate_symbol_gpu.py:41,52`；`tables/detected_mask_audit.csv` |
| AC5 | 明确 SNR 注入位置与物理含义 | 通过 | Q5；`snr_path_audit.json`；`symbol_level_gpu.py:82-94` |
| AC6 | 实测 5-20 dB 与扩展低 SNR 响应 | 通过 | `tables/`、`figures/`（配对 probe -20…20 dB） |
| AC7 | 未修改正式 sensing/预测/训练状态 | 通过 | 源 hash/mtime 未变；缓存 SHA256 未变；无训练进程；GPU 空闲 |
| AC8 | 能回答与真正多目标 ISAC 检测目标的差距 | 通过 | 第 3 节"下一轮建议"与新检测/共享回波差距清单 |

---

## 5. 合规与未修改声明

- 未修改任何正式 sensing 源码：16 个审计文件 hash 全部等于静态审计基准（`checks.json`）。
- 未修改 `configs/symbol_frontend.json`（hash 789168e8…，mtime 2026-09-12 21:00）。
- 未修改现有缓存：train/V_select 8 个 input 文件 SHA256 与历史记录一致，mtime 2026-09-12 21:13。
- 未修改 GNN/QGNN/GPT-2、未启动任何训练；审计期间 GPU 占用 0%。
- 未读取 V_confirm/test 的状态或标签；probe 只枚举 `train`/`V_select` split。
  （`SourceEpisodes` 构造函数仍会装载 episode 清单与共享 source mmap，与生产 loader 相同，
  但本审计未对其中的 V_confirm/test 目标调用 `at_time`，也未打开其缓存文件。）
- 新增文件仅为诊断脚本与报告：`code/06_sensing_diagnostics/*`、
  `reports/sensing_audit/sens_audit_01/*`。

## 6. 复现命令

```bash
cd /home/dell/YrM/ICCT
/home/dell/YrM/envs/ICCT/bin/python -u code/06_sensing_diagnostics/audit_multitarget_frontend.py
/home/dell/YrM/envs/ICCT/bin/python -u code/06_sensing_diagnostics/probe_snr_response.py
```

产物：`summary.json`、`checks.json`、`call_chain.md`、`truth_leakage_audit.csv`、
`snr_path_audit.json`、`probe_summary.json`、`tables/`（7 个 CSV）、`figures/`（4 个 PNG）、
`source_snapshot/`（服务器独有的入口脚本快照）。