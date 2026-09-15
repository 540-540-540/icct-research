# INTERFACE_SPEC.md — 接口、张量合同与数据模式（V1 冻结草案）

本轮只冻结接口，不实现。所有路径相对服务器项目根 `/home/dell/YrM/ICCT`。
`DECISIONS.md` 的编号在本文引用。

## 1. 规划模块布局（最小新增集合）

```text
frontend/
  sensing/
    __init__.py
    waveform.py        PaperWaveform（沿用现有 dataclass）+ 阵列/几何常量
    simulator.py       shared echo 生成（A 域调用；输入仅 GT 状态 + 波形/阵列/功率配置）
    detector.py        3D CA-CFAR + NMS + 峰值参数 + 单站几何换算（B 域）
    coords.py          polar↔Cartesian、协方差传播、角度 wrap（B 域）
  fusion/
    association.py     Mahalanobis 门控 + 分量匹配 + 融合（B 域）
  tracking/
    cv_kf.py           CV Kalman + 航迹状态机 + slot 管理（B 域）
  pack_shared_dataset.py   缓存打包（B 产物 → inputs/labels/metadata）
  check_shared_dataset.py  独立验收（结构/因果/三域隔离；T1–T7）
scripts/
  run_f01e_shared.py       正式入口（用户手动启动）
  calibrate_f01e_snr.py    校准 sweep 入口（只产报告，不冻结正式档位）
configs/
  shared_frontend.json     新 frontend 配置（revision=B01-shared-3bs-v1）
```

保留文件与改造方式见 `FILE_REFACTOR_MAP.csv`。`frontend/symbol_dataset.py` 继续作为唯一模型输入 loader
（KEEP_AND_MODIFY，增加 `--data` 根目录参数；默认仍指向旧 `data/f01d` 直到下游切换）。

## 2. 通用约定

- 数值：模拟/感知计算 `float64/complex128`（CUDA，沿用 A03 约定，避免 float32 网格边缘翻转）。
- 落盘缓存：`float32`（state_hat/labels）与 `bool`（masks），与 F01-D 完全一致。
- 时间：毫秒整数 `time_ms`（A01 全局时钟）；序列内 timestamp 为 float64 秒，`Δt=0.1 s`。
- 坐标：全局局部笛卡尔（A01 原点），米；角度以 BS 视轴为 0，逆时针为正，弧度。
- 种子：`seed = int.from_bytes(SHA256(f"2026:{episode}:{frame}:{slot}:{aspect}")[:8],'little') & (2^63−1)`；
  `aspect` 用于区分波形/相位/噪声流；同一目标跨 SNR 复用同一组种子（配对比较），
  与旧规则（`...:101`）的差异在 `DECISIONS.md` D18 记录。
- 任何 B 域函数**不得**接受 target list / GT ID / 真值位置速度参数。

## 3. 感知接口

### 3.1 simulator（A 域）

```python
def synthesize_shared(
    positions: Tensor[N_t, 2],      # 真值，模拟器内部
    velocities: Tensor[N_t, 2],
    stations: Tensor[3, 2],
    boresights: Tensor[3],          # rad
    waveform: PaperWaveform,
    array: ArrayConfig,             # elements=16, spacing_wavelengths=0.5
    rcs_m2: Tensor[N_t],
    snr_ref_db: float,
    episode: int, frame: int,       # 目标 ID 仅用于种子派生，不进入信道模型
    slots: Tensor[N_t],             # 同上，仅种子派生
) -> dict:
    "Y": Tensor[3, A, K, N] complex128   # 共享多目标回波（求和后噪声）
    "X": Tensor[3, K, N] complex128      # 每 BS 自己的 QPSK 网格（含符号约定）
    "alpha": Tensor[3, N_t]              # 诊断用逐 BS 幅度（标注为 simulator-side）
    "visible": Tensor[3, N_t] bool
```

### 3.2 detector（B 域）

```python
def process_baseline(Y_b: Tensor[A,K,N], X_b: Tensor[K,N], station_id: int,
                     waveform, array, config) -> dict:
    "power": Tensor[Nr,Nv,Na] float64          # 诊断/CFAR 中间量
    "detections": list[Detection]
    "counters": dict                            # candidates_before_cap, suppressed, overflow, ...
```

`Detection`（Python dict / npz 记录，字段冻结）：

```text
station_id: int16
time_ns: int64
r_m: float64                # 斜距（插值后）
vr_mps: float64             # 径向速度（诊断/V1.1）
u: float64                  # 方向余弦
bearing_rad: float64        # 相对视轴
x_m, y_m: float64           # 单站局部笛卡尔
C_xy: float64[2,2]          # 位置协方差（polar→Cartesian 传播，标定参数）
peak_power, noise_floor: float64
peak_to_noise_db: float64
cfar_score: float64
grid_index: int32[3]
```

### 3.3 association/fusion（B 域）

```python
def fuse_frame(detections_by_bs: dict[int, list[Detection]], config) -> list[Observation]

Observation = {
  "time_ns": int64, "x_m": float64, "y_m": float64, "C_xy": float64[2,2],
  "bs_mask": int8,                 # bit0/1/2
  "n_bs": int8,
  "quality_db": float64,           # max peak_to_noise_db
}
```

### 3.4 tracker（B 域）

```python
class CvKalmanTracker:
    def __init__(self, config): ...
    def step(self, observations: list[Observation], time_ns: int) -> list[TrackRecord]

TrackRecord = {
  "track_key": int, "slot": int|-1, "state_hat": float64[4], "P": float64[4,4],
  "track_exists": bool, "detected": bool, "confirmed": bool,
  "misses": int, "age": int, "n_bs": int8, "quality_db": float64,
}
```

## 4. 缓存与入口

### 4.1 新缓存路径（不覆盖 F01-D）

```text
data/f01e/
  sequences/snr_{s}/episode_{NNN}.npz      # 每 episode 199 帧：state_hat[199,8,4]f32,
                                           # track_exists[199,8]bool, detected[199,8]bool,
                                           # timestamp[199]f64
  inputs/{split}_snr_{s}.npz               # [S,20,8,4]f32 + masks + timestamp（与 F01-D 同键）
  labels/{split}.npz                       # future_position[S,20,8,2]f32 + label_valid[S,20,8]bool
  metadata/{split}.json                    # 路由字段（episode/source_keys/source_groups/origin_ms）+
                                           # gt_match 诊断（cost、公共帧、ID switch 计数）
  diagnostics/snr_{s}/episode_{NNN}.json   # 计数、失败、校准哈希
  normalization.json                       # 只由 train 拟合（沿用 F01-D 规则）
  echoes/                                  # 可选诊断样本：仅少量选定 (episode,frame) 的 Y
```

`configs/shared_frontend.json`（新，`revision=B01-shared-3bs-v1`；旧 `symbol_frontend.json` 不动）：

```json
{
  "revision": "B01-shared-3bs-v1",
  "waveform": {"fc": 24e9, "B": 93.1e6, "K": 256, "N": 256, "T": 12.375e-6, "c": 299792458.0},
  "array": {"elements": 16, "spacing_wavelengths": 0.5, "fft_size": 64, "window": "rect"},
  "geometry_source": "reports/f01a/geometry.json",
  "visibility": {"half_angle_deg": 70, "range_m": [10, 300], "height_difference_m": 5},
  "power": {"snr_ref_db": null, "reference_range_m": 100.0, "reference_rcs_m2": 10.0,
            "fixed_rcs_m2": 10.0, "noise_variance": 1.0},
  "detector": {
    "oversample": 2, "cfar_train": [6, 6, 8], "cfar_guard": [2, 2, 4],
    "target_false_alarms_per_bs_frame": 0.5, "max_candidates": 32,
    "nms_radius": [2, 2, 4], "interpolation": true
  },
  "association": {"gate_chi2_2dof": 9.21, "max_detections_per_bs": 32},
  "tracker": {"dt": 0.1, "q_a_m2_s3": null, "birth_inflation": 4.0,
              "confirm_hits": 2, "confirm_window": 3, "confirm_requires_nbs2": true,
              "max_missed": 5, "max_confirmed": 8, "slot_cooldown_cycles": 10},
  "snr_levels_db": null,
  "calibration": {"status": "pending", "report": null}
}
```

`snr_ref_db`、`snr_levels_db`、`q_a_m2_s3` 与 CFAR 阈值类参数在校准后由用户在评审后冻结（D20）。

### 4.2 入口

```bash
# 结构构建（可在校准前，用小 SNR 集合 + 少量 episode 冒烟）
python -m frontend.sensing.simulator ...        # 库调用；正式入口：
python scripts/run_f01e_shared.py --stages generate,sense,track,pack,validate
# 校准 sweep（只写 reports/，不写正式档位）
python scripts/calibrate_f01e_snr.py --snr-ref-db -10 -5 0 5 10 15 20 25 30 35 40
```

## 5. 下游兼容

| 项 | 约定 | 与 F01-D 关系 |
|---|---|---|
| `state_hat` | `[S,20,8,4] float32`，缺失补 0 | 不变 |
| `track_exists` | `[S,20,8] bool` | 不变（新语义见 §6） |
| `detected` | `[S,20,8] bool` | 不变（新语义见 §6） |
| `timestamp` | `[S,20] float64`，0.1 s 网格，A01 origins | 不变 |
| 模型输入 loader | `SharedPredictionInputs` 白名单 4 键 | 不变 |
| normalization | 新 train 四档重拟合 | 需重算（REBUILD-03 内完成） |
| 旧缓存 | `data/f01d` 原样保留，作为 oracle/ideal-sensing baseline | 不覆盖、不修改 |
| 新缓存 | `data/f01e` | 下游切换与重训在 REBUILD-04 决策，本轮不做 |

## 6. `detected` / `track_exists` 语义（冻结）

```text
track_exists = 1  ⇔ 该 slot 的航迹 alive（confirmed 且 misses < max_missed）
detected     = 1  ⇔ 本帧该航迹获得了真实 measurement update（任意 n_bs≥1）
允许且预期出现：track_exists=1, detected=0（coast/漏检）
tentative 航迹：两者都写 0（不输出、不占 slot）
```

## 7. 契约与校验规则（写入时强制）

- `np.any(detected & ~track_exists)` 必须为 False（沿用 F01-D 校验）。
- `state_hat[~track_exists] == 0`。
- `timestamp` 严格 0.1 s 均匀；每 episode 199 帧、origin 窗口 20 帧与 A01 网格精确对齐。
- 输入 npz 只允许 4 键白名单；metadata 的路由字段不允许模型 loader 读取。
- 任何 B 域模块 import 图中不得出现 `frontend.echo_source`、`frontend.scene_manifest`、
  `data/f01_source`（静态检查 + T2）。
- 旧缓存与旧入口不改；`scripts/run_f01d_gpu.py` 等只保留历史可复现性。