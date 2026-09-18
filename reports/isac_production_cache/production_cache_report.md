# Production ISAC Sensing Cache 构建与验证报告

- **构建时间 (UTC)**: `2026-09-18T06:59:31Z`
- **ISAC 版本**: `AUTOMATUM-CONTROLLED-ISAC-V2-FROZEN` (`FROZEN`)
- **确定性重建检验**: `PASS`
- **全量验证结论**: `ALL PASS`

---

## 1. Split 规模与覆盖审计

| Split | 预测样本数 (Windows) | History 行数 (未去重) | 唯一状态数 (Unique Keys) | 去重压缩比 | Cache 形状 | 缺失数 | 重复数 | 非有限值数 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `train` | 10,366 | 769,400 | 49,765 | 15.46x | `[5, 49765, 4]` | 0 | 0 | 0 |
| `val` | 893 | 69,520 | 4,779 | 14.55x | `[5, 4779, 4]` | 0 | 0 | 0 |
| `test` | 1,135 | 95,440 | 5,727 | 16.66x | `[5, 5727, 4]` | 0 | 0 | 0 |

---

## 2. 五档 SNR 误差指标与严格单调性

| Split | 指标 | +10 dB | +5 dB | 0 dB | -5 dB | -10 dB | 单调性 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `train` | Position RMSE | 0.1308 m | 0.2326 m | 0.4137 m | 0.7357 m | 1.3090 m | `PASS` |
| `train` | Velocity RMSE | 0.1590 m/s | 0.2828 m/s | 0.5028 m/s | 0.8943 m/s | 1.5908 m/s | `PASS` |
| `val` | Position RMSE | 0.1433 m | 0.2548 m | 0.4531 m | 0.8059 m | 1.4342 m | `PASS` |
| `val` | Velocity RMSE | 0.1748 m/s | 0.3108 m/s | 0.5526 m/s | 0.9827 m/s | 1.7480 m/s | `PASS` |
| `test` | Position RMSE | 0.1198 m | 0.2130 m | 0.3787 m | 0.6733 m | 1.1974 m | `PASS` |
| `test` | Velocity RMSE | 0.1470 m/s | 0.2613 m/s | 0.4647 m/s | 0.8264 m/s | 1.4697 m/s | `PASS` |

---

## 3. Frontend 数值精确一致性 (Frontend Equality)

在每个 split 均匀抽取 200 个状态重新调用 `frontend.controlled_isac.automatum_frontend.sense_vehicle`，与 cache 逐数值比对：

| Split | 检验状态数 | 最大位置偏差 (m) | 最大速度偏差 (m/s) | 判定 |
| :--- | :--- | :--- | :--- | :--- |
| `train` | 200 | `0.00e+00` | `0.00e+00` | **PASS** |
| `val` | 200 | `0.00e+00` | `0.00e+00` | **PASS** |
| `test` | 200 | `0.00e+00` | `0.00e+00` | **PASS** |

---

## 4. Cache 文件与校验哈希

| Split | 相对路径 | SHA256 |
| :--- | :--- | :--- |
| `train` | `data/automatum_t_crossing/isac_v2/train/sensing_cache.npz` | `d38d26987798746039738a9d93b9833d9512bad576d41b8a9307e1fc58f331ce` |
| `val` | `data/automatum_t_crossing/isac_v2/val/sensing_cache.npz` | `b4edbb5cdabae79f31ef394b0b5d265029788a6ec9bc6848a85777f2ff3c9268` |
| `test` | `data/automatum_t_crossing/isac_v2/test/sensing_cache.npz` | `438e681b5191308d9f31272fbfab4eda7e6b75ad92158a1c3c7b06b8fe4c9a13` |

