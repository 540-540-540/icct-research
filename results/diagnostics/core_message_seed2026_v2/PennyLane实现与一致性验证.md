# PennyLane 实现与一致性验证

## 结论

当前六量子比特核心已经用 PennyLane 独立实现，并通过电路输出、输入梯度、量子参数梯度、冻结图模型检查点以及 RTX 4090 前向/反向传播验证。判定阈值为最大绝对误差 `1e-6`，所有检查通过。

这意味着后续可以在保持当前线路结构和已训练权重不变的条件下，将量子后端切换为 PennyLane。量子比特数仍固定为 6；4/6/8 比特消融留到后续进行。

## 实现范围

- `pennylane_core.py`：PennyLane 版六量子比特核心。
- `model.py`：增加 `quantum_backend="pennylane"` 可选入口，原默认后端仍为 `torch`。
- `validate_pennylane.py`：可重复执行的一致性和 GPU 梯度验证。
- `pennylane_validation/parity.json`：机器可读验证结果。
- `requirements_pennylane.txt`：远程 Python 3.8 环境的固定依赖版本。

PennyLane 线路与保留模型逐门一致：每层先进行 6 个数据编码 `RY`，再进行每比特 `RZ-RY-RZ` 可训练旋转，然后执行 6 条环形 CNOT；共 3 层。输出顺序为 6 个单比特 `Z` 期望值和 6 个相邻比特 `ZZ` 期望值。

## 验证结果

| 检查项 | 最大绝对误差 | 判定 |
|---|---:|---|
| 电路输出 | 4.8846e-8 | 通过 |
| 对输入角度的梯度 | 2.7511e-7 | 通过 |
| 对 54 个量子参数的梯度 | 3.6377e-7 | 通过 |
| 冻结检查点未来轨迹 | 4.7684e-7 | 通过 |
| 冻结检查点节点特征 | 4.7684e-7 | 通过 |
| 冻结检查点注意力 | 2.9802e-7 | 通过 |
| 邻接矩阵差异元素数 | 0 | 通过 |

冻结检查点为 `development/quantum_graph_selected.pt` 的第 12 轮模型。GPU 烟雾测试在 NVIDIA GeForce RTX 4090 上完成，量子核心梯度最大绝对值为 `0.0804603`，输出和梯度均为有限值。

这些误差属于不同状态矢量实现及浮点运算顺序造成的数值舍入量级，远小于 `1e-6` 验收阈值，没有发现门顺序、比特顺序、测量顺序或梯度连接错误。

## 运行环境与方式

服务器原实验环境使用 Python 3.8.10，而当前最新版 PennyLane 要求更高版本的 Python。为避免修改原实验环境，服务器上建立了独立环境：

```text
/home/js_cn/sensing/venv_pennylane
PennyLane 0.31.1
PyTorch 2.4.1+cu121
NumPy 1.23.5
autoray 0.6.3
```

构建 PennyLane 图模型：

```python
from model import build_graph

model = build_graph("quantum", quantum_backend="pennylane")
```

远程重复验证：

```bash
cd /home/js_cn/sensing/diagnostics/core_message_seed2026_v2
/home/js_cn/sensing/venv_pennylane/bin/python validate_pennylane.py
```

## 实验口径

此前保存的开发集和测试集 ADE/FDE 结果来自自研 PyTorch 解析态矢量后端。此次验证证明 PennyLane 在同一线路与同一权重下复现其计算，但没有重新训练，也没有重新读取测试集。因此不能把此前结果直接写成“由 PennyLane 训练得到”；论文中可以写成使用独立 PennyLane 后端完成了数值与梯度交叉验证。

下一阶段若正式采用 PennyLane 产生论文实验，应先在开发集进行有限 shots 和噪声实验，再决定是否需要用 PennyLane 重新训练。解析模式仍然是模拟器运行，不代表真实量子处理器。

## 版本依据

PennyLane 0.31.1 的官方 PyPI 元数据包含 Python 3.8 支持；当前 PennyLane 版本要求 Python 3.11 及以上：

- https://pypi.org/project/PennyLane/0.31.1/
- https://pypi.org/project/pennylane/

