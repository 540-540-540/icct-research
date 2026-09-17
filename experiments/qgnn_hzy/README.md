# H 初始化的 Z/Y 编码 QGNN

本版本入口为 `from experiments.qgnn_hzy.graph import HZYQGNNGraph`，版本标识为 `h-once-rz-ry-rz-ry-rz-ring-z-zz-v1`。

每个量子核心先将六个比特制备为零态，各执行一次 H；随后重复三轮：`RZ(theta_q) → RY(theta_q+6) → RZ(phi_lq) → RY(vartheta_lq) → RZ(omega_lq)`，接原顺序 CNOT 环 `0→1→2→3→4→5→0`。读出顺序仍是六个 Z 和六个相邻 ZZ。H 位于重上传循环之外，每个核心总计六次。两个图层各有 54 个量子参数，总计 108 个，图输入、消息传递、掩码和参数初始化沿用 inherited 版本。

当前 Z/ZZ 读出下，末轮六个 `RZ(omega)` 不影响测量结果；本次保留原参数旋转与读出，54 是每个核心参数张量的元素数，不等同于 54 个有效自由度。

这是新电路。已有 inherited/Q54、motionframe 或 readout 结果不证明本版本的 ADE/FDE；旧代码、结果和检查点保持原样。虽然参数形状相同，不能将旧检查点无标识地加载到此版本并沿用旧结果。本目录不提供或调用旧训练入口；后续实验应从新初始化开始，并使用专属配置、结果目录和版本契约。

在项目根目录运行检查：

```text
python -m experiments.qgnn_hzy.check_graph --output reports/qgnn_hzy/graph_checks.json
```

检查仅使用合成张量，执行真实 PennyLane 电路，核对完整门序与角度映射、批量输出、两层量子梯度、图掩码和置换；有 CUDA 时检查 CPU/CUDA 输出和梯度一致性。不加载数据集或历史检查点，不更新参数，不启动正式训练。性能效果待后续训练验证。
