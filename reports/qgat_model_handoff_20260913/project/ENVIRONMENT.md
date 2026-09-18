# ICCT 服务器专用环境

项目目录：`/home/dell/YrM/ICCT`。专用环境：`/home/dell/YrM/envs/ICCT`。

```bash
cd /home/dell/YrM/ICCT
source /home/dell/YrM/envs/ICCT/bin/activate
python scripts/check_environment.py
OPENBLAS_NUM_THREADS=1 python -m frontend.check_ofdm
OPENBLAS_NUM_THREADS=1 python -m frontend.run_echo_audit
```

脚本也可以直接使用 `/home/dell/YrM/envs/ICCT/bin/python`，无需激活。依赖锁定于 `requirements-lock.txt`；`requirements.txt`列出直接依赖。安装日志保存在 `reports/ICCT_environment_install.log`，实际校核记录为 `reports/ICCT_environment.json`。

这是标准 Python venv，不继承其他环境的第三方包，也不修改其他项目环境。基础 Python 3.11.15 来自 `/home/dell/YrM/envs/quantum-course-design`；因此仍依赖该路径中的解释器及标准库，迁移时应以独立 Python 3.11 重建 venv，再安装锁定依赖。

环境检查覆盖两张 GPU 的计算与梯度、本地 GPT-2 权重加载及 LoRA 梯度、PennyLane 双精度自动微分，以及包依赖一致性。检查不执行优化器更新、不保存训练权重，不代表 QGNN 数学检查或训练门禁通过。两张 24GB 显卡仍按独立设备使用，不能视为一张 48GB 显卡。

原 `reports/environment.json` 保留为 F00 时点的借用环境审计记录；后续运行以专用环境记录为准。

验收结果（2026-09-12）：包依赖无冲突；两张 GPU、本地 GPT-2/LoRA 前向与反向、PennyLane 自动微分均通过。已移除本次初建的小写 `icct` 环境，保留大写 `ICCT`。
