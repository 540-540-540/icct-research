# ICCT：三基站ISAC车辆追踪与预测

当前执行依据：[研究方案书](研究方案书.md)与[ROADMAP](ROADMAP.md)，版本1.3（A03，SNR修订；模型沿用A02）。

## 当前状态
F00及F01-A已执行。用户选择以Wei2024符号级论文为主的逐车独立感知，目标已分离且对应已知，再组成最多8车共同预测输入。A02的Q24/G128定义和训练矩阵不变。

当前A03-symbol-v3-cuda-positive-snr，SNR为5/10/15/20dB、标称20dB。GPU独立验收已通过，episode0/95整合试跑已完成；正式全量未启动、F01-D未验收。用户将在服务器终端手动启动并监视，助手不得代为启动正式全量。未开始预测训练。 固定复噪声方差1，以信号幅度10^(SNR_dB/20)设定原始符号SNR。旧A03-symbol-v1负档位快照、核心及因果成绩保留为历史，见[版本报告](reports/symbol_level/A03_REPORT.md)，不能当作当前正档位成绩。

旧F01-B和C06是叠加回波/匿名前端的历史结果，保留[历史F01-C报告](reports/f01c/F01C_REPORT.md)，其20dB与覆盖/误检门槛不用于A03。A01源划分与共同几何继续复用，V_select历史开发曝光仍保留。

## 实际运行位置
- 专用环境：`/home/dell/YrM/envs/ICCT`；激活：`source /home/dell/YrM/envs/ICCT/bin/activate`。依赖、复现与验证见[环境说明](ENVIRONMENT.md)。
- 服务器项目：/home/dell/YrM/ICCT，SSH别名YrM_TwYhB。所有实践在服务器执行。
- 原始数据：data/Lankershim_Vehicle_Trajectories.csv。
- 新预处理：frontend/scene_manifest.py，配置configs/f01a.json；共同几何frontend/station_geometry.py。
- 当前修订：[A03及历史修订记录](amendment_log.md)。
- 源侧衍生数据：data/f01_source/，不能直接作为模型估计输入。
- GPT-2：models/gpt2/，已下载并通过完整性与CPU前向检查。
- 现有code/与docs/历史记录属于旧实验参考，不能直接作为新协议运行入口。

## 历史资料保留范围
[接手任务书](docs/GPT-6-pro研究交接与深度调研任务书.md)、[历史实验版本](docs/历史实验版本索引.md)保留用于来源、曝光和旧失败证据追溯。随着新流程替代旧内容，逐项清理确认无用途的文档；本轮更新过时README入口，没有批量删除历史证据。

## 同步边界
本地E:/NJUPT/ICCT会议用于文档访问。Syncthing选择性同步代码、文档及配置；data/models/results/archive不自动同步，reports/frontend等新增目录是否同步由现有.stignore决定。服务器产物以服务器文件为准。

## F01-D完整缓存（GPU验收与整合试跑完成，正式全量待用户手动启动）
GPU验收与整合试跑完成，正式全量待用户手动启动，计划写入`data/f01d/inputs/`、`labels/`、`metadata/`、`sequences/`，统一loader为`frontend/symbol_dataset.py`。标准化只用train四档SNR的有效缓存状态项同权拟合。当前未验收通过，未开始预测训练；测试缓存仅固定前端生产和封存，不用于开发选择。

## GPU实施（开发校验中）
GPU生产使用`frontend/generate_symbol_gpu.py`与`frontend/symbol_level_gpu.py`，双RTX4090独立批处理；完整切窗使用`frontend/pack_symbol_dataset.py pack`，统一loader及标准化使用`frontend/symbol_dataset.py normalize`。旧CPU全量generate入口禁用；CPU只负责源读取、路由和写盘，标准化数值也使用torch CUDA。GPU独立验收PASS，见`reports/symbol_gpu/validation.json`；episode0/95整合试跑见`reports/f01d/gpu_smoke_generation.json`。这不代表正式全量完成。

## 用户手动运行
正式入口为`python scripts/run_f01d_gpu.py`，由用户在服务器项目根目录、专用ICCT环境中手动执行。入口依次运行生成→pack→normalize→validate，进度显示在前台，并写入`reports/f01d/run_status.json`与`reports/f01d/F01D_REPORT.md`。当前正式任务未启动，不提前写完整缓存通过；助手只准备入口和说明，不代启动或自动监视正式全量。
