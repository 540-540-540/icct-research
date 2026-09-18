# ICCT：三基站ISAC车辆追踪与预测

当前执行依据：[研究方案书](研究方案书.md)与[ROADMAP](ROADMAP.md)，版本1.2（A02）。

## 当前状态
F00服务器资产审计已执行。F01-A已完成，A01修订解决确认集过度删除和几何盲区；F01-B已实现三站流式OFDM回波并通过小规模校核，保存12个审计立方体（约204MiB）。C06共同前端在固定12个训练及全部25个V_select场景通过标称20dB工程验收；开发覆盖91.43%、位置/速度RMSE 1.091m/1.257m/s、全确认池虚假比例4.04%。F01-D尚未执行，未训练预测模型。详见[F01-C报告](reports/f01c/F01C_REPORT.md)。确认集样本有限、跨段坐标差异及三站重叠变化保留为局限。详见服务器[F01-A报告](reports/f01a/F01A_REPORT.md)和[F01-B报告](reports/f01b/F01B_REPORT.md)。

## 实际运行位置
- 专用环境：`/home/dell/YrM/envs/ICCT`；激活：`source /home/dell/YrM/envs/ICCT/bin/activate`。依赖、复现与验证见[环境说明](ENVIRONMENT.md)。
- 服务器项目：/home/dell/YrM/ICCT，SSH别名YrM_TwYhB。所有实践在服务器执行。
- 原始数据：data/Lankershim_Vehicle_Trajectories.csv。
- 新预处理：frontend/scene_manifest.py，配置configs/f01a.json；共同几何frontend/station_geometry.py。
- 当前修订：[A01记录](amendment_log.md)。
- 源侧衍生数据：data/f01_source/，不能直接作为模型估计输入。
- GPT-2：models/gpt2/，已下载并通过完整性与CPU前向检查。
- 现有code/与docs/历史记录属于旧实验参考，不能直接作为新协议运行入口。

## 历史资料保留范围
[接手任务书](docs/GPT-6-pro研究交接与深度调研任务书.md)、[历史实验版本](docs/历史实验版本索引.md)保留用于来源、曝光和旧失败证据追溯。随着新流程替代旧内容，逐项清理确认无用途的文档；本轮更新过时README入口，没有批量删除历史证据。

## 同步边界
本地E:/NJUPT/ICCT会议用于文档访问。Syncthing选择性同步代码、文档及配置；data/models/results/archive不自动同步，reports/frontend等新增目录是否同步由现有.stignore决定。服务器产物以服务器文件为准。
