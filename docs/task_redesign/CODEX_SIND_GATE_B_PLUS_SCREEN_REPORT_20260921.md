# SinD Gate B+ quantum-primary development screen

该表仅用于开发筛选，不是正式论文证据；test 未访问。

| rank | candidate | mean ADE | mean FDE4 | mean J (aux) | worst-seed margin vs Strong Graph | wins ADE+FDE in both seeds |
|---:|---|---:|---:|---:|---:|---|
| 1 | full_multiscale | 1.1317 | 3.0166 | 2.6400 | -4.023% | False |
| 2 | full_multiscale_fde_weight_1 | 1.1583 | 3.0271 | 2.6719 | -5.405% | False |
| 3 | full_multiscale_fde_weight_0.75 | 1.1446 | 3.0266 | 2.6579 | -5.443% | False |
| 4 | neighbor_residual | 1.2121 | 3.1355 | 2.7799 | -8.359% | False |
| 5 | multiscale_quantum_view | 1.1980 | 3.1550 | 2.7755 | -8.861% | False |
| 6 | dual_quantum_view | 1.1994 | 3.1714 | 2.7851 | -9.477% | False |
| 7 | fused_multiscale | 1.2164 | 3.1987 | 2.8158 | -10.775% | False |
| 8 | cross_order_multiscale | 1.2248 | 3.2286 | 2.8391 | -11.813% | False |

Selected for next iteration: `full_multiscale`.

完整保留所有候选；ADE/FDE 为共同主指标。选择依据为候选相对 Strong Graph 的最差 seed、最差主指标改善率；均值仅用于同分排序，J 只作辅助描述。
进入正式确认前，必须补齐 matched classical comparator，并在未参与本次筛选的新 seed 上冻结验证。
