# SMACv2 / SMAX 基线结果

> 实验还在跑，这份是**快照**。全部跑完后用
> `.venv/bin/python experiments/analyze.py` 重新生成表格和图。

配置：JaxMARL / SMAX（SMAC 的纯 JAX 重实现，带 SMACv2 的随机单位场景），
128 并行环境 × 128 步/更新，1e7 环境步，3 个随机种子。
机器：Apple M6（12 核，16 GB），JAX CPU 后端，4 路并发。

## 已完成

| 算法 | 场景 | 种子 | 胜率 | 回报 | 耗时 |
|---|---|---:|---:|---:|---:|
| MAPPO | smacv2_5_units | 0 | 80.67% | 1.749 | 37 min |
| MAPPO | smacv2_5_units | 1 | 83.01% | 1.782 | 37 min |
| MAPPO | smacv2_5_units | 2 | 80.91% | 1.753 | 37 min |
| MAPPO | smacv2_10_units | 0 | 75.74% | 1.683 | 71 min |
| IPPO | smacv2_5_units | 0 | 58.47% | 1.490 | 26 min |

## 阶段性观察

- **MAPPO 在 smacv2_5_units 上 81.5% ± 1.0%，IPPO 只有 58.5%。**
  同一个环境、同样的步数预算，唯一的结构差异是评论家有没有看到全局状态。
  SMACv2 因为随机化了出生位置和兵种，局部观测里的信息比 SMAC 更少，
  集中式评论家带来的增益在这里尤其明显。
- **三个种子的方差很小**（1.0 个百分点），说明这个差距不是种子噪声。
- **10 个单位反而比 5 个单位好训**（75.7% vs 80.7% 是同一量级），
  但单个 run 的墙钟时间是 5_units 的近两倍——计算量随智能体数超线性增长。

## 待跑

| 批次 | 内容 | 预算 |
|---|---|---|
| 第一批剩余 | mappo_smacv2_10_units s1/s2、ippo_smacv2_5_units s1/s2、ippo_smacv2_10_units s0-s2 | 1e7 |
| 第二批 | PQN-VDN（值分解）on smacv2_5_units，3 个种子 | 5e6 |

PQN-VDN 的预算砍到 5e6 是因为它慢得多：实测约 2k 步/秒，而 PPO 系是 5~8k 步/秒。
对比时会把 MAPPO/IPPO 的曲线同样截到 5e6 再比。

## 原始数据

`results/` 下每个 run 一对文件：

- `<algo>_<map>_s<seed>.jsonl`：每个更新点的完整指标（逐步曲线）
- `<algo>_<map>_s<seed>.summary.json`：该 run 的最终汇总

指标键名在各算法间不统一（PPO 系是 `returns`/`win_rate`，
Q 学习系是 `returned_episode_returns`/`returned_won_episode`），
`experiments/analyze.py` 负责统一后再聚合成 mean ± std。
