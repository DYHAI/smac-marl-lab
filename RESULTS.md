# SMACv2 / SMAX 基线结果

环境：JaxMARL **SMAX**（SMAC 的纯 JAX 重实现，带 SMACv2 的随机兵种/随机出生位置场景）
算法：MAPPO · IPPO · PQN-VDN
机器：Apple M6（12 核，16 GB），JAX CPU 后端
配置：128 并行环境 × 128 步/更新，每个组合 3 个随机种子，全部用仓库默认超参（没有做调参）

> 预算说明：PPO 系跑满 **1e7** 环境步，PQN-VDN 只跑 **5e6**
> （它每个环境步贵约 8 倍）。所以下面给**两个视角**：
> 各自的最终成绩，以及截到统一 5e6 预算的公平对比。

## 最终成绩（各自的完整预算）

| 场景 | 算法 | 预算 | **胜率** | 回报 |
|---|---|---:|---:|---:|
| smacv2_5_units | **MAPPO** | 1e7 | **81.5% ± 1.1%** | 1.762 |
| smacv2_5_units | **PQN-VDN** | 5e6 | **71.8% ± 2.1%** | 1.610 |
| smacv2_5_units | IPPO | 1e7 | 63.2% ± 3.5% | 1.540 |
| smacv2_10_units | **MAPPO** | 1e7 | **77.0% ± 2.3%** | 1.696 |
| smacv2_10_units | IPPO | 1e7 | 33.5% ± 3.3% | 1.157 |

逐种子：

| 场景 | 算法 | seed 0 | seed 1 | seed 2 |
|---|---|---:|---:|---:|
| smacv2_5_units | MAPPO | 80.7% | 83.0% | 80.9% |
| smacv2_5_units | PQN-VDN | 69.5% | 73.5% | 72.3% |
| smacv2_5_units | IPPO | 58.5% | 64.4% | 66.7% |
| smacv2_10_units | MAPPO | 75.7% | 75.1% | 80.1% |
| smacv2_10_units | IPPO | 37.0% | 34.4% | 29.2% |

## 公平对比（统一截到 5e6 环境步）

| 场景 | 算法 | 胜率 |
|---|---|---:|
| smacv2_5_units | MAPPO | 73.5% ± 1.3% |
| smacv2_5_units | **PQN-VDN** | **71.8% ± 2.1%** |
| smacv2_5_units | IPPO | 46.6% ± 1.6% |
| smacv2_10_units | MAPPO | 67.4% ± 1.4% |
| smacv2_10_units | IPPO | 10.8% ± 0.8% |

图：`results/compare_win_rate_at5e6.png`、`compare_return_at5e6.png`

## 四个结论

### 1. 集中式评论家的增益随队伍规模放大

同样 1e7 预算，MAPPO 与 IPPO 的差距：

| 场景 | 差距 |
|---|---:|
| smacv2_5_units（5 个智能体） | **+18.3 点** |
| smacv2_10_units（10 个智能体） | **+43.5 点** |

唯一的结构差异就是评论家看不看全局状态。SMACv2 因为随机化了出生位置和兵种，
单个智能体的局部观测里信息比 SMAC 更少，所以共享全局信息更值钱——
而且**人越多越值钱**。

### 2. IPPO 有明显的规模失效

IPPO 在 10 个单位上**反而比 5 个单位差**：33.5% vs 63.2%（1e7）。
在 5e6 预算下更极端：10.8% vs 46.6%，几乎是"没学会"。

这是独立评论家的典型失败模式：智能体越多，每个个体单靠自己那点局部观测
越难判断全局局势。MAPPO 在同样扩容下基本持平（77.0% vs 81.5%）。

### 3. PQN-VDN 用一半预算几乎追平 MAPPO

5e6 预算下 PQN-VDN 71.8%，MAPPO 73.5%——只差 1.7 个点，
在种子方差（±2.1%）之内，`mean - std` 区间是重叠的。

同时它比 IPPO 高：同预算高 **25.2 点**，即使 IPPO 用两倍预算（1e7）也还高 8.6 点。

**这一点和 SMACv2 论文里的主流观察不完全一致**——那篇论文报告的是
"随机化削弱了同质性假设，值分解方法普遍打不过 PPO 系"。
我们这里值分解没有崩，值得深挖。可能的解释（都还没验证）：

- **网络容量**：PQN-VDN 默认是 `HIDDEN_SIZE=512` × 2 层，比 MAPPO 的 128 宽 4 倍。
  在"环境步"这个横轴上它占便宜，因为每一步的计算量更大。
- **TD(λ)**：它用的是 λ=0.85 的多步回报 + 4 epochs × 16 minibatches 的充分复用，
  样本效率天然高。
- **SMACv2 场景**：SMAX 的三档随机场景和论文里按种族细分的场景不完全等价。

### 4. 但 PQN-VDN 的代价是 8 倍计算量

| | 5e6 步墙钟 |
|---|---:|
| MAPPO | 约 18 min |
| PQN-VDN | **147 min** |

而且它**完全不吃并行**：实测同时跑 3 个 PQN-VDN 进程，
聚合吞吐 2,184 步/秒；单独跑一个是 2,093 步/秒——**3 个并行 ≈ 1 个单跑，
白烧 11.5 个核**。它卡在内存带宽上，不是卡在核数。

所以它是"样本效率高、计算效率低"。在真机上如果算力是瓶颈，
MAPPO 在 18 分钟里达到的水平，PQN-VDN 要跑 2.5 小时。

## 复现

```bash
# PPO 系（MAPPO / IPPO）
.venv/bin/python experiments/scheduler.py --engine ppo --concurrency 6 --steps 1e7 \
    --algo mappo,ippo --map smacv2_5_units,smacv2_10_units --seed 0,1,2

# 值分解系
.venv/bin/python experiments/scheduler.py --engine ql --concurrency 3 --steps 5e6 \
    --algo pqn_vdn_rnn --map smacv2_5_units --seed 0,1,2

# 出表出图（--max-step 用来把不同预算截齐）
.venv/bin/python experiments/analyze.py --max-step 5e6 --suffix _at5e6
.venv/bin/python experiments/analyze.py --suffix _full
```

## 诚实声明（这些结论的边界）

- **每个组合只有 3 个种子**，够看出 10 个百分点以上的差距，不够分辨 2~3 个点的差异。
- **没有做超参搜索**，全部是仓库默认配置。IPPO 的 `ENT_COEF=0.01`、`LR=4e-3` 与
  MAPPO 的 `ENT_COEF=0`、`LR=2e-3` 不同，所以这不是"只变了评论家"的严格消融。
- **预算不对等**：PQN-VDN 是 5e6，PPO 是 1e7。所以给了两个视角，
  但 PQN-VDN 的完整 1e7 表现是未知的。
- **单机单次运行**，没有跨硬件验证。

原始数据在 `results/`，每个 run 一对文件：`.jsonl` 是逐更新的完整曲线，
`.summary.json` 是最终汇总。
