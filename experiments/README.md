# SMACv2 上的 MARL 基线复现

这层目录是本仓库（fork 自 [bold-lab-ai/JaxMARL](https://github.com/bold-lab-ai/JaxMARL)）
新增的实验脚手架和结果，用来在一台 Apple Silicon Mac 上跑 SMACv2 的 MARL SOTA 基线。

## 为什么用 SMAX 而不是官方 SMACv2

官方 [oxwhirl/smacv2](https://github.com/oxwhirl/smacv2) 依赖 StarCraft II 游戏本体
（走 Blizzard 的 ML API + DeepMind 的 PySC2）。在这台 ARM Mac 上这条路走不通：

| 问题 | 具体情况 |
|---|---|
| SC2 4.10 的 mac 包已下架 | `blzdistsc2-a.akamaihd.net/MacOS/SC2.4.10.zip` 返回 404 |
| 只能用 x86 版 | Blizzard 没有为 SMAC 需要的版本提供 arm64 原生包，只能靠 Rosetta 2 跑 x86_64 |
| 性能与稳定性 | Rosetta 下跑 RTS 引擎，训练吞吐会大幅下降，且 SMAC 的并行环境会让它更糟 |

**SMAX** 是 JaxMARL 里对 SMAC 的纯 JAX 重实现：战斗模拟全部用 JAX 写，不需要游戏本体，
在 CPU 上就能跑，而且原生带 SMACv2 的场景：

```
smacv2_5_units / smacv2_10_units / smacv2_20_units
```

这三个场景保留了 SMACv2 相对 SMAC 的核心改动——**随机出生位置 + 随机兵种生成**
（`smacv2_position_generation` / `smacv2_unit_type_generation`），所以研究问题是一致的。

## 三个算法

覆盖两大流派，都是这个领域公认的强基线：

| 算法 | 流派 | 关键差异 | 实现位置 |
|---|---|---|---|
| **MAPPO** | 策略梯度 + 集中式评论家 | 评论家看全局 `world_state`，执行时只靠局部观测 | `baselines/MAPPO/mappo_rnn_smax.py` |
| **IPPO** | 策略梯度 + 独立评论家 | 每个智能体一个评论家，完全不共享全局信息 | `baselines/IPPO/ippo_rnn_smax.py` |
| **PQN-VDN** | 值分解（离线 Q-learning） | 并行化 Q 网络 + VDN 混合网络 | `baselines/QLearning/pqn_vdn_rnn.py` |

MAPPO / IPPO 用来回答"共享全局信息值多少"；PQN-VDN 是仓库 README 自己点名的
"目前 Q-Learning 里回报和训练速度最好的基线"，代表完全不同的算法家族。

## 环境

| 项 | 值 |
|---|---|
| 机器 | Apple M6，12 核，16 GB 统一内存 |
| Python | 3.14.8（Homebrew） |
| jax / jaxlib | 0.11.2（CPU 后端） |
| flax / optax / distrax / gymnax | 0.12.10 / 0.2.8 / 0.1.9 / 0.0.9 |
| 并发 | 4 个训练进程（每个 JAX CPU 进程约吃 3 个核） |

```bash
cd marl-lab
/opt/homebrew/bin/python3 -m venv .venv
.venv/bin/pip install -e ".[algs]"
```

## 怎么跑

```bash
# 单个 run
.venv/bin/python experiments/run_baseline.py \
    --algo mappo --map smacv2_5_units --steps 1e7 --seed 0 \
    --out results/mappo_smacv2_5_units_s0.jsonl

# 整批（自动跳过已完成的 run，可中断重跑）
.venv/bin/python experiments/scheduler.py --engine ppo --concurrency 4 --steps 1e7 \
    --algo mappo,ippo --map smacv2_5_units,smacv2_10_units --seed 0,1,2

# 值学习那一派
.venv/bin/python experiments/scheduler.py --engine ql --concurrency 3 --steps 5e6 \
    --algo pqn_vdn_rnn --map smacv2_5_units --seed 0,1,2

# 汇总出图和表
.venv/bin/python experiments/analyze.py
```

## 三个必须知道的坑

### 1. hydra 在 Python 3.14 上直接崩

仓库自带的入口脚本都挂着 `@hydra.main`，而 hydra-core 1.3.7（当前最新正式版）
在 Python 3.14 上会这样挂掉：

```
File ".../hydra/_internal/utils.py", line 547, in get_args_parser
    parser.add_argument("--shell-completion", help=LazyCompletionHelp(), ...)
File ".../argparse.py", line 1562, in _check_help
ValueError: badly formed help string
```

`LazyCompletionHelp` 这个自定义 help 对象过不了 3.14 新增的 `_check_help` 校验，
连 `main()` 都进不去。

**绕过方式**：失败的只有 hydra 的命令行解析那一步，`make_train(config)` 本身是干净的。
所以 `run_baseline.py` / `run_qlearning.py` 直接把配置写成 dict 喂进去，
自己调 `jax.jit`，完全不走 hydra。

### 2. 指标只从 `wandb.log` 出来

这些脚本把训练指标塞在 `jax.experimental.io_callback` 里、通过 `wandb.log` 输出，
不联网就一条数据都拿不到。解决办法是把 `wandb.log` 换成一个写 JSONL 的本地函数，
顺便按 update 实时打印进度——既不用 wandb 账号，也不用等训练结束才看得到曲线。

PQN-VDN 还多一层：它的日志被
`if config["WANDB_MODE"] != "disabled"` 挡着，所以配置里不能留 `disabled`，
得设成 `offline`（`wandb.init` / `wandb.log` 已经被换掉了，不会真的联网）。

### 3. shell 的 `jobs -p` 不能用来数并发

最初用 `while [ "$(jobs -p | wc -l)" -ge 4 ]` 控并发，结果卡死在等待分支：
**已经结束但还在 shell job 表里的任务照样被算成"在跑"**，于是调度器认为并发永远满，
后面的任务不再启动。实际表现是 12 个 run 只发出 7 个，第 4 个 CPU 槽白空了半小时。

现在换成 `experiments/scheduler.py`，用 `subprocess.Popen.poll()` 判断谁真的活着。

## 文件说明

| 文件 | 作用 |
|---|---|
| `run_baseline.py` | 跑 MAPPO / IPPO，绕开 hydra，指标落 JSONL |
| `run_qlearning.py` | 跑 PQN-VDN（值学习系），同样绕开 hydra |
| `scheduler.py` | 按并发上限排队跑整批实验，幂等（有 summary 就跳过） |
| `analyze.py` | 汇总所有 JSONL，统一键名、按种子求 mean±std、出对比图 |
| `run_all.sh` | 一次跑完全部实验（第一批 PPO，第二批 PQN-VDN） |
| `../deploy/com.playai.marl-lab.plist` | 把 `run_all.sh` 交给 launchd 托管 |

### 为什么用 launchd 而不是 `nohup ... &`

放在终端会话里起的后台进程，会话被回收时会一起被带走——实际踩过一次：
15:38 那批 4 个训练进程全没了，内存还很宽裕，不是 OOM。
launchd 托管的进程不属于任何 shell 会话，能稳定跑完。

## 结果

见 `results/` 与仓库根目录的 `RESULTS.md`。

## 出处与许可

- 上游：[bold-lab-ai/JaxMARL](https://github.com/bold-lab-ai/JaxMARL)（Apache-2.0），
  SMAX 及其论文见 `jaxmarl/environments/smax/README.md`。
- 本 fork 保留了上游的 `LICENSE` 和完整提交历史，新增内容同样以 Apache-2.0 发布。
- SMAC / SMACv2 原始论文：
  [SMAC](https://arxiv.org/abs/1902.04043) ·
  [SMACv2](https://arxiv.org/abs/2212.07489) ·
  [MAPPO](https://arxiv.org/abs/2103.01955) ·
  [QMIX](https://arxiv.org/abs/1803.11485)
