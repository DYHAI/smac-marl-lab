#!/bin/sh
# 一次跑完全部实验：先 MAPPO/IPPO，再 PQN-VDN。
#
# 为什么要交给 launchd 而不是 nohup &：
#   放在终端会话里起的后台进程，会话被回收时会一起被带走
#   （已经踩过一次：15:38 那批 4 个训练进程全没了，不是内存问题）。
#   launchd 托管的进程不属于任何 shell 会话，能稳定跑完。
#
# 幂等：已经产出 .summary.json 的 run 会跳过，中断后重跑不会白干。
set -u

APP=/Users/dingding/srv/marl-lab
cd "$APP" || exit 1
mkdir -p results logs

echo "===== $(date '+%Y-%m-%d %H:%M:%S') MARL 实验开始 ====="

# 第一批：MAPPO / IPPO × {5_units, 10_units} × 3 seeds，1e7 步。
# 并发 4 是有依据的：每个 JAX CPU 进程大约吃 3 个核，这台机器 12 核。
.venv/bin/python experiments/scheduler.py --engine ppo --concurrency 4 --steps 1e7 \
    --algo mappo,ippo --map smacv2_5_units,smacv2_10_units --seed 0,1,2

echo "===== $(date '+%H:%M:%S') 第一批结束，开始 PQN-VDN ====="

# 第二批：值学习那一派。慢得多（实测约 2k 步/秒，PPO 是 5~8k），
# 所以预算砍到 5e6，只跑 5_units、3 个种子。
.venv/bin/python experiments/scheduler.py --engine ql --concurrency 3 --steps 5e6 \
    --algo pqn_vdn_rnn --map smacv2_5_units --seed 0,1,2

echo "===== $(date '+%Y-%m-%d %H:%M:%S') 全部实验结束 ====="
ls -1 results/*.summary.json 2>/dev/null | wc -l | tr -d ' ' | sed 's/$/ 个 run 完成/'
