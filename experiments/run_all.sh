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

# 全部 10 个 run 放进同一个队列，让短任务去填空出来的槽。
# 分两批串行跑过一版，实测总时长更长（第一批 84 min + 第二批 56 min），
# 合并后短任务（ippo_5_units 只要 30 min）能提前腾出槽位给长任务。
#
# 并发 6 的依据：BENCHMARK.md 的扫描显示 6 进程聚合 23,306 步/秒（8 进程是
# 24,593，只多 5%），但单 run 快 26%，而且内存只占 6×1.18=7.1 GB。
.venv/bin/python experiments/scheduler.py --engine ppo --concurrency 6 \
    --steps 1e7 --steps-ql 5e6 \
    --algo mappo,ippo,pqn_vdn_rnn \
    --map smacv2_5_units,smacv2_10_units --map-ql smacv2_5_units \
    --seed 0,1,2

echo "===== $(date '+%Y-%m-%d %H:%M:%S') 全部实验结束 ====="
ls -1 results/*.summary.json 2>/dev/null | wc -l | tr -d ' ' | sed 's/$/ 个 run 完成/'
