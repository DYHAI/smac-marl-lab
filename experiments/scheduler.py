#!/usr/bin/env python
"""实验调度器：按并发上限跑一批训练任务，已完成（有 .summary.json）的自动跳过。

为什么不用 shell 的 `jobs -p` 来数并发：
  之前那版 run_matrix.sh 用 `while [ "$(jobs -p | wc -l)" -ge N ]` 控并发，
  结果卡死在等待分支——已经结束的任务仍留在 shell 的 job 表里被算成"在跑"，
  于是后面的任务永远不启动，12 个 run 只发出 7 个，第 4 个槽白空着。
  这里改用 subprocess.Popen + poll()，谁真的在跑一目了然。

用法：
  python experiments/scheduler.py --engine ppo --concurrency 4 --steps 1e7 \
      --algo mappo,ippo --map smacv2_5_units,smacv2_10_units --seed 0,1,2
  python experiments/scheduler.py --engine ql  --concurrency 3 --steps 5e6 \
      --algo pqn_vdn_rnn --map smacv2_5_units --seed 0,1,2
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = REPO / ".venv" / "bin" / "python"
RESULTS = REPO / "results"
LOGS = REPO / "logs"

# 每种算法对应哪个 runner
ENGINE_FOR_ALGO = {
    "mappo": "ppo",
    "ippo": "ppo",
    "pqn_vdn_rnn": "ql",
    "qmix_rnn": "ql",
}

QL_ALG_CONFIG = {
    "pqn_vdn_rnn": "pqn_vdn_rnn_smax",
    "qmix_rnn": "ql_rnn_smax",
}


def tag_for(algo: str, map_name: str, seed: int) -> str:
    short = {"pqn_vdn_rnn": "pqnvdn", "qmix_rnn": "qmix"}.get(algo, algo)
    return f"{short}_{map_name}_s{seed}"


def steps_for(algo: str, steps_ppo: float, steps_ql: float) -> float:
    return steps_ppo if ENGINE_FOR_ALGO[algo] == "ppo" else steps_ql


def build_command(algo: str, map_name: str, seed: int, steps: float, out: Path) -> list[str]:
    engine = ENGINE_FOR_ALGO[algo]
    if engine == "ppo":
        return [
            str(PY), str(REPO / "experiments" / "run_baseline.py"),
            "--algo", algo, "--map", map_name,
            "--steps", repr(steps), "--seed", str(seed), "--out", str(out),
        ]
    return [
        str(PY), str(REPO / "experiments" / "run_qlearning.py"),
        "--script", algo, "--alg", QL_ALG_CONFIG[algo], "--map", map_name,
        "--steps", repr(steps), "--seed", str(seed), "--out", str(out),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["ppo", "ql"], required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--steps", type=float, required=True)
    parser.add_argument("--steps-ql", type=float, default=None,
                        help="值学习系单独用不同预算（不给就跟 --steps 一致）")
    parser.add_argument("--algo", required=True, help="逗号分隔")
    parser.add_argument("--map", required=True, help="逗号分隔")
    parser.add_argument("--map-ql", default=None,
                        help="值学习系单独用不同场景列表（不给就跟 --map 一致）")
    parser.add_argument("--seed", required=True, help="逗号分隔")
    parser.add_argument("--dry-run", action="store_true", help="只打印队列，不启动")
    args = parser.parse_args()

    RESULTS.mkdir(exist_ok=True)
    LOGS.mkdir(exist_ok=True)

    algos = [a.strip() for a in args.algo.split(",") if a.strip()]
    maps = [m.strip() for m in args.map.split(",") if m.strip()]
    maps_ql = ([m.strip() for m in args.map_ql.split(",") if m.strip()]
               if args.map_ql else maps)
    seeds = [int(s) for s in args.seed.split(",") if s.strip()]

    queue = []
    for algo in algos:
        algo_maps = maps if ENGINE_FOR_ALGO[algo] == "ppo" else maps_ql
        for map_name in algo_maps:
            for seed in seeds:
                tag = tag_for(algo, map_name, seed)
                out = RESULTS / f"{tag}.jsonl"
                if (RESULTS / f"{tag}.summary.json").exists():
                    print(f"[skip] {tag} 已有结果", flush=True)
                    continue
                steps = steps_for(algo, args.steps, args.steps_ql if args.steps_ql else args.steps)
                queue.append((tag, build_command(algo, map_name, seed, steps, out), out))

    print(f"队列 {len(queue)} 个 run，并发 {args.concurrency}", flush=True)
    if args.dry_run:
        for tag, command, _out in queue:
            print(f"  {tag:<32} steps={command[command.index('--steps') + 1]}")
        return 0
    running: list[tuple[str, subprocess.Popen, object]] = []
    started_at = time.time()

    def report(name: str, process: subprocess.Popen, log_handle) -> None:
        log_handle.close()
        code = process.returncode
        flag = "✓" if code == 0 else f"✗ exit={code}"
        print(f"[{time.strftime('%H:%M:%S')}] {flag} {name}  "
              f"(已跑 {int(time.time() - started_at)}s)", flush=True)

    while queue or running:
        while queue and len(running) < args.concurrency:
            tag, command, _out = queue.pop(0)
            log_handle = (LOGS / f"{tag}.log").open("w", encoding="utf-8")
            print(f"[{time.strftime('%H:%M:%S')}] [launch] {tag}", flush=True)
            process = subprocess.Popen(command, cwd=str(REPO), stdout=log_handle,
                                       stderr=subprocess.STDOUT)
            running.append((tag, process, log_handle))

        still: list[tuple[str, subprocess.Popen, object]] = []
        for tag, process, log_handle in running:
            if process.poll() is None:
                still.append((tag, process, log_handle))
            else:
                report(tag, process, log_handle)
        running = still
        if queue or running:
            time.sleep(5)

    print(f"[{time.strftime('%H:%M:%S')}] 全部结束", flush=True)
    done = len(list(RESULTS.glob("*.summary.json")))
    print(f"results/ 下共 {done} 个 summary", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
