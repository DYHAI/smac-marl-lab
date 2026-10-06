#!/usr/bin/env python
"""测 MAPPO 在这台机器上的真实吞吐：并发 N 个进程，看每秒能跑多少环境步。

跑法：每个进程都跑固定的步数（默认 524288 = 32 次更新），
墙钟时间取 N 个进程里最长的那个，聚合吞吐 = N × 步数 / 墙钟。

顺带每 5 秒采一次 CPU 占用，看是不是真的把核吃满了。
注意包含 JIT 编译时间，所以另跑一次单更新测编译开销再扣掉，给"稳态吞吐"。

用法：python experiments/benchmark.py --nodes 1,2,4,6 --steps 524288
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = REPO / ".venv" / "bin" / "python"
BENCH_DIR = REPO / "results" / "bench"


def sample_cpu() -> float:
    """当前机器上所有 python 训练进程的 %CPU 之和。"""
    try:
        out = subprocess.run(
            ["ps", "-Ao", "%cpu=,comm="], capture_output=True, text=True, timeout=10
        ).stdout
    except subprocess.SubprocessError:
        return 0.0
    total = 0.0
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        percent, _, command = line.partition(" ")
        if "Python.app" in command or command.endswith("python"):
            try:
                total += float(percent)
            except ValueError:
                pass
    return total


def run_phase(nodes: int, steps: float, algo: str, map_name: str, extra_env: dict) -> dict:
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    import os

    env = dict(os.environ)
    env.update(extra_env)

    processes = []
    started = time.time()
    for index in range(nodes):
        out = BENCH_DIR / f"bench_{algo}_{map_name}_n{nodes}_{index}.jsonl"
        log = BENCH_DIR / f"bench_{algo}_{map_name}_n{nodes}_{index}.log"
        handle = log.open("w", encoding="utf-8")
        process = subprocess.Popen(
            [str(PY), str(REPO / "experiments" / "run_baseline.py"),
             "--algo", algo, "--map", map_name, "--steps", repr(steps),
             "--seed", str(index), "--out", str(out)],
            cwd=str(REPO), stdout=handle, stderr=subprocess.STDOUT, env=env,
        )
        processes.append((process, handle, out))

    cpu_samples = []
    while any(p.poll() is None for p, _h, _o in processes):
        cpu_samples.append(sample_cpu())
        time.sleep(5)
    wall = time.time() - started

    rates = []
    for process, handle, out in processes:
        handle.close()
        summary = out.with_suffix(".summary.json")
        if summary.exists():
            data = json.loads(summary.read_text())
            rates.append(data["steps_per_second"])

    mean_rate = statistics.fmean(rates) if rates else 0.0
    return {
        "nodes": nodes,
        "steps_per_node": steps,
        "wall_seconds": round(wall, 1),
        "per_run_steps_per_sec": round(mean_rate, 1),
        "aggregate_steps_per_sec": round(mean_rate * nodes, 1),
        "cpu_percent_mean": round(statistics.fmean(cpu_samples), 0) if cpu_samples else 0,
        "cpu_percent_max": round(max(cpu_samples), 0) if cpu_samples else 0,
        "ok": len(rates),
    }


def measure_compile(steps: float, algo: str, map_name: str, extra_env: dict) -> float:
    """跑一次极短训练，用来估计 JIT 编译的固定开销（秒）。"""
    result = run_phase(1, steps, algo, map_name, extra_env)
    compile_estimate = max(0.0, result["wall_seconds"] - steps / max(result["per_run_steps_per_sec"], 1e-6))
    return compile_estimate, result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", default="1,2,4,6")
    parser.add_argument("--steps", type=float, default=524288)
    parser.add_argument("--algo", default="mappo")
    parser.add_argument("--map", dest="map_name", default="smacv2_5_units")
    parser.add_argument("--xla-flags", default=None, help="覆盖 XLA_FLAGS")
    parser.add_argument("--json", default="results/bench/summary.json")
    args = parser.parse_args()

    extra_env = {}
    if args.xla_flags is not None:
        extra_env["XLA_FLAGS"] = args.xla_flags

    print(f"=== {args.algo} on {args.map_name} ===")
    print(f"每个进程 {args.steps:.0f} 步；XLA_FLAGS={extra_env.get('XLA_FLAGS', '(默认)')}\n")

    # 先量一次编译开销：1 次更新 = 16384 步，绝大部分时间都是编译
    compile_seconds, short = measure_compile(16384, args.algo, args.map_name, extra_env)
    print(f"JIT 编译开销约 {compile_seconds:.1f}s "
          f"（单更新跑 {short['wall_seconds']}s / 16384 步）\n")

    rows = []
    for nodes in [int(n) for n in args.nodes.split(",") if n.strip()]:
        result = run_phase(nodes, args.steps, args.algo, args.map_name, extra_env)
        steady = args.steps / max(result["wall_seconds"] - compile_seconds, 1e-6)
        result["per_run_steady_steps_per_sec"] = round(steady, 1)
        result["aggregate_steady_steps_per_sec"] = round(steady * nodes, 1)
        rows.append(result)
        print(f"并发 {nodes:>2}: 墙钟 {result['wall_seconds']:>6.1f}s  "
              f"单run {result['per_run_steps_per_sec']:>7.0f} 步/秒  "
              f"聚合 {result['aggregate_steps_per_sec']:>7.0f} 步/秒  "
              f"含编译")
        print(f"           稳态 单run {steady:>7.0f} 步/秒  "
              f"聚合 {steady * nodes:>7.0f} 步/秒  "
              f"CPU 均值 {result['cpu_percent_mean']:.0f}% / 峰值 {result['cpu_percent_max']:.0f}%")

    out_path = REPO / args.json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"compile_seconds": compile_seconds, "rows": rows}, ensure_ascii=False, indent=2))
    print(f"\n原始数据：{out_path}")

    best = max(rows, key=lambda r: r["aggregate_steady_steps_per_sec"])
    print(f"聚合吞吐最高出现在并发 {best['nodes']}：{best['aggregate_steady_steps_per_sec']:.0f} 步/秒")
    print(f"  此时单个 run 到 1e6 步需要 "
          f"{1e6 / best['per_run_steady_steps_per_sec'] / 60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    sys.exit(main())
