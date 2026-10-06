#!/usr/bin/env python
"""汇总 results/*.jsonl，出对比图和一张表。

难点是三个算法的指标键名不统一：
  MAPPO / IPPO  ->  returns / win_rate
  PQN-VDN       ->  returned_episode_returns / returned_won_episode
                    （还有 test_* 版本，是带 epsilon=0 的贪心评测，更干净）
这里统一成 win_rate / return 两个名字，PQN-VDN 优先用 test_* 那条。

用法：python experiments/analyze.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"

ALGO_LABEL = {
    "mappo": "MAPPO",
    "ippo": "IPPO",
    "pqn_vdn_rnn": "PQN-VDN",
}

# 同一张图上每条线的颜色（按算法固定，和哪张地图无关）
ALGO_COLOR = {
    "mappo": "#1f77b4",
    "ippo": "#ff7f0e",
    "pqn_vdn_rnn": "#2ca02c",
}


def normalize(record: dict) -> tuple[float, float, float] | None:
    step = record.get("env_step")
    if step is None:
        return None

    win = record.get("win_rate")
    ret = record.get("returns")
    if win is None:
        win = record.get("test_returned_won_episode")
        if win is None:
            win = record.get("returned_won_episode")
    if ret is None:
        ret = record.get("test_returned_episode_returns")
        if ret is None:
            ret = record.get("returned_episode_returns")

    if win is None or ret is None:
        return None
    return float(step), float(ret), float(win)


def load_runs() -> dict[tuple[str, str, int], list[tuple[float, float, float]]]:
    runs = defaultdict(list)
    for path in sorted(RESULTS.glob("*.jsonl")):
        if path.name.startswith("smoke"):
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                point = normalize(record)
                if point is None:
                    continue
                key = (record["algo"], record["map"], int(record.get("seed", 0)))
                runs[key].append(point)
    for key in runs:
        runs[key].sort(key=lambda p: p[0])
    return runs


def aggregate(runs, grid: np.ndarray):
    """把每个算法的多条 seed 曲线插值到同一组 step 上，算 mean/std。"""
    buckets: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    for (algo, map_name, _seed), points in runs.items():
        steps = np.array([p[0] for p in points], dtype=float)
        if len(steps) < 2:
            continue
        wins = np.array([p[2] for p in points], dtype=float)
        rets = np.array([p[1] for p in points], dtype=float)
        limit = min(grid[-1], steps[-1])
        mask = grid <= limit
        interp_win = np.full(grid.shape, np.nan)
        interp_ret = np.full(grid.shape, np.nan)
        interp_win[mask] = np.interp(grid[mask], steps, wins)
        interp_ret[mask] = np.interp(grid[mask], steps, rets)
        buckets[(algo, map_name)].append(np.vstack([interp_win, interp_ret]))

    out = {}
    for key, stack in buckets.items():
        arr = np.stack(stack)  # (seeds, 2, steps)
        out[key] = {
            "seeds": arr.shape[0],
            "win_mean": np.nanmean(arr[:, 0, :], axis=0),
            "win_std": np.nanstd(arr[:, 0, :], axis=0),
            "ret_mean": np.nanmean(arr[:, 1, :], axis=0),
            "ret_std": np.nanstd(arr[:, 1, :], axis=0),
        }
    return out


def last_finite(values: np.ndarray) -> float:
    """取最后一个有效值。不同 run 的训练长度不一样，直接取 grid[-1] 会是 NaN。"""
    finite = np.flatnonzero(np.isfinite(values))
    if finite.size == 0:
        return float("nan")
    return float(values[finite[-1]])


def main() -> int:
    runs = load_runs()
    if not runs:
        print("还没有任何结果")
        return 1

    maps = sorted({key[1] for key in runs})
    algos = sorted({key[0] for key in runs})
    max_step = max(p[0] for points in runs.values() for p in points)
    grid = np.linspace(0, max_step, 200)

    agg = aggregate(runs, grid)

    print(f"读入 {len(runs)} 条 (算法, 地图, 种子) 曲线，maps={maps}, algos={algos}")
    print(f"最长训练到 {max_step:.3g} 环境步\n")
    print(f"{'地图':<20}{'算法':<12}{'种子':>4}  {'最终胜率(mean±std)':>22}  {'最终回报':>10}")
    print("-" * 74)
    summary = {}
    for map_name in maps:
        for algo in algos:
            key = (algo, map_name)
            if key not in agg:
                continue
            entry = agg[key]
            win_m, win_s = last_finite(entry["win_mean"]), last_finite(entry["win_std"])
            ret_m = last_finite(entry["ret_mean"])
            print(f"{map_name:<20}{ALGO_LABEL.get(algo, algo):<12}{entry['seeds']:>4}  "
                  f"{win_m:>12.1%} ± {win_s:<6.1%}  {ret_m:>10.3f}")
            summary[f"{algo}|{map_name}"] = {
                "algo": algo,
                "map": map_name,
                "seeds": entry["seeds"],
                "final_win_rate_mean": float(win_m),
                "final_win_rate_std": float(win_s),
                "final_return_mean": float(ret_m),
            }
    print()

    with (RESULTS / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    # ---- 画图 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for metric, ylabel, fname in (
        ("win", "Evaluation win rate", "compare_win_rate.png"),
        ("ret", "Episodic return", "compare_return.png"),
    ):
        fig, axes = plt.subplots(1, len(maps), figsize=(6.0 * len(maps), 4.4), squeeze=False)
        for col, map_name in enumerate(maps):
            ax = axes[0][col]
            for algo in algos:
                key = (algo, map_name)
                if key not in agg:
                    continue
                entry = agg[key]
                mean = entry[f"{metric}_mean"]
                std = entry[f"{metric}_std"]
                color = ALGO_COLOR.get(algo, None)
                label = f"{ALGO_LABEL.get(algo, algo)} (n={entry['seeds']})"
                ax.plot(grid, mean, color=color, linewidth=1.8, label=label)
                ax.fill_between(grid, mean - std, mean + std, color=color, alpha=0.18)
            ax.set_title(map_name)
            ax.set_xlabel("environment steps")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
            if metric == "win":
                ax.set_ylim(bottom=0)
            ax.legend(fontsize=8)
        fig.suptitle(f"JaxMARL / SMAX — {ylabel}", fontsize=12)
        fig.tight_layout()
        out = RESULTS / fname
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"图已写出：{out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
