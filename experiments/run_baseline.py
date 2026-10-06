#!/usr/bin/env python
"""跑 JaxMARL 自带的 MARL 基线（MAPPO / IPPO），把训练曲线实时落到 JSONL。

为什么不用仓库自带的入口：
  baselines/*/[mappo|ippo]_rnn_smax.py 都挂着 @hydra.main，而 hydra-core 1.3.7
  （当前最新的正式版）在 Python 3.14 上会因为 argparse 的 help 校验直接抛
  "badly formed help string"，连 main() 都进不去。

  但失败的只有 hydra 的命令行解析那一步。make_train(config) 本身是干净的，
  所以这里直接把 config 写成 dict 喂进去，自己 jit、自己收指标。

  顺带解决第二个问题：这些脚本的指标只通过 io_callback 里的 wandb.log 出来。
  这里把 wandb.log 换成一个本地捕获函数，于是不联网也能拿到每一步的数据。

用法：
  python experiments/run_baseline.py --algo mappo --map smacv2_5_units \
      --steps 1000000 --out results/mappo_smacv2_5_units.jsonl
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

ALGOS = {
    "mappo": ("baselines/MAPPO/mappo_rnn_smax.py", "config/mappo_homogenous_rnn_smax.yaml"),
    "ippo": ("baselines/IPPO/ippo_rnn_smax.py", "config/ippo_rnn_smax.yaml"),
}


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(f"baseline_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def to_plain(value):
    """numpy/jax 标量 -> python 标量（JSON 友好）。"""
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist"):
        out = value.tolist()
        return out if isinstance(out, (int, float, str, bool)) else None
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", required=True, choices=sorted(ALGOS))
    parser.add_argument("--map", required=True, help="SMAX 场景名，如 smacv2_5_units / 2s3z / 3m")
    parser.add_argument("--steps", type=float, required=True, help="总环境步数 TOTAL_TIMESTEPS")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--out", required=True, help="输出 JSONL 路径")
    parser.add_argument("--set", action="append", default=[], help="覆盖任意配置项 key=value")
    args = parser.parse_args()

    # 必须在 import 之前设好：这些脚本里有 ${oc.env:WANDB_ENTITY}
    os.environ.setdefault("WANDB_ENTITY", "local")
    os.environ.setdefault("WANDB_PROJECT", "local")
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")

    import jax
    import wandb
    from omegaconf import OmegaConf

    script_rel, cfg_rel = ALGOS[args.algo]
    script_path = REPO / "baselines" / ("MAPPO" if args.algo == "mappo" else "IPPO") / Path(script_rel).name
    cfg_path = script_path.parent / cfg_rel

    cfg = OmegaConf.load(cfg_path)
    config = OmegaConf.to_container(cfg, resolve=False)
    # wandb 相关的键留着没用，去掉免得 to_container 之外的地方再碰到
    for key in ("ENTITY", "PROJECT", "WANDB_MODE", "NUM_SEEDS"):
        config.pop(key, None)

    config["MAP_NAME"] = args.map
    config["TOTAL_TIMESTEPS"] = args.steps
    config["SEED"] = args.seed
    config["NUM_ENVS"] = args.num_envs
    config["NUM_STEPS"] = args.num_steps
    for item in args.set:
        key, _, value = item.partition("=")
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = value
        config[key.strip()] = parsed

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    module = load_module(script_path)

    captured: list[dict] = []
    handle = out_path.open("w", encoding="utf-8")

    def capture(payload, *rest, **kwargs):
        record = {}
        for key, value in payload.items():
            plain = to_plain(value)
            if plain is not None:
                record[key] = plain
        record["algo"] = args.algo
        record["map"] = args.map
        record["seed"] = args.seed
        captured.append(record)
        handle.write(json.dumps(record) + "\n")
        handle.flush()
        step = record.get("env_step", 0)
        ret = record.get("returns")
        win = record.get("win_rate")
        ret_s = f"{ret:7.2f}" if isinstance(ret, float) else "   n/a "
        win_s = f"{win:5.3f}" if isinstance(win, float) else " n/a "
        print(f"[{args.algo}/{args.map}] step={step:>10.0f}  return={ret_s}  win={win_s}", flush=True)

    # 指标只从 io_callback 里的 wandb.log 出来，这里换成本地捕获，彻底不碰网络
    wandb.log = capture
    wandb.init = lambda **kwargs: None
    wandb.finish = lambda **kwargs: None
    wandb.define_metric = lambda *a, **k: None

    print(f"=== {args.algo} on {args.map} ===", flush=True)
    print(f"envs={config['NUM_ENVS']} steps_per_env={config['NUM_STEPS']} "
          f"total={config['TOTAL_TIMESTEPS']:.0f} seed={args.seed} "
          f"jax_backend={jax.default_backend()}", flush=True)

    train = jax.jit(module.make_train(config))
    started = time.time()
    result = train(jax.random.PRNGKey(args.seed))
    jax.block_until_ready(result)
    elapsed = time.time() - started
    handle.close()

    summary = {
        "algo": args.algo,
        "map": args.map,
        "seed": args.seed,
        "total_timesteps": config["TOTAL_TIMESTEPS"],
        "envs": config["NUM_ENVS"],
        "steps_per_env": config["NUM_STEPS"],
        "updates": len(captured),
        "wall_seconds": round(elapsed, 1),
        "steps_per_second": round(config["TOTAL_TIMESTEPS"] / elapsed, 1),
        "final_return": captured[-1].get("returns") if captured else None,
        "final_win_rate": captured[-1].get("win_rate") if captured else None,
    }
    print("=== done ===", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    with (out_path.with_suffix(".summary.json")).open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
