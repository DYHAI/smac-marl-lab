#!/usr/bin/env python
"""跑 JaxMARL 的 Q-Learning 基线（PQN-VDN / IQL / VDN / QMIX），同样绕开 hydra。

和 run_baseline.py 一个套路：仓库脚本的指标只从 io_callback 里的 wandb.log 出来，
这里把它换成写 JSONL 的本地捕获，于是不联网也能拿到逐步曲线。

为什么选 PQN-VDN：仓库自己的 README 写着
  "At the moment, PQN-VDN should be the most performant baseline for
   Q-Learning in terms of returns and training speed."
而且它和 PPO 系是两种完全不同的算法家族（值分解 vs 策略梯度），对比起来更有意思。

用法：
  python experiments/run_qlearning.py --alg pqn_vdn_rnn_smax --map smacv2_5_units \
      --steps 10000000 --seed 0 --out results/pqnvdn_smacv2_5_units_s0.jsonl
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

SCRIPTS = {
    "pqn_vdn_rnn": ("baselines/QLearning/pqn_vdn_rnn.py", "pqn_vdn_rnn_smax"),
    "qmix_rnn": ("baselines/QLearning/qmix_rnn.py", "ql_rnn_smax"),
    "transf_qmix": ("baselines/QLearning/transf_qmix.py", "transf_qmix_smax"),
}


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(f"ql_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def to_plain(value):
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", required=True, choices=sorted(SCRIPTS))
    parser.add_argument("--alg", required=True, help="config/alg 下的 yaml 名（不带后缀）")
    parser.add_argument("--map", required=True)
    parser.add_argument("--steps", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--set", action="append", default=[], help="覆盖 alg.* 键，如 NUM_ENVS=64")
    args = parser.parse_args()

    os.environ.setdefault("WANDB_ENTITY", "local")
    os.environ.setdefault("WANDB_PROJECT", "local")
    os.environ.setdefault("WANDB_MODE", "disabled")

    import jax
    import wandb
    from omegaconf import OmegaConf

    script_path = REPO / SCRIPTS[args.script][0]
    module = load_module(script_path)

    base = OmegaConf.to_container(OmegaConf.load(script_path.parent / "config" / "config.yaml"), resolve=False)
    alg = OmegaConf.to_container(
        OmegaConf.load(script_path.parent / "config" / "alg" / f"{args.alg}.yaml"), resolve=False
    )
    alg["MAP_NAME"] = args.map
    alg["TOTAL_TIMESTEPS"] = args.steps
    for item in args.set:
        key, _, value = item.partition("=")
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = value
        alg[key.strip()] = parsed

    base["SEED"] = args.seed
    base["NUM_SEEDS"] = 1
    base["HYP_TUNE"] = False
    base["SAVE_PATH"] = None
    base["alg"] = alg

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    handle = out_path.open("w", encoding="utf-8")
    captured: list[dict] = []

    def capture(payload, *rest, **kwargs):
        record = {k: to_plain(v) for k, v in payload.items()}
        record = {k: v for k, v in record.items() if v is not None}
        record["algo"] = args.script
        record["map"] = args.map
        record["seed"] = args.seed
        captured.append(record)
        handle.write(json.dumps(record) + "\n")
        handle.flush()
        step = record.get("update_steps", 0)
        ret = record.get("returns")
        win = record.get("win_rate")
        ret_s = f"{ret:7.2f}" if isinstance(ret, float) else "   n/a "
        win_s = f"{win:5.3f}" if isinstance(win, float) else " n/a "
        print(f"[{args.script}/{args.map}] update={step:>5}  return={ret_s}  win={win_s}", flush=True)

    # 注意：PQN-VDN 的日志被 `if config["WANDB_MODE"] != "disabled"` 挡着，
    # 所以这里不能留 disabled，否则整个训练一条指标都不会出来。
    # wandb.init / wandb.log 上面已经换成不联网的本地函数，改成 offline 是安全的。
    merged_mode = "offline"

    wandb.log = capture
    wandb.init = lambda **kwargs: None
    wandb.finish = lambda **kwargs: None
    wandb.define_metric = lambda *a, **k: None

    merged = {**base, **alg}
    merged["WANDB_MODE"] = merged_mode
    env, env_name = module.env_from_config(copy.deepcopy(merged))
    print(f"=== {args.script} on {env_name} ===", flush=True)
    print(f"envs={merged['NUM_ENVS']} steps={merged['NUM_STEPS']} "
          f"total={merged['TOTAL_TIMESTEPS']:.0f} seed={args.seed} backend={jax.default_backend()}", flush=True)

    train = jax.jit(jax.vmap(module.make_train(merged, env)))
    rngs = jax.random.split(jax.random.PRNGKey(args.seed), base["NUM_SEEDS"])
    started = time.time()
    out = train(rngs)
    jax.block_until_ready(out)
    elapsed = time.time() - started
    handle.close()

    summary = {
        "algo": args.script,
        "map": args.map,
        "seed": args.seed,
        "env_name": env_name,
        "total_timesteps": merged["TOTAL_TIMESTEPS"],
        "envs": merged["NUM_ENVS"],
        "steps_per_env": merged["NUM_STEPS"],
        "wall_seconds": round(elapsed, 1),
        "steps_per_second": round(merged["TOTAL_TIMESTEPS"] / elapsed, 1),
        "final_return": captured[-1].get("returns") if captured else None,
        "final_win_rate": captured[-1].get("win_rate") if captured else None,
    }
    print("=== done ===", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    with out_path.with_suffix(".summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
