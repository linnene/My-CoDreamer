"""统一重评：所有 checkpoint 在**同一批固定局**上评，不重新训练。

================================================================================
为什么必须做
================================================================================
8-seed 对照里，每个 run 的评测环境是 `make_env(..., seed=config.seed + id)`
（`dreamer.py:321`）——**每个 run 评估的是不同的局**。

于是"CoDreamer 比 IDreamer 低 3.65"这个数里，混着**局难度**：
s7 的 CoDreamer 拿到 −8.66，可能只是它抽到的那 16 局更难。
**八对数字不是配对的**，配对检验（同一批局）根本没做过。

本脚本把这一项消掉：用 `env.seed(k)` 强制每个 checkpoint 面对**完全相同**的
初始条件序列 k = 0..E−1，其余（动作选择规则、局数、步数上限）全部固定。

================================================================================
它不能修什么
================================================================================
只消"局不同"这一项。**训练本身的随机性（权重不同）消不掉** —— 那是 seed 的作用。
所以本脚本回答的是：**3.65 里有多少是"抽到的局不同"**，不是"CoDreamer 是否更好"。

用法::

    python scripts/eval_unified.py --episodes 32 \\
      --checkpoint idreamer=/tmp/runs/vmas_rep_idreamer_s1/latest.pt \\
      --checkpoint codreamer=/tmp/runs/vmas_rep_codreamer_s1/latest.pt
"""

from __future__ import annotations

import argparse
import functools
import pathlib
import sys
import tempfile
import types

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
D3T = REPO / "third_party" / "dreamerv3-torch"
sys.path.insert(0, str(D3T))

import dreamer as dreamer_mod  # noqa: E402


class _SilentLogger:
    step = 0

    def scalar(self, *a, **k):
        pass

    def write(self, *a, **k):
        pass


def _sampled_mode(self, *a, **k):
    """让 actor.mode() 改吐**采样**动作。

    `Dreamer._policy` 在 training=False 时写死 `actor.mode()`（贪心）。论文
    **没说**评测用贪心还是采样，所以两种都要能跑，才能知道这个未说明项有多敏感。
    做法是把这个**实例**的 mode 换掉，不动类、不动仓库代码。
    """
    return self(*a, **k).sample()


def rollout(agent, env, env_seed, max_steps, n_agents):
    """一局。返回每 agent 平均的**未折扣**回报。"""
    env.seed(env_seed)
    obs = env.reset()
    state = None
    total = np.zeros(n_agents, dtype=np.float64)
    done = False
    steps = 0
    while not done and steps < max_steps:
        flat_done = np.zeros(n_agents, dtype=bool)
        action, state = agent(obs, flat_done, state)
        acts = action["action"]
        if torch.is_tensor(acts):
            acts = acts.detach().cpu().numpy()
        idx = np.asarray(acts).argmax(-1).astype(np.int64).reshape(-1)
        obs, reward, done, _ = env.step(idx)
        total += np.asarray(reward, dtype=np.float64).reshape(-1)
        steps += 1
    return float(total.mean()), steps


def evaluate(path: pathlib.Path, env_seeds, args) -> dict:
    ckpt = torch.load(path, map_location=args.device, weights_only=False)
    cfg = argparse.Namespace(**ckpt["config"])
    cfg.device = args.device

    with tempfile.TemporaryDirectory() as tmp:
        env = dreamer_mod.make_env(cfg, "eval", 0)
        cfg.num_agents = getattr(env, "num_agents", 1)
        acts = env.action_space
        cfg.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
        cfg.evaldir = cfg.evaldir or str(pathlib.Path(tmp) / "eval_eps")
        agent = dreamer_mod.Dreamer(
            env.observation_space, env.action_space, cfg, _SilentLogger(), None
        ).to(cfg.device)
        inc = agent.load_state_dict(ckpt["agent_state_dict"], strict=False)
        if inc.missing_keys or inc.unexpected_keys:
            raise SystemExit(f"{path} 结构不匹配")
        agent.eval()
        if args.action_mode == "sample":
            a = agent._task_behavior.actor
            a.mode = types.MethodType(_sampled_mode, a)
        policy = functools.partial(agent, training=False)

        rets, lens = [], []
        with torch.no_grad():
            for k in env_seeds:
                r, n = rollout(policy, env, int(k), args.max_steps, cfg.num_agents)
                rets.append(r)
                lens.append(n)
        env.close()

    return {
        "returns": np.array(rets),
        "lengths": np.array(lens),
        "variant": ckpt["config"].get("comm_variant", "?"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", action="append", default=[], help="名称=路径")
    ap.add_argument("--run-dir", action="append", default=[],
                    help="名称=目录：评该目录下所有 ckpt_*.pt 与 latest.pt，"
                         "按 interval 列出并取最大值（论文的选择规则）")
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--action-mode", choices=("greedy", "sample"), default="greedy")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    env_seeds = np.arange(args.seed_base, args.seed_base + args.episodes)
    print(f"固定局：seed {env_seeds[0]}..{env_seeds[-1]}（共 {len(env_seeds)} 局，"
          f"全部 checkpoint 共用）")
    print(f"动作选择：{args.action_mode}   单局上限：{args.max_steps} 步\n")

    # ---------------------------------------------------------- run-dir 模式
    if args.run_dir:
        for item in args.run_dir:
            name, _, d = item.partition("=")
            d = pathlib.Path(d)
            cks = sorted(d.glob("ckpt_*.pt"))
            if (d / "latest.pt").exists():
                cks.append(d / "latest.pt")
            if not cks:
                print(f"{name}: 目录里没有 ckpt_*.pt（ckpt_every 没打开？）")
                continue
            print(f"=== {name} ===  {len(cks)} 个检查点")
            rows = []
            for c in cks:
                r = evaluate(c, env_seeds, args)
                rows.append((c.stem, r["returns"].mean(),
                             r["returns"].std(ddof=1), r["variant"]))
            for nm, m, s, v in rows:
                print(f"  {nm:<22} mean {m:+.4f}  std {s:.4f}  variant={v}")
            best = max(rows, key=lambda x: x[1])
            last = rows[-1]
            print(f"  → 全程最大（**论文的选择规则**）：{best[0]}  {best[1]:+.4f}")
            print(f"  → 最终点（我们的选择规则）    ：{last[0]}  {last[1]:+.4f}")
            print(f"  → 两者之差 {best[1] - last[1]:+.4f}\n")
        return 0

    res = {}
    for item in args.checkpoint:
        name, _, path = item.partition("=")
        r = evaluate(pathlib.Path(path).expanduser(), env_seeds, args)
        res[name] = r
        print(f"{name:<28} variant={r['variant']:<10} "
              f"mean {r['returns'].mean():+.4f}  "
              f"std {r['returns'].std(ddof=1):.4f}  "
              f"中位 {np.median(r['returns']):+.4f}  "
              f"均值局长 {r['lengths'].mean():.1f}")

    print("\n" + "=" * 74)
    # 配对：同一批局，所以逐局作差是合法的（未配对检验会白扔掉配对信息）
    names = list(res)
    if len(names) == 2:
        a, b = res[names[0]]["returns"], res[names[1]]["returns"]
        d = a - b
        se = d.std(ddof=1) / np.sqrt(len(d))
        t = d.mean() / se if se > 0 else float("nan")
        print(f"配对差（{names[0]} − {names[1]}）：{d.mean():+.4f}"
              f"  SE {se:.4f}  t = {t:.2f}  n = {len(d)}")
        print(f"  逐局胜/平/负：{(d>0).sum()} / {(d==0).sum()} / {(d<0).sum()}")
        print("  ⚠️ 这是**同一批局**下的配对比较，与未配对的 Welch t 不可直接比 ——")
        print("     配对把'局难度'这一项从误差里扣掉了，SE 通常显著更小。")
    else:
        print("给两个 checkpoint 才能算配对差。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
