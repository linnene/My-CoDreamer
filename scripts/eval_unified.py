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


class _SampledDist:
    """把一个分布包起来，把它的 `.mode()` 换成 `.sample()`。

    ⚠️ 第一版**patch 错了对象**：改的是 `agent._task_behavior.actor` 这个 Actor
    **模块**的 `mode`。但 `Dreamer._policy` 里的 `actor` 是
        actor = self._task_behavior.actor(feat, graph=graph)   # ← 这是分布
        action = actor.mode()
    所以真正被调用的是**分布**的 `.mode()`，模块的 `mode` 从来没人调。
    后果是 sampled 那一轮**每个数字与 greedy 逐位相同** —— 一个看起来像结果的空操作。

    `__getattr__` 转发其余一切，`_policy` 后面的 `log_prob` 照常工作。
    """

    def __init__(self, dist):
        self._dist = dist

    def mode(self):
        return self._dist.sample()

    def __getattr__(self, name):
        return getattr(self._dist, name)


def _patch_sampled(actor_module):
    """让这个 Actor 模块每次前向都返回 `_SampledDist`。"""
    def forward(self, *a, **k):
        return _SampledDist(self._orig_forward(*a, **k))

    actor_module._orig_forward = actor_module.forward
    actor_module.forward = types.MethodType(forward, actor_module)


def rollout(agent, env, env_seed, max_steps, n_agents):
    """一局。返回每 agent 平均的**未折扣**回报，以及这条动作轨迹（供自检用）。"""
    env.seed(env_seed)
    obs = env.reset()
    state = None
    total = np.zeros(n_agents, dtype=np.float64)
    done = False
    steps = 0
    trace = []
    while not done and steps < max_steps:
        flat_done = np.zeros(n_agents, dtype=bool)
        action, state = agent(obs, flat_done, state)
        acts = action["action"]
        if torch.is_tensor(acts):
            acts = acts.detach().cpu().numpy()
        idx = np.asarray(acts).argmax(-1).astype(np.int64).reshape(-1)
        if steps < 5:
            trace.append(tuple(idx.tolist()))
        obs, reward, done, _ = env.step(idx)
        total += np.asarray(reward, dtype=np.float64).reshape(-1)
        steps += 1
    return float(total.mean()), steps, trace


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
            _patch_sampled(agent._task_behavior.actor)
        policy = functools.partial(agent, training=False)

        # ★ 自检：干预必须**确实改变了动作** ★
        # 第一版 patch 错了对象，sampled 那一轮与 greedy **逐位相同**，
        # 看起来像一个干净的结果。没有这条断言，那个空操作会直接进台账。
        if args.action_mode == "sample":
            _, _, tr = rollout(policy, env, int(env_seeds[0]), 5, cfg.num_agents)
            agent._task_behavior.actor.forward = agent._task_behavior.actor._orig_forward
            _, _, tr_g = rollout(policy, env, int(env_seeds[0]), 5, cfg.num_agents)
            _patch_sampled(agent._task_behavior.actor)
            if tr == tr_g:
                raise SystemExit(
                    "自检失败：sample 与 greedy 的前 5 步动作完全相同 —— "
                    "干预没有生效（几乎一定是 patch 的对象不对）。"
                )

        rets, lens = [], []
        with torch.no_grad():
            for k in env_seeds:
                r, n, _ = rollout(policy, env, int(k), args.max_steps, cfg.num_agents)
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
