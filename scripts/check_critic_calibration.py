"""Critic 校准：V(s_t) 到底对不对得上真实回报？

================================================================================
为什么需要这个脚本
================================================================================
训练日志里出现过这样的组合：

    value_mean ≈ 10 ~ 19     而     eval_return ≈ 0.1 ~ 0.8

直接相除说"高估几十倍"是**不能下的结论**，因为：

  1. critic 的分布是 `symlog_disc`，`mode()` 走 `DiscDist`，在 symlog 桶上加权后
     再 `symexp` 回来 —— 是原始空间没错，这一点查过（tools.py:469）。
  2. 但 `reward_EMA: True`，critic 的目标是 `(target - offset) / scale`
     （models.py:729），`scale = clip(ema_vals[1]-ema_vals[0], min=1.0)`。
     所以它的输出对应的是**归一化后**的回报，不是原始回报。

于是有两个必须分开的可能：
  (a) 只是单位不同 —— 把 V 反归一化后和 G 对得上；
  (b) critic 真的发散了 —— 反归一化后仍然对不上。

两者对项目的含义完全不同：(a) 什么都不用改，(b) 意味着 actor 的 advantage
（= 归一化回报 − 归一化基线）本身是错的，策略会被推向任意方向。

================================================================================
做法
================================================================================
在**固定的 held-out 轨迹**上（同一批 npz，两个 checkpoint 共用）：

    V_t  = critic(posterior 特征 at t)          ← 只吃观测，是状态价值
    G_t  = Σ_{k≥0} γ^k · r_{t+k}                ← 该时刻往后真实收到的折扣回报

然后同时报**原始空间**与**归一化空间**（用 checkpoint 自带的 ema_vals）的
均值、比值与相关系数。

相关系数比均值更重要：即使整体有偏移，只要 V 与 G 高度相关，优势信号的**排序**
仍是对的；相关系数崩掉才是真的把策略推错方向。

用法::

    python scripts/check_critic_calibration.py \\
        --checkpoint IDreamer=/tmp/runs/.../latest.pt \\
        --checkpoint CoDreamer=/tmp/runs/.../latest.pt \\
        --episodes /tmp/runs/shared_rand
"""

from __future__ import annotations

import argparse
import functools
import pathlib
import sys
import tempfile

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
D3T = REPO / "third_party" / "dreamerv3-torch"
sys.path.insert(0, str(D3T))

import dreamer as dreamer_mod  # noqa: E402
import models  # noqa: E402
import tools  # noqa: E402
import tools_marl  # noqa: E402


class _SilentLogger:
    step = 0

    def scalar(self, *a, **k):
        pass

    def write(self, *a, **k):
        pass


def load_checkpoint(path: pathlib.Path, device: str) -> dict:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    for key in ("agent_state_dict", "config"):
        if key not in ckpt:
            raise SystemExit(f"{path} 里没有 {key}")
    return ckpt


def build_agent(config, obs_space, act_space):
    return dreamer_mod.Dreamer(
        obs_space, act_space, config, _SilentLogger(), None
    ).to(config.device)


def episodes_dir(args) -> pathlib.Path:
    """拿到一批 held-out episode 的目录，没有就现采一批写进去。"""
    if args.episodes:
        p = pathlib.Path(args.episodes)
        if list(p.glob("*.npz")):
            return p
    raise SystemExit(
        "需要一个已有的 held-out 目录（--episodes）。"
        "先用 eval_world_model.py --rollout random --rollout-dir 采一批。"
    )


def calibrate(path: pathlib.Path, args) -> dict:
    ckpt = load_checkpoint(path, args.device)
    cfg = argparse.Namespace(**ckpt["config"])
    cfg.device = args.device

    with tempfile.TemporaryDirectory() as tmp:
        env = dreamer_mod.make_env(cfg, "eval", 0)
        cfg.num_agents = getattr(env, "num_agents", 1)
        acts = env.action_space
        cfg.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
        cfg.evaldir = cfg.evaldir or str(pathlib.Path(tmp) / "eval_eps")
        agent = build_agent(cfg, env.observation_space, env.action_space)
        inc = agent.load_state_dict(ckpt["agent_state_dict"], strict=False)
        if inc.missing_keys or inc.unexpected_keys:
            raise SystemExit(f"{path} 结构不匹配，拒绝评测")
        agent.eval()

        eps = tools.load_episodes(episodes_dir(args), limit=10**6)
        dataset = tools_marl.make_graph_dataset(eps, cfg)
        batches = [next(dataset) for _ in range(args.batches)]
        env.close()

    wm = agent._wm
    behavior = agent._task_behavior
    gamma = float(cfg.discount)

    # checkpoint 自带的归一化参数（训练时就是这个）
    ema = ckpt.get("ema_vals")
    if ema is None and hasattr(behavior, "ema_vals"):
        ema = behavior.ema_vals
    ema = torch.as_tensor(ema, dtype=torch.float32).flatten() if ema is not None else None
    offset = float(ema[0]) if ema is not None else 0.0
    scale = float(max(ema[1] - ema[0], 1.0)) if ema is not None else 1.0

    vs, gs = [], []
    with torch.no_grad():
        for data in batches:
            obs = wm.preprocess(data)
            graph = models.graph_from_data(obs, cfg.device)
            embed = wm.encoder(obs)
            post, _ = wm.dynamics.observe(
                embed, obs["action"], obs["is_first"], graph=graph
            )
            feat = wm.dynamics.get_feat(post)          # (B,T,N,F)
            # 与 _train 一致：feat 保持 (B,T,N,·)，MLP 只动末维、GNN 读倒数第二维
            v = models._dist_mean(behavior.value(feat, graph=graph))   # (B,T,N)

            # 真实折扣回报：G_t = Σ_{k≥0} γ^k r_{t+k}（终止后不再累加）
            rew = obs["reward"]                        # (B,T,N)
            cont = obs["cont"]                         # (B,T,N)
            disc = gamma * cont
            T = rew.shape[1]
            g = torch.zeros_like(v)
            run = torch.zeros_like(v)
            for t in reversed(range(T)):
                run = rew[:, t] + disc[:, t] * run
                g[:, t] = run

            vs.append(v.reshape(-1).cpu())
            gs.append(g.reshape(-1).cpu())

    V = torch.cat(vs).numpy()
    G = torch.cat(gs).numpy()

    def stats(a):
        return (
            float(np.mean(a)), float(np.std(a)),
            float(np.percentile(a, 5)), float(np.percentile(a, 95)),
        )

    corr = float(np.corrcoef(V, G)[0, 1]) if V.std() > 0 and G.std() > 0 else float("nan")
    # 归一化空间：critic 学的就是 (G - offset)/scale
    Vn, Gn = (V - offset) / scale, (G - offset) / scale
    corr_n = (
        float(np.corrcoef(Vn, Gn)[0, 1])
        if Vn.std() > 0 and Gn.std() > 0 else float("nan")
    )

    return {
        "path": str(path), "variant": cfg.comm_variant,
        "offset": offset, "scale": scale,
        "V": V, "G": G, "corr_raw": corr, "corr_norm": corr_n,
        "stats_V": stats(V), "stats_G": stats(G),
        "stats_Vn": stats(Vn), "stats_Gn": stats(Gn),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", action="append", required=True, help="名称=路径")
    ap.add_argument("--episodes", default=None,
                    help="已有的 held-out npz 目录（两个 checkpoint 必须用同一批）")
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    if not args.episodes:
        raise SystemExit("--episodes 必填：校准必须在固定轨迹上做")

    out = []
    for item in args.checkpoint:
        name, _, path = item.partition("=")
        print(f"\n=== {name or path} ===")
        r = calibrate(pathlib.Path(path).expanduser(), args)
        out.append((name or path, r))
        print(f"  归一化参数（来自 checkpoint）: offset={r['offset']:.4f} scale={r['scale']:.4f}")
        print(f"  {'空间':<10}{'mean':>12}{'std':>12}{'p5':>12}{'p95':>12}")
        for label, s in [("V 原始", r["stats_V"]), ("G 原始", r["stats_G"]),
                         ("V 归一", r["stats_Vn"]), ("G 归一", r["stats_Gn"])]:
            print(f"  {label:<10}{s[0]:>12.4f}{s[1]:>12.4f}{s[2]:>12.4f}{s[3]:>12.4f}")
        print(f"  相关系数 原始空间 {r['corr_raw']:+.4f}   归一化空间 {r['corr_norm']:+.4f}")

    print("\n" + "=" * 74)
    print("怎么读：")
    print("  · 均值差得远但**相关系数高** → 只是尺度/偏移问题，advantage 的排序仍可用")
    print("  · 相关系数接近 0 或为负     → critic 没有在预测回报，actor 的基线是噪声")
    print("  · 归一化空间才是 critic 真正学的东西，先看那一列")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
