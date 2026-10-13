#!/usr/bin/env python3
"""Sequential Estimate Game 的可视化 —— 一条命令产出一张图，用来"看见"这个游戏。

这个游戏是抽象协调游戏，没有空间布局：
  - 每个 agent 有一个 [0,1] 的局部状态 s_i
  - 一张随机邻接矩阵（每 episode 静态）决定谁影响谁
  - 每步每个 agent 要从 |A| 个动作里选一个，代表 [0,1] 上的一个等距区间
  - 目标是猜中 y_i —— 由【自己】和【邻居】的状态共同决定的值

用法（在仓库根目录）：
    python scripts/viz_seg.py --out figures/seg_preview.png
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")  # 无头环境
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "dreamerv3-torch"))

from envs.seg import SequentialEstimateGame, Spec  # noqa: E402


def setup_cjk_font() -> str | None:
    """挑一个能出中文的字体。没有就退回去（图里中文会变方框，但不影响运行）。"""
    candidates = [
        "Noto Sans CJK JP",
        "Noto Sans CJK SC",
        "Noto Sans CJK TC",
        "Source Han Sans SC",
        "WenQuanYi Zen Hei",
        "SimHei",
        "Heiti SC",
        "PingFang SC",
        "Arial Unicode MS",
    ]
    available = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False  # 否则负号也是方框
            return name
    return None


def pick_seed(spec: Spec, want_isolated: bool = True, trials: int = 400) -> int:
    """挑一个'有看头'的种子：最好既有连通的、也有孤立的 agent。"""
    for s in range(trials):
        g = SequentialEstimateGame(spec, seed=s)
        g.reset()
        deg = np.array([len(nb) for nb in g.neighbors])
        if not want_isolated:
            return s
        if (deg == 0).any() and (deg > 0).any():
            return s
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/seg_preview.png")
    ap.add_argument("--seed", type=int, default=-1, help="-1 表示自动挑")
    args = ap.parse_args()

    font = setup_cjk_font()
    print(f"CJK font: {font or '（未找到，中文会显示为方框）'}")

    spec = Spec()
    seed = pick_seed(spec) if args.seed < 0 else args.seed

    game = SequentialEstimateGame(spec, seed=seed)
    obs = game.reset()
    n, a = spec.num_agents, spec.num_actions
    adj = game.adjacency.copy()

    # 打一局，记录轨迹
    states, targets, rewards, actions = [game.state.copy()], [], [], []
    for t in range(spec.horizon):
        y = game.target()
        # 用一个"半好"的策略：一半时候选对，一半时候随机 —— 这样奖励不会全是 0
        exact = np.clip(np.floor(y * a + 1e-9).astype(int), 0, a - 1)
        rng = np.random.default_rng(seed * 100 + t)
        act = np.where(rng.random(n) < 0.5, exact, rng.integers(0, a, n))
        _, rew, done, _ = game.step(act)
        targets.append(y)
        rewards.append(rew)
        actions.append(act)
        states.append(game.state.copy())

    states = np.array(states)          # (T+1, N)
    targets = np.array(targets)        # (T, N)
    rewards = np.array(rewards)        # (T, N)
    actions = np.array(actions)        # (T, N)
    neighbours = game.neighbors
    nb_mean = np.zeros_like(states)
    for t in range(spec.horizon):
        for i, nbs in enumerate(neighbours):
            nb_mean[t, i] = states[t][nbs].mean() if nbs.size else states[t][i]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        f"Sequential Estimate Game  (n={n}, T={spec.horizon}, |A|={a}, "
        f"ρ={spec.edge_density}, η={spec.local_state_ratio}, seed={seed})",
        fontsize=13,
    )

    # ---------------------------------------------------------- ① 邻接矩阵
    ax = axes[0, 0]
    ax.imshow(adj, cmap="Blues", vmin=0, vmax=1.4)
    for i in range(n):
        for j in range(n):
            if adj[i, j]:
                ax.text(j, i, "×", ha="center", va="center", color="white", fontsize=14)
    ax.set_title("① 邻接矩阵 A（× = 相连）")
    ax.set_xlabel("agent j")
    ax.set_ylabel("agent i")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    for i, nbs in enumerate(neighbours):
        deg = len(nbs)
        label = f"deg={deg}" + ("   ⚠孤立" if deg == 0 else "")
        ax.text(n + 0.15, i, label, va="center", fontsize=9,
                color="crimson" if deg == 0 else "black")

    # ---------------------------------------------------------- ② 状态演化
    ax = axes[0, 1]
    for i in range(n):
        ax.plot(range(spec.horizon + 1), states[:, i], "-o", label=f"agent {i}")
    ax.set_title("② 局部状态 s_i 随时间的演化")
    ax.set_xlabel("timestep")
    ax.set_ylabel("s_i")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    # ---------------------------------------------------------- ③ 目标与动作区间
    ax = axes[0, 2]
    t_show = 0
    ax.add_patch(Rectangle((0, -1.2), 1, 1.2, facecolor="#f5f5f5", edgecolor="none"))
    cmap = plt.get_cmap("tab10")
    for k in range(a):
        lo, hi = k / a, (k + 1) / a
        ax.add_patch(Rectangle((lo, -0.18), hi - lo, 0.16, facecolor=cmap(k),
                               alpha=0.35, edgecolor="gray", lw=0.8))
        ax.text((lo + hi) / 2, -0.10, f"a={k}", ha="center", va="center", fontsize=8)
    for i in range(n):
        y = targets[t_show, i]
        ax.plot([y], [i], "v", color=cmap(actions[t_show, i]), ms=11,
                markeredgecolor="black", lw=0.7)
        ax.text(1.03, i, f"y={y:.3f}", va="center", fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"agent {i}" for i in range(n)])
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlim(-0.03, 1.18)
    ax.set_title(f"③ t={t_show} 时，各 agent 的目标 y 与所选区间\n(▼颜色 = 选的动作)")
    ax.set_xlabel("目标值 y  /  动作区间 [a/A, (a+1)/A)")
    ax.grid(axis="x", alpha=0.3)

    # ---------------------------------------------------------- ④ 目标由什么组成
    ax = axes[1, 0]
    t_show = 1
    s_now = states[t_show]
    x = np.arange(n)
    own = spec.local_state_ratio * s_now
    nbr = (1 - spec.local_state_ratio) * nb_mean[t_show]
    y_now = targets[t_show]
    ax.bar(x, own, label=f"η·s_i  (自己的贡献, η={spec.local_state_ratio})",
           color="#4c72b0")
    ax.bar(x, nbr, bottom=own,
           label=f"(1−η)·mean(邻居)  (邻居的贡献)", color="#dd8452")
    ax.plot(x, y_now, "k_", ms=22, mew=3, label="目标 y_i（两者之和）")
    for i in range(n):
        ax.text(i, -0.09, "孤立→" if len(neighbours[i]) == 0 else "",
                ha="center", fontsize=8, color="crimson")
    ax.set_xticks(x)
    ax.set_xticklabels([f"agent {i}" for i in range(n)])
    ax.set_ylim(-0.15, 1.05)
    ax.set_title(f"④ t={t_show}：目标 y 由【自己】与【邻居】共同决定")
    ax.set_ylabel("贡献")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # ---------------------------------------------------------- ⑤ 奖励
    ax = axes[1, 1]
    im = ax.imshow(rewards, cmap="Reds", aspect="auto", vmin=rewards.min(), vmax=0)
    for t in range(spec.horizon):
        for i in range(n):
            ax.text(i, t, f"{rewards[t, i]:.2f}", ha="center", va="center",
                    fontsize=8, color="black")
    ax.set_title("⑤ 奖励 r_i（恒 ≤ 0，最优 = 0）")
    ax.set_xlabel("agent i")
    ax.set_ylabel("timestep")
    ax.set_xticks(range(n))
    fig.colorbar(im, ax=ax, fraction=0.046)

    # ---------------------------------------------------------- ⑥ 奖励分布
    ax = axes[1, 2]
    for i in range(n):
        ax.plot(range(spec.horizon), rewards[:, i], "-o", label=f"agent {i}")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.text(0.02, 0.02, "0 = 猜中（最优）", transform=ax.transAxes, fontsize=8)
    ax.set_title("⑥ 各 agent 每步的奖励")
    ax.set_xlabel("timestep")
    ax.set_ylabel("reward")
    ax.set_xticks(range(spec.horizon))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"saved: {out}")
    print(f"seed={seed}  degrees={[len(nb) for nb in neighbours]}  "
          f"mean_reward={rewards.mean():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
