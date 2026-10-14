"""Phase 2（环境切断）实验的四联图。

================================================================================
为什么要四张图而不是一条曲线
================================================================================
「切断环境后策略还在涨」本身**不足以说明世界模型学到了真东西**。它有两种截然
不同的解释：

    策略真的在利用世界模型学到的规律          →  好
    策略在利用世界模型的误差（自举误差累积）  →  假繁荣

只看回报曲线分不出这两者。所以同时画出**世界模型的预测损失**：

    ② 回报在涨  +  ③ 损失平稳   →  真学到了东西
    ② 回报在涨  +  ③ 损失飙高   →  策略在骗自己
    ② 回报不动  +  ③ 损失平稳   →  世界模型不够用，想象里学不到新东西

这正是 CoDreamer 的主张能被检验的地方：如果它的世界模型更准，③ 就该更低，
② 就该更持久。

================================================================================
四张图
================================================================================
① 回报 vs 训练步      评测回报 + 训练回报，标出环境切断点
② Phase 2 放大        只看切断之后那一段（纯想象训练）
③ 世界模型预测损失    reward / state / cont 三条，标出切断点
④ 部署评测分布        来自 eval_policy.py --out-json（可选）

用法::

    python scripts/plot_phase2.py \\
        --run "IDreamer=~/runs/idream_phase2" \\
        --run "CoDreamer=~/runs/codream_phase2" \\
        --deploy figures/deploy.json \\
        --out figures/phase2.png
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

COLORS = ["#3b6ea5", "#c1554b", "#5a9367", "#8a6fb0", "#c08a2e"]


def load_metrics(run_dir: pathlib.Path) -> list[dict]:
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        raise SystemExit(f"找不到 {path}")
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def series(rows: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    xs = [r["step"] for r in rows if key in r]
    ys = [r[key] for r in rows if key in r]
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def cutoff_step(rows: list[dict]) -> float | None:
    """环境切断发生在哪一步。env_cutoff 由 dreamer.py 写成 0/1。"""
    hit = [r["step"] for r in rows if r.get("env_cutoff") == 1.0]
    return float(min(hit)) if hit else None


def smooth(y: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or y.size < w:
        return y
    kernel = np.ones(w) / w
    pad = np.concatenate([np.full(w - 1, y[0]), y])
    return np.convolve(pad, kernel, mode="valid")


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 2 实验四联图")
    ap.add_argument("--run", action="append", required=True, metavar="名字=目录")
    ap.add_argument("--deploy", default=None, help="eval_policy.py --out-json 的产物")
    ap.add_argument("--out", required=True)
    ap.add_argument("--smooth", type=int, default=1, help="训练回报的滑动窗口")
    args = ap.parse_args()

    runs = []
    for item in args.run:
        name, _, d = item.partition("=")
        if not d:
            d, name = name, pathlib.Path(name).name
        rows = load_metrics(pathlib.Path(d).expanduser())
        runs.append((name, rows))
        print(f"  {name}: {len(rows)} 行, 切断点 {cutoff_step(rows)}")

    deploy = None
    if args.deploy:
        p = pathlib.Path(args.deploy).expanduser()
        if p.exists():
            deploy = json.loads(p.read_text())
        else:
            print(f"  [警告] 找不到 {p}，第 ④ 张图跳过")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_panels = 4 if deploy else 3
    # sharex：① 与 ③ 的横轴范围不同（损失日志会延续到 Phase 2），
    # 不共享的话两根切断竖线在两图里位置不一致，很容易看错。
    fig, axes = plt.subplots(1, n_panels, figsize=(5.2 * n_panels, 4.4), sharex=True)
    if n_panels == 1:
        axes = [axes]

    # ---------------------------------------------------------- ① 全周期
    ax = axes[0]
    for i, (name, rows) in enumerate(runs):
        c = COLORS[i % len(COLORS)]
        xs, ys = series(rows, "eval_return")
        if xs.size:
            ax.plot(xs, ys, "-o", color=c, label=f"{name} eval", markersize=3.5)
        xs, ys = series(rows, "train_return")
        if xs.size:
            ax.plot(xs, smooth(ys, args.smooth), color=c, alpha=0.3, linewidth=1.1,
                    label=f"{name} train (smoothed)")
        cut = cutoff_step(rows)
        if cut is not None:
            ax.axvline(cut, color=c, linestyle="--", linewidth=1.1, alpha=0.75)
    ax.axhline(0.0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("training step")
    ax.set_ylabel("episode return (0 = optimal)")
    ax.set_title("(1) full run\n(dashed line = environment cut off)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)

    # ---------------------------------------------------------- ② Phase 2 放大
    ax = axes[1]
    for i, (name, rows) in enumerate(runs):
        cut = cutoff_step(rows)
        if cut is None:
            continue
        c = COLORS[i % len(COLORS)]
        xs, ys = series(rows, "eval_return")
        m = xs >= cut
        if m.sum() >= 2:
            ax.plot(xs[m], ys[m], "-o", color=c, markersize=4, label=name)
            ax.annotate(
                f"{ys[m][0]:+.3f}",
                (xs[m][0], ys[m][0]), fontsize=7, color=c,
                xytext=(3, 4), textcoords="offset points",
            )
            ax.annotate(
                f"{ys[m][-1]:+.3f}  (Δ{ys[m][-1] - ys[m][0]:+.3f})",
                (xs[m][-1], ys[m][-1]), fontsize=7, color=c,
                xytext=(3, -10), textcoords="offset points",
            )
    ax.axhline(0.0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("training step")
    ax.set_ylabel("eval return")
    ax.set_title("(2) after the environment is cut off\n(imagination only, no new data)")
    ax.grid(alpha=0.25)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8)

    # ---------------------------------------------------------- ③ 世界模型损失
    ax = axes[2]
    for i, (name, rows) in enumerate(runs):
        c = COLORS[i % len(COLORS)]
        for key, ls in [("reward_loss", "-"), ("state_loss", "--"), ("cont_loss", ":")]:
            xs, ys = series(rows, key)
            if xs.size:
                ax.plot(xs, ys, ls, color=c, linewidth=1.2,
                        label=f"{name} {key.replace('_loss','')}")
        cut = cutoff_step(rows)
        if cut is not None:
            ax.axvline(cut, color=c, linestyle="--", linewidth=1.1, alpha=0.75)
    ax.set_xlabel("training step")
    ax.set_ylabel("prediction loss (lower is better)")
    ax.set_title("(3) world model prediction loss\n(climbing after the cut => imagination is self-deceiving)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)

    # ---------------------------------------------------------- ④ 部署分布
    if deploy:
        ax = axes[3]
        names = list(deploy.keys())
        data = [np.asarray(deploy[n]["scores"], dtype=np.float64) for n in names]
        bp = ax.boxplot(
            data, labels=names, widths=0.5, patch_artist=True, showfliers=False,
            medianprops=dict(color="black", linewidth=1.6),
        )
        for j, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(COLORS[j % len(COLORS)])
            patch.set_alpha(0.45)
        rng = np.random.default_rng(0)
        for j, (scores, name) in enumerate(zip(data, names), start=1):
            x = j + rng.uniform(-0.16, 0.16, size=scores.size)
            ax.scatter(x, scores, s=5, alpha=0.3, linewidths=0,
                       color=COLORS[(j - 1) % len(COLORS)], zorder=3)
            ax.scatter([j], [scores.mean()], marker="D", s=32,
                       color=COLORS[(j - 1) % len(COLORS)],
                       edgecolor="white", linewidth=0.8, zorder=4)
        ax.axhline(0.0, color="gray", linestyle=":", linewidth=0.8)
        ax.set_ylabel("episode return")
        ax.set_title("(4) deployment evaluation\n(same fixed episodes, diamond = mean)")
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Cutting the environment off: how long can the world model carry the policy?", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = pathlib.Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"\n  图已保存: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
