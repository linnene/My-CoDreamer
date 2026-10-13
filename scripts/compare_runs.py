#!/usr/bin/env python3
"""把多次 run 的学习曲线叠在一起对比。

用途不止这一次：步骤 2 要拿 CoDreamer 比 IDreamer，也是同一套图。

用法：
    python scripts/compare_runs.py \
        --run "pretrain=100(旧)=~/runs/seg_base" \
        --run "pretrain=500(新)=~/runs/seg_pretrain500" \
        --out figures/ab_pretrain.png

`--run` 的格式是 `标签=日志目录`，可以给多个（至少一个）。
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]


def setup_cjk_font() -> str | None:
    for name in ["Noto Sans CJK JP", "Noto Sans CJK SC", "Noto Sans CJK TC",
                 "Source Han Sans SC", "WenQuanYi Zen Hei", "SimHei",
                 "Heiti SC", "PingFang SC", "Arial Unicode MS"]:
        if name in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return name
    return None


def load(logdir: pathlib.Path) -> list[dict]:
    p = logdir / "metrics.jsonl"
    if not p.is_file():
        raise FileNotFoundError(p)
    rows = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def series(rows, key):
    pts = [(r["step"], r[key]) for r in rows if key in r and r.get("step") is not None]
    if not pts:
        return np.array([]), np.array([])
    x, y = zip(*pts)
    return np.asarray(x, float), np.asarray(y, float)


def smooth(y, w):
    if w <= 1 or len(y) < w:
        return y
    return np.convolve(y, np.ones(w) / w, mode="valid")


def split_segments(x, y, jump_factor=5.0):
    """按 x 轴大跳跃切段，避免把预填(x=0)与训练(从 x≈1000 起)之间连成假直线。"""
    if len(x) < 2:
        return [(x, y)]
    gaps = np.diff(x)
    pos = gaps[gaps > 0]
    if not len(pos):
        return [(x, y)]
    cuts = np.flatnonzero(gaps > jump_factor * float(np.median(pos))) + 1
    return [(a, b) for a, b in zip(np.split(x, cuts), np.split(y, cuts)) if len(a)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True,
                    metavar="标签=日志目录", help="可给多个")
    ap.add_argument("--out", default="figures/compare.png")
    ap.add_argument("--smooth", type=int, default=20)
    ap.add_argument("--title", default=None)
    ap.add_argument("--tail", type=int, default=0,
                    help="只用最后 N 条 train_return 算末尾均值（0=用全部尾部 10%%）")
    args = ap.parse_args()

    font = setup_cjk_font()
    print(f"CJK font: {font or '（未找到）'}")

    runs = []
    for spec in args.run:
        if "=" not in spec:
            print(f"格式错误（应为 标签=目录）: {spec}")
            return 2
        label, path = spec.split("=", 1)
        logdir = pathlib.Path(path).expanduser()
        rows = load(logdir)
        runs.append({"label": label, "logdir": logdir, "rows": rows})
        print(f"  {label:<28} {len(rows)} 行  {logdir}")

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.4))

    # ---------------- ① train_return 滑动平均
    ax = axes[0]
    for i, run in enumerate(runs):
        x, y = series(run["rows"], "train_return")
        if not len(x):
            continue
        c = COLORS[i % len(COLORS)]
        ax.plot(x, y, ls="none", marker=".", ms=2, alpha=0.18, color=c)
        for sx, sy in split_segments(x, y):
            ys = smooth(sy, args.smooth)
            if not len(ys):
                continue
            ax.plot(sx[len(sy) - len(ys):], ys, lw=2.0, color=c, label=run["label"])
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_title(f"① train_return（散点=每局，粗线=滑动平均 w={args.smooth}）")
    ax.set_xlabel("logger.step（训练进度）")
    ax.set_ylabel("回报（0 = 满分）")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    # ---------------- ② eval_return
    ax = axes[1]
    for i, run in enumerate(runs):
        x, y = series(run["rows"], "eval_return")
        if not len(x):
            continue
        c = COLORS[i % len(COLORS)]
        ax.plot(x, y, "o-", color=c, ms=7, lw=1.6, label=run["label"])
        for xi, yi in zip(x, y):
            ax.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=8, color=c)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_title("② eval_return（论文真正用来比优劣的指标）")
    ax.set_xlabel("logger.step")
    ax.set_ylabel("回报")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    # ---------------- ③ model_loss
    ax = axes[2]
    for i, run in enumerate(runs):
        x, y = series(run["rows"], "model_loss")
        if len(x):
            ax.plot(x, y, lw=1.8, color=COLORS[i % len(COLORS)], label=run["label"])
    ax.set_title("③ 世界模型总损失 model_loss")
    ax.set_xlabel("logger.step")
    ax.set_ylabel("loss")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    fig.suptitle(args.title or "  vs  ".join(r["label"] for r in runs), fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"saved: {out}")

    # ---------------- 数字对比表
    print()
    print(f"{'run':<30} {'train 前10%':>12} {'train 后10%':>12} "
          f"{'train 总均':>11} {'eval 末值':>10} {'model_loss 末值':>15}")
    for run in runs:
        x, y = series(run["rows"], "train_return")
        if len(y) >= 10:
            k = max(10, len(y) // 10)
            head, tail, mean = y[:k].mean(), y[-k:].mean(), y.mean()
        else:
            head = tail = mean = float("nan")
        ex, ey = series(run["rows"], "eval_return")
        mx, my = series(run["rows"], "model_loss")
        print(f"{run['label']:<30} {head:>12.3f} {tail:>12.3f} "
              f"{mean:>11.3f} {(ey[-1] if len(ey) else float('nan')):>10.3f} "
              f"{(my[-1] if len(my) else float('nan')):>15.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
