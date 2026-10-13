#!/usr/bin/env python3
"""把 metrics.jsonl 画成学习曲线。

用法（在实验机上，或把 metrics.jsonl 拉到本地后）：
    python scripts/plot_metrics.py --logdir ~/runs/seg_base --out figures/seg_base.png
    python scripts/plot_metrics.py --logdir ~/runs/seg_base --out x.png --smooth 20
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(logdir: pathlib.Path) -> list[dict]:
    p = logdir / "metrics.jsonl"
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


def series(rows: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    xs = [r["step"] for r in rows if key in r and r["step"] is not None]
    ys = [r[key] for r in rows if key in r and r["step"] is not None]
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def smooth(y: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or len(y) < w:
        return y
    k = np.ones(w) / w
    return np.convolve(y, k, mode="valid")


def setup_cjk_font() -> str | None:
    candidates = [
        "Noto Sans CJK JP", "Noto Sans CJK SC", "Noto Sans CJK TC",
        "Source Han Sans SC", "WenQuanYi Zen Hei", "SimHei",
        "Heiti SC", "PingFang SC", "Arial Unicode MS",
    ]
    available = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return name
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--out", default="figures/metrics.png")
    ap.add_argument("--smooth", type=int, default=15,
                    help="train_return 的滑动平均窗口（0/1 表示不平滑）")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    font = setup_cjk_font()
    logdir = pathlib.Path(args.logdir).expanduser()
    rows = load(logdir)
    if not rows:
        print(f"没有读到任何指标: {logdir/'metrics.jsonl'}")
        return 1
    print(f"CJK font: {font or '（未找到）'}   metrics 行数: {len(rows)}")

    fig, axes = plt.subplots(2, 2, figsize=(15, 8))

    # ---------------- ① train_return
    ax = axes[0, 0]
    x, y = series(rows, "train_return")
    if len(x):
        ax.plot(x, y, alpha=0.25, lw=0.8, color="tab:blue", label="每局回报")
        ys = smooth(y, args.smooth)
        if len(ys) == len(y):
            ax.plot(x, ys, lw=1.2, color="tab:blue")
        else:
            ax.plot(x[args.smooth - 1:], ys, lw=2.0, color="tab:blue",
                    label=f"滑动平均 (w={args.smooth})")
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.text(0.02, 0.96, "0 = 满分（猜中目标）", transform=ax.transAxes,
                fontsize=9, va="top")
    ax.set_title("① 训练回报 train_return（每完成一局记一个点）")
    ax.set_xlabel("agent 步")
    ax.set_ylabel("回报")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    # ---------------- ② eval_return
    ax = axes[0, 1]
    x, y = series(rows, "eval_return")
    if len(x):
        ax.plot(x, y, "o-", color="tab:orange", ms=7, lw=1.5)
        for xi, yi in zip(x, y):
            ax.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=8)
        ax.axhline(0, color="black", lw=0.8, ls="--")
    else:
        ax.text(0.5, 0.5, "没有 eval_return", ha="center", transform=ax.transAxes)
    ax.set_title("② 评测回报 eval_return（策略用 mode，非采样）")
    ax.set_xlabel("agent 步")
    ax.set_ylabel("回报")
    ax.grid(alpha=0.3)

    # ---------------- ③ 世界模型损失
    ax = axes[1, 0]
    for key, color in [("model_loss", "tab:red"), ("state_loss", "tab:green"),
                       ("reward_loss", "tab:purple"), ("cont_loss", "tab:brown")]:
        x, y = series(rows, key)
        if len(x):
            ax.plot(x, y, lw=1.3, color=color, label=key)
    ax.set_title("③ 世界模型损失")
    ax.set_xlabel("agent 步")
    ax.set_ylabel("loss")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # ---------------- ④ 训练进度信息
    ax = axes[1, 1]
    x, y = series(rows, "dataset_size")
    if len(x):
        ax.plot(x, y, lw=1.5, color="tab:cyan", label="dataset_size（agent 步）")
    x2, y2 = series(rows, "update_count")
    if len(x2):
        ax2 = ax.twinx()
        ax2.plot(x2, y2, lw=1.2, color="tab:gray", label="update_count")
        ax2.set_ylabel("梯度更新次数", color="tab:gray")
    ax.set_title("④ 训练进度")
    ax.set_xlabel("agent 步")
    ax.set_ylabel("数据集规模")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    title = args.title or f"{logdir}  —  IDreamer @ Sequential Estimate Game (N=4)"
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
