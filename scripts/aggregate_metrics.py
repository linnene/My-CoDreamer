"""按论文的口径聚合：Median / IQM / Mean，附 bootstrap 95% CI。

================================================================================
口径对齐
================================================================================
论文 Figure 4 的 caption 是 "Reported from left to right are: Median, IQM, Mean."
其 D.1 定义每个 (任务 m, 训练 run n) 得到一个分数 x_{m,n}，聚合统计量在 **run** 上算。
所以这里的单位是 **每个 seed 的均值**（该 seed 在全部固定局上的平均回报），
不是逐局样本 —— 同一 seed 的 32 局高度相关，当独立样本会把 CI 算窄。

**IQM**（Interquartile Mean）= 去掉最低 25% 与最高 25% 之后的均值。
它存在的理由正是**对重尾稳健** —— 而本项目观察到的臂间差异恰好是重尾
（CoDreamer σ 约 3，IDreamer 约 1）。所以这个统计量会**系统性地压缩**我们的差值，
这与审计里"取全程最优"那条同向。

**它不是什么**：论文的 x_{m,n} 是**跨三个任务归一化到 [0,1]** 之后的分；
我们只有 flocking 一个任务、且未归一化。所以两张图**只能比方向与相对间隔，
不能比绝对高度**。

用法::

    python scripts/aggregate_metrics.py /tmp/runs/eval_greedy.json
"""

from __future__ import annotations

import json
import math
import sys

import numpy as np

METRICS = ("median", "iqm", "mean")


def iqm(x: np.ndarray) -> float:
    lo, hi = np.quantile(x, [0.25, 0.75])
    mid = x[(x >= lo) & (x <= hi)]
    return float(mid.mean()) if mid.size else float(np.median(x))


def stats(x: np.ndarray) -> dict:
    return {"median": float(np.median(x)), "iqm": iqm(x), "mean": float(x.mean())}


def bootstrap(x: np.ndarray, n_boot: int = 20000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    draws = {m: np.empty(n_boot) for m in METRICS}
    for i in range(n_boot):
        s = stats(x[idx[i]])
        for m in METRICS:
            draws[m][i] = s[m]
    return {m: (float(np.quantile(draws[m], 0.025)),
                float(np.quantile(draws[m], 0.975))) for m in METRICS}


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/runs/eval_greedy.json"
    data = json.load(open(path))
    runs = data["runs"]
    arms = {}
    for name, r in runs.items():
        arms.setdefault(r["variant"], []).append(np.mean(r["returns"]))

    print(f"来源 {path}   动作选择 {data['action_mode']}   "
          f"每 seed {len(runs[next(iter(runs))]['returns'])} 局   "
          f"每臂 {len(next(iter(arms.values())))} 个 seed\n")

    out = {}
    for arm in ("idreamer", "codreamer"):
        x = np.array(sorted(arms[arm]))
        s = stats(x)
        ci = bootstrap(x)
        out[arm] = (s, ci, x)
        print(f"{arm}   n={len(x)}  seed 均值 = {np.round(x, 3).tolist()}")
        for m in METRICS:
            print(f"   {m:<7} {s[m]:+.4f}   95% CI [{ci[m][0]:+.4f}, {ci[m][1]:+.4f}]")
        print()

    print("差值（idreamer − codreamer，正 = CoDreamer 更差）")
    for m in METRICS:
        d = out["idreamer"][0][m] - out["codreamer"][0][m]
        # 两组各自 bootstrap 太慢；直接用均值差的 CI（独立样本近似）
        a = out["idreamer"][2]
        b = out["codreamer"][2]
        rng = np.random.default_rng(1)
        ds = np.empty(20000)
        for i in range(20000):
            ai = rng.integers(0, len(a), len(a))
            bi = rng.integers(0, len(b), len(b))
            ds[i] = stats(a[ai])[m] - stats(b[bi])[m]
        lo, hi = np.quantile(ds, [0.025, 0.975])
        print(f"   {m:<7} {d:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]   "
              f"{'CI 不含 0' if lo * hi > 0 else 'CI 含 0'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
