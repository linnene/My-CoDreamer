"""独立部署评测：加载 checkpoint，在全新环境里跑大样本，只看不学。

================================================================================
它回答什么问题
================================================================================
训练时的 `eval_return` 只给一个均值，而且插在训练流程里。要做「策略部署出去
到底什么水平」这类判断，需要的是**分布**：中位数、四分位距、逐局分数。

更重要的是能**在同一批固定的局上对比多个 checkpoint** —— 比如
「环境切断那一刻的策略」vs「仅靠世界模型想象继续训练之后的策略」。
这两者是整个 Phase 2 实验的核心对比，而训练日志里它们分别落在
`cutoff.pt` 和 `latest.pt` 两个文件上。

================================================================================
为什么要每个 checkpoint 重建环境
================================================================================
环境（`SequentialEstimateGameEnv`）在每次 `reset()` 时从自身的 RNG 抽一个新种子。
如果复用同一批 env 实例依次评测多个 checkpoint，第二个 checkpoint 拿到的就是
**另一批局**，两者的分数不可比。

所以每个 checkpoint 都用 `dreamer.make_env` 重建全新实例。由于种子是
`config.seed + id`，重建出来的局序列**完全相同**，对比才是公平的。

================================================================================
口径与训练时一致
================================================================================
仍然走 `tools_marl.simulate_marl(is_eval=True, episodes=...)` —— 与训练中的评测
**同一条代码路径**，所以这里的数字可以直接和 `metrics.jsonl` 里的 `eval_return`
对齐。`--episodes` 的单位与 `eval_episode_num` 相同，是 **agent 局数**：
一队 N 个 agent 同时结束算 N 局。

用法::

    python scripts/eval_policy.py \\
        --checkpoint "切断时=~/runs/idream_phase2/cutoff.pt" \\
        --checkpoint "想象训练后=~/runs/idream_phase2/latest.pt" \\
        --episodes 400 --out figures/deploy.png
"""

from __future__ import annotations

import argparse
import functools
import pathlib
import sys
import tempfile

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
D3T = REPO / "third_party" / "dreamerv3-torch"
sys.path.insert(0, str(D3T))

import torch  # noqa: E402

import dreamer as dreamer_mod  # noqa: E402
import tools  # noqa: E402
import tools_marl  # noqa: E402


class CaptureLogger:
    """把 simulate_marl 写出的标量收在内存里，不落盘。"""

    def __init__(self):
        self.scalars: dict[str, list[float]] = {}
        self.step = 0
        self.eval_scores: list[float] = []
        self.eval_lengths: list[int] = []

    def scalar(self, key, value):
        self.scalars.setdefault(key, []).append(float(value))

    def write(self, *a, **k):
        pass

    def video(self, *a, **k):
        pass


class _NoDataset:
    """Dreamer 需要 dataset，但评测只做前向（training=False），永远取不到它。"""

    def __iter__(self):
        raise RuntimeError("eval_policy 不应触发训练：dataset 被访问了")


def load_checkpoint(path: pathlib.Path, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "config" not in ckpt:
        raise SystemExit(
            f"{path} 里没有 'config'。\n"
            "这是旧格式的 checkpoint（早于把 config 一并保存的那次改动）。\n"
            "没有它就推不出 gnn_layers / 环境规格等关键超参，无法可靠重建模型。\n"
            "请用新代码重新训练一次。"
        )
    return ckpt


def build_agent(config, device: str):
    """按 config 重建环境与 agent（结构与训练时完全一致）。"""
    config.device = device
    envs = [dreamer_mod.make_env(config, "eval", i) for i in range(config.envs)]
    acts = envs[0].action_space
    config.num_agents = getattr(envs[0], "num_agents", 1)
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
    agent = dreamer_mod.Dreamer(
        envs[0].observation_space,
        envs[0].action_space,
        config,
        CaptureLogger(),
        _NoDataset(),
    ).to(device)
    agent.requires_grad_(requires_grad=False)
    for e in envs:
        try:
            e.close()
        except Exception:
            pass
    return agent


def evaluate(agent, config, episodes: int, device: str) -> dict:
    """在一个**全新**环境实例上跑 episodes 局，返回分数分布。"""
    envs = [dreamer_mod.make_env(config, "eval", i) for i in range(config.envs)]
    logger = CaptureLogger()
    cache: dict = {}
    policy = functools.partial(agent, training=False)
    with tempfile.TemporaryDirectory() as tmp:
        tools_marl.simulate_marl(
            policy,
            envs,
            cache,
            tmp,
            logger,
            is_eval=True,
            limit=config.dataset_size,
            episodes=episodes,
        )
    for e in envs:
        try:
            e.close()
        except Exception:
            pass

    scores = np.asarray(logger.eval_scores, dtype=np.float64)
    if scores.size == 0:
        raise SystemExit(
            "没有收到任何评测分数 —— 检查 episodes 是否 >= 一个队伍里的 agent 数"
        )
    return {
        "scores": scores,
        "n": int(scores.size),
        "mean": float(scores.mean()),
        "std": float(scores.std(ddof=1)) if scores.size > 1 else 0.0,
        "median": float(np.median(scores)),
        "q1": float(np.percentile(scores, 25)),
        "q3": float(np.percentile(scores, 75)),
        "min": float(scores.min()),
        "max": float(scores.max()),
        # 均值的 95% 置信区间（正态近似）
        "ci": (
            1.96 * float(scores.std(ddof=1)) / np.sqrt(scores.size)
            if scores.size > 1
            else 0.0
        ),
    }


def parse_checkpoints(items: list[str]) -> list[tuple[str, pathlib.Path]]:
    out = []
    for item in items:
        if "=" in item:
            name, _, path = item.partition("=")
        else:
            path = item
            name = pathlib.Path(item).parent.name + "/" + pathlib.Path(item).stem
        p = pathlib.Path(path).expanduser()
        if not p.exists():
            raise SystemExit(f"checkpoint 不存在: {p}")
        out.append((name, p))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="独立部署评测（加载 checkpoint，只看不学）"
    )
    ap.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="[名字=]路径",
        help="可重复；每个 checkpoint 都在同一批固定的局上评测",
    )
    ap.add_argument("--episodes", type=int, default=200, help="agent 局数（同 eval_episode_num）")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None, help="箱线图输出路径")
    ap.add_argument(
        "--out-json", default=None, help="逐局分数的 JSON，供 plot_phase2.py 用"
    )
    args = ap.parse_args()

    checkpoints = parse_checkpoints(args.checkpoint)

    print("独立部署评测")
    print("=" * 78)

    first = load_checkpoint(checkpoints[0][1], args.device)
    config = argparse.Namespace(**first["config"])
    if config.device != args.device:
        print(f"  （checkpoint 里记的是 {config.device}，按 --device {args.device} 跑）")
    config.device = args.device
    # 评测不训练，也不存盘
    config.eval_episode_num = args.episodes

    print(
        f"  环境     task={config.task}  agents={config.seg_num_agents}  "
        f"horizon={config.seg_horizon}  graph_obs={config.seg_graph_obs}"
    )
    print(f"  评测     {args.episodes} agent 局 / checkpoint，环境每次重建（同一批局）")
    print()

    agent = None
    active_key = None
    n_params = 0

    def config_key(cfg):
        """模型结构相关的超参。变了就必须重建模型。"""
        return (
            cfg.task, cfg.gnn_layers, cfg.gnn_head_layers, cfg.gnn_heads,
            cfg.seg_num_agents, cfg.seg_graph_obs, cfg.seg_num_actions,
            cfg.seg_horizon, cfg.dyn_stoch, cfg.dyn_deter, cfg.units,
        )

    results = []
    for name, path in checkpoints:
        ckpt = load_checkpoint(path, args.device)
        cfg = argparse.Namespace(**ckpt["config"])
        cfg.device = args.device
        cfg.eval_episode_num = args.episodes

        key = config_key(cfg)
        if key != active_key:
            if agent is not None:
                print(
                    f"  -- config 变化（gnn={cfg.gnn_layers} "
                    f"head={cfg.gnn_head_layers}）-> 重建模型"
                )
            agent = build_agent(cfg, args.device)
            n_params = sum(p.numel() for p in agent.parameters())
            active_key = key
            print(
                f"  模型     rssm_gnn={cfg.gnn_layers}  head_gnn={cfg.gnn_head_layers}  "
                f"参数量 {n_params:,}"
            )

        missing, unexpected = agent.load_state_dict(ckpt["agent_state_dict"], strict=False)
        real_missing = [k for k in missing if not k.startswith("_")]
        if real_missing:
            raise SystemExit(
                f"{path} 的权重与当前 config 不匹配，缺 {len(real_missing)} 个键："
                f"{real_missing[:5]}..."
            )
        if unexpected:
            print(f"  [警告] {name}: 多出 {len(unexpected)} 个键被忽略")
        agent.eval()
        r = evaluate(agent, cfg, args.episodes, args.device)
        r["name"] = name
        r["path"] = str(path)
        results.append(r)
        print(
            f"  {name:<20s} n={r['n']:<5d} 均值 {r['mean']:+.4f} ± {r['ci']:.4f}  "
            f"中位数 {r['median']:+.4f}  IQR [{r['q1']:+.3f}, {r['q3']:+.3f}]  "
            f"[{r['min']:+.3f}, {r['max']:+.3f}]"
        )

    print()

    if len(results) > 1:
        print()
        print("  两两对比（同一批局，配对差值）：")
        base = results[0]
        for other in results[1:]:
            d = other["scores"] - base["scores"]
            # 配对：同一批局，所以直接看逐局差值
            se = d.std(ddof=1) / np.sqrt(d.size) if d.size > 1 else 0.0
            print(
                f"    {other['name']} − {base['name']}: "
                f"均值差 {d.mean():+.4f} ± {1.96 * se:.4f}  "
                f"（逐局看：{int((d > 0).sum())}/{d.size} 局更好）"
            )

    if args.out:
        plot(results, pathlib.Path(args.out).expanduser())
    if args.out_json:
        import json

        p = pathlib.Path(args.out_json).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {
                    r["name"]: {
                        "path": r["path"],
                        "scores": r["scores"].tolist(),
                        "mean": r["mean"],
                        "median": r["median"],
                        "q1": r["q1"],
                        "q3": r["q3"],
                        "ci": r["ci"],
                    }
                    for r in results
                },
                ensure_ascii=False,
                indent=1,
            )
        )
        print(f"  分数已保存: {p}")
    return 0


def plot(results, out: pathlib.Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(5.0, 1.6 * len(results) + 3), 4.6))
    data = [r["scores"] for r in results]
    labels = [r["name"] for r in results]
    bp = ax.boxplot(
        data, labels=labels, widths=0.5, patch_artist=True, showfliers=False,
        medianprops=dict(color="black", linewidth=1.6),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("#c9d9f0")
        patch.set_alpha(0.8)
    # 叠上逐局散点，让人看到分布形状而不是只有箱体
    rng = np.random.default_rng(0)
    for i, (scores, color) in enumerate(zip(data, ["#3b6ea5", "#c1554b", "#5a9367", "#8a6fb0"]), start=1):
        x = i + rng.uniform(-0.16, 0.16, size=scores.size)
        ax.scatter(x, scores, s=6, alpha=0.35, color=color, linewidths=0, zorder=3)
        ax.scatter([i], [scores.mean()], marker="D", s=34, color=color,
                   edgecolor="white", linewidth=0.8, zorder=4)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel("episode return（0 = 满分）")
    ax.set_title("独立部署评测：同一批局上的策略表现\n"
                 "（箱体 = 四分位距，菱形 = 均值，散点 = 逐局）")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"\n  图已保存: {out}")


if __name__ == "__main__":
    raise SystemExit(main())
