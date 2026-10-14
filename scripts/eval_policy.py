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

================================================================================
不确定度按【队伍】聚合，不是按 agent
================================================================================
同一队的 N 个 agent 共享环境状态与邻接关系，连奖励都是同一份团队奖励。
它们的回报**不是独立样本** —— 直接把 N×T 个 agent 局当样本算置信区间，
区间会被算得过窄（相当于把同一个观测重复计了 N 遍）。

所以逐局分数先按队伍求均值，再在**队伍**上算均值的 95% 置信区间。
对外报告的口径（`mean` / `ci`）一律是队伍级；`agent_mean` 等保留原来的
agent 级口径，供与旧结果对照。

================================================================================
配对比较的前提：种子必须一致
================================================================================
配对差值只有在于两个 checkpoint 跑的是**同一批局**时才成立。局序列由
`config.seed + id` 决定，所以这里会**断言**所有 checkpoint 的 seed 相同；
不同就直接报错，而不是默默给出一个无效的配对结论。

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

    # [CoDreamer] 按队伍聚合。
    #
    # 同一队的 N 个 agent 共享环境状态与邻接关系，连奖励都是同一份团队奖励，
    # 所以它们是**相关样本**。把 N×T 个 agent 局当独立样本算置信区间，相当于
    # 把同一个观测重复计了 N 遍 —— 区间会窄到 sqrt(N) 倍。
    team_ids = np.asarray(getattr(logger, "eval_team_ids", []))
    if team_ids.size == scores.size and scores.size > 0:
        team_order = list(dict.fromkeys(team_ids.tolist()))
        team_scores = np.array(
            [scores[team_ids == t].mean() for t in team_order], dtype=np.float64
        )
    else:
        # 拿不到队 id（旧版 rollout 或非多智能体环境）就退化成 agent 级。
        # 明确标出来，避免下游把窄区间当真。
        team_order = None
        team_scores = scores

    def stats(a):
        a = np.asarray(a, dtype=np.float64)
        return {
            "n": int(a.size),
            "mean": float(a.mean()),
            "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "median": float(np.median(a)),
            "q1": float(np.percentile(a, 25)),
            "q3": float(np.percentile(a, 75)),
            "min": float(a.min()),
            "max": float(a.max()),
            # 均值的 95% 置信区间（正态近似）
            "ci": (
                1.96 * float(a.std(ddof=1)) / np.sqrt(a.size) if a.size > 1 else 0.0
            ),
        }

    out = stats(team_scores)
    out["unit"] = "team" if team_order is not None else "agent"
    out["n_teams"] = len(team_order) if team_order is not None else int(scores.size)
    # 逐 agent 的原始分数：箱线图与散点图用（看分布形状）。
    # 统计量（mean / ci / median）一律用上面的队伍级分数。
    out["scores"] = scores
    out["team_scores"] = team_scores
    if team_order is not None:
        out["team_ids"] = team_order
        # 保留 agent 级口径供与旧结果对照
        agent_stats = stats(scores)
        out["agent_mean"] = agent_stats["mean"]
        out["agent_ci"] = agent_stats["ci"]
    return out


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
            cfg.task, cfg.comm_variant,
            cfg.gnn_layers, cfg.gnn_head_layers, cfg.gnn_behavior_layers,
            cfg.gnn_heads,
            cfg.seg_num_agents, cfg.seg_graph_obs, cfg.seg_num_actions,
            cfg.seg_horizon, cfg.dyn_stoch, cfg.dyn_deter, cfg.units,
        )

    results = []
    seeds = set()
    for name, path in checkpoints:
        ckpt = load_checkpoint(path, args.device)
        cfg = argparse.Namespace(**ckpt["config"])
        cfg.device = args.device
        cfg.eval_episode_num = args.episodes
        seeds.add(cfg.seed)

        key = config_key(cfg)
        if key != active_key:
            if agent is not None:
                print(
                    f"  -- config 变化（{cfg.comm_variant}: rssm={cfg.gnn_layers} "
                    f"head={cfg.gnn_head_layers} behavior={cfg.gnn_behavior_layers}）"
                    f"-> 重建模型"
                )
            agent = build_agent(cfg, args.device)
            n_params = sum(p.numel() for p in agent.parameters())
            active_key = key
            print(
                f"  模型     {cfg.comm_variant}  rssm_gnn={cfg.gnn_layers}  "
                f"head_gnn={cfg.gnn_head_layers}  "
                f"behavior_gnn={cfg.gnn_behavior_layers}  "
                f"参数量 {n_params:,}（含 value 的 EMA 副本 _slow_value，"
                f"所以不能直接与论文表对照）"
            )

        # [CoDreamer] ★ 严格加载 ★
        #
        # 这里曾经写的是：
        #     missing, unexpected = agent.load_state_dict(..., strict=False)
        #     real_missing = [k for k in missing if not k.startswith("_")]
        # Dreamer 的子模块叫 _wm / _task_behavior，**每一个真参数的键都以 _ 开头**，
        # 于是这个过滤器把全部缺失键都筛掉了 —— 校验恒为空，永远不报错。
        # 结构不匹配时会默默用随机权重评测。现在任何 missing / unexpected 都直接失败。
        incompatible = agent.load_state_dict(ckpt["agent_state_dict"], strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise SystemExit(
                f"{path} 的权重与当前模型结构不匹配。\n"
                f"  缺 {len(incompatible.missing_keys)} 个键: "
                f"{incompatible.missing_keys[:6]}\n"
                f"  多 {len(incompatible.unexpected_keys)} 个键: "
                f"{incompatible.unexpected_keys[:6]}\n"
                f"  模型的 comm_variant={cfg.comm_variant} "
                f"gnn=(rssm {cfg.gnn_layers}, head {cfg.gnn_head_layers}, "
                f"behavior {cfg.gnn_behavior_layers})。\n"
                f"  拒绝用随机权重继续评测。"
            )
        agent.eval()
        r = evaluate(agent, cfg, args.episodes, args.device)
        r["name"] = name
        r["path"] = str(path)
        results.append(r)
        extra = ""
        if "agent_mean" in r:
            extra = f"  (agent 级口径 {r['agent_mean']:+.4f} ± {r['agent_ci']:.4f})"
        print(
            f"  {name:<20s} 队={r['n_teams']:<4d} agent={r['n']:<5d} "
            f"均值 {r['mean']:+.4f} ± {r['ci']:.4f}  "
            f"中位数 {r['median']:+.4f}  IQR [{r['q1']:+.3f}, {r['q3']:+.3f}]  "
            f"[{r['min']:+.3f}, {r['max']:+.3f}]{extra}"
        )

    # [CoDreamer] 配对比较的前提是同一批局。局序列由 config.seed + id 决定，
    # seed 不同就不是同一批局，配对差值无意义 —— 直接拒绝，不给无效结论。
    if len(seeds) > 1:
        raise SystemExit(
            f"各 checkpoint 的训练 seed 不同（{sorted(seeds)}），跑的不是同一批局，"
            f"配对比较无效。请只比较同一次训练的不同 checkpoint。"
        )

    print()

    if len(results) > 1:
        print()
        print("  两两对比（同一批局，配对差值；按【队伍】配对，不是按 agent）：")
        base = results[0]
        for other in results[1:]:
            d = other["team_scores"] - base["team_scores"]
            # 配对：同一批局，所以直接看逐队差值
            se = d.std(ddof=1) / np.sqrt(d.size) if d.size > 1 else 0.0
            print(
                f"    {other['name']} − {base['name']}: "
                f"均值差 {d.mean():+.4f} ± {1.96 * se:.4f}  "
                f"（逐队看：{int((d > 0).sum())}/{d.size} 队更好）"
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
                        "unit": r["unit"],
                        "n_teams": r["n_teams"],
                        "scores": r["scores"].tolist(),
                        "team_scores": r["team_scores"].tolist(),
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
    ax.set_ylabel("episode return (0 = optimal)")
    ax.set_title("Deployment evaluation: same fixed episodes for every checkpoint\n"
                 "(box = IQR, diamond = mean, dots = individual episodes)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"\n  图已保存: {out}")


if __name__ == "__main__":
    raise SystemExit(main())
