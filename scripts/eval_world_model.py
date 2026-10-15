"""Held-out 上的**多步 open-loop** 预测误差 —— 判断世界模型"想没想对"。

================================================================================
为什么不能用 replay 上的训练损失
================================================================================
`plot_phase2.py` 的第三张图画的是 `metrics.jsonl` 里的 `reward_loss` / `state_loss`，
并据此判断"策略有没有在利用模型误差"。这个推论**不成立**，因为：

  1. 那是**训练损失** —— 同一份 replay 被反复优化，损失下降可以只是记住了它；
  2. 它用的是 **posterior** 隐状态，而 posterior 每一步都**看过当步观测**
     （见 WorldModel._train：`feat = get_feat(post)`）。
     所以它衡量的是"给定当前帧，重建当前帧"，**不是**对未来开环预测的能力。

真正的问题——"世界模型想象出来的轨迹准不准"——只有 open-loop 能回答：

    用 posterior 吃 context 步（允许看观测）→ 取某一时刻的状态
    → 之后**只喂真实动作、不再看任何观测**，让 prior 往前滚 K 步
    → 把第 k 步的预测与真值比

而且必须用**没参与训练**的数据（`evaldir` 的 rollout 与 `traindir` 是两批，
天然 held-out）。

图也用**冻结的起点图**（论文要求想象阶段复用起点图），所以这个误差反映的正是
想象轨迹所处的条件。

================================================================================
四个量
================================================================================
    reward_mae    预测奖励 vs 真实奖励（世界模型最直接有用的东西）
    cont_mae      预测继续概率 vs 真实（1 - is_terminal）
    kl            rollout 出来的 prior 与真实 posterior 的 KL —— 隐状态漂移
    state_mse    用 prior 特征重建观测的 MSE（decoder）—— 状态漂移的另一面

用法::

    python scripts/eval_world_model.py \
        --checkpoint "IDreamer=/tmp/runs/idreamer_s0_base/latest.pt" \
        --checkpoint "CoDreamer=/tmp/runs/codreamer_s0_base/latest.pt" \
        --horizons 1 2 4 8 16
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


def load_checkpoint(path: pathlib.Path, device: str) -> dict:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    for key in ("agent_state_dict", "config"):
        if key not in ckpt:
            raise SystemExit(f"{path} 里没有 {key}; 是不是旧格式的 checkpoint？")
    return ckpt


def build_agent(config, obs_space, act_space):
    agent = dreamer_mod.Dreamer(obs_space, act_space, config, _SilentLogger(), None)
    return agent.to(config.device)


class _SilentLogger:
    """simulate_marl 需要一个 logger，但这里不需要写盘。"""

    step = 0

    def scalar(self, *a, **k):
        pass

    def write(self, *a, **k):
        pass


class _UniformPolicy:
    """与任何模型无关的均匀随机策略 —— 用于采集**共享**轨迹。

    它不依赖 checkpoint，所以两个世界模型评在同一批数据上时，谁都没有主场优势。
    代价是这些状态离训练分布更远（见 heldout_batch 的说明）。
    """

    def __init__(self, num_actions: int, num_agents: int, seed: int = 0):
        self.num_actions = num_actions
        self.num_agents = num_agents
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs, done, state):
        batch = obs["state"].shape[0]
        idx = self.rng.integers(0, self.num_actions, size=batch)
        action = np.eye(self.num_actions, dtype=np.float32)[idx]
        return {"action": action, "logprob": np.zeros(batch, np.float32)}, None


def heldout_batch(
    config,
    agent=None,
    repeats: int = 20,
    teams: int = 40,
    policy=None,
    out_dir=None,
):
    """跑一批全新的 episode 作为 held-out，返回 repeats 个图批次。

    采集者二选一：
      - `agent`：用**策略自己**跑（默认）。数据在分布内，且从未参与任何梯度更新。
      - `policy`：用指定的、与模型无关的策略跑（配合 `out_dir` 复用同一批数据）。

    为什么必须能指定 `policy`（加于 2026-10-15）：
    两个 checkpoint 各自 on-policy 采集时，**两个世界模型根本不在同一批数据上比**。
    此时"CoDreamer 的 state_mse 差 4 倍"里同时含着两样东西——模型能力的差异，
    和两个策略访问的状态分布不同。把两者分开的唯一办法，是让它们在**同一批**
    (观测, 动作, 真值未来) 上各算一次。

    两条路都要留着并分别标注：共享轨迹上没有主场优势，但状态离训练分布更远；
    on-policy 分布内，却只回答"各自访问分布上的误差"。混着用会得到两个都不能
    回答的数。

    为什么不能用 `evaldir`：`tools_marl.simulate_marl` 在 `is_eval=True` 时
    **不落盘**（那个分支里没有 save_episodes），所以 evaldir 是空的。

    为什么不能用训练 replay：它的每一帧都被梯度更新看过，拿它算预测误差等于
    在训练集上报数。
    """

    def rollout(directory):
        envs = [dreamer_mod.make_env(config, "eval", i) for i in range(config.envs)]
        try:
            tools_marl.simulate_marl(
                policy if policy is not None
                else functools.partial(agent, training=False),
                envs,
                {},
                directory,
                _SilentLogger(),
                is_eval=False,      # False 才会落盘，之后按与训练**完全相同**的格式读回来
                limit=10**6,
                episodes=teams * config.num_agents,
            )
        finally:
            for e in envs:
                try:
                    e.close()
                except Exception:
                    pass

    if out_dir is not None:
        # 共享轨迹：目录里已有 npz 就复用，不重采
        path = pathlib.Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        if not list(path.glob("*.npz")):
            rollout(path)
        eps = tools.load_episodes(path, limit=10**6)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            rollout(tmp)
            eps = tools.load_episodes(tmp, limit=10**6)

    if not eps:
        raise SystemExit("held-out rollout 没写出任何 episode")
    dataset = tools_marl.make_graph_dataset(eps, config)
    return [next(dataset) for _ in range(repeats)]


def openloop(wm, data, config, horizons, max_horizon=None):
    """薄包装 —— 唯一实现在 `models.openloop_error`，训练日志用的是同一份。

    这里曾经有一份拷贝，那等于让训练时的口径和事后分析的口径各自演化。
    """
    return models.openloop_error(wm, data, horizons)


def plot(results: dict, out: pathlib.Path, groups: list[str] | None = None) -> None:
    """误差随 horizon 的曲线。名字里的 -sN 后缀用来分组（IDreamer / CoDreamer）。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.parent.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("cont_mae", "continuation prediction error (MAE)"),
        ("kl", "open-loop latent drift (KL prior vs posterior)"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(6.0 * len(metrics), 4.4))
    colors = {"IDreamer": "#3b6ea5", "CoDreamer": "#c1554b"}
    # 无 CJK 字体，一律英文标签（否则渲染成方块）
    for ax, (key, title) in zip(axes, metrics):
        for name, rows in results.items():
            base = name.split("-")[0]
            hs = sorted(int(h) for h in rows)
            ys = [rows[str(h) if str(h) in rows else h][key] for h in hs]
            ax.plot(hs, ys, marker="o", ms=4, lw=1.4, alpha=0.85,
                    color=colors.get(base, None), label=name)
        ax.set_xlabel("open-loop horizon (steps without observations)")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        if key == "cont_mae":
            ax.legend(fontsize=7, ncol=1)
    fig.suptitle(
        "held-out open-loop prediction: no observations after the start, frozen graph",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"  图已保存: {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", action="append", required=True, help="名称=路径")
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--batches", type=int, default=20, help="重复采样几批")
    ap.add_argument("--teams", type=int, default=40, help="held-out rollout 采几支队")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--rollout", choices=["self", "random"], default="self",
        help="self=各自 on-policy 采集（默认，分布内但两个模型不在同一批数据上）；"
             "random=用均匀随机策略采一批共享轨迹，所有 checkpoint 评在同一批数据上",
    )
    ap.add_argument("--rollout-dir", default=None, help="共享轨迹的落地目录（复用/复现用）")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out", default=None, help="误差-水平线图（png）")
    args = ap.parse_args()

    ckpts = []
    for item in args.checkpoint:
        name, _, path = item.partition("=")
        ckpts.append((name or pathlib.Path(path).stem, pathlib.Path(path).expanduser()))

    horizons = sorted(set(args.horizons))
    H = max(horizons)

    # 共享轨迹目录：存在即复用，所以所有 checkpoint 评的是**同一批** npz。
    #
    # 三种跑法，回答三个不同的问题：
    #   --rollout random                    一批中性轨迹，谁都没有主场优势
    #   --rollout self --rollout-dir D      由**第一个** checkpoint 的策略采集，其余复用
    #                                       → 把采集者写在前面，即可让两个模型都评在
    #                                         某一方的 on-policy 分布上（主场优势测试）
    #   --rollout self（不给目录）           各自 on-policy —— 唯一的不可跨模型比的跑法
    shared_dir = None
    if args.rollout == "random":
        shared_dir = args.rollout_dir or tempfile.mkdtemp(prefix="shared_heldout_")
        print(
            f"\n共享轨迹模式：均匀随机策略采集，所有 checkpoint 评在同一批数据上\n"
            f"  目录 {shared_dir}\n"
            f"  注意：这些状态离各自的训练分布更远，与 on-policy 的数字**不可混用**"
        )
    elif args.rollout_dir:
        shared_dir = args.rollout_dir
        print(
            f"\n共享轨迹模式：由**第一个** checkpoint（{ckpts[0][0]}）的策略采集，"
            f"其余 checkpoint 评在同一批数据上\n"
            f"  目录 {shared_dir}\n"
            f"  这一批对 {ckpts[0][0]} 是 on-policy 分布，对其余模型是 off-policy —— "
            f"正是用来检验差距里有多少来自分布"
        )
    else:
        print("\non-policy 模式：每个 checkpoint 用**自己的策略**采集轨迹")
        print("  注意：两个模型因此不在同一批数据上 —— 跨模型的差距里含着分布差异")

    results = {}
    for name, path in ckpts:
        ckpt = load_checkpoint(path, args.device)
        cfg = argparse.Namespace(**ckpt["config"])
        cfg.device = args.device
        print(
            f"\n=== {name}  ({path})\n"
            f"    {cfg.comm_variant}  gnn: rssm={cfg.gnn_layers} "
            f"head={cfg.gnn_head_layers} behavior={cfg.gnn_behavior_layers}"
        )
        with tempfile.TemporaryDirectory() as tmp:
            env = dreamer_mod.make_env(cfg, "eval", 0)
            cfg.num_agents = getattr(env, "num_agents", 1)
            acts = env.action_space
            cfg.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
            cfg.evaldir = cfg.evaldir or str(pathlib.Path(tmp) / "eval_eps")
            agent = build_agent(cfg, env.observation_space, env.action_space)
            inc = agent.load_state_dict(ckpt["agent_state_dict"], strict=False)
            if inc.missing_keys or inc.unexpected_keys:
                raise SystemExit(
                    f"{path} 结构不匹配：缺 {len(inc.missing_keys)} 多 "
                    f"{len(inc.unexpected_keys)} —— 拒绝用随机权重评测"
                )
            agent.eval()
            wm = agent._wm

            if shared_dir is None:
                batches = heldout_batch(cfg, agent, args.batches, args.teams)
            else:
                # random 模式才换采集者；否则用本 checkpoint 自己的策略
                # （目录是持久的，所以实际只有第一个会真正采集）
                collector = (
                    _UniformPolicy(
                        cfg.num_actions, cfg.num_agents, seed=getattr(cfg, "seed", 0)
                    )
                    if args.rollout == "random"
                    else None
                )
                batches = heldout_batch(
                    cfg, agent, args.batches, args.teams,
                    policy=collector, out_dir=shared_dir,
                )
            acc: dict[int, list[dict]] = {h: [] for h in horizons}
            for data in batches:
                got = openloop(wm, data, cfg, set(horizons), H)
                for h in horizons:
                    acc[h].append(got[h])
        env.close()

        print(f"    {'horizon':>8}  {'reward_mae':>11}  {'cont_mae':>9}  {'kl':>8}  {'state_mse':>10}")
        rows = {}
        for h in horizons:
            m = {
                key: float(np.mean([b[key] for b in acc[h]]))
                for key in ("reward_mae", "cont_mae", "kl", "state_mse")
            }
            rows[h] = m
            print(
                f"    {h:>8}  {m['reward_mae']:>11.4f}  {m['cont_mae']:>9.4f}  "
                f"{m['kl']:>8.4f}  {m['state_mse']:>10.4f}"
            )
        results[name] = rows

    if args.out_json:
        import json

        p = pathlib.Path(args.out_json).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(results, ensure_ascii=False, indent=1))
        print(f"\n  已保存 {p}")
    if args.out:
        plot(results, pathlib.Path(args.out).expanduser())

    print(
        "\n  说明：全部是 held-out 上的**开环**误差 —— 只喂真实动作、不再看观测。"
        "\n  曲线随 horizon 上升多少，才是世界模型\"想得准不准\"。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
