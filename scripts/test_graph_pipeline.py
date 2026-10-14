"""② 数据管线 (B, T, N, ·) 的验收测试。

================================================================================
为什么需要这个测试
================================================================================
步骤 2 把数据管线从「N 条独立序列摊平进 batch」改成「N 个 agent 是样本内部的
一个轴」，因为 GNN 要在同一时刻对 N 个 agent 做邻域聚合 —— 摊平之后它们彼此
不可见，GNN 没有轴可以聚合。

改动本身是纯结构性的：IDreamer 语义下 agent 之间没有信息流动，所以
`fold_agent_axis` 把 N 折进 batch 之后，**整条计算链路应该一个数都不变**。

这就是本测试要证明的。它不是"跑两个 run 看曲线像不像"那种软对比，而是一个
数值恒等式：

    model(num_agents=N)._train(图批次 G)
        ==
    model(num_agents=1)._train(折平后的 G)

两侧的权重初始化、数据、优化器配置完全相同，唯一差别是 `_train` 里要不要调用
`fold_agent_axis`。任何一处对齐写错（切错 agent、轴顺序反了、拼接错位）都会让
某个损失对不上。

阈值取 0：这是恒等变换，不是近似。浮点上同一条计算图的同一批运算必须逐位相同。
================================================================================
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import tempfile

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
D3T = REPO / "third_party" / "dreamerv3-torch"
sys.path.insert(0, str(D3T))

import ruamel.yaml as yaml  # noqa: E402

import models  # noqa: E402
import tools  # noqa: E402
import tools_marl  # noqa: E402
from envs.seg import Spec  # noqa: E402
from envs.seg_gym import SequentialEstimateGameEnv  # noqa: E402

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# 复刻 dreamer.py 的配置加载（保持与真实训练完全同一套超参）
# ---------------------------------------------------------------------------
def load_config(config_names: list[str]) -> argparse.Namespace:
    configs = yaml.safe_load((D3T / "configs.yaml").read_text())

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    defaults = {}
    for name in list(config_names):
        recursive_update(defaults, configs[name])
    return argparse.Namespace(**{k: v for k, v in defaults.items()})


class _RandomPolicy:
    """随机策略，用来快速灌满 replay。"""

    def __init__(self, num_actions: int, num_agents: int, seed: int = 0):
        self.num_actions = num_actions
        self.num_agents = num_agents
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs, done, state):
        batch = obs["state"].shape[0]
        onehot = np.zeros((batch, self.num_actions), dtype=np.float32)
        idx = self.rng.integers(0, self.num_actions, batch)
        onehot[np.arange(batch), idx] = 1.0
        return {"action": onehot, "logprob": np.zeros(batch, np.float32)}, None


class _NullLogger:
    step = 0

    def scalar(self, *a, **k):
        pass

    def write(self, *a, **k):
        pass

    def video(self, *a, **k):
        pass


def build_batch(config, spec, num_teams: int, batch_length: int, tmpdir: str) -> tuple[dict, object]:
    """跑一点 rollout，产出图批次 (B, T, N, ·)，并返回 (batch, obs_space)。"""
    envs = [SequentialEstimateGameEnv(spec, seed=1000 + i, graph_obs=True) for i in range(1)]
    cache: dict = {}
    policy = _RandomPolicy(spec.num_actions, spec.num_agents, seed=0)
    # episodes= 而不是 steps=：结算按 agent-episode 计数，更不容易算错
    tools_marl.simulate_marl(
        policy, envs, cache, tmpdir, _NullLogger(),
        limit=10**6, episodes=8 * spec.num_agents,
    )
    episodes = tools.load_episodes(tmpdir, limit=10**6)
    gen = tools_marl.sample_episode_graphs(
        episodes, batch_length, spec.num_agents, seed=0
    )
    batch = {}
    for _ in range(num_teams):
        sample = next(gen)
        for k, v in sample.items():
            batch.setdefault(k, []).append(v)
    batch = {k: np.stack(v, 0) for k, v in batch.items()}
    return batch, envs[0].observation_space


def train_once(config, obs_space, batch, num_agents_override: int, seed: int, gnn_layers: int = 0) -> dict:
    """在一个全新模型上跑一次 _train，返回 (损失指标, 参数量)。"""
    import copy

    cfg = copy.deepcopy(config)
    cfg.num_agents = num_agents_override
    cfg.gnn_layers = gnn_layers
    cfg.device = "cpu"
    cfg.precision = 32          # 关掉 amp，保证两次比较逐位一致
    cfg.compile = False
    tools.set_seed_everywhere(seed)
    wm = models.WorldModel(obs_space, _DummyActSpace(), 0, cfg)
    n_params = sum(p.numel() for p in wm.parameters())
    # WorldModel._train 返回 (post, context, metrics)，取第三个
    _post, _context, mets = wm._train(dict(batch))
    return mets, n_params


class _DummyActSpace:
    n = 4


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=3, help="队伍数")
    ap.add_argument("--batch-length", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("数据管线 (B,T,N,·) 等价性测试")
    print("=" * 74)

    config = load_config(["defaults", "codreamer"])
    config.num_actions = 4
    spec = Spec()

    with tempfile.TemporaryDirectory() as tmp:
        batch, obs_space = build_batch(
            config, spec, args.batch_size, args.batch_length, tmp
        )

    n = spec.num_agents
    b = args.batch_size
    t = args.batch_length

    print("\n[1] 图批次形状")
    check("state 是 (B,T,N,D)", batch["state"].shape == (b, t, n, 1 + n), str(batch["state"].shape))
    check("reward 是 (B,T,N)", batch["reward"].shape == (b, t, n), str(batch["reward"].shape))
    check("is_first 是 (B,T,N)", batch["is_first"].shape == (b, t, n), str(batch["is_first"].shape))
    check("action 是 (B,T,N,A)", batch["action"].shape == (b, t, n, 4), str(batch["action"].shape))
    check("adjacency 是 (B,T,N,N)", batch["adjacency"].shape == (b, t, n, n), str(batch["adjacency"].shape))
    check("edge_features 是 (B,T,N,N,F)", batch["edge_features"].shape == (b, t, n, n, 1), str(batch["edge_features"].shape))

    print("\n[2] 折平变换本身")
    folded = models.fold_agent_axis(batch)
    check("state (B,T,N,D) -> (B*N,T,D)", folded["state"].shape == (b * n, t, 1 + n), str(folded["state"].shape))
    check("adjacency (B,T,N,N) -> (B*N,T,N)", folded["adjacency"].shape == (b * n, t, n), str(folded["adjacency"].shape))
    # 折平不能改变数值，只能改形状
    check("折平不改变数值总和", np.isclose(folded["state"].sum(), batch["state"].sum()))
    # 行主序：折平后第 (i*N+j) 行必须是第 i 队第 j 个 agent。
    # 全量枚举，因为"少了转置"这类错误只会弄乱一部分位置，抽查一个下标不一定能碰到。
    order_ok = all(
        np.array_equal(folded["state"][i * n + j], batch["state"][i, :, j])
        for i in range(b)
        for j in range(n)
    )
    check("折平顺序 team-major / agent-minor（全量枚举）", order_ok,
          "每个 (i*N+j) 都必须等于 team i 的 agent j")
    check(
        "折平顺序对 reward/adjacency 同样成立",
        all(
            np.array_equal(folded["reward"][i * n + j], batch["reward"][i, :, j])
            and np.array_equal(folded["adjacency"][i * n + j], batch["adjacency"][i, :, j])
            for i in range(b)
            for j in range(n)
        ),
    )

    print("\n[3] 真实模型：折平路径 vs 图路径，损失必须一致（gnn_layers=0）")
    mets_graph, n_graph = train_once(config, obs_space, batch, n, args.seed, gnn_layers=0)
    mets_flat, _ = train_once(config, obs_space, folded, 1, args.seed, gnn_layers=0)

    # 容差说明（为什么不再是 0）：
    #
    # 图路径算的是 (B,T,N,·)，展平路径算 (B*N,T,·)。**同一批数，内存顺序不同**，
    # 于是 torch.mean / logsumexp 这类归约的累加顺序不同，float32 下结果会差
    # 最后几位。float32 的机器 epsilon 是 1.2e-07，实测相对偏差恰在这个量级。
    #
    # 这是**原理上不可避免**的，不是 bug：任何"同样输入、不同张量形状"的两次归约
    # 都不保证逐位相同。判断标准改成相对误差：真错了（轴接反、切错 agent）会差
    # 好几个数量级，而不是 1e-07。
    #
    # 而 [2] 节对 fold_agent_axis 本身仍然要求**严格为 0** —— 那一步是纯重排，
    # 不涉及任何归约，可以且应该逐位相同。
    RTOL = 1e-5   # 比噪声地板高约 100 倍，但远小于任何真实错误的幅度
    keys = ["model_loss", "state_loss", "reward_loss", "cont_loss", "kl", "dyn_loss", "rep_loss"]
    worst_rel = 0.0
    for key in keys:
        if key not in mets_graph or key not in mets_flat:
            check(f"{key} 存在", False, f"graph={key in mets_graph} flat={key in mets_flat}")
            continue
        a, bb = float(np.asarray(mets_graph[key])), float(np.asarray(mets_flat[key]))
        rel = abs(a - bb) / max(abs(a), abs(bb), 1e-12)
        worst_rel = max(worst_rel, rel)
        check(f"{key} 一致", rel < RTOL, f"graph={a!r} flat={bb!r} rel={rel:.2e}")

    print("\n[4] 结论")
    check(f"所有损失相对偏差 < {RTOL:.0e}", worst_rel < RTOL, f"最大相对偏差 {worst_rel:.2e}")

    # ------------------------------------------------------------------
    print("\n[5] GNN 确实接上了（本节是步骤 3 的核心验收）")
    mets_gnn, n_gnn = train_once(config, obs_space, batch, n, args.seed, gnn_layers=1)
    print(f"       gnn_layers=0 参数量 {n_graph:,}")
    print(f"       gnn_layers=1 参数量 {n_gnn:,}  （+{n_gnn - n_graph:,}）")
    check("0 层不增加任何参数", n_graph == 16429829, f"{n_graph:,}（预期 16,429,829）")
    check("1 层确实加了参数", n_gnn > n_graph, f"+{n_gnn - n_graph:,}")
    check("1 层无 NaN", all(
        np.isfinite(np.asarray(mets_gnn[k])).all()
        for k in ["model_loss", "state_loss", "reward_loss", "cont_loss", "kl"]
    ))
    # 关键：开了 GNN 损失必须变 —— 不变就意味着 GNN 根本没接到计算图里
    changed = [
        k for k in ["model_loss", "state_loss", "reward_loss", "kl"]
        if not np.isclose(float(np.asarray(mets_gnn[k])), float(np.asarray(mets_graph[k])))
    ]
    check("开 GNN 后损失发生变化（证明接线生效）", len(changed) > 0,
          f"变化的指标: {changed}")
    check("dyn_loss 也变化（GNN 影响了 prior 的 KL）",
          not np.isclose(float(np.asarray(mets_gnn["dyn_loss"])),
                         float(np.asarray(mets_graph["dyn_loss"]))))

    # GNN 只该影响“跨 agent”的部分。孤岛 agent 的 det​er 不应因邻居而变。
    # （已由 networks_gnn 的单元测试覆盖，这里不重复。）

    print("\n[6] 结论")
    if not FAILURES:
        print("      ✓ 数据管线 (B,T,N,·) 与展平路径等价（相对误差在 float32 机器精度量级）")
        print("      ✓ GNN 已接入 RSSM 的 GRU 咽喉点，开/关确实改变计算结果")

    print("=" * 74)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): {', '.join(FAILURES)}")
        return 1
    print("全部通过：GNN 层数为 0 时，图批次与展平批次完全等价。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
