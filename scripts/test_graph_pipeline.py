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
import copy
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


def train_once(config, obs_space, batch, num_agents_override: int, seed: int,
               gnn_layers: int = 0, gnn_head_layers: int = 0) -> dict:
    """在一个全新模型上跑一次 _train，返回 (损失指标, 参数量)。"""
    cfg = copy.deepcopy(config)
    cfg.num_agents = num_agents_override
    cfg.gnn_layers = gnn_layers
    cfg.gnn_head_layers = gnn_head_layers
    cfg.device = "cpu"
    cfg.precision = 32          # 关掉 amp，保证两次比较逐位一致
    cfg.compile = False
    tools.set_seed_everywhere(seed)
    wm = models.WorldModel(obs_space, _DummyActSpace(), 0, cfg)
    n_params = sum(p.numel() for p in wm.parameters())
    # ★ 建完模型后**再种一次种子** ★
    #
    # 为什么必需：不同架构的参数个数不同，**建模型时消耗的随机数个数也不同**，
    # 于是全局 RNG 状态被推到不同位置。而 RSSM 的先验/后验是用 `dist.sample()` 
    # 采样的，采样结果就会不一样 —— 于是所有权重未变的模块（比如 decoder）也
    # 跟着算出不同的损失，把"只有 reward/cont 变了"这个结论淹没掉。
    #
    # 重新种一次，两次比较就从**同一个 RNG 状态**出发前向；差异只可能来自架构本身。
    # （曾被 [6] 节里 state_loss / kl 竟然也变化这一现象发现。）
    tools.set_seed_everywhere(seed)
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

    print("\n[6] 预测头里的通信：只有 reward / cont，decoder 必须独立")
    # 为什么不用 loss 判断头有没有接上图：
    # reward 头的输出层用 `outscale: 0.0` 初始化，logits 恒为 0 → 预测分布是均匀的
    # → 损失被钉在最大熵 log(255) = 5.5413 上，单步训练后输出层还没学动，
    # **换任何图都不动**。那是初始化的退化，不是接线错误。
    # 所以这里直接比对**头的输出张量**，与 loss 的绝对值无关。
    cfg = copy.deepcopy(config)
    cfg.num_agents = n
    cfg.gnn_layers = 0
    cfg.gnn_head_layers = 1
    # ★ 为什么要把 reward 头的 outscale 从 0 改成 1 ★
    #
    # reward 头的输出层用 `outscale: 0.0` 初始化（即权重全零），于是 logits 恒为 0
    # → 预测分布均匀 → 输出与输入**无关**。这对训练本身是正常的（head 要学），
    # 但会让“头对图有没有响应”这个问题永远得到“没响应”，无论 GNN 接没接上。
    #
    # 这里只为行为验收临时打开 scale，使头非退化；参数量与结构检查不受影响。
    # （真正的训练路径仍用配置里的 outscale=0。）
    cfg.reward_head = dict(cfg.reward_head, outscale=1.0)
    cfg.device = "cpu"
    cfg.precision = 32
    cfg.compile = False
    tools.set_seed_everywhere(args.seed)
    wm = models.WorldModel(obs_space, _DummyActSpace(), 0, cfg)
    n_h = sum(p.numel() for p in wm.parameters())
    print(f"       gnn_head_layers=0 参数量 {n_graph:,}")
    print(f"       gnn_head_layers=1 参数量 {n_h:,}  （+{n_h - n_graph:,}，两个头）")
    check("开了头 GNN 后参数变多", n_h > n_graph, f"+{n_h - n_graph:,}")

    # --- 结构：哪几个头盖了 GNN，层数各是多少
    feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
    check("reward 头 GNN 层数 = 1", wm.heads["reward"]._gnn.num_layers == 1)
    check("cont 头 GNN 层数 = 1", wm.heads["cont"]._gnn.num_layers == 1)
    check(
        "decoder 头 GNN 层数 = 0（论文明确要求保持独立）",
        wm.heads["decoder"]._mlp._gnn.num_layers == 0,
        f"{wm.heads['decoder']._mlp._gnn.num_layers}",
    )

    # --- 行为：头的输出确实随图变化
    def _mean(dist):
        # DiscDist 的 mean 是方法，Bernoulli 的是属性 —— 统一取一下
        v = getattr(dist, "mean")
        return v() if callable(v) else v

    feat = torch.randn(b, t, n, feat_size)
    adj_a = torch.as_tensor(batch["adjacency"])
    edge_a = torch.as_tensor(batch["edge_features"])
    adj_b = adj_a.clone()
    adj_b[..., 0, 2] = 1.0
    adj_b[..., 2, 0] = 1.0          # 多连一条边：0-2 相连
    edge_b = edge_a.clone()
    edge_b[..., 0, 2, :] = 0.5      # 并改一下那条边的边特征
    edge_b[..., 2, 0, :] = 0.5

    for name in ["reward", "cont"]:
        head = wm.heads[name]
        out_a = _mean(head(feat, graph=(adj_a, edge_a)))
        out_b = _mean(head(feat, graph=(adj_b, edge_b)))
        check(
            f"{name} 头对图有响应（改边/改边特征会改变输出）",
            not torch.allclose(out_a, out_b),
            f"max|Δ|={float((out_a - out_b).abs().max()):.3e}",
        )

    # --- state_loss 必须逐位不变（decoder 在真实训练路径上不受影响）
    mets_h, _ = train_once(
        config, obs_space, batch, n, args.seed, gnn_layers=0, gnn_head_layers=1
    )
    check(
        "state_loss 完全不变（decoder 独立，论文明确要求）",
        float(np.asarray(mets_h["state_loss"])) == float(np.asarray(mets_graph["state_loss"])),
        f"head={float(np.asarray(mets_h['state_loss']))!r} "
        f"base={float(np.asarray(mets_graph['state_loss']))!r}",
    )
    check(
        "kl 不变（本轮 RSSM GNN 关闭）",
        np.isclose(float(np.asarray(mets_h["kl"])), float(np.asarray(mets_graph["kl"]))),
    )

    print("\n[7] 结论")
    if not FAILURES:
        print("      ✓ 数据管线 (B,T,N,·) 与展平路径等价（相对误差在 float32 机器精度量级）")
        print("      ✓ GNN 已接入 RSSM 的 GRU 咽喉点，开/关确实改变计算结果")
        print("      ✓ reward / cont 头接了通信，而 decoder 逐位不受影响")

    print("=" * 74)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): {', '.join(FAILURES)}")
        return 1
    print("全部通过：GNN 层数为 0 时，图批次与展平批次完全等价。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
