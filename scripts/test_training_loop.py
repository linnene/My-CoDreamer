"""回路测试：训练循环本身，而不是它里面的模块。

================================================================================
为什么必须有这个文件
================================================================================
`test_graph_pipeline.py` 有 566 行断言，全绿，却漏掉了两个致命错误：

  1. 预训练时 `update_behavior` 与 `update_world_model` **同时**为 False，
     那 500 步是 500 次不更新任何参数的前向。日志照旧打印"预训练完成"。
  2. `Dreamer._policy()` 从不传图，于是收集数据用的动作来自一个看不见图的
     actor，而世界模型是带着图训练的 —— CoDreamer 在真实执行中从未通信。

两个都长在**集成点**上：模块各自都对，接起来错。而 `test_graph_pipeline.py`
测的全是模块（GATv2 形状、数据折平、RSSM 前向、参数量守恒），一条回路断言都没有。

这个文件补的就是那一层。它**不检查任何内部张量的形状**，只检查从外面能观测到的
行为 —— 这与上面两个 bug 的存在形式一致，也只有这种断言能抓住它们。

================================================================================
八条断言
================================================================================
  [1] 预训练：世界模型权重必须**变**，actor 权重必须**逐位不变**
  [2] 非预训练：两者都必须变（否则 [1] 会因为"什么都不学"而假通过）
  [3] 冻结世界模型：世界模型必须不变，actor 必须变
  [4] 执行路径带图（codreamer）：改邻接矩阵必须改变动作 / 对数概率
  [5] 执行路径的惰性（idreamer）：同上但必须**逐位不变**
  [6] comm_variant 单一开关确实同时驱动三处 GNN，且能按需覆盖
  [7] 消融臂的参数量恒等式：三处 GNN 互不串味（机械可查）
  [8] 通信的 **reach**：信息走几跳（结构，不是敏感性）

[2] 和 [5] 是**对照**，不可缺少：[1] 说"actor 不变"，只有 [2] 证明"不是所有东西
都不变"才有意义；[4] 说"改图有用"，只有 [5] 证明"这个敏感性来自 GNN 而不是
来自别的路径"才有意义。

[4]/[5] 与 [8] 测的不是一回事，两者都要：[4]/[5] 测**敏感性**（改邻接矩阵，
动作变不变），[8] 测**结构**（哪几跳有、哪几跳无）。后者能抓住前者抓不到的错 ——
**GNN 接错了位置，但仍然对邻接矩阵敏感**。

用法::

    python scripts/test_training_loop.py
"""

from __future__ import annotations

import argparse
import copy
import pathlib
import shutil
import sys
import tempfile

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
D3T = REPO / "third_party" / "dreamerv3-torch"
sys.path.insert(0, str(D3T))
sys.path.insert(0, str(REPO / "scripts"))

import ruamel.yaml as yaml  # noqa: E402

import dreamer as dreamer_mod  # noqa: E402
import models  # noqa: E402
import tools  # noqa: E402
import tools_marl  # noqa: E402

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(name)


def load_config(config_names: list[str]) -> argparse.Namespace:
    """复刻 dreamer.py 的配置加载（与真实训练同一套超参）。"""
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
    return argparse.Namespace(**defaults)


class _RandomPolicy:
    """随机策略，用来快速灌满 replay。返回格式与 Dreamer._policy 一致。"""

    def __init__(self, num_actions: int, num_agents: int, seed: int = 0):
        self.num_actions = num_actions
        self.num_agents = num_agents
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs, done, state):
        batch = obs["state"].shape[0]
        idx = self.rng.integers(0, self.num_actions, size=batch)
        action = np.eye(self.num_actions, dtype=np.float32)[idx]
        return {"action": action, "logprob": np.zeros(batch, np.float32)}, None


def snapshot(module) -> dict:
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def max_diff(a: dict, b: dict) -> float:
    keys = set(a) | set(b)
    assert set(a) == set(b), f"参数集合不同: {set(a) ^ set(b)}"
    return max((a[k] - b[k]).abs().max().item() for k in keys)


# ---------------------------------------------------------------------------
# 搭一个最小的真实训练环境：环境 -> replay -> 图数据集 -> Dreamer
# ---------------------------------------------------------------------------
def build_run(
    tmp: pathlib.Path,
    *,
    variant: str,
    pretrain: int,
    seed: int = 0,
    gnn: tuple[int, int, int] | None = None,
):
    cfg = copy.deepcopy(load_config(["defaults", "codreamer"]))
    cfg.envs = 1                 # 1 支队
    cfg.batch_size = 1           # batch_size 的单位是队伍数
    cfg.batch_length = 4
    cfg.prefill = 8
    cfg.pretrain = pretrain
    cfg.seg_graph_obs = True
    cfg.seg_num_agents = 4
    cfg.comm_variant = variant
    # -1 = 跟随 comm_variant（与 yaml 默认一致）。消融时传 gnn=(rssm, head, behavior)。
    cfg.gnn_layers, cfg.gnn_head_layers, cfg.gnn_behavior_layers = (
        gnn if gnn is not None else (-1, -1, -1)
    )
    cfg.device = "cpu"
    cfg.precision = 32
    cfg.compile = False
    cfg.video_pred_log = False
    cfg.action_repeat = 1
    cfg.steps = 1000
    cfg.train_ratio = 512
    cfg.eval_every = 10**9
    cfg.log_every = 10**9
    cfg.dataset_size = 10**6
    cfg.expl_until = 0.0         # 保证不走探索策略分支
    cfg.logdir = str(tmp / "logdir")
    cfg.traindir = str(tmp / "train_eps")
    cfg.evaldir = str(tmp / "eval_eps")
    pathlib.Path(cfg.logdir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(cfg.traindir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(cfg.evaldir).mkdir(parents=True, exist_ok=True)

    # 走真实的那一个解析函数，而不是在测试里复制一遍
    dreamer_mod.resolve_comm_variant(cfg)

    tools.set_seed_everywhere(seed)
    env = dreamer_mod.make_env(cfg, "train", 0)
    # main() 在拿到环境后才回填这两个键，测试必须复刻，否则 Dreamer 拿不到动作数
    cfg.num_agents = getattr(env, "num_agents", 1)
    acts = env.action_space
    cfg.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    # 用随机策略灌 replay
    #
    # 注意第 4 个参数是**存盘目录**，必须就是随后 load_episodes 读的那个。
    # 两边用不同的目录会得到"replay 里有 episode，但找不到完整队伍"。
    cache: dict = {}
    tools_marl.simulate_marl(
        _RandomPolicy(cfg.num_actions, cfg.num_agents, seed=seed),
        [env],
        cache,
        cfg.traindir,
        tools.Logger(pathlib.Path(cfg.logdir), 0),
        is_eval=False,
        limit=cfg.dataset_size,
        episodes=cfg.prefill * cfg.num_agents,
    )

    logger = tools.Logger(pathlib.Path(cfg.logdir), 0)
    train_eps = tools.load_episodes(cfg.traindir, limit=cfg.dataset_size)
    dataset = tools_marl.make_graph_dataset(train_eps, cfg)
    agent = dreamer_mod.Dreamer(
        env.observation_space, env.action_space, cfg, logger, dataset
    ).to(cfg.device)
    return cfg, env, agent


def flat_obs(env, seed: int = 0):
    """取一份 (batch, ·) 形状的观测 —— 与 simulate_marl 铺平后的形状一致。

    n_teams=1 时 simulate_marl 的 `np.concatenate([o[k] for o in obs], 0)` 就是
    环境自己返回的那一份。
    """
    tools.set_seed_everywhere(seed)
    obs = env.reset()
    out = {}
    for k, v in obs.items():
        if "log_" in k:
            continue
        out[k] = np.asarray(v)
    return out


# ---------------------------------------------------------------------------
def section_pretrain(tmp: pathlib.Path) -> None:
    print("\n[1][2] 预训练到底训没训东西")
    with tempfile.TemporaryDirectory() as d:
        cfg, env, agent = build_run(pathlib.Path(d), variant="idreamer", pretrain=5)
        obs = flat_obs(env)
        done = np.ones(cfg.num_agents, dtype=bool)

        # 第一次调用就是预训练槽位（_should_pretrain 是 tools.Once）。
        # 注意必须**只看第一次**：第二次开始就是正常训练，两者会混在一起。
        wm_before = snapshot(agent._wm)
        actor_before = snapshot(agent._task_behavior.actor)
        agent(obs, done, training=True)
        wm_delta = max_diff(wm_before, snapshot(agent._wm))
        actor_delta = max_diff(actor_before, snapshot(agent._task_behavior.actor))
        check("预训练 5 步：世界模型**变**了", wm_delta > 0.0, f"max|Δ|={wm_delta:.3e}")
        check(
            "预训练 5 步：actor **逐位不变**",
            actor_delta == 0.0,
            f"max|Δ|={actor_delta:.3e}",
        )

        # ★ 对照组，不能省 ★
        # 只有上面两条时，"actor 不变"可能是因为整条回路压根没在学。
        # 预训练结束后 actor 必须开始动，这才能证明上面那个 0 是真的 0。
        wm_b2 = snapshot(agent._wm)
        actor_b2 = snapshot(agent._task_behavior.actor)
        for _ in range(3):
            agent(obs, done, training=True)
        wm_d2 = max_diff(wm_b2, snapshot(agent._wm))
        actor_d2 = max_diff(actor_b2, snapshot(agent._task_behavior.actor))
        check(
            "对照：预训练结束后 actor 开始更新（否则上面的 0 是假通过）",
            actor_d2 > 0.0,
            f"max|Δ|={actor_d2:.3e}",
        )
        check("对照：世界模型继续更新", wm_d2 > 0.0, f"max|Δ|={wm_d2:.3e}")


def section_freeze(tmp: pathlib.Path) -> None:
    print("\n[3] 冻结世界模型：只训策略")
    with tempfile.TemporaryDirectory() as d:
        cfg, env, agent = build_run(pathlib.Path(d), variant="idreamer", pretrain=0)
        obs = flat_obs(env)
        done = np.ones(cfg.num_agents, dtype=bool)
        agent(obs, done, training=True)          # 先把 Once 预训练位走掉
        agent.set_world_model_frozen(True)
        wm_before = snapshot(agent._wm)
        actor_before = snapshot(agent._task_behavior.actor)
        for _ in range(3):
            agent(obs, done, training=True)
        wm_delta = max_diff(wm_before, snapshot(agent._wm))
        actor_delta = max_diff(actor_before, snapshot(agent._task_behavior.actor))
        check("冻结后世界模型逐位不变", wm_delta == 0.0, f"max|Δ|={wm_delta:.3e}")
        check("冻结后 actor 仍在更新", actor_delta > 0.0, f"max|Δ|={actor_delta:.3e}")


def section_execution_graph(tmp: pathlib.Path) -> None:
    print("\n[4][5] 执行路径（_policy）是否真的用到图")
    for variant, want_sensitive in [("codreamer", True), ("idreamer", False)]:
        with tempfile.TemporaryDirectory() as d:
            cfg, env, agent = build_run(pathlib.Path(d), variant=variant, pretrain=0)
            obs = flat_obs(env)
            n = cfg.num_agents
            batch = obs["adjacency"].shape[0]

            # ★ 先把形状钉死 ★
            # 环境的观测里 adjacency 是**整张** (N, N)（同一份给所有 agent），
            # simulate_marl 铺平后是 (n_teams*N, N) —— 每行是某队矩阵的一行。
            # GNN 要的是 agent 轴显式存在的 (n_teams, N, N)。
            # 这里如果形状错了，下面"改图不影响"的断言会因为错误的原因通过。
            n_teams = batch // n
            check(
                "铺平后的 adjacency 是 (batch, N)（一堆行）",
                obs["adjacency"].ndim == 2 and obs["adjacency"].shape[1] == n,
                f"得到 {obs['adjacency'].shape}，期望 ({batch}, {n})",
            )

            # 补 agent 轴 —— 这正是 Dreamer._policy 里 add_agent_axis 做的事
            unflat = dict(obs)
            unflat["adjacency"] = obs["adjacency"].reshape(n_teams, n, n)
            unflat["edge_features"] = obs["edge_features"].reshape(
                n_teams, n, n, obs["edge_features"].shape[-1]
            )
            g = models.execution_graph(unflat, n, "cpu")
            check(
                "补上 agent 轴后得 (n_teams, N, N)",
                g is not None and tuple(g[0].shape) == (n_teams, n, n),
                f"得到 {None if g is None else tuple(g[0].shape)}",
            )
            # 每支队伍那一块必须等于它自己的邻接矩阵
            expect = obs["adjacency"].reshape(n_teams, n, n)
            check(
                "每支队伍的矩阵等于原矩阵",
                np.array_equal(expect, g[0].numpy()),
            )
            # ★ 不带 agent 轴时必须报错 ★
            # 这个形状已经错过三次，且三次都是 GNN 内部报错而不是调用点报错。
            try:
                models.execution_graph(obs, n, "cpu")
                check("缺 agent 轴时 execution_graph 会报错（而不是传给 GNN）", False)
            except ValueError:
                check("缺 agent 轴时 execution_graph 会报错（而不是传给 GNN）", True)

            # 翻一条有向边。铺平布局下每个 agent 各占一行。
            adj0 = np.asarray(obs["adjacency"], dtype=np.float32).copy()
            adj1 = adj0.copy()
            adj1[0, 1] = 1.0 - adj1[0, 1]
            obs0 = dict(obs, adjacency=adj0)
            obs1 = dict(obs, adjacency=adj1)
            changed = float(np.abs(adj0 - adj1).max())

            def run(o):
                # obs_step 的后验会采样 → 必须每次重新播种，否则比的是 RNG
                tools.set_seed_everywhere(1234)
                with torch.no_grad():
                    out, _ = agent._policy(o, None, training=False)
                return out["action"].clone(), out["logprob"].clone()

            a0, l0 = run(obs0)
            a1, l1 = run(obs1)
            da = float((a0 - a1).abs().max())
            dl = float((l0 - l1).abs().max())

            if want_sensitive:
                check(
                    f"{variant}: 改邻接矩阵会改变动作 / 对数概率（图进了执行路径）",
                    dl > 1e-7 or da > 0.0,
                    f"边改动量={changed:.1f}  max|Δaction|={da:.3e}  max|Δlogprob|={dl:.3e}",
                )
            else:
                check(
                    f"{variant}: 改邻接矩阵**逐位不影响**任何东西（图不漏进 IDreamer）",
                    da == 0.0 and dl == 0.0,
                    f"max|Δaction|={da:.3e}  max|Δlogprob|={dl:.3e}",
                )


def section_variant_switch() -> None:
    print("\n[6] comm_variant 单一开关")
    for variant, want in [("idreamer", 0), ("codreamer", 1)]:
        cfg = load_config(["defaults", "codreamer"])
        cfg.comm_variant = variant
        cfg.gnn_layers = -1
        cfg.gnn_head_layers = -1
        cfg.gnn_behavior_layers = -1
        dreamer_mod.resolve_comm_variant(cfg)
        got = (cfg.gnn_layers, cfg.gnn_head_layers, cfg.gnn_behavior_layers)
        check(f"{variant} -> 三处 GNN 全为 {want}", got == (want, want, want), f"得到 {got}")

    # 显式覆盖仍然可用（做消融）
    cfg = load_config(["defaults", "codreamer"])
    cfg.comm_variant = "codreamer"
    cfg.gnn_layers = 0
    cfg.gnn_head_layers = -1
    cfg.gnn_behavior_layers = -1
    dreamer_mod.resolve_comm_variant(cfg)
    check(
        "显式指定可以覆盖 comm_variant（消融用）",
        (cfg.gnn_layers, cfg.gnn_head_layers, cfg.gnn_behavior_layers) == (0, 1, 1),
        f"得到 {(cfg.gnn_layers, cfg.gnn_head_layers, cfg.gnn_behavior_layers)}",
    )

    # 非法值必须报错
    cfg = load_config(["defaults", "codreamer"])
    cfg.comm_variant = "half"
    try:
        dreamer_mod.resolve_comm_variant(cfg)
        check("非法 comm_variant 会报错", False)
    except SystemExit:
        check("非法 comm_variant 会报错", True)


# ---------------------------------------------------------------------------
# [7] 消融臂的参数恒等式 —— 机械可查的"没串味"检查
# ---------------------------------------------------------------------------
#
# 论文的核心结果是**两级通信的分解**（WM Comm / AC Comm / CoDreamer / IDreamer），
# 而我们此前只跑过三处全 0 与三处全 1 这两个端点。做那四个臂之前先钉死一件事：
# 打开哪一处的 GNN，就只能改变那一处的参数量。
#
# 这些恒等式不需要训练就能查。一旦不成立，四臂实验的全部结论都是错的 ——
# 它们会变成"某个我没意识到的地方也被改了"的对照。
ABLATION_ARMS = {
    # 名字         (rssm, head, behavior)
    "idreamer": (0, 0, 0),
    "wm_comm": (1, 1, 0),
    "ac_comm": (0, 0, 1),
    "codreamer": (1, 1, 1),
}


def n_params(module) -> int:
    return sum(p.numel() for p in module.parameters())


def section_ablation_params(root: pathlib.Path) -> None:
    print("\n[7] 消融臂：三处 GNN 的参数量互不串味")
    counts = {}
    for name, gnn in ABLATION_ARMS.items():
        tmp = root / f"abl_{name}"
        tmp.mkdir(parents=True, exist_ok=True)
        _, env, agent = build_run(tmp, variant="idreamer", pretrain=1, gnn=gnn)
        counts[name] = {
            "world": n_params(agent._wm),
            "actor": n_params(agent._task_behavior.actor),
            "value": n_params(agent._task_behavior.value),
        }
        env.close()
        c = counts[name]
        print(
            f"    {name:<10} gnn={gnn}  world={c['world']:,}  "
            f"actor={c['actor']:,}  value={c['value']:,}"
        )

    # 世界模型只由 rssm / head 决定 —— 与 behavior 无关
    check(
        "世界模型参数量：idreamer == ac_comm（都不含 WM 的 GNN）",
        counts["idreamer"]["world"] == counts["ac_comm"]["world"],
        f"{counts['idreamer']['world']:,} vs {counts['ac_comm']['world']:,}",
    )
    check(
        "世界模型参数量：wm_comm == codreamer（都含 WM 的 GNN）",
        counts["wm_comm"]["world"] == counts["codreamer"]["world"],
        f"{counts['wm_comm']['world']:,} vs {counts['codreamer']['world']:,}",
    )
    # actor / critic 只由 behavior 决定 —— 与 WM 的 GNN 无关
    for key in ("actor", "value"):
        check(
            f"{key} 参数量：idreamer == wm_comm（都不含 AC 的 GNN）",
            counts["idreamer"][key] == counts["wm_comm"][key],
            f"{counts['idreamer'][key]:,} vs {counts['wm_comm'][key]:,}",
        )
        check(
            f"{key} 参数量：ac_comm == codreamer（都含 AC 的 GNN）",
            counts["ac_comm"][key] == counts["codreamer"][key],
            f"{counts['ac_comm'][key]:,} vs {counts['codreamer'][key]:,}",
        )
    # 存在性对照：GNN 确实加了参数。
    # 少了这条，上面五条会因为"处处相等"而假通过 —— 比如 GNN 根本没接上时。
    check(
        "对照：打开 GNN 确实增加了参数量（否则上面的相等是假通过）",
        counts["codreamer"]["world"] > counts["idreamer"]["world"]
        and counts["codreamer"]["actor"] > counts["idreamer"]["actor"],
        f"world +{counts['codreamer']['world'] - counts['idreamer']['world']:,}  "
        f"actor +{counts['codreamer']['actor'] - counts['idreamer']['actor']:,}",
    )


def section_comm_reach() -> None:
    """[8] 通信的 **reach**：信息走几跳。

    与 [4]/[5] 的分工，值得写清楚，因为两者抓的错不一样：

        [4]/[5]  测**敏感性** —— 改邻接矩阵，动作分布变不变
        [8]      测**结构**   —— 哪几跳有、哪几跳无

    后者能抓住前者抓不到的错：**GNN 接错了位置，但仍对邻接矩阵敏感**。
    比如把聚合插在 encoder 之后、或者只接了一半的 agent，敏感性照样是有的。

    而且它是二值判据（1 跳有 / 2 跳恰好没有），因此跨环境可比、不随 seed 漂、
    不需要噪声本底 —— 这正是这个项目里"回报对比"缺的那几样性质。
    实现复用 `scripts/probe_comm_reach.py`，不在这里复制一份测量逻辑。

    为什么放在回路测试而不是留成独立脚本：台账记着，566 行模块级断言全绿却漏掉
    两个致命 bug，**因为它们长在集成点上**，而通信接线就是集成点（执行路径的三次
    形状 bug 全出在那里）。独立脚本要靠人记得跑，这一层不行。
    """
    print()
    print("\n[8] 通信的 reach（结构，不是敏感性）")
    sys.path.insert(0, str(REPO / "scripts"))
    import probe_comm_reach as probe_mod
    from eval_policy import CaptureLogger, _NoDataset

    def run(variant: str, layers=None):
        cfg = copy.deepcopy(load_config(["defaults", "codreamer"]))
        cfg.task = "overcooked_demo"
        cfg.envs = 1
        cfg.comm_variant = variant
        # agent 数由布局决定（forced_coordination_3p = 3），不能在这里覆盖。
        # C=3 是实测选的：三个 agent 被锁在 x=1 / x=3..4 / x=6 三个区，切比雪夫距离
        # 恒为 2~3（相邻区）和 5（两端），所以 C=3 得到**恒定的链**：边数恒为 2，
        # 两端恒隔 2 跳。C=1 会得到空图（0 条边），2 跳那一行就整个没了。
        cfg.overcooked_comm_radius = 3.0
        cfg.overcooked_graph_obs = True
        cfg.device = "cpu"          # 与本节其余部分一致，不占 GPU
        cfg.seed = 0
        if layers is not None:
            cfg.gnn_layers, cfg.gnn_head_layers, cfg.gnn_behavior_layers = layers
        dreamer_mod.resolve_comm_variant(cfg)
        env = dreamer_mod.make_env(cfg, "eval", 0)
        cfg.num_agents = env.num_agents
        cfg.num_actions = env.action_space.n
        torch.manual_seed(0)
        agent = dreamer_mod.Dreamer(
            env.observation_space, env.action_space, cfg,
            CaptureLogger(), _NoDataset(),
        ).to(cfg.device)
        agent.requires_grad_(requires_grad=False)
        agent.eval()
        # 3 个 agent 下 2 跳的 (i,j) 对本来就少（24 局只有约 20 个），
        # 提到 40 局把 2 跳那一行的样本量抬一档 —— 代价只是 batch 变大，
        # 前向**次数**不变（循环是按 (t, 有序对) 走的，与局数无关）。
        out = probe_mod.probe(
            agent, env, device=cfg.device, n_episodes=40, horizon=4, seed=0
        )
        env.close()
        got = (cfg.gnn_layers, cfg.gnn_head_layers, cfg.gnn_behavior_layers)
        return out, got

    # ---- 对照：一层 GNN 都没有 → 任何跨 agent 影响必须**恰好**为 0 ----
    r0, lay0 = run("idreamer")
    check("idreamer 三处 GNN 层数都是 0", lay0 == (0, 0, 0), str(lay0))
    check(
        "idreamer: deter 变化在每一跳上恰好为 0",
        float(np.max(r0["det_delta"])) == 0.0,
        f"max={float(np.max(r0['det_delta'])):.3e}",
    )
    check(
        "idreamer: 动作侧（翻转率与 logprob）在每一跳上恰好为 0",
        float(np.max(r0["flip"])) == 0.0
        and float(np.max(r0["flip_n"])) == 0.0
        and float(np.max(r0["lp_delta"])) == 0.0,
    )

    # ---- CoDreamer：actor 那层 GNN 的 reach 恰好 1 跳 ----
    r, lay = run("codreamer")
    check("codreamer 三处 GNN 层数都是 1", lay == (1, 1, 1), str(lay))

    # 样本存在性守卫。**这条是必需的不是装饰**：下面两条"恰好为 0"的断言在
    # 没有对应样本时是**空断言**，会假通过 —— 台账里"护栏本身是假的"那一类。
    check("1 跳样本存在", r["lp_n"][1] > 0, f"{r['lp_n'][1]}")
    check(
        "2 跳样本存在（否则下一条是空断言）",
        r["lp_n"].shape[0] > 2 and r["lp_n"][2] > 0,
        f"{r['lp_n'].tolist()}",
    )

    # ★ 结构性主张落在**连续量**上 ★
    #
    # 早先这条用的是翻转率。翻转率是阈值化的量（argmax 会把弱影响吃掉），
    # 所以"没测到翻转"既可能是"真的没有通路"，也可能只是样本不够 ——
    # 那就得靠样本量论证，而 (i,j) 对在同一局里共享状态、并不独立，
    # 有效样本量最多是 局数 × 可用 t 数，设计效应可达数倍。
    #
    # logprob 是连续量，于是"恰好没变"是一条 **float 精确**断言：
    # 若 2 跳真有通路，第一个样本就会给出非零值（浮点不会把真实依赖舍入成 0）。
    # 独立性、设计效应、簇相关统统不进入 —— 问题被消掉，而不是被修好。
    check(
        "1 跳：信息确实传过去了（deter 变、logprob 变、动作翻）",
        r["rel"][1] > 0.0 and r["lp_delta"][1] > 0.0 and r["flip"][1] > 0.0,
        f"deter {r['rel'][1]:.4f}  logprob {r['lp_delta'][1]:.3e}  "
        f"翻转 {r['flip'][1]:.4f}",
    )
    # ⚠️ 这条失败时的**第一问是「是不是有人把层数改成 2 了」**，不是
    # 「是不是掩码泄漏了」—— 它本质是一句关于 GNN 层数的断言：1 层的 reach
    # 到不了 2 跳。别去 masked_fill 那几行找一个并不存在的 bug。
    # 前提：这个 0 来自**硬掩码**（结构性无通路），不是算术相消。
    # 若将来有人把硬掩码换成软门控，这条会失败 —— **那时失败是正确行为**。
    check(
        "2 跳：当前步动作侧**恰好**没变（actor 那层只传 1 跳）",
        float(r["lp_delta"][2]) == 0.0 and float(r["flip"][2]) == 0.0,
        f"logprob {float(r['lp_delta'][2]):.3e}  翻转 {float(r['flip'][2]):.4f}",
    )
    # ⚠️ 这里早先写的是"3 跳：deter 恰好为 0"，守卫写成 `shape[0] <= 3 or ...`，
    # 两头都不对：**行不存在时那个守卫恒真 → 空断言**（正是本节其他地方在防的
    # "假通过"），而 detail 的 f-string 又会被**提前求值**成 IndexError
    # （`cond` 的短路救不了 f-string）。而且 3 个 agent 的最大跳数就是 2，
    # 根本没有第 3 行 —— 这条断言在这个配置下不可能有意义。
    # 换成对"表覆盖了哪些跳数"的显式断言。
    n_hops = int(r["det_delta"].shape[0])
    check(
        "跳数表覆盖 0..2（3 个 agent 最大只能隔 2 跳，没有第 3 行可查）",
        n_hops == 3,
        f"{n_hops} 行；2 跳样本 {int(r['cnt_n'][2])}",
    )
    # 复合：RSSM 一跳 + actor 一跳 = 2 跳。下一步的动作在 2 跳上应当仍有影响。
    #
    # ⚠️ 这里必须用**连续量**（logprob 相对移动），不能用翻转率。复合信号走过两跳
    # 之后衰减到翻不动 argmax —— 实测 52 个 2 跳样本里翻转率恰好是 0.0000，而那
    # 只说明"移得不够大到改 argmax"，不说明没有传播。翻转率是阈值化的量，和样本量
    # 绑在一起；连续量没有这个问题。1 跳那一档早就是这个口径了，这里当时漏改了。
    lp_rel_n = (
        r["lp_delta_n"][2] / r["lp_base_n"][2] if r["lp_base_n"][2] > 0 else 0.0
    )
    check(
        "复合后 2 跳仍有影响（RSSM 1 跳 + actor 1 跳）",
        lp_rel_n > 0.0,
        f"logprob 相对移动 {lp_rel_n:.3e}（翻转率 {r['flip_n'][2]:.4f}，"
        f"样本 {r['cnt_n'][2]} / {r['lp_n_n'][2]}）",
    )

    # ★ 消融臂不是独立的通信开关 ★
    #
    # `action_t` 是下一步 GRU 的**输入**，所以在想象里 actor 的输出也进入世界模型的
    # 下一步动力学（`_imagine`: action = policy(feat) → img_step(state, action)）。
    # 于是 ac_comm（rssm=0 head=0 behavior=1）**不等于**"世界模型没有通信"：
    # RSSM 那层 GNN 确实是恒等，但邻居信息仍经动作通道流进了 deter_{t+1}。
    #
    # 这条直接决定怎么读四个消融臂的结果 —— `idreamer / wm_comm / ac_comm /
    # codreamer` 这组命名读起来像四档独立开关，实际不是。
    r_ac, lay_ac = run("codreamer", layers=(0, 0, 1))
    check("ac_comm 的 RSSM 那层确实是恒等（rssm=0）", lay_ac == (0, 0, 1), str(lay_ac))
    check(
        "ac_comm: 世界模型的隐状态**仍然**拿到邻居信息（经 action_t）",
        r_ac["rel"][1] > 0.0,
        f"1 跳 deter {r_ac['rel'][1]:.4f}（rssm 的 GNN 关着）",
    )
    # 注意这里**不能**断言 2 跳恰好为 0：动态图上"图 t 与图 t+1 不同"会让
    # 图 t 上只隔 1 跳的对落进 2 跳那一行（实测 0.003580）。要干净地判"有没有
    # 第二条通路"必须用**静态图**（SEG 上同一配置实测恰好 0）。
    check(
        "ac_comm: 当前步动作侧在 2 跳恰好没变（actor 只有 1 层 GNN）",
        float(r_ac["lp_delta"][2]) == 0.0,
        f"logprob {float(r_ac['lp_delta'][2]):.3e}",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmp-root", default=None)
    args = ap.parse_args()

    root = pathlib.Path(args.tmp_root or tempfile.mkdtemp(prefix="loop_test_"))
    root.mkdir(parents=True, exist_ok=True)
    print(f"临时目录 {root}")
    try:
        section_variant_switch()
        section_ablation_params(root)
        section_pretrain(root)
        section_freeze(root)
        section_execution_graph(root)
        section_comm_reach()
    finally:
        if args.tmp_root is None:
            shutil.rmtree(root, ignore_errors=True)

    print("\n" + "=" * 74)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): " + ", ".join(FAILURES))
        return 1
    print("全部通过：预训练确实在训世界模型，图确实进了训练/想象/执行三条路径。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
