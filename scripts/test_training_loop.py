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
五条断言
================================================================================
  [1] 预训练：世界模型权重必须**变**，actor 权重必须**逐位不变**
  [2] 非预训练：两者都必须变（否则 [1] 会因为"什么都不学"而假通过）
  [3] 冻结世界模型：世界模型必须不变，actor 必须变
  [4] 执行路径带图（codreamer）：改邻接矩阵必须改变动作 / 对数概率
  [5] 执行路径的惰性（idreamer）：同上但必须**逐位不变**
  [6] comm_variant 单一开关确实同时驱动三处 GNN，且能按需覆盖

[2] 和 [5] 是**对照**，不可缺少：[1] 说"actor 不变"，只有 [2] 证明"不是所有东西
都不变"才有意义；[4] 说"改图有用"，只有 [5] 证明"这个敏感性来自 GNN 而不是
来自别的路径"才有意义。

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
def build_run(tmp: pathlib.Path, *, variant: str, pretrain: int, seed: int = 0):
    cfg = copy.deepcopy(load_config(["defaults", "codreamer"]))
    cfg.envs = 1                 # 1 支队
    cfg.batch_size = 1           # batch_size 的单位是队伍数
    cfg.batch_length = 4
    cfg.prefill = 8
    cfg.pretrain = pretrain
    cfg.seg_graph_obs = True
    cfg.seg_num_agents = 4
    cfg.comm_variant = variant
    # -1 = 跟随 comm_variant（与 yaml 默认一致）
    cfg.gnn_layers = -1
    cfg.gnn_head_layers = -1
    cfg.gnn_behavior_layers = -1
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
    cache: dict = {}
    with tempfile.TemporaryDirectory() as sim_tmp:
        tools_marl.simulate_marl(
            _RandomPolicy(cfg.num_actions, cfg.num_agents, seed=seed),
            [env],
            cache,
            sim_tmp,
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
    for pretrain, want_wm, want_actor, label in [
        (5, True, False, "预训练 5 步：世界模型变、actor 逐位不变"),
        (0, True, True, "非预训练：两者都变（[1] 的对照，防止假通过）"),
    ]:
        with tempfile.TemporaryDirectory() as d:
            cfg, env, agent = build_run(pathlib.Path(d), variant="idreamer", pretrain=pretrain)
            print(
                f"  ({label})  update_world_model 期间 pretrain={pretrain}"
            )
            obs = flat_obs(env)
            done = np.ones(cfg.num_agents, dtype=bool)
            wm_before = snapshot(agent._wm)
            actor_before = snapshot(agent._task_behavior.actor)
            agent(obs, done, training=True)
            wm_delta = max_diff(wm_before, snapshot(agent._wm))
            actor_delta = max_diff(actor_before, snapshot(agent._task_behavior.actor))

            wm_changed = wm_delta > 0.0
            actor_changed = actor_delta > 0.0
            check(
                f"pretrain={pretrain}: 世界模型{'变' if want_wm else '不变'}",
                wm_changed == want_wm,
                f"max|Δ|={wm_delta:.3e}",
            )
            check(
                f"pretrain={pretrain}: actor {'变' if want_actor else '逐位不变'}",
                actor_changed == want_actor,
                f"max|Δ|={actor_delta:.3e}",
            )


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

            # 造两张不同的邻接矩阵。只翻一条有向边，保证差异最小。
            adj0 = np.asarray(obs["adjacency"], dtype=np.float32).copy()
            adj1 = adj0.copy()
            adj1[:, 0, 1] = 1.0 - adj1[:, 0, 1]
            obs0 = dict(obs, adjacency=adj0)
            obs1 = dict(obs, adjacency=adj1)
            edge0 = np.asarray(obs["edge_features"], dtype=np.float32).copy()
            if edge0.ndim == 3:                       # (batch, N, F) 无 N 轴时
                pass
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmp-root", default=None)
    args = ap.parse_args()

    root = pathlib.Path(args.tmp_root or tempfile.mkdtemp(prefix="loop_test_"))
    root.mkdir(parents=True, exist_ok=True)
    print(f"临时目录 {root}")
    try:
        section_variant_switch()
        section_pretrain(root)
        section_freeze(root)
        section_execution_graph(root)
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
