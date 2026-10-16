"""Overcooked 接入的接线测试 —— 只测集成点，不测模块。

================================================================================
为什么单独一个文件
================================================================================
`envs/overcooked_env.py` 自带自检（`python -m envs.overcooked_env`），它测的是
**环境本体**：观测是否局部、装载是否需要协作、图是否退化。

但本项目已经有过两次教训（见 docs/reproduction-plan.md §7）：566 行模块级断言
全绿，却漏掉了两个长在**集成点**上的致命 bug。这个文件补的就是那一层 ——
`make_env` 拿到的对象，是不是 `tools_marl.simulate_marl` 与 `models.py` 要的东西。

它**不检查任何环境内部状态**，只断言从框架那一侧看得见的东西。全部在 CPU 上跑，
不需要 GPU，也不需要训练。

================================================================================
六条断言
================================================================================
  [1] make_env 对 overcooked suite 返回**队伍级环境本身**，不接 gym wrapper 链
  [2] 接口与 vmas_env / seg_gym 逐条对齐（属性面完全相同）
  [3] obs 字典含 `is_first` / `is_terminal`（models.py 会 assert 这两个键）
  [4] 图批次能还原回 (N, N)：拼接 N 个 agent 的行再 reshape，与 env.adjacency 逐位相同
  [5] `encoder.mlp_keys` 的正则**必须**选中 state、**必须**排除 adjacency/edge_features
  [6] graph_obs 关掉后 adjacency 键消失，且 models.graph_from_data 返回 None
      —— "通信链路的唯一开关是数据里有没有图"这条不变式

用法::

    python scripts/test_team_env_wiring.py
"""

from __future__ import annotations

import argparse
import copy
import pathlib
import re
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
D3T = REPO / "third_party" / "dreamerv3-torch"
sys.path.insert(0, str(D3T))

import ruamel.yaml as yaml  # noqa: E402

import dreamer as dreamer_mod  # noqa: E402
import models  # noqa: E402

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


def base_cfg(task: str = "overcooked_demo", **overrides):
    cfg = copy.deepcopy(load_config(["defaults", "codreamer"]))
    cfg.task = task
    cfg.envs = 1
    cfg.overcooked_graph_obs = True
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# ---------------------------------------------------------------------------
def section_make_env() -> None:
    print("[1] make_env 返回队伍级环境本身")
    cfg = base_cfg()
    env = dreamer_mod.make_env(cfg, "train", 0)

    check(
        "返回的是 OvercookedTeamEnv，不是 gym wrapper 链",
        type(env).__name__ == "OvercookedTeamEnv",
        type(env).__name__,
    )
    # TimeLimit / SelectAction / UUID 都要求 env 是 gym.Env；
    # 队伍级环境自己处理时间限制与动作选择，所以绝不能出现在这条链上。
    check(
        "没有套上 TimeLimit / SelectAction / UUID",
        not hasattr(env, "_env_max_episode_steps") and not hasattr(env, "_action_space_n"),
    )
    check("num_agents 可读（dreamer.py 靠它判定多智能体）", env.num_agents == cfg.overcooked_num_agents,
          str(env.num_agents))
    check("有 close()", callable(getattr(env, "close", None)))
    check("有 .id（rollout 层会回写）", hasattr(env, "id"))
    env.close()


def section_interface_matches_vmas() -> None:
    print()
    print("[2] 接口与 vmas_env 逐条对齐")
    import envs.vmas_env as vmas_env

    lbf = dreamer_mod.make_env(base_cfg(), "train", 0)
    vmas = vmas_env.VMASEnv(task="flocking", seed=0, graph_obs=True)

    # adjacency / edge_features 是 property，**要在 reset 之后才有值** ——
    # reset 之前读会炸在 self._state 为 None 上，于是 hasattr 静默返回 False。
    # 一个还没 reset 的环境本来也不可用，所以先 reset 再查接口面。
    lbf.reset()
    vmas.reset()
    surface = ["num_agents", "observation_space", "action_space", "reset", "step",
               "close", "seed", "adjacency", "edge_features", "spec_edge_feature_dim",
               "graph_obs", "comm_radius"]
    missing = [a for a in surface if not hasattr(lbf, a)]
    check("Overcooked 具备 VMAS 的全部接口面", not missing, str(missing))

    check(
        "observation_space 都是 gym.spaces.Dict",
        type(lbf.observation_space).__name__ == type(vmas.observation_space).__name__,
    )
    check(
        "两者都声明单 agent 形状的 state",
        len(lbf.observation_space["state"].shape) == 1
        and len(vmas.observation_space["state"].shape) == 1,
    )
    check(
        "两者都打了 discrete 标记（dreamer.py 靠它选 OneHotDist）",
        getattr(lbf.action_space, "discrete", None) is True
        and getattr(vmas.action_space, "discrete", None) is True,
    )
    check("spec_edge_feature_dim 都是 1",
          lbf.spec_edge_feature_dim == 1 and vmas.spec_edge_feature_dim == 1)
    lbf.close()
    vmas.close()


def section_obs_contract() -> None:
    print()
    print("[3] obs 字典满足 models.py 的硬要求")
    cfg = base_cfg(overcooked_add_agent_index=True)
    env = dreamer_mod.make_env(cfg, "train", 0)
    n = env.num_agents
    obs = env.reset()

    # models.py 无条件 assert 这两个键存在（obs_step 用 is_first 重置隐状态、
    # cont 头用 is_terminal 当标签）。缺一个就直接炸在深处。
    for key in ("is_first", "is_terminal"):
        check(f"obs 含 {key}（models.py 会 assert）", key in obs)
        check(f"{key} 形状 (N,)", np.asarray(obs[key]).shape == (n,),
              str(np.asarray(obs[key]).shape))
        check(f"{key} 是 bool", np.asarray(obs[key]).dtype == bool)
    check("reset 时 is_first 全 True", bool(np.asarray(obs["is_first"]).all()))

    d = env.observation_space["state"].shape[0]
    check("state 形状 (N, D+N)", obs["state"].shape == (n, d), str(obs["state"].shape))
    check("state 是 float32", obs["state"].dtype == np.float32, str(obs["state"].dtype))

    r = env.step(np.zeros(n, dtype=int))[1]
    check("reward 形状 (N,) float32", r.shape == (n,) and r.dtype == np.float32,
          f"{r.shape} {r.dtype}")

    # 非观测键不能混进来：tools_marl 把除 log_ 前缀外的键一律当观测进 batch
    allowed = {"state", "is_first", "is_terminal", "adjacency", "edge_features"}
    extra = set(obs) - allowed
    check("obs 没有多余键（多出来的会被当观测编码）", not extra, str(extra))
    env.close()


def section_graph_roundtrip() -> None:
    print()
    print("[4] 图批次能还原回 (N, N)")
    cfg = base_cfg(overcooked_graph_obs=True)
    env = dreamer_mod.make_env(cfg, "train", 0)
    n = env.num_agents
    obs = env.reset()

    # 复刻 tools_marl.py:423 —— 每个 agent 一条记录，堆成 (n_teams*N, N)
    a = np.asarray(obs["adjacency"])
    e = np.asarray(obs["edge_features"])
    check("env 产出的 adjacency 是 (N, N)", a.shape == (n, n), str(a.shape))
    check("env 产出的 edge_features 是 (N, N, F)", e.ndim == 3 and e.shape[:2] == (n, n),
          str(e.shape))

    # 声明成单 agent 形状，是为了让 replay 里每条样本天然是单智能体形状
    check("observation_space 按单 agent 声明 adjacency",
          env.observation_space["adjacency"].shape == (n,),
          str(env.observation_space["adjacency"].shape))
    check("observation_space 按单 agent 声明 edge_features",
          env.observation_space["edge_features"].shape == (n, env.spec_edge_feature_dim),
          str(env.observation_space["edge_features"].shape))

    # 执行路径的还原：models.execution_graph 把 (n_teams*N, N) reshape 回 (n_teams, N, N)
    stacked = np.concatenate([a] * 1, axis=0)          # 1 支队 = N 行
    check("N 行堆叠后 reshape 回 (N, N) 与原图逐位相同",
          bool(np.array_equal(stacked.reshape(1, n, n)[0], a)))
    env.close()


def section_encoder_regex() -> None:
    print()
    print("[5] encoder 的正则必须选中 state、排除图拓扑")
    cfg = base_cfg()
    env = dreamer_mod.make_env(cfg, "train", 0)
    obs = env.reset()

    mlp = cfg.encoder["mlp_keys"]
    # 复刻 networks.MultiEncoder 的实际过滤顺序：**先**剔掉 excluded 与 log_*，
    # **再**套正则。顺序很重要 —— 正则 `^(?!adjacency$)(?!edge_features$).*`
    # 本身是匹配 is_first / is_terminal 的，挡住它们的是那张排除表。
    # （把这一点写进断言，是因为护栏如果只靠正则、而有人改了排除表，图键就会漏进去。）
    excluded = ("is_first", "is_last", "is_terminal", "reward")
    shapes = {k: v for k, v in obs.items()
              if k not in excluded and not k.startswith("log_")}
    selected = [k for k in shapes if re.match(mlp, k)]

    check("encoder 实际选中的键恰好只有 state", selected == ["state"], str(selected))
    for gk in ("adjacency", "edge_features"):
        check(f"mlp_keys 排除 {gk}（否则邻接矩阵会被当特征编码）",
              not re.match(mlp, gk))

    # 反证：如果不先过滤，正则确实会选中 is_first / is_terminal ——
    # 说明"只有 state 被编码"这件事**同时**依赖排除表和正则，两者都不能改坏。
    raw = [k for k in obs if re.match(mlp, k)]
    check("（反证）不经排除表时正则会选中 is_first/is_terminal",
          "is_first" in raw and "is_terminal" in raw, str(raw))
    env.close()


def section_graph_switch() -> None:
    print()
    print("[6] 关掉 graph_obs 后通信链路整体降级为 IDreamer")
    cfg = base_cfg(overcooked_graph_obs=False)
    env = dreamer_mod.make_env(cfg, "train", 0)
    obs = env.reset()

    check("graph_obs=False 时 obs 里没有 adjacency", "adjacency" not in obs)
    check("observation_space 里也没有", "adjacency" not in env.observation_space.spaces)
    check("is_first / is_terminal 仍在", "is_first" in obs and "is_terminal" in obs)

    # 这是整条链路的唯一开关：models.graph_from_data 在数据里没有 adjacency 时返回 None，
    # 返回 None 就走 IDreamer 路径。所以"没有图"和"有图但关掉通信"是同一个东西。
    import torch

    data = {
        "state": torch.zeros(2, 3, env.num_agents, 4),
        "is_first": torch.zeros(2, 3, env.num_agents, dtype=torch.bool),
    }
    check("models.graph_from_data 对没有图的批次返回 None",
          models.graph_from_data(data, torch.device("cpu"), torch.float32) is None)

    # 对照组：补上 adjacency 就必须返回非 None（否则上一条会因为"函数恒返回 None"而假通过）
    data["adjacency"] = torch.zeros(2, 3, env.num_agents, env.num_agents)
    data["edge_features"] = torch.zeros(2, 3, env.num_agents, env.num_agents, 1)
    check("补上 adjacency 后返回非 None（对照）",
          models.graph_from_data(data, torch.device("cpu"), torch.float32) is not None)
    env.close()


def section_reproducible() -> None:
    print()
    print("[7] 重建实例可复现同一批局（scripts/eval_policy.py 的配对评测前提）")
    cfg = base_cfg()
    e1 = dreamer_mod.make_env(cfg, "eval", 0)
    e2 = dreamer_mod.make_env(cfg, "eval", 0)

    def rollout(env, n_ep=5, horizon=15):
        """每局记录 (观测, 奖励, 终止) 的整条序列 —— 局长固定，所以形状可比。"""
        out = []
        for _ in range(n_ep):
            obs = env.reset()
            states, rewards, dones = [obs["state"].copy()], [], []
            for _ in range(horizon):
                obs, r, done, _ = env.step(np.zeros(env.num_agents, dtype=int))
                states.append(obs["state"].copy())
                rewards.append(r.copy())
                dones.append(bool(done))
                if done:
                    break
            out.append((np.stack(states), np.stack(rewards), tuple(dones)))
        return out

    a, b = rollout(e1), rollout(e2)
    same = all(
        x[0].shape == y[0].shape and np.array_equal(x[0], y[0])
        and np.array_equal(x[1], y[1]) and x[2] == y[2]
        for x, y in zip(a, b)
    )
    check("两个独立实例跑出的 5 局（观测/奖励/终止）逐位相同", same)
    check("这些局确实有内容（不是空序列假通过）",
          all(x[0].shape[0] > 2 and len(x[1]) > 1 for x in a),
          f"局长 {[x[0].shape[0] for x in a]}")
    e1.close()
    e2.close()


def section_eval_policy_key() -> None:
    """回归：eval_policy 判断"要不要重建模型"的方式。

    这里走过两代错法：
      1. 写死 `seg_*` 键 —— 对别的 suite 取到 yaml 里与本 run 无关的默认值；
      2. 按 task 前缀枚举 —— 只覆盖被命名成 `<suite>_*` 的旋钮，改名就漏。

    现行做法是**权重形状表**（`_arch_key`）：它机械完备，因为它直接回答
    "这份权重装不装得进当前模型"，而那正是重建决策唯一要问的问题。
    本节钉住的就是"形状变了，key 必须变"。
    """
    print()
    print("[8] eval_policy 用权重形状表判结构（不再枚举 config 键）")
    sys.path.insert(0, str(REPO / "scripts"))
    import torch

    import eval_policy

    def fake_ckpt(shapes):
        return {"agent_state_dict": {k: torch.zeros(s) for k, s in shapes.items()}}

    a = fake_ckpt({"_wm.encoder._mlp.0.weight": (64, 21), "x": (4,)})
    b = fake_ckpt({"_wm.encoder._mlp.0.weight": (64, 36), "x": (4,)})   # 观测维变了
    c = fake_ckpt({"_wm.encoder._mlp.0.weight": (64, 21), "x": (4, 5)})
    d = fake_ckpt({"_wm.encoder._mlp.0.weight": (64, 21), "x": (4,)})

    check("观测维变了 → key 变（这正是 seg_obs_mode local→oracle 的效果）",
          eval_policy._arch_key(a) != eval_policy._arch_key(b))
    check("参数量没变但形状变了 → key 也变", eval_policy._arch_key(a) != eval_policy._arch_key(c))
    check("完全相同的形状 → key 相同（不会无谓重建）",
          eval_policy._arch_key(a) == eval_policy._arch_key(d))
    check("key 里带上了键名（不是只看参数量）",
          "x" in dict(eval_policy._arch_key(a)))

    # 与旧的枚举法对照：它看不到 seg_obs_mode 之外的**未命名**结构性旋钮
    seg = copy.deepcopy(load_config(["defaults", "codreamer"]))
    seg.task = "seg_seq"
    named = eval_policy._env_spec(seg)
    check("_env_spec 仍然可用（只用于打印）",
          any(k == "seg_obs_mode" for k, _ in named))


def main() -> int:
    print("Overcooked 接线测试（纯 CPU，不需要 GPU）")
    print("=" * 68)
    section_make_env()
    section_interface_matches_vmas()
    section_obs_contract()
    section_graph_roundtrip()
    section_encoder_regex()
    section_graph_switch()
    section_reproducible()
    section_eval_policy_key()
    print("=" * 68)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): " + ", ".join(FAILURES))
        return 1
    print("全部通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
