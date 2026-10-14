"""Sequential Estimate Game 的 gym 包装 —— 队伍级（team-level）环境。

================================================================================
归属说明
================================================================================
本文件**不属于上游 dreamerv3-torch**（不在 UPSTREAM_MANIFEST.sha256 里），
是 CoDreamer 复现的一部分，由我们编写。

================================================================================
为什么是"队伍级"而不是"单智能体级"
================================================================================
上游 `tools.simulate` 的收集循环是：

    results = [e.step(a) for e, a in zip(envs, action)]

它**逐个**调用每个 env 的 step，且要求**立即**返回。但多智能体的世界必须拿到
**全部 N 个动作**才能前进一步。因此"把每个 agent 伪装成一个独立 env"这条路
走不通：当 env[0].step() 被调用时，它无从得知 env[1..N-1] 的动作。

所以本环境以**整个队伍**为单位对外提供接口：
    reset()          -> 全部 N 个 agent 的观测
    step(actions)    -> 全部 N 个 agent 的观测/奖励/done

数据收集由 `tools_marl.py` 里我们自己的 rollout 负责，它会把每次 step 产生的
N 条记录**分别写入 replay**（N 条独立的 episode 流）—— 这正是 IDreamer 的语义：
每个 agent 有自己的世界模型、只用自己的经验训练、彼此不观测。

================================================================================
与框架的接口约定
================================================================================
`observation_space` / `action_space` 声明的是**单个 agent 的视角**：

    observation_space["state"] = Box(D + N)   D 维观测 + N 维 one-hot agent index
    action_space               = Discrete(A)

声明成单智能体形状是刻意的：上游 `WorldModel.__init__` 用
`{k: tuple(v.shape) for k, v in obs_space.spaces.items()}` 推导 encoder 的输入维度。
因为 N 条记录是**分别**存进 replay 的，训练时每条样本天然就是单智能体形状，
encoder 永远看不到 agent 轴。**这是"reshape 展平"方案的落地点。**

实际 `reset()` / `step()` 返回的数组**带一个 agent 轴**，形状为 (N, ...)，
由我们的 rollout 负责拆开。

================================================================================
one-hot agent index
================================================================================
论文对 IDreamer 的明确要求：
    "we concatenate agent indices as a one-hot encoding to all observations in
     non-visual environments, enabling the world model to differentiate between agents"

因为参数共享，N 个 agent 共用一套世界模型权重；不拼索引的话模型无从知道自己
在服务哪个 agent。本环境无条件拼接（Estimate Game 是向量观测环境）。

================================================================================
奖励语义
================================================================================
返回 shape (N,) 的**逐 agent 局部奖励**。论文在 Estimate Game 上就是用局部奖励
训练的（"we train agents using their local rewards"），不同于 Melting Pot 的
全局求和。`global_mode` 可切换以便做对照。
"""

from __future__ import annotations

from dataclasses import replace

import gym
import numpy as np

from envs.seg import SequentialEstimateGame, Spec

__all__ = ["SequentialEstimateGameEnv"]


# =============================================================================
# 观测模式
# =============================================================================
#   "local"   只看自己： obs = [s_i] + one-hot(i)                维度 = 1 + N
#   "oracle"  直接给全部：obs = [s_0..s_{N-1}] + one-hot(i)      维度 = N + N
#
# oracle 是用来算【上界】的：它能看到所有 agent 的状态，因此能精确算出目标
# y_i = eta*s_i + (1-eta)*mean(neighbours)，必定选对区间、每步得 0 分。
#
# 用途：如果 oracle 与 local 的差距很小，说明这个环境根本没有给通信留空间，
# 那么"加上 GNN 会不会更好"这个问题在此环境上就是不可测的。
OBS_MODES = ("local", "oracle")


# =============================================================================
# 图结构（GNN 需要的拓扑）
# =============================================================================
# CoDreamer 与 IDreamer 的唯一区别就是"通信"。通信必须作用在一张图上，而
# 论文明确规定：**图是环境给的，不是学出来的**（§2.2.1）：
#
#     "an edge is created if ||p_i - p_j|| <= C,
#      where C is the communication range set by the environment"
#
# 所以环境必须吐出拓扑。本环境把它作为两个【逐 agent】的观测键附加在 obs dict
# 里（每个 agent 只拿自己那一行）：
#
#     "adjacency"      (N,)     邻接矩阵的第 i 行 —— 谁是我的邻居
#     "edge_features"  (N, F)   边特征的第 i 行 —— 我与各邻居的距离
#
# 为什么按"行"而不是给整个 (N,N) 矩阵：`tools_marl.simulate_marl` 写入 replay 时
# 是 `{key: val[k] for k in agents}` 逐个 agent 切的。给出 (N,N) 的矩阵，切到
# 第 k 个 agent 自然就是第 k 行 —— 和现有机制完全契合，不需要任何特例。
# 训练时把同一队 N 个 agent 的行堆起来，就还原出完整的 A 和 E。
#
# ⚠️ 这两个键**要声明在 observation_space 里**（否则空间描述不完整），但必须在
# encoder/decoder 的 mlp_keys 里**排除掉** —— 否则 `mlp_keys: '.*'` 会试图把
# 邻接矩阵当成特征去编码。见 configs.yaml 的 codreamer 块。
GRAPH_OBS_KEYS = ("adjacency", "edge_features")


class SequentialEstimateGameEnv:
    """队伍级环境。对外接口刻意贴近 gym，但不继承 gym.Env（多智能体不是 gym 语义）。"""

    metadata = {}

    def __init__(
        self,
        spec: Spec | None = None,
        seed: int | None = None,
        add_agent_index: bool = True,
        obs_mode: str = "local",
        graph_obs: bool = False,
    ) -> None:
        if obs_mode not in OBS_MODES:
            raise ValueError(f"unknown obs_mode: {obs_mode} (expected one of {OBS_MODES})")
        self.spec = spec or Spec()
        self.game = SequentialEstimateGame(self.spec, seed=seed)
        self._add_agent_index = add_agent_index
        self._obs_mode = obs_mode
        self._graph_obs = bool(graph_obs)
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        n = self.spec.num_agents
        d = self.spec.obs_size
        # 单个 agent 的观测维度
        if obs_mode == "oracle":
            d_eff = n * d          # 全部 agent 的状态
        else:
            d_eff = d              # 只有自己的
        agent_dim = d_eff + (n if add_agent_index else 0)

        spaces = {
            "state": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(agent_dim,), dtype=np.float32
            ),
        }
        if self._graph_obs:
            # 邻接矩阵第 i 行（0/1）与边特征第 i 行。
            # 形状都是【单 agent 视角】，与 "state" 的约定一致。
            spaces["adjacency"] = gym.spaces.Box(
                low=0.0, high=1.0, shape=(n,), dtype=np.float32
            )
            spaces["edge_features"] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(n, self.spec_edge_feature_dim),
                dtype=np.float32,
            )
        self.observation_space = gym.spaces.Dict(spaces)
        self.action_space = gym.spaces.Discrete(self.spec.num_actions)
        # 上游约定：离散环境自己给动作空间打这个标记（atari.py / crafter.py /
        # memorymaze.py / OneHotAction 都这么做）。dreamer.py 靠它选择
        # OneHotDist 作为预填阶段的随机策略；配置里也要相应设
        # actor: {dist: 'onehot'}。
        self.action_space.discrete = True
        self.reward_range = (-np.inf, 0.0)

        # 常数：one-hot agent index，形状 (N, N)
        self._agent_onehot = np.eye(n, dtype=np.float32)

    # ------------------------------------------------------------------ 属性

    @property
    def num_agents(self) -> int:
        return self.spec.num_agents

    @property
    def spec_edge_feature_dim(self) -> int:
        return self.game.edge_feature_dim

    @property
    def graph_obs(self) -> bool:
        return self._graph_obs

    @property
    def adjacency(self) -> np.ndarray:
        return self.game.adjacency

    # ------------------------------------------------------------------ 组装观测

    def _make_obs(self, raw: np.ndarray, is_first: bool, is_terminal: bool) -> dict:
        """把游戏返回的 (N, obs_size) 观测组装成框架要的 dict。

        raw 是全部 N 个 agent 的状态。每个 agent 看到的东西取决于 obs_mode：
          local  -> 只有自己那一行
          oracle -> 整个 raw 拍平（所有 agent 的状态）
        然后按需拼上自己的 one-hot 索引。
        """
        n = self.spec.num_agents
        if self._obs_mode == "oracle":
            # (N, obs_size) -> (N, N*obs_size)，每行都是全部 agent 的状态
            flat = raw.reshape(1, -1).repeat(n, axis=0)
        else:
            flat = raw
        parts = [flat]
        if self._add_agent_index:
            parts.append(self._agent_onehot)
        state = np.concatenate(parts, axis=1).astype(np.float32)
        assert state.shape == (n, self.observation_space["state"].shape[0]), (
            state.shape,
            self.observation_space["state"].shape,
        )
        out = {
            "state": state,
            "is_first": np.full(n, is_first, dtype=bool),
            "is_terminal": np.full(n, is_terminal, dtype=bool),
        }
        if self._graph_obs:
            # 只给"我自己那一行" —— rollout 按 agent 切分后，每个 agent 的
            # episode 流里就是它自己的图视图，堆起来即还原整张图。
            out["adjacency"] = self.game.adjacency.astype(np.float32)          # (N, N)
            out["edge_features"] = self.game.edge_features()                    # (N, N, F)
        return out

    # ------------------------------------------------------------------ 接口

    def reset(self) -> dict:
        """重开一局，返回 (N, ·) 形状的观测字典。

        注意：与 gym 不同，这里**没有随机化 seed 的入口** —— 由 rollout 层控制，
        以确保 N 个 agent 看到同一局。
        """
        if self._seed is not None:
            seed = int(self._rng.integers(0, 2**31 - 1))
            raw = self.game.reset(seed=seed)
        else:
            raw = self.game.reset()
        return self._make_obs(raw, is_first=True, is_terminal=False)

    def step(self, actions: np.ndarray) -> tuple[dict, np.ndarray, bool, dict]:
        """走一步。

        Args:
            actions: (N,) 整数动作下标

        Returns:
            obs:  (N, ·) 的观测字典，`is_first` 全 False
            reward: (N,) 逐 agent 局部奖励
            done: bool（整局是否结束，所有 agent 同时结束）
            info: dict
        """
        raw, reward, done, info = self.game.step(actions)
        obs = self._make_obs(raw, is_first=False, is_terminal=bool(done))
        return obs, reward.astype(np.float32), bool(done), info

    def close(self) -> None:
        pass


# ============================================================================
# 自检 —— 本地不跑实验，断言写死在代码里。
#   在 third_party/dreamerv3-torch 目录下： python -m envs.seg_gym
# （不能用 `python envs/seg_gym.py` —— 那样 sys.path[0] 是 envs/，`import envs.seg` 会失败）
# ============================================================================


def _self_test() -> None:
    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        print(f"  [{'ok  ' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
        if not cond:
            failures.append(name)

    print("SequentialEstimateGameEnv 自检")
    print("-" * 68)

    env = SequentialEstimateGameEnv(Spec(), seed=0)
    n, a, d = env.num_agents, env.action_space.n, env.spec.obs_size

    # --- 空间声明必须是【单 agent 视角】
    check("obs space 是单 agent 形状", env.observation_space["state"].shape == (d + n,),
          str(env.observation_space["state"].shape))
    check("action space 是 Discrete", isinstance(env.action_space, gym.spaces.Discrete))
    check("action space 带 .discrete 标记（上游约定）",
          getattr(env.action_space, "discrete", False) is True)
    check("action space size", env.action_space.n == a, str(env.action_space.n))

    # --- reset
    obs = env.reset()
    check("reset 返回的 state 带 agent 轴", obs["state"].shape == (n, d + n), str(obs["state"].shape))
    check("state dtype float32", obs["state"].dtype == np.float32, str(obs["state"].dtype))
    check("is_first 全 True", bool(obs["is_first"].all() and obs["is_first"].shape == (n,)))
    check("is_terminal 全 False", bool((~obs["is_terminal"]).all()))

    # --- one-hot 索引确实拼在后面
    idx = obs["state"][:, d:]
    check("one-hot 索引形状", idx.shape == (n, n), str(idx.shape))
    check("one-hot 是单位阵", bool(np.allclose(idx, np.eye(n, dtype=np.float32))))
    check("one-hot 每行恰好一个 1", bool((idx.sum(1) == 1).all()))

    # --- step
    acts = np.zeros(n, dtype=int)
    obs2, rew, done, info = env.step(acts)
    check("step reward 形状 (N,)", rew.shape == (n,), str(rew.shape))
    check("step reward <= 0", bool((rew <= 1e-9).all()), str(rew))
    check("step done 是 bool", isinstance(done, (bool, np.bool_)))
    check("step obs is_first 全 False", bool((~obs2["is_first"]).all()))
    check("step obs 带 agent 轴", obs2["state"].shape == (n, d + n), str(obs2["state"].shape))
    check("info 带 adjacency", "adjacency" in info)

    # --- done 语义：horizon 步后结束，且 is_terminal 打开
    env.reset()
    for t in range(env.spec.horizon):
        obs_t, rew_t, done_t, _ = env.step(np.full(n, t % a, dtype=int))
    check("horizon 步后 done", bool(done_t))
    check("结束时 is_terminal 全 True", bool(obs_t["is_terminal"].all()))

    # --- 动作越界要报错（而不是静默出错）
    env.reset()
    for bad in (np.full(n, -1), np.full(n, a)):
        try:
            env.step(bad)
            check(f"越界动作 {bad[0]} 应报错", False)
        except ValueError:
            check(f"越界动作 {bad[0]} 报错", True)

    # --- 动作形状错误要报错
    try:
        env.reset()
        env.step(np.zeros(n + 1, dtype=int))
        check("错误动作形状应报错", False)
    except ValueError:
        check("错误动作形状报错", True)

    # --- 关闭索引拼接后维度变窄
    env2 = SequentialEstimateGameEnv(Spec(), seed=0, add_agent_index=False)
    o2 = env2.reset()
    check("关掉索引后维度 = D", o2["state"].shape == (n, d), str(o2["state"].shape))
    check("关掉索引后空间也变窄",
          env2.observation_space["state"].shape == (d,),
          str(env2.observation_space["state"].shape))

    # --- 一局内 N 个 agent 各有自己的局部状态（不应该相等），但共享同一张邻接矩阵
    env3 = SequentialEstimateGameEnv(Spec(), seed=0)
    o3 = env3.reset()
    locals3 = o3["state"][:, :d]
    check("N 个 agent 的局部状态各不相同",
          len({tuple(v) for v in locals3}) == n, str(locals3.ravel()))
    adj = env3.adjacency
    check("整个队伍共享一张邻接矩阵", adj.shape == (n, n), str(adj.shape))
    check("邻接矩阵无自环", bool(not np.diag(adj).any()))

    # --- 图结构（graph_obs）—— CoDreamer 的 GNN 需要拓扑
    envg = SequentialEstimateGameEnv(Spec(), seed=0, graph_obs=True)
    check("graph_obs 声明 adjacency", "adjacency" in envg.observation_space.spaces)
    check("graph_obs 声明 edge_features", "edge_features" in envg.observation_space.spaces)
    check(
        "graph_obs: adjacency 空间是单 agent 形状 (N,)",
        envg.observation_space["adjacency"].shape == (n,),
        str(envg.observation_space["adjacency"].shape),
    )
    check(
        "graph_obs: edge_features 空间是单 agent 形状 (N,F)",
        envg.observation_space["edge_features"].shape == (n, 1),
        str(envg.observation_space["edge_features"].shape),
    )
    og = envg.reset()
    check("graph_obs: reset 带 agent 轴 (N,N)",
          og["adjacency"].shape == (n, n), str(og["adjacency"].shape))
    check("graph_obs: edge_features 带 agent 轴 (N,N,F)",
          og["edge_features"].shape == (n, n, 1), str(og["edge_features"].shape))
    check("graph_obs: dtype float32", og["adjacency"].dtype == np.float32,
          str(og["adjacency"].dtype))
    # 关键性质：每个 agent 那一行堆起来 == 完整的 A（这是"按 agent 存、按队还原"的基础）
    check(
        "graph_obs: N 行可还原完整邻接矩阵",
        bool(np.array_equal(og["adjacency"], envg.adjacency.astype(np.float32))),
    )
    # step 之后图仍在（且在 episode 内不变）
    og2, _, _, _ = envg.step(np.zeros(n, dtype=int))
    check("graph_obs: step 后仍带图", "adjacency" in og2)
    check("graph_obs: episode 内邻接不变",
          bool(np.array_equal(og["adjacency"], og2["adjacency"])))
    check("graph_obs: state 维度不受影响",
          og["state"].shape == (n, d + n), str(og["state"].shape))
    # 默认关闭时不能出现这两个键（保证旧行为逐位不变）
    envl = SequentialEstimateGameEnv(Spec(), seed=0)
    ol = envl.reset()
    check("默认不带图结构键（旧行为不变）",
          not any(k in ol for k in GRAPH_OBS_KEYS), str(list(ol)))
    check("默认 observation_space 不含图结构",
          not any(k in envl.observation_space.spaces for k in GRAPH_OBS_KEYS))

    print("-" * 68)
    if failures:
        print(f"FAILED ({len(failures)}): {', '.join(failures)}")
        raise SystemExit(1)
    print("全部通过。")


if __name__ == "__main__":
    _self_test()
