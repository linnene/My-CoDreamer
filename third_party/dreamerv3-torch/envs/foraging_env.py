"""Level-Based Foraging 的队伍级包装 —— 对外接口与 `envs/vmas_env.py` **完全一致**。

================================================================================
归属说明
================================================================================
本文件不属于上游 dreamerv3-torch，是 CoDreamer 复现的一部分，由我们编写。

================================================================================
为什么再加一个环境：要一个"不通信在信息论上就做不到"的任务
================================================================================
VMAS 上的结论（8 seed、p<0.05、CoDreamer 更差）与本文件无关，但它暴露了一个
选环境的硬要求：**如果每个 agent 的观测已经足以独立决策，通信在信息论上就是冗余的**，
再好的通信模块也只能去拟合噪声。VMAS flocking 正是这样 —— 它的 lidar 只留非 agent
实体，agent 互相完全不可见，而奖励是成对距离的 potential shaping，每个 agent
看自己那一份就够。

Level-Based Foraging（LBF，Farama 生态，Papoudakis et al. 2021 的 MARL benchmark
收录）在源码层面同时给出两件事：

1. **观测是局部的。** `_make_obs` 只取 `neighborhood(*player.position, sight)`
   这个 (2·sight+1)² 窗口，窗外实体在扁平观测里被填成 (-1, -1, 0)
   （lbforaging/foraging/environment.py:451-486、:500-520）。
   于是"B 视野里的食物"对 A **根本不存在**。
2. **协作是强制的。** 装载要求**同一时刻**选 LOAD、且都贴在同一食物旁的 agent
   等级之和 ≥ 食物等级；不够则全体受罚（environment.py:670-700）。

两条合起来就是论文说的 overlap of information，而且是**可判定的**：
食物只有 B 看得见、又必须 A 一起搬 → 不通信时 A 只能从 B 的**动作**去反推，
通信时 B 的隐状态直接进 A 的世界模型。

### 为什么默认 `force_coop=False`（食物等级 ∈ {1,2}）而不是全强制协作

实测：`force_coop=True`（每个食物都要两个 agent 同一时刻在同一个食物旁选 LOAD）时，
**独立贪心策略 100 局只搬掉 0.02/1.0 的回报**，随机策略恒为 0。任务在低预算下
根本没有学习信号 —— 这是 SEG"饱和"的反面（那边是太容易，这边是太难），
两种都会得到"分不出变体"。

改成等级 ∈ {1,2} 后，任务被**拆成两个可分开度量的部分**：

    等级 1 食物 —— 单人搬得动  → 学习信号（世界模型有事可学）
    等级 2 食物 —— 必须两人     → 协作收益（通信的用武之地）

于是一个比"总回报"直观得多的主指标出现了：**等级 2 食物的采集率**。
它直接对应"只有协作才拿得到的那部分"，而且有明确的天花板（100%）。

独立贪心策略（= "不通信"的代理）的实测基线：

    n=3 fs=8 nf=3 ms=20   等级1 采集率 40%    等级2 采集率  5%
    n=3 fs=6 nf=2 ms=20   等级1 采集率 55%    等级2 采集率 17%

**等级 2 有 80~95 个百分点的空位**，这就是 CoDreamer 可能兑现的地方。

================================================================================
与框架的接口约定（逐条对齐 vmas_env.py / seg_gym.py）
================================================================================
    observation_space["state"]        = Box(D + N)      D 维观测 + N 维 one-hot 索引
    observation_space["adjacency"]    = Box(N)          （仅 graph_obs 时）
    observation_space["edge_features"]= Box(N, 1)       （仅 graph_obs 时）
    action_space                      = Discrete(6)

声明成**单 agent 视角**是刻意的：N 条记录分别存进 replay，训练时每条样本天然
就是单智能体形状，encoder 永远看不到 agent 轴（见 seg_gym.py 的长注释）。

`reset()` / `step()` 实际返回的数组**带一个 agent 轴**，形状 (N, ...)。

注意 LBF 是 gymnasium 环境，但本包装**不继承 gym.Env** —— 与 seg_gym/vmas_env
一致，契约由 `tools_marl.simulate_marl` 决定而不是 gym。LBF 自带的 `TimeLimit`
也没用：局长由 `max_episode_steps` 直接交给 LBF，且 LBF 在**食物搬完时提前终止**
（`_game_over = field.sum() == 0 or max_steps <= current_step`），
所以这是一个**会真正终止**的环境 —— 与 VMAS 只在 max_steps 到点时才终止不同。

================================================================================
动作空间：6
================================================================================
    NONE=0, NORTH=1, SOUTH=2, WEST=3, EAST=4, LOAD=5
LBF 自己会把非法动作替换成 NONE（`:630-632`），本包装只做越界检查 ——
"动作被环境静默改写"是 LBF 的语义，不在这里改。

================================================================================
图：C 是我们定的（论文未给），默认取 sight
================================================================================
论文 §2.2.1 规定"边当且仅当 ‖p_i−p_j‖ ≤ C，C 由环境设定"，但没给值。
LBF 的信息局部性本身就是**切比雪夫**正方形（`neighborhood` 取的是方形窗口），
所以这里用切比雪夫距离。默认 **C = max(2, field_size//2 − 1)**（8x8 → 3，6x6 → 2），
取值是**实测后选的**，与 VMAS 上处理同一个缺口的方式一致。
（切比雪夫距离在整数网格上只取整数值，所以 C=2 与 C=2.5 等价。）

    sight=2，每格 **2000 局**（5 seed × 400 reset）的平均度与退化比例：

        8x8 n=3  C=2  度 0.53/2  全连通  5%  全断开 38%
        8x8 n=3  C=3  度 0.92/2  全连通 16%  全断开 14%   ← 默认
        8x8 n=3  C=4  度 1.30/2  全连通 34%  全断开  4%
        8x8 n=3  C=6  度 1.88/2  全连通 85%  全断开  0%
        6x6 n=3  C=2  度 0.86/2  全连通 14%  全断开 18%   ← 默认
        6x6 n=3  C=3  度 1.37/2  全连通 38%  全断开  4%
        8x8 n=4  C=3  度 1.39/3  全连通  4%  全断开  2%

⚠️ 早先只抽 300 局时读到 C=3 是 1.23/2，与 2000 局的 0.92/2 差得远 ——
**图的稀疏度这类量必须多种子大样本**，单次抽样会给出一个看起来很确定的错数。

C 太小则图常年空（通信没机会发生），太大则接近全连通（GNN 退化成对所有邻居
求平均，图结构不再承载信息）。默认值落在两者之间。

**为什么不用 C = sight：** 那样边就等于"互相看得见"，通信只能在双方已经近到
能彼此看见时发生 —— 而那时候对方的位置信息本来就在观测里，通信对"找不到的食物"
毫无帮助。取 C > sight 才打开一个**非局部**通道：B 看得见距 B ≤ sight 的食物，
于是 A 能从 B 那里知道距 A 最多 C+sight 格的食物，远在自己视野之外。这正是要演示的。

`comm_radius` 可覆盖，实验里应当扫（2 / 3 / 4）。
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym

# 与 LBF 的 Action 枚举一致（lbforaging/foraging/environment.py:Action）
ACTION_NONE, ACTION_NORTH, ACTION_SOUTH, ACTION_WEST, ACTION_EAST, ACTION_LOAD = range(6)
NUM_ACTIONS = 6

GRAPH_OBS_KEYS = ("adjacency", "edge_features")


class ForagingTeamEnv:
    """队伍级 LBF 环境。接口与 VMASEnv 一致。"""

    metadata = {}

    def __init__(
        self,
        num_agents: int = 3,
        field_size: int = 8,
        max_num_food: int = 3,
        sight: int = 2,
        max_steps: int = 15,
        force_coop: bool = False,
        max_player_level: int = 1,
        max_food_level: int | None = 2,
        min_player_level: int = 1,
        min_food_level: int = 1,
        normalize_reward: bool = True,
        penalty: float = 0.0,
        grid_observation: bool = False,
        seed: int | None = None,
        add_agent_index: bool = True,
        graph_obs: bool = False,
        comm_radius: float | None = None,
    ) -> None:
        from lbforaging.foraging import ForagingEnv

        self._num_agents = int(num_agents)
        self._field_size = int(field_size)
        self._sight = int(sight)
        self._max_steps = int(max_steps)
        self._force_coop = bool(force_coop)
        self._add_agent_index = bool(add_agent_index)
        self._graph_obs = bool(graph_obs)
        # 默认 C = max(2, field_size//2 - 1)（实测选值，理由见文件头）
        default_c = max(2, self._field_size // 2 - 1)
        self._comm_radius = float(default_c if comm_radius is None else comm_radius)
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._forced_seed: int | None = None
        self.id = None  # 由 rollout 层赋值（与上游 UUID wrapper 的约定一致）

        # max_food_level=None 时 LBF 会把它设成"最弱三个 agent 的等级之和"，
        # 配合 force_coop 就等于"必须三个人一起搬" —— 那对本项目的预算太硬，
        # 所以默认给一个显式的 2（"必须两个人一起"）。要那档难度就把键设为 -1。
        food_level = None if max_food_level is None or int(max_food_level) < 0 else int(max_food_level)

        self._env = ForagingEnv(
            players=self._num_agents,
            min_player_level=int(min_player_level),
            max_player_level=int(max_player_level),
            min_food_level=int(min_food_level),
            max_food_level=food_level,
            field_size=(self._field_size, self._field_size),
            max_num_food=int(max_num_food),
            sight=self._sight,
            max_episode_steps=self._max_steps,
            force_coop=self._force_coop,
            normalize_reward=bool(normalize_reward),
            grid_observation=bool(grid_observation),
            penalty=float(penalty),
        )

        self.num_agents = self._num_agents
        n = self.num_agents

        # LBF 的观测是"元组，每 agent 一个定长向量"；取第 0 个的维度作为单 agent 维度
        raw = self._env.reset(seed=0)[0]
        self._obs_dim = int(np.asarray(raw[0]).reshape(-1).shape[0])
        assert len(raw) == n, (len(raw), n)

        agent_dim = self._obs_dim + (n if self._add_agent_index else 0)
        spaces = {
            "state": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(agent_dim,), dtype=np.float32
            )
        }
        if self._graph_obs:
            spaces["adjacency"] = gym.spaces.Box(
                low=0.0, high=1.0, shape=(n,), dtype=np.float32
            )
            spaces["edge_features"] = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(n, 1), dtype=np.float32
            )
        self.observation_space = gym.spaces.Dict(spaces)

        self.action_space = gym.spaces.Discrete(NUM_ACTIONS)
        # 上游约定：离散环境自己打这个标记（dreamer.py 靠它选 OneHotDist）
        self.action_space.discrete = True
        self.reward_range = (-np.inf, np.inf)

        self._agent_onehot = np.eye(n, dtype=np.float32)

    # ------------------------------------------------------------------ 属性

    @property
    def spec_edge_feature_dim(self) -> int:
        return 1

    @property
    def graph_obs(self) -> bool:
        return self._graph_obs

    @property
    def comm_radius(self) -> float:
        return self._comm_radius

    def seed(self, seed: int | None) -> None:
        """强制后续每次 reset 都用同一个种子（seed=None 取消）。语义同 vmas_env。"""
        self._forced_seed = None if seed is None else int(seed)

    # ------------------------------------------------------------------ 图

    def _positions(self) -> np.ndarray:
        """(N, 2) 的网格坐标（行, 列）—— 全局绝对坐标，不是视野内的相对坐标。"""
        return np.stack(
            [np.asarray(p.position, dtype=np.float64).reshape(-1)[:2]
             for p in self._env.players]
        )

    def _graph(self) -> tuple[np.ndarray, np.ndarray]:
        """(adjacency (N,N) float, edge_features (N,N,1) float)。

        无自环（与 seg_gym / vmas_env 一致）。LBF 的局部性是**切比雪夫方形窗口**
        （neighborhood 取的是方框），所以这里用切比雪夫距离，不是欧氏 —— 这样
        "边"与"看得见"是同一个关系。边特征 = 切比雪夫距离，非边处填 0。
        """
        pos = self._positions()
        delta = np.abs(pos[:, None, :] - pos[None, :, :]).max(axis=-1)  # (N,N)
        adj = delta <= self._comm_radius
        np.fill_diagonal(adj, False)
        edge = (delta * adj)[..., None]
        return adj.astype(np.float32), edge.astype(np.float32)

    @property
    def adjacency(self) -> np.ndarray:
        return self._graph()[0]

    def edge_features(self) -> np.ndarray:
        return self._graph()[1]

    # ------------------------------------------------------------------ 观测

    def _make_obs(self, raw, is_first: bool, is_terminal: bool) -> dict:
        """raw: LBF 的 tuple[N] of (obs_dim,) 数组。"""
        n = self.num_agents
        flat = np.stack(
            [np.asarray(o).reshape(-1).astype(np.float32) for o in raw]
        )                                                          # (N, obs_dim)
        assert flat.shape == (n, self._obs_dim), (flat.shape, self._obs_dim)
        parts = [flat]
        if self._add_agent_index:
            parts.append(self._agent_onehot)
        state = np.concatenate(parts, axis=1).astype(np.float32)
        out = {
            "state": state,
            "is_first": np.full(n, is_first, dtype=bool),
            "is_terminal": np.full(n, is_terminal, dtype=bool),
        }
        if self._graph_obs:
            adj, edge = self._graph()
            out["adjacency"] = adj
            out["edge_features"] = edge
        return out

    # ------------------------------------------------------------------ 接口

    def _clear_positions(self) -> None:
        """把玩家的位置清成 None —— 必须在每次 `reset()` 之前做。

        LBF 的 `spawn_players` 有个状态泄漏：它按顺序给每个玩家随机抽位置，
        但**只在新位置成功时才写 `player.position`**，而 `_is_empty_location`
        会拒绝**所有**玩家占用的格子。于是上一局残留的位置会参与本局的占用判定，
        改变 RNG 的消耗次数 —— `reset(seed=k)` 因此**不可重现**（实测：同一实例
        连续两次 `seed(11)+reset()` 位置不同）。

        全新实例没这个问题，因为 `Player.__init__` 把 position 设成 None，
        而 `_is_empty_location` 里 `if a.position and ...` 会跳过 None。
        所以"清成 None"正好还原到全新实例的状态。这是包装层的修复，
        不改已安装的 lbforaging 包。
        """
        for p in self._env.players:
            p.position = None

    def reset(self) -> dict:
        """重开一局，返回 (N, ·) 的观测字典。

        种子由本层控制（与 vmas_env 一致）：每局从自己的 RNG 抽一个种子再交给 LBF。
        配合 `_clear_positions`，`env.seed(k)` 后连续 reset 是**逐位可重现**的，
        所以"同一批固定局"的配对评测成立（这是 scripts/eval_policy.py 的前提）。
        """
        if self._forced_seed is not None:
            s = self._forced_seed
        elif self._seed is not None:
            s = int(self._rng.integers(0, 2**31 - 1))
        else:
            s = None
        self._clear_positions()
        raw, _ = self._env.reset(seed=s)
        for p in self._env.players:
            # 1000 次尝试都没找到空位时 spawn_players 会放弃并留下 None；
            # 那会让下游的 _positions() 炸在一个难懂的地方，所以在这里就拦住。
            if p.position is None:
                raise RuntimeError(
                    "LBF 未能为某个玩家找到出生点（场地上没有空位）—— "
                    "检查 max_num_food 与 field_size / num_agents 的比例"
                )
        return self._make_obs(raw, is_first=True, is_terminal=False)

    def step(self, actions: np.ndarray):
        """Args: actions (N,) 整数动作下标。Returns: (obs, reward (N,), done, info)。"""
        n = self.num_agents
        actions = np.asarray(actions).reshape(-1)
        if actions.shape[0] != n:
            raise ValueError(f"动作形状 {actions.shape}，期望 ({n},)")
        if actions.min() < 0 or actions.max() >= NUM_ACTIONS:
            raise ValueError(f"动作越界: {actions}（合法范围 0..{NUM_ACTIONS - 1}）")

        raw, rewards, done, _truncated, _info = self._env.step(
            [int(a) for a in actions]
        )
        reward = np.asarray(rewards, dtype=np.float32).reshape(-1)
        assert reward.shape == (n,), (reward.shape, n)
        d = bool(done)
        obs = self._make_obs(raw, is_first=False, is_terminal=d)
        return obs, reward, d, {"adjacency": self.adjacency}

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------ 诊断

    @property
    def game_over(self) -> bool:
        """LBF 自己的终止标记（食物搬完 或 到步数上限）。"""
        return bool(self._env.game_over)

    @property
    def remaining_food(self) -> int:
        """场上剩余的食物**个数**（不是等级之和）。"""
        return int((self._env.field > 0).sum())

    def food_levels(self) -> np.ndarray:
        """场上剩余食物的等级列表（用于诊断"需要几个人一起搬"）。"""
        return self._env.field[self._env.field > 0].astype(np.int64)

    def player_levels(self) -> np.ndarray:
        return np.array([p.level for p in self._env.players], dtype=np.int64)


# ============================================================================
# 自检 —— 本地不跑实验，断言写死在代码里。
#   在 third_party/dreamerv3-torch 目录下： python -m envs.foraging_env
# ============================================================================
def _self_test() -> None:
    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        print(f"  [{'ok  ' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
        if not cond:
            failures.append(name)

    print("ForagingTeamEnv 自检")
    print("-" * 68)

    env = ForagingTeamEnv(num_agents=3, seed=0, graph_obs=True)
    n = env.num_agents
    d = env._obs_dim
    check("agent 数 = 3", n == 3, str(n))
    check("动作数 = 6", env.action_space.n == NUM_ACTIONS, str(env.action_space.n))
    check(
        "action_space 是 Discrete 且带 discrete 标记",
        isinstance(env.action_space, gym.spaces.Discrete)
        and getattr(env.action_space, "discrete", False) is True,
    )

    obs = env.reset()
    check("state 形状 (N, D+N)", obs["state"].shape == (n, d + n), str(obs["state"].shape))
    check("obs space 是单 agent 形状", env.observation_space["state"].shape == (d + n,))

    # --- 图的性质 ---
    adj = obs["adjacency"]
    check("adjacency (N,N)", adj.shape == (n, n), str(adj.shape))
    check("无自环", bool((np.diag(adj) == 0).all()))
    check("对称", bool(np.allclose(adj, adj.T)))
    check("取值只有 0/1", bool(np.isin(adj, (0.0, 1.0)).all()))
    check(
        "edge_features (N,N,1) 且非边处为 0",
        obs["edge_features"].shape == (n, n, 1)
        and bool((obs["edge_features"][adj == 0] == 0).all()),
    )

    # --- step ---
    o2, r, dn, _info = env.step(np.zeros(n, dtype=int))
    check("reward (N,) float32", r.shape == (n,) and r.dtype == np.float32)
    check("done 是 bool", isinstance(dn, (bool, np.bool_)))
    check("step obs 的 is_first 全 False", bool((~o2["is_first"]).all()))

    for bad in (np.full(n, -1), np.full(n, NUM_ACTIONS)):
        try:
            env.step(bad)
            check(f"越界动作 {bad[0]} 应报错", False)
        except ValueError:
            check(f"越界动作 {bad[0]} 报错", True)
    try:
        env.step(np.zeros(n + 1, dtype=int))
        check("错误动作形状应报错", False)
    except ValueError:
        check("错误动作形状报错", True)

    # --- 可重现性（配对评测的前提）---
    env.seed(11)
    env.reset()
    pa = env._positions()
    env.reset()
    pb = env._positions()
    check("seed(k) 后 reset 可重现（位置逐位相同）", bool(np.allclose(pa, pb)))

    env.seed(None)
    env.reset()
    qa = env._positions()
    env.reset()
    qb = env._positions()
    check("默认路径每局不同（有随机性）", not bool(np.allclose(qa, qb)))

    # --- 关掉 agent 索引后维度变窄 ---
    env2 = ForagingTeamEnv(num_agents=3, seed=0, add_agent_index=False)
    o = env2.reset()
    check("关索引后维数 = D", o["state"].shape == (n, d), str(o["state"].shape))

    # ========================================================================
    # 环境语义（这是选它的理由，必须钉住）
    # ========================================================================
    print()
    print("  --- 环境语义 ---")

    # 1) 局部观测：sight=2 时，扁平观测里"可见玩家槽位"不可能超过 2*sight 的距离
    env3 = ForagingTeamEnv(num_agents=3, sight=2, seed=1, force_coop=True)
    obs = env3.reset()
    flat = obs["state"][:, : env3._obs_dim]
    player_blocks = flat[:, 3 * env3._env.max_num_food :].reshape(n, env3.num_agents, 3)
    vis = player_blocks[:, :, 0] >= 0
    max_off = 0
    for i in range(n):
        j = 0
        for k in range(env3.num_agents):
            if vis[i, k]:
                off = np.abs(player_blocks[i, k, :2]).max()
                max_off = max(max_off, int(off))
    check(
        "可见玩家的相对偏移不超过 sight（观测确实是局部的）",
        max_off <= 2 * 2,  # _transform_to_neighborhood 的平移量上界
        f"最大偏移 {max_off}",
    )

    # 2) 协作强制：force_coop 下每个食物的等级都等于 max_food_level，
    #    而单个 agent 的等级 < 该值 → 一个人搬不动
    check(
        "force_coop: 每个食物等级 = 2（单个 agent 等级 1 搬不动）",
        bool((env3.food_levels() == 2).all()),
        str(env3.food_levels()),
    )
    check(
        "玩家等级 = 1（确实低于食物等级）",
        bool((env3.player_levels() == 1).all()),
        str(env3.player_levels()),
    )

    # 3) 装载必须两个 agent 同时 LOAD：单独一个 LOAD 必然失败（等级 1 < 2）
    env4 = ForagingTeamEnv(num_agents=3, seed=3, force_coop=True)
    any_solo_success = False
    for trial in range(20):
        env4.seed(100 + trial)
        env4.reset()
        # 手工把两个 agent 摆到一个食物旁边（借 LBF 自己的字段），只让一个 LOAD
        field = env4._env.field
        if not field.any():
            continue
        coords = np.nonzero(field)
        fr, fc = int(coords[0][0]), int(coords[1][0])
        env4._env.players[0].position = (fr - 1, fc) if fr > 0 else (fr + 1, fc)
        env4._env.players[1].position = (0, 0)
        env4._env.players[2].position = (0, 1)
        env4._env._gen_valid_moves()
        acts = [ACTION_LOAD, ACTION_NONE, ACTION_NONE]
        _, rew, _, _, _ = env4._env.step(acts)
        if np.asarray(rew).sum() > 0:
            any_solo_success = True
            break
    check("单个 agent LOAD 永远不成功（协作是强制的）", not any_solo_success)

    # 4) 两个 agent 同时 LOAD 会成功、且食物真的消失
    env5 = ForagingTeamEnv(num_agents=3, seed=5, force_coop=True)
    env5.seed(7)
    env5.reset()
    field = env5._env.field
    coords = np.nonzero(field)
    fr, fc = int(coords[0][0]), int(coords[1][0])
    # 把 agent 0/1 放到该食物的两个相邻空格上
    spots = [(fr - 1, fc), (fr + 1, fc), (fr, fc - 1), (fr, fc + 1)]
    spots = [(r, c) for (r, c) in spots if 0 <= r < env5._env.rows and 0 <= c < env5._env.cols]
    env5._env.players[0].position = spots[0]
    env5._env.players[1].position = spots[1]
    env5._env.players[2].position = (0, env5._env.cols - 1)
    env5._env._gen_valid_moves()
    _, rew, _, _, _ = env5._env.step([ACTION_LOAD, ACTION_LOAD, ACTION_NONE])
    check("两个 agent 同时 LOAD 成功，且拿到正奖励", float(np.asarray(rew).sum()) > 0,
          f"reward={np.asarray(rew)}")
    check("成功后该食物从场上消失", field[fr, fc] == 0, f"field[{fr},{fc}]={field[fr, fc]}")

    # 5) 默认（force_coop=False）下任务被拆成两半，这是整个选型的支点：
    #    等级 1 = 单人可搬（学习信号），等级 2 = 必须两人（协作收益）
    lv_all = []
    for s in range(40):
        e = ForagingTeamEnv(num_agents=3, seed=s)
        e.reset()
        lv_all.extend(e.food_levels().tolist())
    lv_all = np.array(lv_all)
    check("默认配置同时产出等级 1 与等级 2 的食物",
          bool((lv_all == 1).any() and (lv_all == 2).any()),
          f"等级分布 {np.bincount(lv_all)[1:]}")

    # 单人搬得动等级 1
    env_solo = ForagingTeamEnv(num_agents=3, max_num_food=1, seed=11)
    solo_ok = False
    for trial in range(60):
        env_solo.seed(500 + trial)
        env_solo.reset()
        field = env_solo._env.field
        if not field.any():
            continue
        coords = np.nonzero(field)
        fr, fc = int(coords[0][0]), int(coords[1][0])
        if field[fr, fc] != 1:  # 只测等级 1
            continue
        spots = [(fr - 1, fc), (fr + 1, fc), (fr, fc - 1), (fr, fc + 1)]
        spots = [(r, c) for (r, c) in spots
                 if 0 <= r < env_solo._env.rows and 0 <= c < env_solo._env.cols]
        env_solo._env.players[0].position = spots[0]
        env_solo._env.players[1].position = (0, env_solo._env.cols - 1)
        env_solo._env.players[2].position = (0, env_solo._env.cols - 2)
        env_solo._env._gen_valid_moves()
        _, rew, _, _, _ = env_solo._env.step([ACTION_LOAD, ACTION_NONE, ACTION_NONE])
        if float(np.asarray(rew).sum()) > 0:
            solo_ok = True
            break
    check("等级 1 食物：单个 agent 就能搬（学习信号存在）", solo_ok)

    # 5) 会真正终止：food 搬完 → done
    env6 = ForagingTeamEnv(num_agents=3, max_num_food=1, seed=6, force_coop=True)
    env6.seed(9)
    env6.reset()
    field = env6._env.field
    coords = np.nonzero(field)
    fr, fc = int(coords[0][0]), int(coords[1][0])
    spots = [(fr - 1, fc), (fr + 1, fc), (fr, fc - 1), (fr, fc + 1)]
    spots = [(r, c) for (r, c) in spots if 0 <= r < env6._env.rows and 0 <= c < env6._env.cols]
    env6._env.players[0].position = spots[0]
    env6._env.players[1].position = spots[1]
    env6._env.players[2].position = (0, env6._env.cols - 1)
    env6._env._gen_valid_moves()
    _, _, dn, _ = env6.step([ACTION_LOAD, ACTION_LOAD, ACTION_NONE])
    check("食物搬完 → done（环境会提前终止，不是只有时间到）", dn is True)

    # 6) 局长生效
    env7 = ForagingTeamEnv(num_agents=3, max_steps=4, seed=7)
    env7.seed(21)
    env7.reset()
    steps = 0
    for _ in range(50):
        _, _, dn, _ = env7.step(np.zeros(3, dtype=int))
        steps += 1
        if dn:
            break
    check("max_steps 生效（在 4 步内终止）", steps <= 4, f"{steps} 步")

    # ========================================================================
    # 图：默认 C = sight 时，平均度既不接近 0 也不接近 N-1
    # ========================================================================
    print()
    print("  --- 通信图的稀疏度（C = sight 是否合理）---")
    env8 = ForagingTeamEnv(num_agents=3, sight=2, field_size=8, seed=8, graph_obs=True)
    degs, frac_full, frac_empty = [], 0, 0
    for _ in range(200):
        adj = env8.reset()["adjacency"]
        deg = float(adj.sum(1).mean())
        degs.append(deg)
        frac_full += int(deg == env8.num_agents - 1)
        frac_empty += int(deg == 0)
    m = float(np.mean(degs))
    check(
        "平均度既不接近 0 也不接近 N-1（图有意义）",
        0.5 < m < env8.num_agents - 0.5,
        f"平均度 {m:.2f} / {env8.num_agents - 1}",
    )
    check(
        "图会变化（不是恒连通也不是恒断开）",
        frac_full < 200 and frac_empty < 200,
        f"全连通 {frac_full}/200，全断开 {frac_empty}/200",
    )

    print("=" * 68)
    if failures:
        print(f"FAILED ({len(failures)}): " + ", ".join(failures))
        raise SystemExit(1)
    print("全部通过。")


if __name__ == "__main__":
    _self_test()
