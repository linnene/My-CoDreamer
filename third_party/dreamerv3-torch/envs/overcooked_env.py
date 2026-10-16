"""Overcooked（胡闹厨房）的队伍级包装 —— 对外接口与 `envs/vmas_env.py` 一致。

================================================================================
归属说明
================================================================================
本文件不属于上游 dreamerv3-torch，是 CoDreamer 复现的一部分，由我们编写。
底层是 `overcooked_ai_py`（HumanCompatibleAI，MIT），v1.1.0。

================================================================================
为什么是 Overcooked
================================================================================
需求是"一款家喻户晓的多人协作游戏，把算法接上去，让通信在起作用这件事直接
看得出来"。它的优势是不需要解释：洋葱、锅、盘子、上菜口，谁在配合、谁在原地
打转，一眼就分得出。

但它有一个**必须自己解决的问题**：**Overcooked 默认是全可观的。** 官方 featurize
给的是整个厨房的状态（所有玩家位置/朝向/手持、所有台面上的东西），那种观测下
通信在信息论上是**冗余**的 —— 每个 agent 看自己那一份就够。这正是 VMAS 上分不出
变体、以及本项目在最开始那个网格环境上要避开的那件事。

所以本包装**不使用**官方的 `lossless_state_encoding` / `featurize_state`，
而是自己写一份**限视野**特征（见下）：每个 agent 只看得到自己周围 `sight` 半径
内的格子，窗外的一切（包括队友）对它不存在。信息不对称由此是**构造出来的**，
不是学出来的。

================================================================================
只依赖 numpy：直接驱动 OvercookedGridworld，不用 OvercookedEnv
================================================================================
`overcooked_ai_py.mdp.overcooked_env` 顶层 `import gym, tqdm`，而本项目装的是
`gym 0.22.0`（vmas 的依赖）。`overcooked_ai_py.mdp.overcooked_mdp` **只依赖
numpy**。所以这里直接用 `OvercookedGridworld`：

    起局    mdp.get_random_start_state_fn(...)  或  mdp.get_standard_start_state()
    转移    mdp.get_state_transition(state, joint_action) -> (new_state, infos)
    奖励    infos["sparse_reward_by_agent"]      （上菜才给，长度 = 玩家数）
    终止    mdp.is_terminal(state)  或  自己数步数

顺带避开了官方 `lossless_state_encoding` 里那处 `num_players == 2` 的假设。

================================================================================
短局长：靠 cook_time，而不是靠拉长 episode
================================================================================
Overcooked 煮一锅汤默认要 `cook_time`（通常 20）个 tick。若局长只有 20 步，
**一锅汤都煮不完**，任务在语义上就是空的 —— 这正是调研里 Melting Pot 被否掉的
原因（`COOKING_TIME = 20` 对上 15~30 帧的回合）。

`cook_time` 是**可配的**（`overcooked_mdp.py:210`：`if 'cook_time' in conf`），
所以默认取 **5**：保留"取洋葱 → 下锅 → 等 → 装盘 → 上菜"的完整结构，但不为
纯等待留出 20 步。局长因此可以压到 20~25，而 `prefill` 必须 ≥ 一局长度，
短局长同时让 prefill 便宜。

**配方（`recipe`）同理默认只用 1 个洋葱**，而不是经典的 3 个。原因同样是步数：
3 个洋葱意味着 3 趟"去分发器 → 回锅"，2 人 25 步的预算里光这一项就要 30+ 步，
整局会在"还没上过一道菜"的状态下结束 —— 那和 `cook_time=20` 是同一个毛病。
把 `recipe=("onion","onion","onion")` 传回来就得到经典设定，但要相应加长局长。

================================================================================
观测：自己写的限视野特征
================================================================================
单 agent 的向量 = 三块拼起来：

  [自己]      朝向 one-hot(4) + 手持 one-hot(5)                       =   9
  [窗口]      (2·sight+1)² 个格子，每格 地形 one-hot(8) + 物体 one-hot(6)
              地形 8 = 7 种 + 出界 1；sight=2 时 25 格                   = 350
  [队友]      其余 N-1 个 agent，每个 (相对 dx, dy) 归一化(2)
              + 可见标志(1) + 手持 one-hot(5)                          = 8·(N-1)

**窗口之外的格子根本不在向量里**（不是填 0 —— 填 0 等于告诉 agent "那里是空的"，
那也是一种信息）。队友若在窗口外，只有"不可见"这一个比特。

`self` 的位置不进向量：它在窗口里恒为中心，绝对坐标对决策没有用，反而给了
一个能绕过局部性的旁路。

================================================================================
图：C 是环境给的（论文未给值），默认 = sight
================================================================================
论文 §2.2.1："边当且仅当 ‖p_i − p_j‖ ≤ C，C 由环境设定"。Overcooked 没有这个
概念，所以 C 是我们选的。默认取 **C = sight = 2**，实测（9x6 三人厨房，2000 局随机起局）：

    C=1  平均度 0.43/2   全连通  1.9%   全断开 46.4%   ← 太稀
    C=2  平均度 1.00/2   全连通 18.4%   全断开  9.1%   ← 默认
    C=3  平均度 1.50/2   全连通 48.4%   全断开  0.0%   ← 太密

语义是"能和看得见的人通信"。但**看得见人 ≠ 知道对方脑子里的东西**（他手上
有没有盘子、他刚才往哪个方向走），信息差仍然存在。
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym

# 与 overcooked_ai_py.mdp.actions.Action.INDEX_TO_ACTION 一致
#   [NORTH, SOUTH, EAST, WEST, STAY, INTERACT]
NUM_ACTIONS = 6
ACTION_INTERACT = 5

# 地形字符 -> one-hot 下标。最后一位留给"出界"，它和"台面"必须分开：
# 把出界当成台面等于凭空告诉 agent 那里有一堵墙。
TERRAIN_CHARS = (" ", "X", "O", "T", "D", "P", "S")
TERRAIN_OOB = len(TERRAIN_CHARS)          # 7 -> 出界
N_TERRAIN = len(TERRAIN_CHARS) + 1        # 8

# 手持 / 台面物体 -> one-hot 下标。"none" 是真实的一类（空手 / 空台面）。
OBJECT_NAMES = ("none", "onion", "tomato", "dish", "soup")
N_OBJECT = len(OBJECT_NAMES)

GRAPH_OBS_KEYS = ("adjacency", "edge_features")


def _obj_index(obj) -> int:
    """ObjectState（或 None）-> 类别下标。汤不分"煮好没有" —— 那是锅的属性，
    在这个特征里由地形格 P 承担。"""
    if obj is None:
        return 0
    name = obj.name
    if name.startswith("soup"):
        return 4
    return {"onion": 1, "tomato": 2, "dish": 3}.get(name, 0)


def _object_class(state, pos) -> int:
    return _obj_index(state.objects.get(tuple(pos)))


class OvercookedTeamEnv:
    """队伍级 Overcooked 环境。接口与 VMASEnv 一致。"""

    metadata = {}

    def __init__(
        self,
        num_agents: int = 2,
        layout: str = "three_player",
        max_steps: int = 25,
        cook_time: int = 5,
        recipe: tuple = ("onion",),
        sight: int = 2,
        add_agent_index: bool = True,
        graph_obs: bool = False,
        comm_radius: float | None = None,
        random_start_pos: bool = True,
        seed: int | None = None,
    ) -> None:
        from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

        self._num_agents = int(num_agents)
        self._max_steps = int(max_steps)
        self._cook_time = int(cook_time)
        self._sight = int(sight)
        self._add_agent_index = bool(add_agent_index)
        self._graph_obs = bool(graph_obs)
        self._random_start_pos = bool(random_start_pos)
        # 默认 C = sight：能和看得见的人通信（理由见文件头）
        self._comm_radius = float(sight if comm_radius is None else comm_radius)
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._forced_seed: int | None = None
        self.id = None

        grid = LAYOUTS[layout](self._num_agents)
        self._layout = layout
        self._recipe = tuple(recipe)
        self._mdp = OvercookedGridworld.from_grid(
            grid,
            base_layout_params={
                "start_all_orders": [{"ingredients": list(self._recipe)}],
                "cook_time": self._cook_time,
            },
        )
        if self._mdp.num_players != self._num_agents:
            raise ValueError(
                f"布局 {layout} 给出 {self._mdp.num_players} 个玩家，"
                f"与 num_agents={self._num_agents} 不符"
            )

        self.num_agents = self._num_agents
        n = self.num_agents
        self._s = self._sight
        self._win = 2 * self._s + 1

        # 单 agent 观测维
        self._d_self = 4 + N_OBJECT
        self._d_window = self._win * self._win * (N_TERRAIN + N_OBJECT)
        self._d_other = 2 + 1 + N_OBJECT
        dim = self._d_self + self._d_window + self._d_other * (n - 1)
        self._obs_dim = dim

        agent_dim = dim + (n if self._add_agent_index else 0)
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
        self._state = None

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
        return np.stack(
            [np.asarray(p.position, dtype=np.float64) for p in self._state.players]
        )

    def _graph(self) -> tuple[np.ndarray, np.ndarray]:
        """(adjacency (N,N), edge_features (N,N,1))。切比雪夫距离，无自环。

        厨房是方格，用切比雪夫与"窗口是方形"这件事一致（同 vmas_env）。
        """
        pos = self._positions()
        delta = np.abs(pos[:, None, :] - pos[None, :, :]).max(axis=-1)
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

    def _window(self, i: int) -> np.ndarray:
        """agent i 的 (sight) 窗口。窗外不在向量里；窗口内但出界 -> OOB 类。"""
        rows, cols = len(self._mdp.terrain_mtx), len(self._mdp.terrain_mtx[0])
        cx, cy = self._state.players[i].position
        out = np.zeros((self._win * self._win, N_TERRAIN + N_OBJECT), dtype=np.float32)
        k = 0
        for dy in range(-self._s, self._s + 1):
            for dx in range(-self._s, self._s + 1):
                x, y = cx + dx, cy + dy
                # 注意 Overcooked 的坐标是 (x=列, y=行)
                if 0 <= y < rows and 0 <= x < cols:
                    ch = self._mdp.terrain_mtx[y][x]
                    out[k, TERRAIN_CHARS.index(ch)] = 1.0
                    out[k, N_TERRAIN + _object_class(self._state, (x, y))] = 1.0
                else:
                    out[k, TERRAIN_OOB] = 1.0
                    out[k, N_TERRAIN] = 1.0          # 出界的格子上没有物体
                k += 1
        return out.reshape(-1)

    def _make_obs(self, is_first: bool, is_terminal: bool) -> dict:
        from overcooked_ai_py.mdp.actions import Direction

        n = self.num_agents
        cx, cy = self._positions()[:, 0], self._positions()[:, 1]
        rows: list[np.ndarray] = []
        for i in range(n):
            p = self._state.players[i]
            own = np.zeros(self._d_self, dtype=np.float32)
            # 朝向 one-hot（Direction.ALL_DIRECTIONS 的顺序：[N, S, E, W]）
            own[Direction.ALL_DIRECTIONS.index(tuple(p.orientation))] = 1.0
            own[4 + _obj_index(p.held_object)] = 1.0
            others = np.zeros(self._d_other * (n - 1), dtype=np.float32)
            k = 0
            for j in range(n):
                if j == i:
                    continue
                dx, dy = cx[j] - cx[i], cy[j] - cy[i]
                vis = max(abs(dx), abs(dy)) <= self._s
                # 溢出保护：归一化到 [-1,1]，并 clamp 到窗口范围
                others[k] = np.clip(dx, -self._s, self._s) / max(self._s, 1)
                others[k + 1] = np.clip(dy, -self._s, self._s) / max(self._s, 1)
                others[k + 2] = 1.0 if vis else 0.0
                if vis:
                    others[k + 3 + _obj_index(self._state.players[j].held_object)] = 1.0
                else:
                    # 看不见就**不给**相对位置 —— 那是信息泄漏
                    others[k] = 0.0
                    others[k + 1] = 0.0
                k += self._d_other
            rows.append(np.concatenate([own, self._window(i), others]))

        flat = np.stack(rows)
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

    def reset(self) -> dict:
        if self._forced_seed is not None:
            s = self._forced_seed
        elif self._seed is not None:
            s = int(self._rng.integers(0, 2**31 - 1))
        else:
            s = None
        if self._random_start_pos:
            # Overcooked 的随机起局内部用全局 np.random，没有自己的 RNG 句柄 ——
            # 只能从外面固定它，否则"同一批固定局"这条配对评测的前提不成立。
            np.random.seed(0 if s is None else s)
            fn = self._mdp.get_random_start_state_fn(random_start_pos=True)
            self._state = fn()
        else:
            self._state = self._mdp.get_standard_start_state()
        self._total_reward = 0.0
        return self._make_obs(is_first=True, is_terminal=False)

    def step(self, actions: np.ndarray):
        """Args: actions (N,) 整数动作下标。Returns: (obs, reward (N,), done, info)。"""
        from overcooked_ai_py.mdp.actions import Action

        n = self.num_agents
        actions = np.asarray(actions).reshape(-1)
        if actions.shape[0] != n:
            raise ValueError(f"动作形状 {actions.shape}，期望 ({n},)")
        if actions.min() < 0 or actions.max() >= NUM_ACTIONS:
            raise ValueError(f"动作越界: {actions}（合法范围 0..{NUM_ACTIONS - 1}）")

        joint = [Action.INDEX_TO_ACTION[int(a)] for a in actions]
        self._state, infos = self._mdp.get_state_transition(self._state, joint)
        reward = np.asarray(infos["sparse_reward_by_agent"], dtype=np.float32).reshape(-1)
        assert reward.shape == (n,), (reward.shape, n)
        self._total_reward += float(reward.sum())
        d = bool(self._mdp.is_terminal(self._state) or self._state.timestep >= self._max_steps)
        obs = self._make_obs(is_first=False, is_terminal=d)
        return obs, reward, d, {"adjacency": self.adjacency}

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------ 诊断

    @property
    def total_reward(self) -> float:
        """本局累计的稀疏奖励（上菜才给）。指标与演示都用它。"""
        return float(self._total_reward)

    def state_string(self) -> str:
        """官方渲染，用于调试与文本回放。"""
        return self._mdp.state_string(self._state)


# ============================================================================
# 布局 —— 边界必须全非空、每行等长、玩家数字恰为 1..n
#   X 台面   O 洋葱   T 番茄   D 盘子   P 锅   S 上菜口   空格 地板   1..n 出生点
# ============================================================================
LAYOUTS = {
    # 官方内置的经典双人厨房（5x4）
    "cramped_room": lambda n: [
        "XXPXX",
        "O  2O",
        "X1  X",
        "XDXSX",
    ],
    # 自写三人厨房（9x6）。尺寸是**实测选的**，判据不是"窗口占多少面积"，而是
    # "站在一个典型位置，锅 / 洋葱 / 盘子 / 上菜口 这几个**任务相关站点**看得见几个"：
    #
    #     7x5  sight=2   平均看得见 80% 的站点；80% 的位置至少漏一个
    #     9x6  sight=2   平均看得见 49% 的站点；**100%** 的位置至少漏一个
    #     9x7  sight=2   平均看得见 39% 的站点；100% 的位置至少漏一个
    #
    # 7x5 上 sight=2 平均看得见八成站点，等于几乎没造出信息差 —— 房子的面积比
    # 骗人（厨房大部分是台面，占面积大不等于看得见锅）。9x6 是"每一处都漏点什么"
    # 的最小尺寸，同时厨房不要大到把交付流程推到 25 步装不下。
    "three_player": lambda n: [
        "XXPXXXXXX",
        "O       O",
        "X       X",
        "X 1 2 3 X",
        "X       X",
        "XDXSXXXXX",
    ],
}


# ============================================================================
# 自检 —— 本地不跑实验，断言写死在代码里。
#   在 third_party/dreamerv3-torch 目录下： python -m envs.overcooked_env
# ============================================================================
def _self_test() -> None:
    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        print(f"  [{'ok  ' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
        if not cond:
            failures.append(name)

    print("OvercookedTeamEnv 自检")
    print("-" * 68)

    env = OvercookedTeamEnv(num_agents=3, seed=0, graph_obs=True)
    n = env.num_agents
    d = env._obs_dim
    check("agent 数 = 3（布局里的玩家数字）", n == 3, str(n))
    check("动作数 = 6", env.action_space.n == NUM_ACTIONS, str(env.action_space.n))
    check(
        "action_space 是 Discrete 且带 discrete 标记",
        isinstance(env.action_space, gym.spaces.Discrete)
        and getattr(env.action_space, "discrete", False) is True,
    )

    obs = env.reset()
    check("state 形状 (N, D+N)", obs["state"].shape == (n, d + n), str(obs["state"].shape))
    check("obs space 是单 agent 形状", env.observation_space["state"].shape == (d + n,))
    check("观测维与文档一致",
          d == env._d_self + env._win**2 * (N_TERRAIN + N_OBJECT) + env._d_other * (n - 1),
          f"{d}")

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

    # --- 图 ---
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

    # ========================================================================
    # 限视野：这是整个包装存在的理由，必须钉死
    # ========================================================================
    print()
    print("  --- 限视野（信息不对称是构造出来的）---")
    def other_block(env, obs):
        """切出"队友"那一块 —— 末尾还有 N 维 agent one-hot，不能一起切进来。"""
        beg = env._d_self + env._win**2 * (N_TERRAIN + N_OBJECT)
        end = beg + env._d_other * (env.num_agents - 1)
        return obs["state"][:, beg:end].reshape(env.num_agents, env.num_agents - 1,
                                                env._d_other)

    env2 = OvercookedTeamEnv(num_agents=3, sight=1, seed=1)
    obs = env2.reset()
    # 队友块：可见标志为 0 的那些，相对位置必须**恰好是 0**，不能泄漏
    blocks = other_block(env2, obs)
    hidden = blocks[:, :, 2] == 0.0
    leaked = np.abs(blocks[:, :, :2])[hidden]
    check("看不见的队友，其相对位置恰好为 0（没有泄漏）",
          bool((leaked == 0.0).all()) if leaked.size else True,
          f"{int(hidden.sum())} 个不可见槽位")

    # 窗口之外的格子不在向量里：观测维恒等于 sight 对应的维数，不随队友走远而变
    check("观测维与 sight 对应（窗口外的格子根本不进向量）",
          env._obs_dim == env._d_self + (2 * 2 + 1) ** 2 * (N_TERRAIN + N_OBJECT)
          + env._d_other * (n - 1))

    # 把两个 agent 放到相距 > sight 的位置，可见标志必须为 0
    env3 = OvercookedTeamEnv(num_agents=2, layout="cramped_room", sight=1, seed=2)
    env3.seed(3)
    env3.reset()
    env3._state.players[0].position = (1, 2)
    env3._state.players[1].position = (3, 1)
    blocks2 = other_block(env3, env3._make_obs(False, False))
    check("相距超过 sight 时，互相都看不见（可见标志 = 0）",
          bool((blocks2[:, :, 2] == 0.0).all()))
    env3._state.players[1].position = (2, 2)
    blocks2 = other_block(env3, env3._make_obs(False, False))
    check("相邻时互相可见（可见标志 = 1）", bool((blocks2[:, :, 2] == 1.0).all()))
    # 反向对照：可见时相对位置**必须非零**，否则上面那条"不可见时为 0"会因为
    # "这个字段恒为 0"而假通过
    check("（反证）可见时相对位置确实非零",
          bool((np.abs(blocks2[:, :, :2]).max() > 0.0)))

    # ========================================================================
    # 游戏机制：cook_time 压缩后，短局长里真的能煮出并上交一锅汤
    # ========================================================================
    print()
    print("  --- 机制（cook_time 压缩是否让短局长有意义）---")
    from overcooked_ai_py.mdp.actions import Action, Direction

    env4 = OvercookedTeamEnv(num_agents=2, layout="cramped_room", max_steps=40,
                             cook_time=3, sight=2, seed=4, random_start_pos=False)

    def drive(env, targets, budget=40):
        """朝目标格走（不绕墙 —— 这张图够空旷，直线可达）。"""
        for _ in range(budget):
            pos = [p.position for p in env._state.players]
            if all(tuple(p) == tuple(t) for p, t in zip(pos, targets)):
                return True
            acts = []
            for (px, py), (tx, ty) in zip(pos, targets):
                if (px, py) == (tx, ty):
                    acts.append(Action.STAY)
                elif px != tx:
                    acts.append(Direction.EAST if tx > px else Direction.WEST)
                else:
                    acts.append(Direction.SOUTH if ty > py else Direction.NORTH)
            env.step(np.array([Action.ACTION_TO_INDEX[a] for a in acts]))
        return False

    def joint(a0, a1=Action.STAY):
        return np.array([Action.ACTION_TO_INDEX[a0], Action.ACTION_TO_INDEX[a1]])

    env4.seed(0)
    env4.reset()
    # cramped_room: 洋葱分发器在 (0,1) 与 (4,1)，锅在 (2,0)，盘子 (1,3)，上菜口 (3,3)
    # 两个 agent 不能站同一格，1 号送去别处待着（cramped_room 的空格很少）
    check("走到洋葱旁", drive(env4, [(1, 1), (3, 1)]),
          str([p.position for p in env4._state.players]))
    # INTERACT 作用于**面朝**的格子，所以要先转向洋葱所在的那一侧（西）
    env4.step(joint(Direction.WEST))
    check("转向西（对着洋葱分发器）",
          tuple(env4._state.players[0].orientation) == tuple(Direction.WEST),
          str(env4._state.players[0].orientation))
    env4.step(joint(Action.INTERACT))
    held0 = env4._state.players[0].held_object
    check("在洋葱分发器旁交互后拿到洋葱（地形 O 有效）",
          held0 is not None and held0.name == "onion", str(held0))

    # 站到锅正下方 (2,1)，面朝北才能对 (2,0) 交互
    check("走到锅下方", drive(env4, [(2, 1), (3, 1)]),
          str([p.position for p in env4._state.players]))
    env4.step(joint(Direction.NORTH))       # 撞墙但朝向改北
    check("朝向北", tuple(env4._state.players[0].orientation) == tuple(Direction.NORTH),
          str(env4._state.players[0].orientation))
    env4.step(joint(Action.INTERACT))
    pot = env4._state.objects.get((2, 0))
    check("洋葱进了锅（地形 P 有效）", pot is not None and pot.name == "soup", str(pot))
    check("刚下锅时还没煮好", not pot.is_ready, f"cooking_tick={pot._cooking_tick}")
    check("此时锅处于 idle（没开火）", pot.is_idle, f"cooking_tick={pot._cooking_tick}")

    # ★ 开火是**第二次**交互：只有玩家**空手**对锅交互时才 begin_cooking()
    #   （overcooked_mdp.py:1168-1172）。这一步不是可有可无的 —— 少了它，
    #   cooking_tick 会一直停在 -1，汤永远煮不好。
    env4.step(joint(Action.INTERACT))
    pot = env4._state.objects.get((2, 0))
    # 注意 tick 不一定是 0：begin_cooking() 置 0 之后，**同一步**的
    # step_environment_effects 就会 cook() 推进到 1。判据是"不再 idle"。
    check("空手再交互一次才开火（cooking_tick 从 -1 变为非负）",
          not pot.is_idle and pot._cooking_tick >= 0, f"cooking_tick={pot._cooking_tick}")

    ready_at = None
    for _ in range(15):
        env4.step(joint(Action.STAY))
        pot = env4._state.objects.get((2, 0))
        if pot is not None and pot.is_ready:
            ready_at = env4._state.timestep
            break
    check(f"等待后汤煮好了（cook_time={env4._cook_time}）", ready_at is not None)
    # 这一条是"短局长可行"的核心：默认 cook_time=20 时同样的流程要 20+ 步纯等待
    check("煮好所需总步数远小于默认 cook_time=20 时的等待量",
          ready_at is not None and ready_at <= 12, f"第 {ready_at} 步")

    env.close()
    env2.close()
    env3.close()
    env4.close()

    print("=" * 68)
    if failures:
        print(f"FAILED ({len(failures)}): " + ", ".join(failures))
        raise SystemExit(1)
    print("全部通过。")


if __name__ == "__main__":
    _self_test()
