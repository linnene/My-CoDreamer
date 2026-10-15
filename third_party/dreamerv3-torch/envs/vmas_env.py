"""VMAS 的队伍级包装 —— 对外接口与 `envs/seg_gym.py` **完全一致**。

================================================================================
归属说明
================================================================================
本文件不属于上游 dreamerv3-torch，是 CoDreamer 复现的一部分，由我们编写。

================================================================================
为什么换到这个环境
================================================================================
Sequential Estimate Game 跑完后结论是：**它饱和了，分不出策略差别**
（六配置扫描 + 三 seed 对照都收敛到同一个回报）。而论文自己的消融也给出了
环境排序（附录 C）：

    Melting Pot   —— 最强：AC Comm / WM Comm 对 IDreamer 的改进**有统计显著性**
    Estimate Game —— 中：作者自承 overlap of information "highly evident by the
                     construction of the environment"
    VMAS          —— 最弱："not highly inter-agent dependent"、
                     "IDreamer and CoDreamer are similar... due to stochasticity"

VMAS 是论文**正文主实验**（Figure 4），所以复现绕不过去。而且它有一个别的环境
给不了的好处：**向量观测、4~5 个 agent、单局 500 步**，在 RTX 3060 上跑得动。

本包装刻意复用 seg_gym 的整套约定，`tools_marl.simulate_marl` 一个字都不用改。

================================================================================
与框架的接口约定（逐条对齐 seg_gym.py）
================================================================================
    observation_space["state"]        = Box(D + N)      D 维观测 + N 维 one-hot 索引
    observation_space["adjacency"]    = Box(N)          （仅 graph_obs 时）
    observation_space["edge_features"]= Box(N, 1)       （仅 graph_obs 时）
    action_space                      = Discrete(9)

声明成**单 agent 视角**是刻意的：N 条记录分别存进 replay，训练时每条样本天然
就是单智能体形状，encoder 永远看不到 agent 轴（见 seg_gym.py 的长注释）。

`reset()` / `step()` 实际返回的数组**带一个 agent 轴**，形状 (N, ...)。

================================================================================
动作空间：9 而不是论文写的 5
================================================================================
论文 Table 8 写 "Discrete (5)"，但 vmas 1.5.2 的离散动作是 `discrete_action_nvec
= [3, 3]`，即**二维各三个取值**（x/y 方向的力 ∈ {-1,0,1}），合计 9 个组合，
对外是**一个**离散索引（`get_agent_action_size == 1`）。

论文用的应是更老的 vmas。同理观测维度也对不上：Discovery 19（论文 21）、
Buzz Wire 6（论文 8）；只有 Flocking 的 18 与论文完全一致。

**这是版本漂移，不是实现错误。** 需要严格对齐时可把 vmas 钉回论文时期的版本，
但那会牵动整条依赖链，所以先把当前版本跑通并记录差异。

================================================================================
图：C 是我们定的（论文未给）
================================================================================
论文 §2.2.1 只规定"边当且仅当 ||p_i - p_j|| <= C"，而 **C 由环境设定** —— 但
VMAS 本身没有通信半径这个东西，论文也没给三个任务各自的 C。所以 C 必须由我们
选，这是复现缺口之一，登记于此。

选法：先量互距分布，再取一个让图**既不空也不满**的值（六配置扫描的教训是：
环境若不给通信留空间，就测不出差别）。

    flocking   N=4  互距中位 1.13   取 C=1.5   平均度 ~2.2/3
    discovery  N=5  互距中位 1.11   取 C=1.5   平均度 ~3.0/4
    buzz_wire  N=2  互距恒 0.49     取 C=0.5   恒连通

⚠️ buzz_wire 的两个 agent 由刚性连杆相连，互距几乎是常数（0.46~0.52），
所以它的图**不随轨迹变化** —— 要么恒连通要么恒断开。这不影响 CoDreamer 的
语义（通信 = 聚合邻居隐状态，不需要拓扑变化），但这个任务因此**不能用来做
"图稀疏度"的消融**。想调 C 用 `--vmas_comm_radius`，或走 configs.yaml。

边特征取欧氏距离（F=1）。论文原话："All graphs have edge features representing
the relative distances between agents"。与 seg_gym 的约定一致（非边处填 0）。

================================================================================
奖励
================================================================================
VMAS 原生给**逐 agent** 奖励 (N,)，直接透传。论文在 VMAS 上用全局和还是局部
奖励没有明说；这里的实现透传原生值，切换留待需要时再加。
"""

from __future__ import annotations

import gym
import numpy as np
import torch

# [CoDreamer] 论文 Table 8 / 9 / 10 的实例化。`comm_radius` 见模块 docstring。
TASKS: dict[str, dict] = {
    "flocking": dict(
        scenario="flocking", n_agents=4, n_obstacles=5, comm_radius=1.5
    ),
    "discovery": dict(
        scenario="discovery",
        n_agents=5,
        n_objectives=7,
        coverage_requirement=2,
        coverage_distance=0.25,
        comm_radius=1.5,
    ),
    "buzz_wire": dict(scenario="buzz_wire", n_agents=2, comm_radius=0.5),
}

GRAPH_OBS_KEYS = ("adjacency", "edge_features")

__all__ = ["VMASEnv", "TASKS"]


class VMASEnv:
    """队伍级 VMAS 环境。接口与 SequentialEstimateGameEnv 一致。"""

    metadata = {}

    def __init__(
        self,
        task: str = "flocking",
        seed: int | None = None,
        add_agent_index: bool = True,
        graph_obs: bool = False,
        comm_radius: float | None = None,
        max_steps: int = 500,
        device: str = "cpu",
    ) -> None:
        import vmas

        if task not in TASKS:
            raise ValueError(f"unknown VMAS task: {task} (expected {sorted(TASKS)})")
        cfg = dict(TASKS[task])
        self.task = task
        self._scenario = cfg.pop("scenario")
        self._comm_radius = float(
            comm_radius if comm_radius is not None else cfg.pop("comm_radius")
        )
        self._scenario_kwargs = cfg  # 其余都是 scenario 参数
        self._max_steps = int(max_steps)
        self._device = device
        self._add_agent_index = bool(add_agent_index)
        self._graph_obs = bool(graph_obs)
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        # 强制种子（见 seed()）。None = 每局从 _rng 抽，与 seg_gym 的语义一致。
        self._forced_seed: int | None = None
        self.id = None  # 由 rollout 层赋值（与上游 UUID wrapper 的约定一致）

        self._vmas = vmas
        self._env = self._make_vmas(seed=seed)
        obs = self._raw_reset()
        self._obs_dim = int(np.asarray(obs[0]).reshape(-1).shape[0])
        self.num_agents = len(self._env.agents)

        n = self.num_agents
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

        # 单个离散索引；VMAS 内部再展成 discrete_action_nvec
        n_acts = int(np.prod(self._env.agents[0].discrete_action_nvec))
        self.action_space = gym.spaces.Discrete(n_acts)
        # 上游约定：离散环境自己打这个标记（dreamer.py 靠它选 OneHotDist）
        self.action_space.discrete = True
        self.reward_range = (-np.inf, np.inf)

        self._agent_onehot = np.eye(n, dtype=np.float32)

    # ------------------------------------------------------------------ 内部

    def _make_vmas(self, seed: int | None):
        env = self._vmas.make_env(
            scenario=self._scenario,
            num_envs=1,                 # 一个 env 实例 = 一支队伍
            device=self._device,
            continuous_actions=False,
            max_steps=self._max_steps,
            seed=0 if seed is None else int(seed),
            **self._scenario_kwargs,
        )
        return env

    def _raw_reset(self):
        return self._env.reset()

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
        """强制后续每次 reset 都用同一个种子（seed=None 取消）。

        默认行为（`_seed` 非 None）已经是"每局从自己的 RNG 抽种子"，可重现但
        每局不同 —— 配对评测要的就是那个。本方法提供的是"每局都一模一样"，
        用于自检与调试。注意 VMAS 自己的 `seed()` 在**内层**，包装层必须转发，
        否则调用方（比如 `scripts/eval_policy.py` 之类想固定局的地方）拿不到。
        """
        self._forced_seed = None if seed is None else int(seed)

    def _graph(self) -> tuple[np.ndarray, np.ndarray]:
        """(adjacency (N,N) float, edge_features (N,N,1) float)。

        无自环（与 seg_gym 一致）。边特征 = 欧氏距离（论文原话），非边处填 0。
        """
        pos = np.stack(
            [a.state.pos[0].detach().cpu().numpy() for a in self._env.agents]
        ).astype(np.float64)                                      # (N, 2)
        dist = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)  # (N, N)
        adj = dist <= self._comm_radius
        np.fill_diagonal(adj, False)                              # 无自环
        edge = (dist * adj)[..., None]
        return adj.astype(np.float32), edge.astype(np.float32)

    @property
    def adjacency(self) -> np.ndarray:
        return self._graph()[0]

    def edge_features(self) -> np.ndarray:
        return self._graph()[1]

    # ------------------------------------------------------------------ 观测

    def _make_obs(self, raw, is_first: bool, is_terminal: bool) -> dict:
        """raw: VMAS 的 list[N] of (1, obs_dim) tensor。"""
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

    def reset(self) -> dict:
        """重开一局，返回 (N, ·) 的观测字典。

        种子由本层控制（与 seg_gym 一致）：每局从自己的 RNG 抽一个种子再交给
        VMAS。`env.seed(k)` + `reset()` 是**跨实例可重现**的（已实测），
        所以"同一批固定局"的配对评测成立。
        """
        if self._forced_seed is not None:
            self._env.seed(self._forced_seed)
        elif self._seed is not None:
            self._env.seed(int(self._rng.integers(0, 2**31 - 1)))
        raw = self._raw_reset()
        return self._make_obs(raw, is_first=True, is_terminal=False)

    def step(self, actions: np.ndarray):
        """Args: actions (N,) 整数动作下标。Returns: (obs, reward (N,), done, info)。"""
        n = self.num_agents
        actions = np.asarray(actions).reshape(-1)
        if actions.shape[0] != n:
            raise ValueError(f"动作形状 {actions.shape}，期望 ({n},)")
        n_acts = self.action_space.n
        if actions.min() < 0 or actions.max() >= n_acts:
            raise ValueError(f"动作越界: {actions}（合法范围 0..{n_acts - 1}）")

        acts = [
            torch.as_tensor([[int(a)]], dtype=torch.long, device=self._env.device)
            for a in actions
        ]
        raw, rew, done, info = self._env.step(acts)
        reward = np.stack(
            [np.asarray(r).reshape(-1)[0] for r in rew]
        ).astype(np.float32)
        d = bool(np.asarray(done).reshape(-1)[0])
        obs = self._make_obs(raw, is_first=False, is_terminal=d)
        return obs, reward, d, {"adjacency": self.adjacency}

    def close(self) -> None:
        pass


# ============================================================================
# 自检 —— 本地不跑实验，断言写死在代码里。
#   在 third_party/dreamerv3-torch 目录下： python -m envs.vmas_env
# ============================================================================
def _self_test() -> None:
    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        print(f"  [{'ok  ' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
        if not cond:
            failures.append(name)

    print("VMASEnv 自检")
    print("-" * 68)

    for task, want_n, want_dim in [
        ("flocking", 4, 18),
        ("discovery", 5, 19),
        ("buzz_wire", 2, 6),
    ]:
        env = VMASEnv(task=task, seed=0, graph_obs=True)
        n = env.num_agents
        d = env._obs_dim
        check(f"{task}: agent 数 = {want_n}", n == want_n, str(n))
        check(f"{task}: 观测维 = {want_dim}（vmas 1.5.2）", d == want_dim, str(d))

        obs = env.reset()
        check(
            f"{task}: state 形状 (N, D+N)",
            obs["state"].shape == (n, d + n),
            str(obs["state"].shape),
        )
        check(f"{task}: obs space 是单 agent 形状",
              env.observation_space["state"].shape == (d + n,))
        check(f"{task}: action space 是 Discrete 且带 discrete 标记",
              isinstance(env.action_space, gym.spaces.Discrete)
              and getattr(env.action_space, "discrete", False) is True)
        check(f"{task}: action 数 = 9（3x3，论文写 5）", env.action_space.n == 9,
              str(env.action_space.n))

        # 图
        adj = obs["adjacency"]
        check(f"{task}: adjacency (N,N)", adj.shape == (n, n), str(adj.shape))
        check(f"{task}: 无自环", bool((np.diag(adj) == 0).all()))
        check(f"{task}: 对称", bool(np.allclose(adj, adj.T)))
        check(f"{task}: 取值只有 0/1", bool(np.isin(adj, (0.0, 1.0)).all()))
        check(f"{task}: edge_features (N,N,1) 且非边处为 0",
              obs["edge_features"].shape == (n, n, 1)
              and bool((obs["edge_features"][adj == 0] == 0).all()))

        # step
        o2, r, dn, info = env.step(np.zeros(n, dtype=int))
        check(f"{task}: reward (N,) float32", r.shape == (n,) and r.dtype == np.float32)
        check(f"{task}: done 是 bool", isinstance(dn, (bool, np.bool_)))
        check(f"{task}: step obs 的 is_first 全 False", bool((~o2["is_first"]).all()))

        # 动作越界必须报错
        for bad in (np.full(n, -1), np.full(n, env.action_space.n)):
            try:
                env.step(bad)
                check(f"{task}: 越界动作 {bad[0]} 应报错", False)
            except ValueError:
                check(f"{task}: 越界动作 {bad[0]} 报错", True)
        try:
            env.step(np.zeros(n + 1, dtype=int))
            check(f"{task}: 错误动作形状应报错", False)
        except ValueError:
            check(f"{task}: 错误动作形状报错", True)

        # 每局可确定重现（配对评测的前提）
        env.seed(11)
        env.reset()
        pos_p = np.stack([x.state.pos[0].cpu().numpy() for x in env._env.agents])
        env.reset()
        pos_q = np.stack([x.state.pos[0].cpu().numpy() for x in env._env.agents])
        check(f"{task}: seed(k) 后 reset 可重现", bool(np.allclose(pos_p, pos_q)))

        # 默认路径（不强制）：同一实例两次 reset 应该**不同**，否则局没有随机性
        # 注：这里的变量名不要用 d / n —— 上面已经把 d 用做观测维度。
        env.seed(None)
        env.reset()
        pos_a = np.stack([x.state.pos[0].cpu().numpy() for x in env._env.agents])
        env.reset()
        pos_b = np.stack([x.state.pos[0].cpu().numpy() for x in env._env.agents])
        check(f"{task}: 默认路径每局不同（有随机性）", not bool(np.allclose(pos_a, pos_b)))

        # 关掉 agent 索引后维度变窄
        env2 = VMASEnv(task=task, seed=0, add_agent_index=False)
        o = env2.reset()
        check(f"{task}: 关索引后维数 = D", o["state"].shape == (n, d), str(o["state"].shape))
        print()

    # 图不空也不满（flocking/discovery 上 C 的选取是否合理）
    env = VMASEnv(task="flocking", seed=0, graph_obs=True)
    degs = []
    for _ in range(30):
        obs = env.reset()
        degs.append(obs["adjacency"].sum(1).mean())
    m = float(np.mean(degs))
    check(
        "flocking: 平均度既不接近 0 也不接近 N-1（图有意义）",
        0.5 < m < env.num_agents - 0.5,
        f"平均度 {m:.2f} / {env.num_agents - 1}",
    )

    print("=" * 68)
    if failures:
        print(f"FAILED ({len(failures)}): " + ", ".join(failures))
        raise SystemExit(1)
    print("全部通过。")


if __name__ == "__main__":
    _self_test()
