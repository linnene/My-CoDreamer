"""Sequential Estimate Game —— CoDreamer 论文附录 B 的自建环境。

================================================================================
归属说明
================================================================================
本文件**不属于上游 dreamerv3-torch**（不在 UPSTREAM_MANIFEST.sha256 里），
是 CoDreamer 复现的一部分，由我们编写。

放在 envs/ 下而不是独立的 codreamer/ 包，是为了零 import 摩擦 ——
dreamer.py 的 make_env 用 `import envs.<name>`，即相对当前工作目录解析。

================================================================================
论文规格
================================================================================
CoDreamer (arXiv:2406.13600v1) 附录 B "Empirical Validation of Modelling
Expressivity"。该环境是对 Kortvelesy & Prorok (2022) 的 Estimate Game 的扩展，
把一个单步博弈变成多步 POMG，其 reward 与 transition **显式依赖邻居**，
因此无法仅凭单个 agent 的信息建模 —— 这正是 CoDreamer 关于"表达力"主张的检验场。

论文 Table 1 给出的实例化参数见下方 Spec 的默认值。

reward（agent i，t 时刻）::

    y_i^t = 2 * ( eta * (s_i^t - 0.5)
                + (1 - eta) * mean_{j in N_i}(s_j^t - 0.5) ) + 0.5
    r_i^t = -max( | bin(a_i^t) - y_i^t | - 1/(2|A|),  0 )

transition::

    s_i^{t+1} = 0.5 * cos( a_val_i^t + eta * s_i^t
                         + (1 - eta) * mean_{j in N_i}(s_j^t) ) + 0.5

其中 ``bin(a) = (a + 0.5) / |A|`` 是动作对应区间的中心，``|A|`` 是离散动作数。
即：agent 要预测一个由自己与邻居共同决定的目标值 ``y_i``，动作被离散成 |A| 个
等距区间；只要落在正确区间内（误差不超过半个区间宽）就得 0 分，否则按超出部分扣分。

================================================================================
论文未明确、本实现自行判定的地方（**复现风险，需在论文对比时留意**）
================================================================================
1. ``transition`` 里的 ``a`` 用的是**归一化区间中心** ``bin(a)``，而不是原始
   动作下标。依据：reward 里明确用了 ``a/|A| + 1/(2|A|)``，两者应指同一量；且
   ``s`` 在 [0,1]，若用原始下标 0..3，cos 的相位会被动作完全支配，
   邻居项将失去意义。可用 ``Spec.transition_action`` 切换以做消融。
2. 邻接矩阵采样方式。论文只说"用边密度 rho 构造随机邻接矩阵"。
   本实现默认在**上三角**上以概率 rho 采样再对称化，使实际边密度恰为 rho；
   若改为对全体元素采样再对称化，实际密度会变成 1-(1-rho)^2 ≈ 0.84。
   可用 ``Spec.adjacency_sampling`` 切换。
3. ``N_i`` 是否包含 i 自身。本实现**不含自身**（对角线恒为 False）。
   依据：reward 把 ``eta * s_i`` 与 ``(1-eta) * mean_{j in N_i}(s_j)`` 分开写，
   说明 N_i 是"别人"。
4. ``N_i`` 为空时的处理。论文承认"某些 agent 在其 reward 和 transition 上
   完全独立"，故空邻域必然出现。本实现把空邻域的邻居均值定义为 **0.0**，
   使公式退化但不产生 NaN；这不一定是论文的做法。
5. 每 episode 的邻接矩阵是**静态**的（论文明确），跨 episode 重新采样。

================================================================================
奖励语义
================================================================================
论文在 Estimate Game 上**用局部（per-agent）奖励训练**，而非像 Melting Pot
那样求和成全局奖励：

    "we train agents using their local rewards in order to thoroughly evaluate
     CoDreamer's ability to learn more independent reward functions that are
     still dependent on non-local information"

因此本环境默认返回 **shape (N,) 的逐 agent 奖励**。注意上游 dreamerv3-torch
的奖励通路假设标量，接入时需要另行处理（见 docs/reproduction-plan.md）。
``global_mode`` 可切换为 "sum" / "min" 以便做对照。
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

__all__ = ["Spec", "SequentialEstimateGame"]


@dataclass(frozen=True)
class Spec:
    """论文 Table 1 的实例化，默认值即论文取值。"""

    num_agents: int = 4          # n
    horizon: int = 5             # T
    obs_size: int = 1            # |s_i|
    num_actions: int = 4         # |A|
    edge_density: float = 0.6    # rho
    local_state_ratio: float = 0.3  # eta

    # 论文未明确的部分，见模块 docstring
    transition_action: str = "normalized"   # "normalized" | "index"
    adjacency_sampling: str = "upper_triangle"  # "upper_triangle" | "symmetric_full"
    global_mode: str = "none"               # "none" | "sum" | "min"

    def __post_init__(self) -> None:
        if self.num_agents < 1:
            raise ValueError("num_agents must be >= 1")
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")
        if self.num_actions < 2:
            raise ValueError("num_actions must be >= 2")
        if not 0.0 <= self.edge_density <= 1.0:
            raise ValueError("edge_density must be in [0, 1]")
        if not 0.0 <= self.local_state_ratio <= 1.0:
            raise ValueError("local_state_ratio must be in [0, 1]")
        if self.transition_action not in ("normalized", "index"):
            raise ValueError(f"unknown transition_action: {self.transition_action}")
        if self.adjacency_sampling not in ("upper_triangle", "symmetric_full"):
            raise ValueError(f"unknown adjacency_sampling: {self.adjacency_sampling}")
        if self.global_mode not in ("none", "sum", "min"):
            raise ValueError(f"unknown global_mode: {self.global_mode}")


class SequentialEstimateGame:
    """纯逻辑实现，不依赖 gym，不依赖 dreamerv3-torch。

    与框架的对接（gym 空间、obs 字典格式、奖励形状）留给后续的包装层，
    以便本模块可以独立测试。
    """

    def __init__(self, spec: Spec | None = None, seed: int | None = None) -> None:
        self.spec = spec or Spec()
        self._rng = np.random.default_rng(seed)
        self._adjacency: np.ndarray | None = None
        self._state: np.ndarray | None = None
        self._t: int = 0

    # ---------------------------------------------------------------- 属性

    @property
    def num_agents(self) -> int:
        return self.spec.num_agents

    @property
    def num_actions(self) -> int:
        return self.spec.num_actions

    @property
    def horizon(self) -> int:
        return self.spec.horizon

    @property
    def observation_shape(self) -> tuple[int, int]:
        """(num_agents, obs_size)"""
        return (self.spec.num_agents, self.spec.obs_size)

    @property
    def adjacency(self) -> np.ndarray:
        """当期 (N, N) 布尔邻接矩阵，无自环，对静止的 episode 恒定。"""
        if self._adjacency is None:
            raise RuntimeError("call reset() before reading adjacency")
        return self._adjacency

    @property
    def neighbors(self) -> list[np.ndarray]:
        """每个 agent 的邻居下标数组列表。"""
        return [np.flatnonzero(row) for row in self.adjacency]

    @property
    def timestep(self) -> int:
        return self._t

    @property
    def state(self) -> np.ndarray:
        """(N,) 当前局部状态 s。"""
        if self._state is None:
            raise RuntimeError("call reset() before reading state")
        return self._state

    # ---------------------------------------------------------------- 采样

    def _sample_adjacency(self) -> np.ndarray:
        n = self.spec.num_agents
        rho = self.spec.edge_density
        a = np.zeros((n, n), dtype=bool)

        if self.spec.adjacency_sampling == "upper_triangle":
            # 上三角采样，实际边密度恰为 rho
            iu = np.triu_indices(n, k=1)
            drawn = self._rng.random(len(iu[0])) < rho
            a[iu[0][drawn], iu[1][drawn]] = True
            a |= a.T
        else:
            # 全元素采样再对称化，实际密度为 1-(1-rho)^2
            raw = self._rng.random((n, n)) < rho
            a = raw | raw.T

        np.fill_diagonal(a, False)
        return a

    # ---------------------------------------------------------------- 核心

    def reset(self, seed: int | None = None) -> np.ndarray:
        """重新采样邻接矩阵与初始状态，返回观测 (N, obs_size)。"""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._adjacency = self._sample_adjacency()
        self._state = self._rng.random(self.spec.num_agents).astype(np.float64)
        self._t = 0
        return self._observation()

    def _observation(self) -> np.ndarray:
        # o_i = s_i（论文："agents' observations being equal to agent local states"）
        return np.repeat(self._state[:, None], self.spec.obs_size, axis=1).astype(
            np.float32
        )

    def _neighbour_mean(self, values: np.ndarray) -> np.ndarray:
        """对每个 agent 求其邻居的均值；空邻域定义为 0.0（见模块 docstring #4）。"""
        out = np.zeros(self.spec.num_agents, dtype=np.float64)
        for i, nbrs in enumerate(self.neighbors):
            if nbrs.size:
                out[i] = float(values[nbrs].mean())
        return out

    def target(self, state: np.ndarray | None = None) -> np.ndarray:
        """论文中的 y_i，(N,)。"""
        s = self.state if state is None else state
        eta = self.spec.local_state_ratio
        nbr_mean = self._neighbour_mean(s)
        return 2.0 * (eta * (s - 0.5) + (1.0 - eta) * (nbr_mean - 0.5)) + 0.5

    def bin_center(self, actions: np.ndarray) -> np.ndarray:
        """动作下标 -> 区间中心，(N,) 且落在 (0, 1)。"""
        return (np.asarray(actions, dtype=np.float64) + 0.5) / self.spec.num_actions

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, bool, dict]:
        """执行一步。

        Args:
            actions: (N,) 整数动作下标，取值 [0, num_actions)。

        Returns:
            obs (N, obs_size) float32
            rewards: (N,) float —— 逐 agent 局部奖励（见 global_mode）
            done: bool
            info: dict，含 adjacency / target
        """
        if self._state is None:
            raise RuntimeError("call reset() before step()")
        if self._t >= self.spec.horizon:
            raise RuntimeError("episode already finished; call reset()")

        actions = np.asarray(actions)
        if actions.shape != (self.spec.num_agents,):
            raise ValueError(
                f"actions shape {actions.shape} != ({self.spec.num_agents},)"
            )
        if not np.issubdtype(actions.dtype, np.integer):
            raise ValueError(f"actions must be integer, got {actions.dtype}")
        if actions.min() < 0 or actions.max() >= self.spec.num_actions:
            raise ValueError(
                f"actions out of range [0, {self.spec.num_actions}): "
                f"min={actions.min()} max={actions.max()}"
            )

        eta = self.spec.local_state_ratio
        n_actions = self.spec.num_actions

        # --- reward：以当前状态算目标，与动作的区间中心比较
        y = self.target()
        centre = self.bin_center(actions)
        half_bin = 1.0 / (2.0 * n_actions)
        rewards = -np.maximum(np.abs(centre - y) - half_bin, 0.0)

        # --- transition
        a_val = centre if self.spec.transition_action == "normalized" else actions.astype(
            np.float64
        )
        s_nbr_mean = self._neighbour_mean(self._state)
        self._state = 0.5 * np.cos(a_val + eta * self._state + (1.0 - eta) * s_nbr_mean) + 0.5
        self._t += 1

        done = self._t >= self.spec.horizon

        if self.spec.global_mode == "sum":
            rewards_out = np.full_like(rewards, rewards.sum())
        elif self.spec.global_mode == "min":
            # 原版 Estimate Game 用"所有局部奖励的最小值"作全局奖励
            rewards_out = np.full_like(rewards, rewards.min())
        else:
            rewards_out = rewards

        info = {"adjacency": self.adjacency.copy(), "target": y}
        return self._observation(), rewards_out, done, info


# ============================================================================
# 自检 —— 本地不可运行任何实验，所以断言必须写死在代码里，
# 让远端（或任何环境）一跑就暴露问题。`python envs/seg.py` 即可执行。
# ============================================================================


def _self_test() -> None:
    import itertools

    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        status = "ok  " if cond else "FAIL"
        print(f"  [{status}] {name}{('  ' + detail) if detail else ''}")
        if not cond:
            failures.append(name)

    print("SequentialEstimateGame 自检")
    print("-" * 68)

    spec = Spec()
    game = SequentialEstimateGame(spec, seed=0)

    # --- 规格默认值必须与论文 Table 1 一致
    check("Table1 n=4", spec.num_agents == 4)
    check("Table1 T=5", spec.horizon == 5)
    check("Table1 |s|=1", spec.obs_size == 1)
    check("Table1 |A|=4", spec.num_actions == 4)
    check("Table1 rho=0.6", spec.edge_density == 0.6)
    check("Table1 eta=0.3", spec.local_state_ratio == 0.3)

    # --- 观测形状与取值范围
    obs = game.reset()
    check("obs shape", obs.shape == (4, 1), str(obs.shape))
    check("obs dtype float32", obs.dtype == np.float32, str(obs.dtype))
    check("obs in [0,1]", bool((obs >= 0).all() and (obs <= 1).all()))

    # --- 邻接矩阵性质
    a = game.adjacency
    check("adjacency shape", a.shape == (4, 4), str(a.shape))
    check("adjacency symmetric", bool((a == a.T).all()))
    check("adjacency no self-loop", bool(not np.diag(a).any()))
    check("adjacency dtype bool", a.dtype == bool, str(a.dtype))

    # --- 边密度（统计意义，用同规格独立采样再断言）
    g2 = SequentialEstimateGame(Spec(num_agents=64), seed=1)
    total_edges = 0
    total_pairs = 0
    for s in range(200):
        g2.reset(seed=1000 + s)
        m = g2.adjacency
        total_edges += int(m.sum() // 2)
        total_pairs += m.shape[0] * (m.shape[0] - 1) // 2
    density = total_edges / total_pairs
    check(
        "edge density ~ rho",
        abs(density - 0.6) < 0.02,
        f"observed {density:.4f}",
    )

    # --- 至少出现过"孤岛 agent"（论文明确说某些 agent 完全独立）
    g3 = SequentialEstimateGame(Spec(num_agents=4), seed=2)
    isolated_seen = False
    for s in range(500):
        g3.reset(seed=2000 + s)
        if any(len(nb) == 0 for nb in g3.neighbors):
            isolated_seen = True
            break
    check("isolated agent occurs", isolated_seen)

    # --- 奖励：正确区间给 0，否则为负；上界恒 <= 0
    game.reset(seed=7)
    y = game.target()
    exact = np.clip(
        np.floor(y * spec.num_actions - 0.5 + 1e-9).astype(int), 0, spec.num_actions - 1
    )
    _, r_best, _, _ = game.step(exact)
    check("best action gives 0", bool(np.allclose(r_best, 0.0, atol=1e-6)), str(r_best))

    game.reset(seed=7)
    y = game.target()
    worst = np.where(y >= 0.5, 0, spec.num_actions - 1)
    _, r_worst, _, _ = game.step(worst)
    check("rewards <= 0", bool((r_worst <= 1e-9).all()), str(r_worst))
    check("worst <= best", bool((r_worst <= r_best + 1e-9).all()))

    # --- episode 长度与 done 语义
    game.reset(seed=11)
    for t in range(spec.horizon):
        _, _, done, _ = game.step(np.zeros(spec.num_agents, dtype=int))
        if t < spec.horizon - 1:
            check(f"not done at t={t}", not done)
    check(f"done at t={spec.horizon}", done)
    try:
        game.step(np.zeros(spec.num_agents, dtype=int))
        check("step after done raises", False)
    except RuntimeError:
        check("step after done raises", True)

    # --- 邻接矩阵在 episode 内静态
    game.reset(seed=13)
    a0 = game.adjacency.copy()
    for _ in range(spec.horizon):
        game.step(np.zeros(spec.num_agents, dtype=int))
    check("adjacency static within episode", bool((game.adjacency == a0).all()))

    # --- 跨 episode 邻接矩阵会重采样
    seen = set()
    for s in range(50):
        game.reset(seed=3000 + s)
        seen.add(game.adjacency.tobytes())
    check("adjacency resampled across episodes", len(seen) > 1, f"{len(seen)} distinct")

    # --- 状态始终在 [0,1]（cos 映射 + 0.5）
    game.reset(seed=17)
    in_range = True
    for _ in range(spec.horizon):
        game.step(game._rng.integers(0, spec.num_actions, spec.num_agents))
        if not ((game.state >= 0).all() and (game.state <= 1).all()):
            in_range = False
    check("state stays in [0,1]", in_range)

    # --- 与邻居的依赖确实存在：改邻居状态应当改变目标值
    game.reset(seed=19)
    s = game.state.copy()
    y_base = game.target()
    nbrs = game.neighbors
    touched = None
    for i, nb in enumerate(nbrs):
        if nb.size:
            touched = (i, nb[0])
            break
    if touched is None:
        check("neighbour dependency (need a non-empty neighbourhood)", False)
    else:
        i, j = touched
        s2 = s.copy()
        s2[j] = 1.0 - s2[j]
        check(
            "target depends on neighbours",
            not np.isclose(game.target(s2)[i], y_base[i]),
            f"agent {i}, neighbour {j}",
        )
        # 且只影响相邻的 agent
        isolated = [k for k, nb in enumerate(nbrs) if nb.size == 0]
        if isolated:
            k = isolated[0]
            check("isolated agent unaffected", np.isclose(game.target(s2)[k], y_base[k]))

    # --- 全局奖励模式
    for mode in ("sum", "min"):
        gm = SequentialEstimateGame(replace(spec, global_mode=mode), seed=23)
        gm.reset()
        _, r, _, _ = gm.step(np.zeros(spec.num_agents, dtype=int))
        check(f"global_mode={mode} broadcast", bool(np.allclose(r, r[0])))

    # --- 可复现性
    def rollout(seed: int) -> np.ndarray:
        g = SequentialEstimateGame(spec, seed=seed)
        g.reset()
        acc = []
        for t in range(spec.horizon):
            _, r, _, _ = g.step(np.full(spec.num_agents, t % spec.num_actions))
            acc.append(r)
        return np.array(acc)

    check("deterministic given seed", bool(np.allclose(rollout(5), rollout(5))))
    check("different seeds differ", not np.allclose(rollout(5), rollout(6)))

    print("-" * 68)
    if failures:
        print(f"FAILED ({len(failures)}): {', '.join(failures)}")
        raise SystemExit(1)
    print("全部通过。")


if __name__ == "__main__":
    _self_test()
