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


class SequentialEstimateGameEnv:
    """队伍级环境。对外接口刻意贴近 gym，但不继承 gym.Env（多智能体不是 gym 语义）。"""

    metadata = {}

    def __init__(
        self,
        spec: Spec | None = None,
        seed: int | None = None,
        add_agent_index: bool = True,
    ) -> None:
        self.spec = spec or Spec()
        self.game = SequentialEstimateGame(self.spec, seed=seed)
        self._add_agent_index = add_agent_index
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        n = self.spec.num_agents
        d = self.spec.obs_size
        # 单个 agent 的观测维度：D 维观测 (+ N 维 one-hot agent index)
        agent_dim = d + (n if add_agent_index else 0)

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(agent_dim,), dtype=np.float32
                ),
            }
        )
        self.action_space = gym.spaces.Discrete(self.spec.num_actions)
        self.reward_range = (-np.inf, 0.0)

        # 常数：one-hot agent index，形状 (N, N)
        self._agent_onehot = np.eye(n, dtype=np.float32)

    # ------------------------------------------------------------------ 属性

    @property
    def num_agents(self) -> int:
        return self.spec.num_agents

    @property
    def adjacency(self) -> np.ndarray:
        return self.game.adjacency

    # ------------------------------------------------------------------ 组装观测

    def _make_obs(self, raw: np.ndarray, is_first: bool, is_terminal: bool) -> dict:
        """把游戏返回的 (N, obs_size) 观测拼上 one-hot 索引，组装成框架要的 dict。"""
        n = self.spec.num_agents
        parts = [raw]
        if self._add_agent_index:
            parts.append(self._agent_onehot)
        state = np.concatenate(parts, axis=1).astype(np.float32)
        assert state.shape == (n, self.observation_space["state"].shape[0]), state.shape
        return {
            "state": state,
            "is_first": np.full(n, is_first, dtype=bool),
            "is_terminal": np.full(n, is_terminal, dtype=bool),
        }

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

    print("-" * 68)
    if failures:
        print(f"FAILED ({len(failures)}): {', '.join(failures)}")
        raise SystemExit(1)
    print("全部通过。")


if __name__ == "__main__":
    _self_test()
