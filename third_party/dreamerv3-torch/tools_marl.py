"""多智能体数据收集 rollout。

================================================================================
归属说明
================================================================================
本文件**不属于上游 dreamerv3-torch**，是 CoDreamer 复现的一部分，由我们编写。

================================================================================
为什么不能用上游的 tools.simulate
================================================================================
上游收集循环的核心是这一行：

    results = [e.step(a) for e, a in zip(envs, action)]      # tools.py:170

它**逐个**调用每个 env 的 step，并要求**立即**返回。但多智能体世界里，
一次状态推进必须拿到**全部 N 个动作**。所以"把每个 agent 伪装成一个独立 env"
这条路走不通 —— env[0].step() 被调用时无从得知 env[1..N-1] 的动作。

于是本文件用【队伍级】环境（envs/seg_gym.py）配自写的收集循环：
    ① 每个队伍走一步，得到全部 N 个 agent 的 (obs, reward, done)
    ② 把 N 条记录**【分别】**写进 replay，成为 N 条独立的 episode 流

================================================================================
为什么"N 条独立 episode 流"是关键
================================================================================
IDreamer 的语义就是：每个 agent 有自己的世界模型、只用自己的经验训练、
彼此不观测。把 N 条流分开写之后：

    - replay 里每条样本天然就是**单智能体形状**
    - encoder / RSSM / 各预测头 / actor-critic **全部不用改**
    - 参数共享是免费的（同一套权重服务所有 agent）

"reshape 展平"的方案就落在这里，代价只是多写这个收集循环。

================================================================================
agent 调用约定：把 N 个 agent 铺平成 batch
================================================================================
每次队伍 step，我们把所有队伍的所有 agent 展平成 (n_teams * N, ·) 传给 agent
（Dreamer），它按 batch 处理，返回同样 batch 大小的动作。然后再拆回每个队伍。

⚠️ 副作用（**重要**）：Dreamer.__call__ 结尾会做 `self._step += len(reset)`，
而这里的 reset 长度是 `n_teams * N`。也就是说 `_step` 每队伍步增加 N，
比单智能体快 N 倍，于是 `_should_train` 调度会触发得频繁 N 倍。

要让"每多少个队伍步做一次梯度更新"与论文一致，配置里的 train_ratio 必须补偿：

    train_ratio_effective = train_ratio_paper / (num_agents * action_repeat)

推导：上游 `_should_train = Every(batch_size*batch_length/train_ratio)` 作用在 `_step` 上，
而单智能体时 `_step = env_step / action_repeat`，多智能体时 `_step = N * team_step`。
两者相差 N*action_repeat 倍，故 interval 需放大同样倍数，即 train_ratio 缩小同样倍数。

（在 configs.yaml 的 codreamer 配置块里有注释说明，别改漏。）
"""

from __future__ import annotations

import numpy as np

from tools import add_to_cache, convert, erase_over_episodes, save_episodes

__all__ = ["agent_episode_id", "simulate_marl"]


def agent_episode_id(team_id: str, agent_index: int) -> str:
    """队伍 id -> 该队伍第 k 个 agent 的 episode id。

    每个 agent 拥有独立的 episode 流，是 IDreamer 的语义要求。
    """
    return f"{team_id}-a{agent_index}"


def _split_action(action, n_teams: int, n_agents: int):
    """把 agent 返回的动作拆成 (n_teams, n_agents, ·) 的 numpy 数组。

    Dreamer._policy 返回的 policy_output 是 dict，其中 "action" 是 torch tensor，
    形状 (batch, act_dim)，batch = n_teams * n_agents。
    其余 key（如 logprob）同样按 batch 组织，会被写进 replay 供日志使用。
    """
    out = {}
    for key, value in action.items():
        arr = np.asarray(value.detach().cpu()) if hasattr(value, "detach") else np.asarray(value)
        if arr.ndim == 0:
            arr = np.full((n_teams * n_agents,), arr.item())
        out[key] = arr.reshape((n_teams, n_agents) + arr.shape[1:])
    return out


def simulate_marl(
    agent,
    envs,
    cache,
    directory,
    logger,
    is_eval=False,
    limit=None,
    steps=0,
    episodes=0,
    state=None,
):
    """多智能体版 tools.simulate。

    envs: 队伍级环境列表（envs/seg_gym.SequentialEstimateGameEnv）。
          每个 env 是一个【队伍】，内部有 num_agents 个 agent。

    状态元组与上游一致： (step, episode, done, length, obs, agent_state, reward)
    但 done / length / obs / reward 都是**按队伍**的，不是按 agent。
    """
    if not envs:
        raise ValueError("need at least one env")
    n_teams = len(envs)
    n_agents = envs[0].num_agents
    for e in envs:
        if e.num_agents != n_agents:
            raise ValueError("all envs must have the same number of agents")
    batch = n_teams * n_agents

    if state is None:
        step, episode = 0, 0
        done = np.ones(n_teams, bool)
        length = np.zeros(n_teams, np.int32)
        obs = [None] * n_teams
        agent_state = None
        reward = [0] * n_teams
    else:
        step, episode, done, length, obs, agent_state, reward = state

    eval_scores: list[float] = []
    eval_lengths: list[int] = []
    eval_done = False

    while (steps and step < steps) or (episodes and episode < episodes):
        # ---------------------------------------------------------- 重开完成的队伍
        if done.any():
            for i in np.flatnonzero(done):
                result = envs[i].reset()
                length[i] = 0
                # 每个 agent 各写一条初始记录，形成 N 条独立 episode 流
                for k in range(n_agents):
                    t = {key: convert(val[k]) for key, val in result.items()}
                    t["reward"] = 0.0
                    t["discount"] = 1.0
                    add_to_cache(cache, agent_episode_id(envs[i].id, k), t)
                obs[i] = result

        # ---------------------------------------------------------- 铺平成 batch
        keys = [k for k in obs[0] if "log_" not in k]
        flat_obs = {k: np.concatenate([o[k] for o in obs], 0) for k in keys}
        assert all(v.shape[0] == batch for v in flat_obs.values()), (
            "flattened obs batch mismatch",
            {k: v.shape for k, v in flat_obs.items()},
        )
        flat_done = np.repeat(done, n_agents)  # (batch,)
        assert flat_done.shape == (batch,), flat_done.shape

        # ---------------------------------------------------------- 问策略
        action, agent_state = agent(flat_obs, flat_done, agent_state)
        if not isinstance(action, dict):
            raise TypeError(
                "simulate_marl expects the policy to return a dict "
                f"(got {type(action)}); Dreamer._policy already does this"
            )
        acts = _split_action(action, n_teams, n_agents)  # (n_teams, n_agents, ·)

        # one-hot 动作 -> 整数下标（环境接口要的是下标）
        onehot = acts.get("action")
        if onehot is None:
            raise KeyError("policy output must contain an 'action' key")
        if onehot.ndim != 3:
            raise ValueError(f"action shape {onehot.shape}, expected (n_teams, n_agents, A)")
        indices = onehot.argmax(-1).astype(np.int64)  # (n_teams, n_agents)

        # ---------------------------------------------------------- 走一步
        results = [envs[i].step(indices[i]) for i in range(n_teams)]
        obs = [r[0] for r in results]
        reward = [r[1] for r in results]          # 每队 (N,)
        team_done = np.array([bool(r[2]) for r in results])
        infos = [r[3] for r in results]

        step += n_teams
        # episode 统一按【agent episode】计数：一局结束 = N 条 agent episode
        episode += int(team_done.sum()) * n_agents
        length += 1
        length *= 1 - team_done

        # ---------------------------------------------------------- 写入 N 条流
        for i in range(n_teams):
            o, r, d, info = results[i]
            for k in range(n_agents):
                transition = {key: convert(val[k]) for key, val in o.items()}
                for key, val in acts.items():
                    transition[key] = convert(val[i][k])
                transition["reward"] = convert(r[k])
                transition["discount"] = info.get(
                    "discount", np.array(1 - float(d), dtype=np.float32)
                )
                add_to_cache(cache, agent_episode_id(envs[i].id, k), transition)

        # ---------------------------------------------------------- 收尾
        if team_done.any():
            for i in np.flatnonzero(team_done):
                if is_eval:
                    # 评测时：N 个 agent 各算一条 episode
                    for k in range(n_agents):
                        ep = cache[agent_episode_id(envs[i].id, k)]
                        eval_lengths.append(len(ep["reward"]) - 1)
                        eval_scores.append(float(np.array(ep["reward"]).sum()))
                else:
                    for k in range(n_agents):
                        key = agent_episode_id(envs[i].id, k)
                        save_episodes(directory, {key: cache[key]})
                        ep_len = len(cache[key]["reward"]) - 1
                        score = float(np.array(cache[key]["reward"]).sum())
                        for lk in list(cache[key].keys()):
                            if "log_" in lk:
                                logger.scalar(lk, float(np.array(cache[key][lk]).sum()))
                                cache[key].pop(lk)
                    step_in_dataset = erase_over_episodes(cache, limit)
                    logger.scalar("dataset_size", step_in_dataset)
                    logger.scalar("train_return", score)
                    logger.scalar("train_length", ep_len)
                    logger.scalar("train_episodes", len(cache))
                    logger.write(step=logger.step)

        if (
            is_eval
            and not eval_done
            and episodes
            and len(eval_scores) >= episodes
        ):
            logger.scalar("eval_return", sum(eval_scores) / len(eval_scores))
            logger.scalar("eval_length", sum(eval_lengths) / len(eval_lengths))
            logger.scalar("eval_episodes", len(eval_scores))
            logger.write(step=logger.step)
            eval_done = True

    if is_eval:
        # 评测时只留最后一条，供 video_pred 使用
        while len(cache) > 1:
            cache.popitem(last=False)

    return (
        step - steps,
        episode - episodes,
        done,
        length,
        obs,
        agent_state,
        reward,
    )


# ============================================================================
# 自检 —— 用一个假的"策略"跑通收集循环，不需要任何模型。
#   在 third_party/dreamerv3-torch 目录下： python -m tools_marl
# ============================================================================


class _FakePolicy:
    """模仿 Dreamer._policy 的接口：吃 (batch, ·) 的 obs，吐 dict 动作。

    随机策略，但会记录被调用的次数与 batch 大小，用来验证铺平/拆分是否正确。
    """

    def __init__(self, num_actions: int, seed: int = 0):
        self.num_actions = num_actions
        self.rng = np.random.default_rng(seed)
        self.calls = []
        self.step_count = 0

    def __call__(self, obs, done, state):
        self.calls.append((obs["state"].shape[0], done.shape[0]))
        batch = obs["state"].shape[0]
        onehot = np.zeros((batch, self.num_actions), dtype=np.float32)
        idx = self.rng.integers(0, self.num_actions, batch)
        onehot[np.arange(batch), idx] = 1.0
        self.step_count += 1
        # 模拟 Dreamer 的 state 传递
        new_state = ({"dummy": np.zeros(batch)}, onehot)
        return {"action": onehot, "logprob": np.zeros(batch, dtype=np.float32)}, new_state


class _FakeLogger:
    def __init__(self):
        self.scalars = {}
        self.step = 0
        self.writes = 0

    def scalar(self, key, value):
        self.scalars.setdefault(key, []).append(value)

    def write(self, step=None):
        self.writes += 1

    def video(self, *a, **k):
        pass


def _self_test() -> None:
    import tempfile

    from envs.seg import Spec
    from envs.seg_gym import SequentialEstimateGameEnv

    failures: list[str] = []

    def check(name, cond, detail=""):
        print(f"  [{'ok  ' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
        if not cond:
            failures.append(name)

    print("simulate_marl 自检")
    print("-" * 68)

    spec = Spec()
    n = spec.num_agents

    # --- 单队伍
    envs = [SequentialEstimateGameEnv(spec, seed=1)]
    cache: dict = {}
    logger = _FakeLogger()
    policy = _FakePolicy(spec.num_actions, seed=0)

    with tempfile.TemporaryDirectory() as tmp:
        res = simulate_marl(
            policy, envs, cache, tmp, logger, limit=10**6, steps=2 * spec.horizon
        )
        step, episode, done, length, obs, agent_state, reward = res

        check("策略被调用次数 = 队伍步数", policy.step_count == 2 * spec.horizon,
              f"{policy.step_count} vs {2*spec.horizon}")
        check("每次调用 batch = N", all(b == n for b, _ in policy.calls),
              str(policy.calls[:3]))
        check("done 长度 = batch", all(d == n for _, d in policy.calls))

        # N 条独立 episode 流
        ids = sorted(cache.keys())
        check("产生了 N 条 episode 流", len(ids) == n, str(ids))
        check("id 命名带 agent 后缀",
              all(i.endswith(f"-a{k}") for k, i in enumerate(ids)))

        # 每条流的长度与内容
        for k, key in enumerate(ids):
            ep = cache[key]
            check(f"流 {k} 有 {spec.horizon + 1} 步（含初始）",
                  len(ep["reward"]) == spec.horizon + 1, str(len(ep["reward"])))
            check(f"流 {k} 首步 reward = 0（初始记录）", float(ep["reward"][0]) == 0.0)
            check(f"流 {k} state 是单智能体形状",
                  np.array(ep["state"]).shape[1:] == (spec.obs_size + n,),
                  str(np.array(ep["state"]).shape))
            check(f"流 {k} action 是 one-hot 且宽度 = |A|",
                  np.array(ep["action"]).shape[1:] == (spec.num_actions,),
                  str(np.array(ep["action"]).shape))
            check(f"流 {k} 后续 reward ≤ 0",
                  bool((np.array(ep["reward"])[1:] <= 1e-9).all()))

        check("episode 计数", episode == 1, str(episode))
        check("logger 记录了 train_return", "train_return" in logger.scalars)

    # --- 多队伍：batch 应为 n_teams * N
    n_teams = 3
    envs2 = [SequentialEstimateGameEnv(spec, seed=100 + i) for i in range(n_teams)]
    cache2: dict = {}
    logger2 = _FakeLogger()
    policy2 = _FakePolicy(spec.num_actions, seed=1)
    with tempfile.TemporaryDirectory() as tmp:
        simulate_marl(policy2, envs2, cache2, tmp, logger2, limit=10**6,
                      steps=2 * spec.horizon)
    check("多队伍时 batch = n_teams*N",
          all(b == n_teams * n for b, _ in policy2.calls), str(policy2.calls[:2]))
    check("多队伍产生 n_teams*N 条流",
          len(cache2) == n_teams * n, str(len(cache2)))

    # --- 评测模式的 episodes 计数
    envs3 = [SequentialEstimateGameEnv(spec, seed=7)]
    cache3: dict = {}
    logger3 = _FakeLogger()
    policy3 = _FakePolicy(spec.num_actions, seed=2)
    with tempfile.TemporaryDirectory() as tmp:
        simulate_marl(policy3, envs3, cache3, tmp, logger3, is_eval=True, episodes=n)
    check("评测模式记录 eval_return", "eval_return" in logger3.scalars,
          str(list(logger3.scalars)))
    check("评测模式 episode 数 = N", len(logger3.scalars.get("eval_episodes", [])) == 1)

    print("-" * 68)
    if failures:
        print(f"FAILED ({len(failures)}): {', '.join(failures)}")
        raise SystemExit(1)
    print("全部通过。")


if __name__ == "__main__":
    _self_test()
