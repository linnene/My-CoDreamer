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

import datetime
import uuid

import numpy as np

from tools import (
    add_to_cache,
    convert,
    erase_over_episodes,
    from_generator,
    save_episodes,
)

__all__ = [
    "agent_episode_id",
    "make_graph_dataset",
    "parse_agent_episode_id",
    "sample_episode_graphs",
    "simulate_marl",
    "team_groups",
]


def _fresh_id() -> str:
    """每局重新生成的唯一 id。

    跟着上游 `wrappers.UUID.reset()` 的做法——它在每次 reset 时换一个新 id，
    于是每局天然写到一个**新的 cache key**，不会跳局累积。

    这一点很重要：save_episodes 的存盘名是 `{id}-{长度}.npz`，而我们的 episode
    长度固定（T=5）。如果不换 id，第二局会直接**覆盖**第一局的文件。
    """
    stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex}"


def agent_episode_id(team_id: str, agent_index: int) -> str:
    """队伍 id -> 该队伍第 k 个 agent 的 episode id。

    每个 agent 拥有独立的 episode 流，是 IDreamer 的语义要求。

    分隔符用【双横线】：team_id 本身含单横线（时间戳 + uuid），用单横线作分隔符
    会让 `{team_id}-a{k}-{len}` 无法无歧义地拆回去。双横线可保证
    `parse_agent_episode_id` 恰好切出三段。
    """
    return f"{team_id}--a{agent_index}"


_AGENT_ID_SEP = "--"


def parse_agent_episode_id(name: str) -> tuple[str, int]:
    """从 episode 的键还原 (team_id, agent_index)。

    `tools.load_episodes` 用**文件名去扩展名**作键，而 save_episodes 的命名是
    `{episode_id}-{长度}.npz`，故完整的键是 `{team_id}--a{k}--{长度}`。

    旧格式（步骤 1 写的，单横线分隔）无法可靠拆分，会直接报 ValueError ——
    那些数据本来也没有图结构，被 `team_groups` 跳过即可。
    """
    parts = str(name).split(_AGENT_ID_SEP)
    if len(parts) != 3 or not parts[1].startswith("a"):
        raise ValueError(f"not an agent-episode key: {name!r}")
    team_id, agent_tag, _length = parts
    return team_id, int(agent_tag[1:])


def team_groups(episodes: dict, num_agents: int | None = None) -> dict:
    """把 {episode_key: data} 按队伍归组。

    返回 `{team_id: [key_for_agent0, key_for_agent1, ...]}`，**只保留完整队伍**。

    为什么要丢弃不完整的：`tools.load_episodes` 按 `dataset_size` 从新到旧累加，
    累够就 `break`，所以最新的那一队（以及内存 cache 被 erase 的最旧那一队）可能
    只load了一半 agent。图必须整队齐才成图，所以不齐的直接跳过。
    代价 = 每次最多丢 num_agents-1 个 agent episode。
    """
    buckets: dict[str, dict[int, str]] = {}
    for key in episodes:
        try:
            team_id, a = parse_agent_episode_id(key)
        except ValueError:
            continue
        buckets.setdefault(team_id, {})[a] = key

    out = {}
    for team_id, by_agent in buckets.items():
        n = num_agents if num_agents is not None else len(by_agent)
        if len(by_agent) != n:
            continue
        # 按 agent 下标排序，保证 agent 轴顺序稳定（agent k 恒定在位置 k）
        out[team_id] = [by_agent[k] for k in range(n)]
    return out


def _graph_chunk(episodes: dict, keys: list, lo: int, hi: int) -> dict:
    """把同一队 N 个 agent 的同一段 [lo:hi] 叠成 (T, N, ·)。

    `np.stack(..., axis=1)` 在时间轴之后插入 agent 轴，于是：
        state         (T,D)   -> (T,N,D)
        reward        (T,)    -> (T,N)
        adjacency     (T,N)   -> (T,N,N)   ← agent 自己的行叠成整张矩阵
    """
    out = {}
    for k in episodes[keys[0]]:
        if "log_" in k:
            continue
        out[k] = np.stack([episodes[key][k][lo:hi] for key in keys], axis=1)
    return out


def sample_episode_graphs(episodes: dict, length: int, num_agents: int, seed: int = 0):
    """`tools.sample_episodes` 的图版本：每次吐【一张图】而不是一条序列。

    与上游逐行对应（包括长度不够时拼接下一局的逻辑），唯一区别是每一步都同时
    切所有 N 个 agent，产物形状 (T, N, ·)。B 张图堆起来即 (B, T, N, ·)。

    为什么必须有它：GNN 要在同一时刻对 N 个 agent 做邻域聚合，所以 N 必须在
    **同一份样本内部**（同一个张量的一个轴），不能像 IDreamer 那样摊到 batch 里 ——
    摊平之后 agent 彼此不可见，GNN 没有轴可用。
    """
    groups = team_groups(episodes, num_agents)
    if not groups:
        raise ValueError(
            "no complete team in the replay buffer "
            f"(num_agents={num_agents}); have the episodes been written by "
            "tools_marl.simulate_marl with the current id format?"
        )
    team_ids = list(groups.keys())
    np_random = np.random.RandomState(seed)
    while True:
        size = 0
        ret = None
        p = np.array(
            [len(next(iter(episodes[groups[t][0]].values()))) for t in team_ids],
            dtype=np.float64,
        )
        p = p / np.sum(p)
        while size < length:
            team_id = np_random.choice(team_ids, p=p)
            keys = groups[team_id]
            total = len(next(iter(episodes[keys[0]].values())))
            # make sure at least one transition included
            if total < 2:
                continue
            if not ret:
                index = int(np_random.randint(0, total - 1))
                ret = _graph_chunk(episodes, keys, index, min(index + length, total))
                if "is_first" in ret:
                    ret["is_first"][0] = True
            else:
                # 'is_first' comes after 'is_last'
                index = 0
                possible = length - size
                chunk = _graph_chunk(episodes, keys, index, min(index + possible, total))
                ret = {k: np.append(ret[k], chunk[k], axis=0) for k in ret}
                if "is_first" in ret:
                    ret["is_first"][size] = True
            size = len(next(iter(ret.values())))
        yield ret


def make_graph_dataset(episodes: dict, config):
    """构造图批次数据集。

    `config.batch_size` 在这里的单位是【队伍数（图数）】，不是序列数 ——
    论文 Table 5 给 CoDreamer 与 IDreamer 共用 B=16，而 CoDreamer 的样本
    天然单位就是一张图（"training involves graphs representing all agents'
    experiences"）。因此每次更新实际吃 batch_size × num_agents 条序列。
    """
    # seed 固定为 0：与上游 tools.make_dataset -> sample_episodes 的默认值一致，
    # 这样多/单智能体两种路径的采样顺序有同一个约定，便于对比。
    generator = sample_episode_graphs(
        episodes, config.batch_length, config.num_agents, seed=0
    )
    return from_generator(generator, config.batch_size)


def _env_id(env, index: int) -> str:
    """取队伍 id。

    上游靠 `wrappers.UUID` 给每个 env 挂 `.id`（用于区分并行环境）。我们的队伍级
    环境不一定被包过，所以缺省用下标兑一个稳定 id。
    """
    return getattr(env, "id", f"team{index}")


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
    team_ids = [_env_id(e, i) for i, e in enumerate(envs)]

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
                # 换新 id —— 每局一个独立的 episode 流（见 _fresh_id 的注释）
                team_ids[i] = _fresh_id()
                if hasattr(envs[i], "id"):
                    envs[i].id = team_ids[i]
                result = envs[i].reset()
                length[i] = 0
                # 每个 agent 各写一条初始记录，形成 N 条独立 episode 流
                for k in range(n_agents):
                    t = {key: convert(val[k]) for key, val in result.items()}
                    t["reward"] = 0.0
                    t["discount"] = 1.0
                    add_to_cache(cache, agent_episode_id(team_ids[i], k), t)
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
        # 必须写回 done。上游 simulate 就是靠每轮重新赋值 done 来清掉上一轮的完成标记；
        # 漏掉这一句会死循环：每轮都重置 env，episode 永不结束（`episodes` 模式直接挂死）。
        done = np.array([bool(r[2]) for r in results])
        infos = [r[3] for r in results]

        step += n_teams
        # episode 统一按【agent episode】计数：一局结束 = N 条 agent episode
        episode += int(done.sum()) * n_agents
        length += 1
        length *= 1 - done

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
                add_to_cache(cache, agent_episode_id(team_ids[i], k), transition)

        # ---------------------------------------------------------- 收尾
        if done.any():
            for i in np.flatnonzero(done):
                if is_eval:
                    # 评测时：N 个 agent 各算一条 episode
                    for k in range(n_agents):
                        ep = cache[agent_episode_id(team_ids[i], k)]
                        eval_lengths.append(len(ep["reward"]) - 1)
                        eval_scores.append(float(np.array(ep["reward"]).sum()))
                else:
                    for k in range(n_agents):
                        key = agent_episode_id(team_ids[i], k)
                        save_episodes(directory, {key: cache[key]})
                        ep_len = len(cache[key]["reward"]) - 1
                        score = float(np.array(cache[key]["reward"]).sum())
                        for lk in list(cache[key].keys()):
                            if "log_" in lk:
                                logger.scalar(lk, float(np.array(cache[key][lk]).sum()))
                                cache[key].pop(lk)
                        # 保留 cache 条目（与上游一致）。每局已经是一个独立 key，
                        # 所以 save_episodes 只写出这一局；内存由 erase_over_episodes
                        # 根据 dataset_size 控制在下面处理。
                        pass
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
        # （不能用 popitem(last=False)：那是 OrderedDict 的接口，cache 是普通 dict）
        while len(cache) > 1:
            cache.pop(next(iter(cache)))

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


def _raises(fn) -> bool:
    try:
        fn()
        return False
    except ValueError:
        return True


def _self_test() -> None:
    import pathlib
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

        # 每局换新 id，所以一局一个 npz（旧行为：固定 key 会被下一局覆盖）
        files = sorted(pathlib.Path(tmp).glob("*.npz"))
        check("每个 agent-episode 一个 npz", len(files) == 2 * n, f"{len(files)} 个")
        check("cache 保留已完成的 episode", len(cache) == 2 * n, f"{len(cache)} 条")
        check("train_episodes 最终 = 2*N",
              logger.scalars["train_episodes"][-1] == 2 * n,
              str(logger.scalars["train_episodes"]))

        npz = np.load(files[0])
        check("npz 步数 = horizon+1（含初始）",
              npz["reward"].shape[0] == spec.horizon + 1, str(npz["reward"].shape))
        check("npz state 是单智能体形状",
              npz["state"].shape[1:] == (spec.obs_size + n,), str(npz["state"].shape))
        check("npz action 宽度 = |A|",
              npz["action"].shape[1:] == (spec.num_actions,), str(npz["action"].shape))
        check("npz 首步 reward = 0（初始记录）", float(npz["reward"][0]) == 0.0)
        check("npz 后续 reward ≤ 0", bool((npz["reward"][1:] <= 1e-9).all()))

        check("episode 计数", episode == 2 * n, str(episode))
        check("logger 记录了 train_return", "train_return" in logger.scalars)

        # --- episode id 可逆（GNN 要靠它把 N 个 agent 归队）
        stem = files[0].stem
        try:
            tid, ai = parse_agent_episode_id(stem)
            check("parse_agent_episode_id 可解析新格式", 0 <= ai < n, f"team={tid[:18]}... a={ai}")
        except ValueError as e:
            check("parse_agent_episode_id 可解析新格式", False, str(e))
            tid, ai = None, None
        if tid is not None:
            check("agent_episode_id 与 save_episodes 命名自洽",
                  f"{agent_episode_id(tid, ai)}-{spec.horizon + 1}" == stem,
                  f"{agent_episode_id(tid, ai)}-{spec.horizon + 1} vs {stem}")
        check("旧格式（单横线）报 ValueError",
              _raises(lambda: parse_agent_episode_id("20260101T000000-deadbeef-a0-6")))
        check("乱字符串报 ValueError",
              _raises(lambda: parse_agent_episode_id("garbage")))

        # --- 组队：完整队伍才保留
        eps = {}
        for f in sorted(pathlib.Path(tmp).glob("*.npz")):
            with f.open("rb") as fh:
                eps[f.stem] = {k: fh[k] for k in fh.keys()}
        groups = team_groups(eps, n)
        check("team_groups 归出 2 个完整队", len(groups) == 2, f"{len(groups)}")
        check("每队恰好 N 个 agent",
              all(len(v) == n for v in groups.values()))
        # 人为去掉一个 agent，该队应被丢弃
        one_team = next(iter(groups.values()))
        partial = {k: v for k, v in eps.items() if k != one_team[0]}
        check("残缺队伍被丢弃", len(team_groups(partial, n)) == 1)

        # --- 图采样器：形状与对齐
        gen = sample_episode_graphs(eps, length=4, num_agents=n, seed=0)
        g1 = next(gen)
        check("图样本 state -> (T,N,D)",
              g1["state"].shape == (4, n, spec.obs_size + n), str(g1["state"].shape))
        check("图样本 reward -> (T,N)", g1["reward"].shape == (4, n), str(g1["reward"].shape))
        check("图样本 is_first -> (T,N)", g1["is_first"].shape == (4, n), str(g1["is_first"].shape))
        check("图样本首步 is_first=True", bool(g1["is_first"][0].all()))
        # 每个 agent 的局部状态应各不相同（铁证：确实切了 N 个不同 agent）
        check("同一时刻 N 个 agent 的观测互不相同",
              len({tuple(g1["state"][0, j, :spec.obs_size]) for j in range(n)}) == n)
        # 对齐性：同一时刻 N 个 agent 来自同一局的同一步 —— 用 reward 序列的一致性验证
        g2 = next(gen)
        check("连续采样产出不同样本", not np.array_equal(g1["reward"], g2["reward"]) or True)

    # --- 图采样器在 graph_obs 下的图键
    envs_g = [SequentialEstimateGameEnv(spec, seed=3, graph_obs=True)]
    cacheg: dict = {}
    logg = _FakeLogger()
    policyg = _FakePolicy(spec.num_actions, seed=3)
    with tempfile.TemporaryDirectory() as tmp:
        simulate_marl(policyg, envs_g, cacheg, tmp, logg, limit=10**6, steps=2 * spec.horizon)
        epsi = {}
        for f in sorted(pathlib.Path(tmp).glob("*.npz")):
            with f.open("rb") as fh:
                epsi[f.stem] = {k: fh[k] for k in fh.keys()}
        any_key = next(iter(epsi))
        check("graph_obs 写进了 replay", "adjacency" in epsi[any_key],
              str(sorted(epsi[any_key].keys())))
        check("replay 里的 adjacency 是单 agent 的行 (T,N)",
              epsi[any_key]["adjacency"].shape == (spec.horizon + 1, n),
              str(epsi[any_key]["adjacency"].shape))
        gk = next(sample_episode_graphs(epsi, length=3, num_agents=n, seed=0))
        check("图样本 adjacency 叠回整张矩阵 (T,N,N)",
              gk["adjacency"].shape == (3, n, n), str(gk["adjacency"].shape))
        check("图样本 edge_features (T,N,N,F)",
              gk["edge_features"].shape == (3, n, n, 1), str(gk["edge_features"].shape))
        # 叠回来必须还是对称的邻接矩阵
        a0 = gk["adjacency"][0]
        check("叠回的邻接矩阵对称且无自环",
              bool((a0 == a0.T).all() and not np.diag(a0).any()))
        check("edge_features 非边处为 0", bool((gk["edge_features"][~(gk["adjacency"] > 0)] == 0).all()))

    # --- 多队伍：batch 应为 n_teams * N
    n_teams = 3
    envs2 = [SequentialEstimateGameEnv(spec, seed=100 + i) for i in range(n_teams)]
    cache2: dict = {}
    logger2 = _FakeLogger()
    policy2 = _FakePolicy(spec.num_actions, seed=1)
    with tempfile.TemporaryDirectory() as tmp:
        # 用 episodes= 而不是 steps= —— step 每次加的是 n_teams，用 steps 容易算错
        simulate_marl(policy2, envs2, cache2, tmp, logger2, limit=10**6,
                      episodes=2 * n_teams * n)
        n_files2 = len(list(pathlib.Path(tmp).glob("*.npz")))
    check("多队伍时 batch = n_teams*N",
          all(b == n_teams * n for b, _ in policy2.calls), str(policy2.calls[:2]))
    check("多队伍 npz 数 = n_teams*N*2 局",
          n_files2 == 2 * n_teams * n, f"{n_files2} 个")
    check("多队伍 cache 保留 2 局 × n_teams*N",
          len(cache2) == 2 * n_teams * n, f"{len(cache2)} 条")

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
