"""通信探针：agent j 的信息，真的传到 agent i 了吗？走几跳？

================================================================================
为什么需要这个，而不是又画一条学习曲线
================================================================================
本项目的记录里，变体间的回报对比是一台坏仪器：σ_seed 是 3.3，而要测的效应量
只有 0.68~3.65，8 个 seed 才勉强显著（docs/reproduction-plan.md「最终结论」）。
而且它测的是**混合物** —— 世界模型质量、策略对模型的利用程度、优化噪声全搅在
一起，出了结果也说不清是哪一环。

这个探针把问题拆到最小：**通信到底传不传得过去，传几跳。**
它是架构的属性，不是学出来的，所以

  - 不需要训练（随机初始化的网络就能测）
  - 不需要 GPU（CPU 上几秒；本项目的 MLP 默认 device='cuda'，见下面"已知坑"）
  - 没有 seed 方差（同一份权重、同一份观测、共同随机数）

而且它是**可证伪的**：论文说"k 层 GNN = 在同一时刻内传播 k 跳"，那么
receptive field 就必须正好等于各段 GNN 层数。多一跳或少一跳都说明接线错了。

================================================================================
怎么做
================================================================================
`Dreamer._policy` 是 (obs, state) 的函数（`dreamer.py:153` 是唯一传图点）。
于是：**同一个 incoming state 下跑两次，只改一个 agent 的观测内容**，两次输出的差
就是"那个 agent 的信息对这个 agent 的影响"。

扰动方式：把 agent j 的**观测内容**（`state[:, :D]`）换成 agent k 的，
但**保留 j 自己的 one-hot 索引**（`state[:, D:]`）。图不动（位置没变）。
这样只有"j 掌握了什么信息"变了，别的一律不动。

================================================================================
★ 必须沿时间滚几步再测，否则测出来恒为 0 ★
================================================================================
Dreamer 的 `obs_step` 里，隐状态是这么来的：

    deter_t = img_step(prev_state_{t-1}, prev_action_{t-1})     ← 通信在这里
    stoch_t = posterior([deter_t, embed(obs_t)])                ← 观测只进这里

**`deter_t` 里根本不含 obs_t，只含上一步的 stoch。** 所以

  - 从 `state=None` 起步的**第一步**，deter 只由初始状态决定，对观测是常数；
    在这一步测"扰动 j 会不会改变 i 的 deter"，两个变体都恒为 0 —— 看起来像
    "RSSM 的通信没接线"，其实只是问错了位置。（实测踩过：绝对差恰好 0.000e+00。）
  - 正确的问法是**晚一步**：扰动 t 时刻 j 的观测 → 看 t+1 时刻 i 的 deter。

于是探针沿时间滚 T 步，在第 t 步扰动、在 t 与 t+1 各测一次。**每个量按它自己那一
时刻的图分组** —— agent 会移动，t 时刻隔 2 跳的两个 agent 在 t+1 可能只隔 1 跳，
统一用一张图会把实际 1 跳的样本混进"2 跳"那一行，让那一行永远不可能是 0。

================================================================================
实测结果（overcooked_demo，3 agent，C=2，60 条轨迹 × 6 步，随机初始化）
================================================================================
    跳数   deter(t+1) 相对变化   动作(t) 翻转   动作(t+1) 翻转   logprob(t) 相对变化   样本数
    --------------------------------------------------------------------------------------
    1           5.557%           6.13%          6.15%           3.07e-02        702
    2           0.114%           0.0000         0.0000        **恰好 0**         94
    IDreamer 每一格都是**恰好 0**

注意 1 跳那一行三个口径**都给出一致的"有"**（deter / 翻转率 / 连续量），而 2 跳
那一行**动作侧恰好是 0、deter 却非零**。这不是矛盾，而是两件事：

  - `动作(t)` / `logprob(t)`：**恰好 0** —— actor 只有 1 层 GNN，2 跳外没有通路，
    掩码是硬的（94 个样本上一个非零值都没有）。
  - `deter(t+1)` = 0.114%：**非零**，因为 `deter` 的 reach 其实是 **2 跳**，不是 1 跳。

⚠️ **探针免训练，所以这些幅度只是"路径存在"的证据，不是"哪个通道更强"的证据。**
上表比的是两个**随机初始化**的模型；训练后的幅度不可由它外推。

### `deter(t+1)` 为什么是 2 跳：两段 GNN 分处两个时刻

这不是"空间上多传了一跳"，也不是图随时间变化造成的。**决定性证据是消融**
（SEG，静态图，同一批 108 个 2 跳样本）：

    rssm=1 head=1 behavior=1   2 跳 deter = 0.000281         非零
    rssm=1 head=0 behavior=0   2 跳 deter = 恰好 0
    rssm=0 head=0 behavior=1   2 跳 deter = 恰好 0

**两段缺一不可**，所以路径必然是复合的：

    扰动 obs_t[j]
      → stoch_t[j] 变
      → 【图 t 上的 actor GNN】把 j 的信息传给 action_t[m]，m ∈ N_t(j)     ← 第 1 跳
      → action_t 是 t+1 时刻 GRU 的**输入**（`x = cat([prev_stoch, prev_action])`）
      → GRU_m 的输出变
      → 【图 t+1 上的 RSSM GNN】再聚合一跳                                  ← 第 2 跳
      → deter_{t+1}

**与图是否随时间变化无关**：SEG 的邻接矩阵只在 `reset()` 里采样一次
（`envs/seg.py:243`），`step()` 从不重采样，局内是静态的，而现象照旧。
3 跳恰好为 0 也自洽 —— 那需要第三段 GNN。

⚠️ **一处被实测否掉的解释，记在这里免得有人重新发明它**：曾把这条写成
"i 的 `deter_t` 里已经有 j 的信息，GRU 把它带到了 t+1"。**这是错的** ——
`deter_t` 可证不受 `obs_t` 影响（它由 `state_{t-1}` 算出，见上一节）。
载体是 `action_t`，而且它走的是 GRU 的**动作输入**，不是循环状态。

**结构性的主张必须落在 logprob 这一列上，不能落在翻转率上。** 翻转率是阈值化的量
（argmax 会把弱影响吃掉），"没测到翻转"既可能是真没通路、也可能只是样本不够；
而 (i,j) 对在同一局里共享状态、并不独立，有效样本量上限是 局数 × 可用 t 数，
设计效应可达数倍 —— 样本量论证在这里站不住。logprob 是连续量，于是"恰好没变"
是一条 **float 精确**断言：真有通路的话第一个样本就会给出非零值。

读法（论文默认 rssm/head/behavior 各 1 层）：

  - **动作(t) 的 reach = 1**，2 跳**恰好 0** → actor 自己那层 GNN 只传一跳。
  - **deter(t+1) 的 reach = 2**，3 跳恰好 0 → RSSM 那层本身只传一跳，但它在
    时间上被 actor 那层**接了一棒**（机制见上一节）。**两段缺一不可**，
    消融证据在同节。
  - **动作(t+1) 的 reach = 2**，3 跳归零 → 同样是复合。

所以"k 层 GNN = 在同一时刻内传播 k 跳"在**每段内部**成立，而**端到端的 reach
不等于单段层数**：两段分处相邻两个时刻的 GNN 会串成一条跨时刻的链
（actor@t → action_t → RSSM@t+1）。这也意味着**测 reach 时不能只看单段层数**，
`(rssm, behavior)` 的组合必须以实测为准。

IDreamer 逐格严格为零则是这条测量的对照：它证明测到的差异确实来自通信，
而不是探针自身的耦合。

================================================================================
已知坑（都不是本探针的 bug，但会咬人）
================================================================================
1. `networks.MLP.__init__` 的 `device` 默认值是**字面量 "cuda"**（networks.py:706），
   而 `config.encoder` 里没有这个键。所以整个代码库**在不暴露 CUDA 的情况下连
   构造都过不去**（`CUDA_VISIBLE_DEVICES=""` 会炸在 `torch.tensor(..., device="cuda")`）。
   本脚本因此需要一块可见的 GPU，但计算量极小。
2. `cfg.comm_variant` 的 yaml 默认是 `'idreamer'`。不显式设它，`codreamer` 的 run
   会静默建成 IDreamer，而探针照样打得出一张表 —— 两行一模一样，看着像
   "通信没效果"。所以这里调 `dreamer.resolve_comm_variant()`（唯一实现），
   并把参数量打出来当机械证据。

用法::

    python scripts/probe_comm_reach.py --task overcooked_demo
    python scripts/probe_comm_reach.py --task overcooked_demo --variant idreamer
    python scripts/probe_comm_reach.py --task overcooked_demo --gnn 1 0 0   # 只 RSSM 通信
"""

from __future__ import annotations

import argparse
import copy
import pathlib
import sys

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
D3T = REPO / "third_party" / "dreamerv3-torch"
sys.path.insert(0, str(D3T))
sys.path.insert(0, str(REPO / "scripts"))

import ruamel.yaml as yaml  # noqa: E402

import dreamer as dreamer_mod  # noqa: E402
from eval_policy import CaptureLogger, _NoDataset  # noqa: E402

OBS_KEYS = ("state", "is_first", "is_terminal", "adjacency", "edge_features")


def load_config(names: list[str]) -> argparse.Namespace:
    configs = yaml.safe_load((D3T / "configs.yaml").read_text())

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    defaults: dict = {}
    for name in names:
        recursive_update(defaults, configs[name])
    return argparse.Namespace(**defaults)


def hop_distances(adj: np.ndarray) -> np.ndarray:
    """(N,N) 的图跳数；不可达为 inf。对角线为 0。N 很小，Floyd–Warshall 足够。"""
    n = adj.shape[0]
    d = np.where(adj > 0, 1.0, np.inf)
    np.fill_diagonal(d, 0.0)
    for k in range(n):
        d = np.minimum(d, d[:, k, None] + d[None, k, :])
    return d


def collect_trajectories(env, n_episodes: int, horizon: int, seed: int) -> list[dict]:
    """滚 n_episodes 条长为 horizon 的随机轨迹，按时间步堆 batch。

    返回 list[T]，每项是 {key: (n_episodes*N, ...)} —— 与 simulate_marl 的布局一致
    （agent 轴铺平进 batch 维；`Dreamer._policy` 会自己还原成 (n_teams, N, ·)）。
    """
    per_step: list[dict[str, list]] = [
        {k: [] for k in OBS_KEYS} for _ in range(horizon)
    ]
    rng = np.random.default_rng(seed)
    kept = 0
    for ep in range(n_episodes):
        # vmas_env / overcooked_env 提供 seed(k) = "每局都用同一个种子"。
        # seg_gym 没有这个方法，退化成用环境自己的 RNG（局仍然各不相同，
        # 只是跨进程不可复现）。探针测的是架构属性，不依赖局的可复现性；
        # 依赖它的地方（scripts/eval_policy.py 的配对评测）走的是另一条路。
        if hasattr(env, "seed"):
            env.seed(seed * 1000 + ep)
        obs = env.reset()
        traj = [{k: np.asarray(obs[k]) for k in OBS_KEYS}]
        for _ in range(horizon - 1):
            a = rng.integers(0, env.action_space.n, size=env.num_agents)
            obs, _r, done, _info = env.step(a)
            traj.append({k: np.asarray(obs[k]) for k in OBS_KEYS})
            if done:
                break
        if len(traj) < horizon:
            # Overcooked 会在食物搬完时提前终止。少数几条短轨迹会破坏"每个时间步一批"
            # 的等长前提，直接丢掉而不是补齐 —— 补齐会造出并不存在的状态。
            continue
        kept += 1
        for t in range(horizon):
            for k in OBS_KEYS:
                per_step[t][k].append(traj[t][k])
    if kept == 0:
        raise SystemExit("一条完整轨迹都没收到，把 horizon 调小或 max_episode_steps 调大")
    return [
        {k: np.concatenate(v, axis=0) for k, v in step.items()} for step in per_step
    ]


def run_policy(agent, obs: dict, state, device: str, seed: int):
    """跑一次 `_policy`，返回 (动作, 下一步 latent 的 deter, 新 state)。

    ★ 必须固定 RNG ★
    `RSSM.obs_step` 会从 posterior **采样** `stoch`，走全局 RNG —— "同一批观测跑
    两次"本来就不是同一个结果。实测踩到过：deter 逐位相同而动作翻转率 65%，
    那张表看着像"通信在起作用"，其实全是重跑噪声。这里用**共同随机数**：同一时间步
    的 baseline 与扰动两次调用前都把种子重置成同一个值，于是没被扰动的 agent 抽到
    逐位相同的 stoch，差值只可能来自扰动。
    """
    tensors = {k: torch.as_tensor(v, device=device) for k, v in obs.items()}
    torch.manual_seed(seed)
    with torch.no_grad():
        # 签名是 _policy(obs, state, training) —— 没有 done 参数。
        out, new_state = agent._policy(tensors, state, False)
    latent, _action = new_state
    # logprob 是策略对自己**贪心动作**的对数概率。它是动作侧的**连续**口径 ——
    # 翻转率是阈值化的（弱影响会被 argmax 吃掉），而连续性让"恰好没变"
    # 变成一条 float 精确断言，不再依赖样本量。
    return (
        out["action"].cpu().numpy(),
        latent["deter"].cpu().numpy(),
        out["logprob"].cpu().numpy(),
        new_state,
    )


def _scalarise(lp: np.ndarray, n_episodes: int, n_agents: int) -> np.ndarray:
    """把 logprob 整成 (n_episodes, n_agents) 的标量表。

    不同 dist 的 logprob 末维可能是 () 或 (1)。形状对不上时直接炸 ——
    静默取错列会让整张表变成一个看似合理但无意义的数。
    """
    v = np.asarray(lp).reshape(n_episodes, n_agents, -1)
    if v.shape[-1] != 1:
        raise SystemExit(f"logprob 末维是 {v.shape[-1]}，期望 1（形状 {np.asarray(lp).shape}）")
    return v[..., 0]


def probe(agent, env, device: str, n_episodes: int, horizon: int, seed: int):
    obs_seq = collect_trajectories(env, n_episodes, horizon, seed)
    n_agents = env.num_agents
    d_obs = env.observation_space["state"].shape[0] - n_agents

    # ---- baseline 滚一遍，留下每个时刻的 incoming state ----
    states: list = [None]
    act_base: list = []
    lp_base: list = []
    for t in range(horizon):
        act, _det, lp, new_state = run_policy(
            agent, obs_seq[t], states[t], device, seed + t
        )
        act_base.append(act)
        lp_base.append(lp)
        states.append(new_state)
    # 第 t 次调用返回的 deter 记为 D_t。注意 D_t = img_step(states[t]) ——
    # 它由**上一步**的 stoch 决定，与 obs_t 无关（见文件头）。
    det_base = [states[t + 1][0]["deter"].cpu().numpy() for t in range(horizon)]

    # 自证：确定性（同一 time step 重跑必须逐位相同）
    act_re, det_re, lp_re, _ = run_policy(
        agent, obs_seq[0], states[0], device, seed + 0
    )
    if not (
        np.array_equal(act_re, act_base[0])
        and np.array_equal(det_re, det_base[0])
        and np.array_equal(lp_re, lp_base[0])
    ):
        raise SystemExit(
            "探针不自洽：同一 (obs, state, seed) 两次前向结果不同。先修确定性再看表。"
        )

    # ---- 逐对扰动 ----
    # ★ 每个量必须按**它自己那一时刻**的图分组 ★
    # agent 会移动：t 时刻隔 2 跳的两个 agent，t+1 时刻可能只隔 1 跳。
    # 如果统一用 t 时刻的跳数，名义上的"2 跳"样本里会混进实际 1 跳的，
    # 于是 2 跳那一行永远不可能是 0 —— 实测踩过：deter 在 2/3 跳还有
    # 0.73% / 0.056% 的"影响"，看着像泄漏，其实是分组分错了。
    #   act(t)                → 用图 t
    #   deter(t+1), act(t+1)  → 用图 t+1
    def hops_at(t):
        adj = obs_seq[t]["adjacency"].reshape(n_episodes, n_agents, n_agents)
        return [hop_distances(adj[e]) for e in range(n_episodes)]

    hop_t = {t: hops_at(t) for t in range(horizon)}

    max_hop = 0
    for t in range(1, horizon - 1):     # 与下面的扰动循环同范围（要用到 t+1）
        for e in range(n_episodes):
            for d in (hop_t[t][e], hop_t[t + 1][e]):
                fin = d[np.isfinite(d)]
                if fin.size:
                    max_hop = max(max_hop, int(fin.max()))

    shape = (max_hop + 1,)
    act_flip = np.zeros(shape)        # 第 t 次调用的动作（即时口径，按图 t 分组）
    act_flip_n = np.zeros(shape)      # 第 t+1 次调用的动作（含 RSSM 传上来的，按图 t+1）
    act_total = np.zeros(shape)       # 图 t 上的样本数
    n_total = np.zeros(shape)         # 图 t+1 上的样本数
    det_delta = np.zeros(shape)
    det_base_sum = np.zeros(shape)
    # 动作侧的**连续**口径：贪心动作未翻转时，logprob 是同一个动作在两个分布下的
    # 对数概率，差值就是分布移动的直接度量。翻转率是阈值化的量，弱影响会被 argmax
    # 吃掉，所以"2 跳恰好没有影响"这条主张必须落在连续量上才不依赖样本量。
    lp_delta = np.zeros(shape)
    lp_base_sum = np.zeros(shape)
    lp_n = np.zeros(shape)
    # 同一套连续口径，用在**第 t+1 次调用**上（复合 2 跳）。
    # 这一档原先只有翻转率，但复合信号走过两跳后衰减得翻不动 argmax —— 实测恰好
    # 0.0000/52 样本，断言因此失败。那是阈值造成的，不是没有传播：同一条主张在
    # 1 跳那一档早就改成连续量了，这里当时漏了。
    lp_delta_n = np.zeros(shape)
    lp_base_n = np.zeros(shape)
    lp_n_n = np.zeros(shape)
    # 扰动第 t 步；动作看第 t 次调用，deter 看第 t+1 次调用。
    # 第 t+1 次调用的 deter 需要 states[t+1]，而 baseline 只滚了 horizon 次调用
    # （下标 0..horizon-1），所以 t 最大到 horizon-2。
    for t in range(1, horizon - 1):
        for i in range(n_agents):
            for j in range(n_agents):
                if i == j:
                    continue
                k = (j + 1) % n_agents
                if k == i:
                    k = (k + 1) % n_agents
                obs_p = {kk: v.copy() for kk, v in obs_seq[t].items()}
                st = obs_p["state"].reshape(n_episodes, n_agents, -1)
                src = obs_seq[t]["state"].reshape(n_episodes, n_agents, -1)
                st[:, j, :d_obs] = src[:, k, :d_obs]
                obs_p["state"] = st.reshape(-1, st.shape[-1])
                # 自证：扰动必须真的改了输入（切片写错会让整张表恒为 0，
                # 而"全零"恰好又是 IDreamer 的正确结果 —— 错法与对法长得一样）
                if np.array_equal(obs_p["state"], obs_seq[t]["state"]):
                    raise SystemExit(f"扰动没有生效 (t={t}, i={i}, j={j}, d_obs={d_obs})")

                # (1) 第 t 次调用：扰动 obs_t → stoch_t 变了，deter_t 不变。
                #     动作在这一次就可能变（actor 的 GNN 聚合的是含 stoch 的 feat）。
                act_p_t, _det_ignore, lp_p_t, state_p_t1 = run_policy(
                    agent, obs_p, states[t], device, seed + t
                )
                # (2) 第 t+1 次调用：喂**原始** obs_{t+1}，但接着扰动过的 state。
                #     这才是 RSSM 的通信落地的地方 —— deter_{t+1} = img_step(state_p_t1)。
                #     少了这一步，测到的就只是"隐状态自己随时间的漂移"，
                #     与扰动无关（实测：IDreamer 会显示出 0.64 的巨大"变化"）。
                act_p_t1, det_p_t1, lp_p_t1, _ = run_policy(
                    agent, obs_seq[t + 1], state_p_t1, device, seed + t + 1
                )

                a0 = act_base[t].reshape(n_episodes, n_agents, -1).argmax(-1)
                a1 = act_p_t.reshape(n_episodes, n_agents, -1).argmax(-1)
                b0 = act_base[t + 1].reshape(n_episodes, n_agents, -1).argmax(-1)
                b1 = act_p_t1.reshape(n_episodes, n_agents, -1).argmax(-1)
                d0 = det_base[t + 1].reshape(n_episodes, n_agents, -1)
                d1 = det_p_t1.reshape(n_episodes, n_agents, -1)
                # logprob 每个样本一个标量；形状不对就直接炸，别静默取错列
                l0 = _scalarise(lp_base[t], n_episodes, n_agents)
                l1 = _scalarise(lp_p_t, n_episodes, n_agents)
                l0n = _scalarise(lp_base[t + 1], n_episodes, n_agents)
                l1n = _scalarise(lp_p_t1, n_episodes, n_agents)

                for e in range(n_episodes):
                    # 动作(t) 用图 t；deter(t+1) 与动作(t+1) 用图 t+1
                    h_now = hop_t[t][e][i, j]
                    h_next = hop_t[t + 1][e][i, j]
                    if np.isfinite(h_now):
                        h = int(h_now)
                        act_total[h] += 1
                        act_flip[h] += int(a1[e, i] != a0[e, i])
                        # 只在贪心动作**没翻**的样本上累计 logprob 差 ——
                        # 那样两次比的是同一个动作，差值就是纯粹的分布移动。
                        # 翻转的样本本身就是"移得很大"的证据，另行计数。
                        if a1[e, i] == a0[e, i]:
                            lp_delta[h] += abs(float(l1[e, i] - l0[e, i]))
                            lp_base_sum[h] += abs(float(l0[e, i]))
                            lp_n[h] += 1
                    if np.isfinite(h_next):
                        h = int(h_next)
                        n_total[h] += 1
                        act_flip_n[h] += int(b1[e, i] != b0[e, i])
                        det_delta[h] += float(np.linalg.norm(d1[e, i] - d0[e, i]))
                        det_base_sum[h] += float(np.linalg.norm(d0[e, i]))
                        if b1[e, i] == b0[e, i]:
                            lp_delta_n[h] += abs(float(l1n[e, i] - l0n[e, i]))
                            lp_base_n[h] += abs(float(l0n[e, i]))
                            lp_n_n[h] += 1

    rel = np.where(det_base_sum > 0, det_delta / np.maximum(det_base_sum, 1e-12), 0.0)
    flip = np.where(act_total > 0, act_flip / np.maximum(act_total, 1), 0.0)
    flip_n = np.where(n_total > 0, act_flip_n / np.maximum(n_total, 1), 0.0)
    return {
        "rel": rel,
        "flip": flip,
        "flip_n": flip_n,
        "cnt_t": act_total.astype(int),
        "cnt_n": n_total.astype(int),
        "det_delta": det_delta,
        "det_base": det_base_sum,
        "lp_delta": lp_delta,
        "lp_base": lp_base_sum,
        "lp_n": lp_n.astype(int),
        "lp_delta_n": lp_delta_n,
        "lp_base_n": lp_base_n,
        "lp_n_n": lp_n_n.astype(int),
    }


def assert_zero_comm_control(layers, res):
    """★ 对照：一处 GNN 都没有时，任何跨 agent 的影响都必须**恰好为 0**。★

    这是整个探针的"测量本身没坏"证明。它抓的正是我犯过的那个错：把扰动调用的
    deter 拿去比 baseline 的**下一步** deter，测到的是隐状态自己的时间漂移而非
    扰动的影响 —— 于是完全没有通信的 IDreamer 也显示出 0.64 的"变化"。
    有这条断言，那种错误会当场炸，而不是写进一份看起来很像结论的表。
    """
    if any(v > 0 for v in layers):
        return
    bad = []
    if float(np.max(res["det_delta"])) != 0.0:
        bad.append(f"deter 变化 {float(np.max(res['det_delta'])):.3e} != 0")
    if float(np.max(res["flip"])) != 0.0:
        bad.append(f"当前步动作翻转 {float(np.max(res['flip'])):.4f} != 0")
    if float(np.max(res["flip_n"])) != 0.0:
        bad.append(f"下一步动作翻转 {float(np.max(res['flip_n'])):.4f} != 0")
    if float(np.max(res["lp_delta"])) != 0.0:
        bad.append(f"logprob 变化 {float(np.max(res['lp_delta'])):.3e} != 0")
    if bad:
        raise SystemExit(
            "对照失败：GNN 层数全为 0（无任何通信），却测到了跨 agent 的影响 ——\n  "
            + "\n  ".join(bad)
            + f"\n样本数 {res['cnt_n'].tolist()}。这说明探针本身有耦合，先修探针。"
        )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="overcooked_demo",
                   help="seg_seq / vmas_flocking / overcooked_demo")
    p.add_argument("--variant", default="codreamer", choices=["idreamer", "codreamer"])
    p.add_argument("--gnn", type=int, nargs=3, metavar=("RSSM", "HEAD", "BEHAVIOR"),
                   default=None, help="逐段覆盖 GNN 层数（默认跟随 comm_variant）")
    p.add_argument("--episodes", type=int, default=24, help="滚多少条轨迹")
    p.add_argument("--horizon", type=int, default=6,
                   help="每条轨迹滚几步；必须 >= 3。可用的扰动时刻是 1..horizon-2 "
                        "（第 0 步的 deter 对观测是常数，最后一步没有下一步可看）")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    # 通信半径是给"把某几跳的样本量拉起来"用的（切比雪夫距离在整数网格上只取
    # 整数值，所以 2 与 2.5 等价）。
    # **没有 --agents**：Overcooked 的 agent 数由布局里的玩家数字决定，与布局
    # 不一致会直接报错 —— 它不是一个能独立调的量。
    p.add_argument("--comm-radius", type=float, default=None,
                   help="覆盖 overcooked_comm_radius")
    args = p.parse_args()

    if args.horizon < 3:
        raise SystemExit(
            "horizon 必须 >= 3：第 0 步的 deter 对观测是常数（晚一步），"
            "最后一步没有下一步可看 deter，中间至少留一个可用时刻"
        )

    cfg = copy.deepcopy(load_config(["defaults", "codreamer"]))
    cfg.task = args.task
    cfg.envs = 1
    cfg.comm_variant = args.variant
    cfg.seed = args.seed
    if args.gnn is not None:
        cfg.gnn_layers, cfg.gnn_head_layers, cfg.gnn_behavior_layers = args.gnn
    cfg.device = args.device
    if args.comm_radius is not None:
        cfg.overcooked_comm_radius = args.comm_radius
    # 探针**必须有图**（它测的就是图上的传播）。各 suite 的开关键名不同，
    # 统一打开 —— seg 的默认是 false，不打开会在取 obs["adjacency"] 时 KeyError。
    for key in ("seg_graph_obs", "vmas_graph_obs", "overcooked_graph_obs"):
        if hasattr(cfg, key):
            setattr(cfg, key, True)
    # ★ 必须调这个 ★ 它是把单一开关展开到三处 GNN 的唯一实现（dreamer.py:434）。
    # 绕过它的后果实测过：两个变体都建成 IDreamer，探针照样打出一张表，两行一样。
    # 只在值 < 0 时改写，所以上面的 --gnn 覆盖仍然生效。
    dreamer_mod.resolve_comm_variant(cfg)

    env = dreamer_mod.make_env(cfg, "eval", 0)
    n = env.num_agents
    cfg.num_agents = n
    cfg.num_actions = env.action_space.n

    torch.manual_seed(args.seed)
    agent = dreamer_mod.Dreamer(
        env.observation_space, env.action_space, cfg, CaptureLogger(), _NoDataset()
    ).to(args.device)
    agent.requires_grad_(requires_grad=False)
    agent.eval()

    layers = (cfg.gnn_layers, cfg.gnn_head_layers, cfg.gnn_behavior_layers)
    assert -1 not in layers, "resolve_comm_variant 没展开干净"
    wm = sum(p.numel() for p in agent._wm.parameters())
    ac = sum(p.numel() for p in agent._task_behavior.actor.parameters())
    va = sum(p.numel() for p in agent._task_behavior.value.parameters())

    print("通信探针 —— 信息从 agent j 传到 agent i 吗，走几跳")
    print("=" * 78)
    print(f"  task={cfg.task}  agents={n}  观测维={env.observation_space['state'].shape[0]}")
    print(f"  variant={args.variant}  GNN 层数 rssm/head/behavior={layers}")
    print(f"  预期：下一步 deter 的 reach = rssm={layers[0]}；"
          f"当前步动作的 reach = rssm({layers[0]}) + behavior({layers[2]})")
    print(f"  参数量 world_model={wm:,} actor={ac:,} value={va:,}"
          f"   （两个变体这几个数必须不同，否则 comm_variant 没生效）")
    print(f"  {args.episodes} 条轨迹 × {args.horizon} 步  seed={args.seed}  device={args.device}")
    print()

    res = probe(
        agent, env, device=args.device, n_episodes=args.episodes,
        horizon=args.horizon, seed=args.seed,
    )
    assert_zero_comm_control(layers, res)
    rel, flip, flip_n = res["rel"], res["flip"], res["flip_n"]
    cnt_t, cnt_n = res["cnt_t"], res["cnt_n"]

    print("  跳数   deter(t+1) 相对变化   动作(t) 翻转   动作(t+1) 翻转   "
          "logprob(t) 相对    logprob(t+1) 相对   样本数(t / t+1)")
    print("  " + "-" * 122)
    for h in range(len(rel)):
        lp_rel = (
            res["lp_delta"][h] / res["lp_base"][h] if res["lp_base"][h] > 0 else 0.0
        )
        lp_rel_n = (
            res["lp_delta_n"][h] / res["lp_base_n"][h]
            if res["lp_base_n"][h] > 0 else 0.0
        )
        print(
            f"  {h:<5d}  {rel[h]:>17.6f}   {flip[h]:>11.4f}   {flip_n[h]:>12.4f}   "
            f"{lp_rel:>15.3e}   {lp_rel_n:>17.3e}   {cnt_t[h]:>7d} / {cnt_n[h]:>6d}"
        )
    print()
    print("  deter(t+1) = 扰动 obs_t 之后，再走一步看到的隐状态变化（连续量）")
    print("              → 只反映 RSSM 里的通信，reach 应等于 rssm 层数")
    print("  动作(t)    = 同一时刻的动作翻转率（阈值化量）")
    print("  动作(t+1)  = 两步之和 → reach 应等于 rssm + behavior 层数")
    print("  logprob    = 贪心动作未翻转的样本上，策略对自己动作的对数概率变化（连续量）")
    print("  ⚠️ 两列 logprob 才是**判据**，两列翻转率只是参考。翻转率要求影响大到改掉")
    print("     argmax，弱影响会被吃掉 —— 实测 2 跳的复合影响在 52/52 样本上都有")
    print("     ~1e-2 的分布移动，翻转率却是 0.0000。用翻转率下结论会得到相反的答案。")
    print("              翻转率是阈值化的、抓不住边缘效应；这条能，而且它在"
          "『恰好没变』上是 float 精确的")
    print("  跳数 0 不会出现（i≠j）。不可达的 (i,j) 对已跳过。")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
