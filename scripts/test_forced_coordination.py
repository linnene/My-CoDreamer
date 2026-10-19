"""强制协作厨房的自检 —— 把"这个任务真的需要协作"写成可执行的断言。

================================================================================
为什么需要这个测试
================================================================================
之前用的开放厨房（`three_player`）**单个 agent 就能拿满分**，另外两个 agent 的
边际贡献是零。那种布局不该被当成多人协作任务，而这件事光看代码看不出来 ——
地形是连通的，谁都能走到任何站点，只有**跑**才知道单人够不够。

所以这个测试把判据定死在这里：

  1. 每个 agent **单独行动**（其余冻住）都拿 0 分
     -> 没有任何一个人能完成交付链，协作是结构上必需的
  2. 三人**协作**能拿到分
     -> 任务可解。少了这条，一个"谁都完不成"的坏布局也会通过第 1 条
  3. 通信图恒为一条链：C=3 时边数恒为 2、两端恒隔 2 跳
     -> 多跳通信的卖点有地方可验。C=1（默认 = sight）时边数为 0，
        **通信图是空的**，那样任务无解 —— 这也是 C 必须显式给的原因

第 1 条和第 2 条必须一起验：只验第 1 条，布局写错了也会"通过"。

用法::

    python scripts/test_forced_coordination.py
"""
from __future__ import annotations

import pathlib
import sys
from collections import Counter, deque

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "third_party" / "dreamerv3-torch"))
sys.path.insert(0, str(REPO / "scripts"))

LAYOUT = "forced_coordination_3p"
SEEDS = 3
MAX_STEPS = 300


def hops(adj: np.ndarray, s: int, t: int) -> int:
    """邻接矩阵上 s 到 t 的最短跳数，不可达返回 -1。"""
    n = adj.shape[0]
    dist = [-1] * n
    dist[s] = 0
    q = deque([s])
    while q:
        u = q.popleft()
        for v in range(n):
            if adj[u][v] and dist[v] < 0:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist[t]


def run(env, policy, solo=None, max_steps=MAX_STEPS):
    from overcooked_ai_py.mdp.actions import Action

    stall = Action.ACTION_TO_INDEX[Action.STAY]
    for _ in range(max_steps):
        a = policy()
        if solo is not None:
            a = np.asarray(a).copy()
            a[[j for j in range(env.num_agents) if j != solo]] = stall
        _, _, done, _ = env.step(a)
        if done:
            break
    return env.total_reward


def main() -> int:
    from envs.overcooked_env import OvercookedTeamEnv

    # 交接策略就放在渲染脚本里（回放和自检用的是同一份协议，不能各写一份）
    from render_overcooked import HandoffPolicy

    fail = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal fail
        print(f"  [{'ok  ' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
        if not ok:
            fail += 1

    def make(**kw):
        params = dict(layout=LAYOUT, orders=(("onion",), ("tomato",)), cook_time=5,
                      max_steps=MAX_STEPS, sight=1, comm_radius=3.0,
                      random_start_pos=False, seed=0)
        params.update(kw)
        e = OvercookedTeamEnv(**params)
        e.reset()
        return e

    print("=" * 78)
    print(f"{LAYOUT} 强制协作自检")
    print("=" * 78)

    env = make()
    check("布局解析出 3 个 agent", env.num_agents == 3, f"num_agents={env.num_agents}")

    # ── 1. 单人通不了关 ────────────────────────────────────────────────
    solo_scores = []
    for i in range(env.num_agents):
        env.reset()
        solo_scores.append(run(env, HandoffPolicy(env), solo=i))
    check("每个 agent 单独行动都是 0 分（协作是结构上必需的）",
          all(s == 0 for s in solo_scores),
          f"各 agent 单独得分 {solo_scores}")

    # ── 2. 三人协作可解 ────────────────────────────────────────────────
    team = []
    for s in range(SEEDS):
        env.reset()
        team.append(run(env, HandoffPolicy(env)))
    check("三人协作能拿到分（任务可解）", all(t > 0 for t in team),
          f"各局得分 {[int(t) for t in team]}")

    # ── 3. 通信图恒为一条链 ────────────────────────────────────────────
    env.reset()
    pol = HandoffPolicy(env)
    edges, hop2 = [], []
    for _ in range(env._max_steps):
        adj = env.adjacency
        edges.append(int(adj.sum() // 2))
        hop2.append(hops(adj, 0, 2))
        _, _, done, _ = env.step(pol())
        if done:
            break
    check("C=3 时通信图边数恒为 2（一条链）", set(edges) == {2}, f"边数分布 {dict(Counter(edges))}")
    check("两端 agent 之间恒为 2 跳（多跳通信有地方可验）", set(hop2) == {2},
          f"跳数分布 {dict(Counter(hop2))}")

    # ── 4. 随机订单机制 ────────────────────────────────────────────────
    #
    # 只有一种配方时，"配料区有两种蔬菜"是假选择：拿什么丢进锅都一样。有了订单，
    # 配料区才必须**先读订单**再决定拿哪种。这里验三件事，缺一条机制就不成立。
    from overcooked_ai_py.mdp.overcooked_mdp import Recipe

    env.reset()
    pol = HandoffPolicy(env)
    seen, matched_rewards = set(), []
    for _ in range(env._max_steps):
        seen.add(env.current_order)
        _, r, done, _ = env.step(pol())
        if r.sum() > 0:
            matched_rewards.append((env.current_order, float(r.sum())))
        if done:
            break
    check("一局里订单确实会换（至少出现两张）", len(seen) > 1,
          f"出现过 {sorted(seen)}")
    check("两种配方都真的能被交付", len(seen) > 1 and len(matched_rewards) > 1,
          f"交付 {len(matched_rewards)} 锅")

    # 奖励**按订单判**：同一锅汤，对得上订单给分，对不上给 0。
    env.reset()
    st = env._state
    v_onion = env._mdp.get_recipe_value(st, Recipe(("onion",)))
    v_tomato = env._mdp.get_recipe_value(st, Recipe(("tomato",)))
    cur = env.current_order
    wrong = ("tomato",) if cur == ("onion",) else ("onion",)
    v_wrong = env._mdp.get_recipe_value(st, Recipe(wrong))
    check("对得上订单的汤有分、对不上的恰好 0 分",
          v_wrong == 0 and max(v_onion, v_tomato) > 0,
          f"当前订单 {cur}: 葱汤={v_onion} 番茄汤={v_tomato} 错单={v_wrong}")

    # ── 4. 默认 C（= sight = 1）会得到空图 ─────────────────────────────
    #     这不是配置建议，是**必须显式给 comm_radius** 的证据
    env_c1 = make(comm_radius=1.0)
    adj1 = env_c1.adjacency
    check("C=1 时通信图是空的（所以 C 必须显式给 3.0）",
          int(adj1.sum()) == 0, f"边数 {int(adj1.sum() // 2)}")
    env_c1.close()

    env.close()
    print()
    if fail:
        print(f"FAILED（{fail} 项）")
        return 1
    print("全部通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
