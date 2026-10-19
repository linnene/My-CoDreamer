"""Overcooked 的脚本策略 —— 演示预热与回放共用同一份。

================================================================================
它为什么在 env 包里，而不是 scripts/
================================================================================
这个策略有两个用途：渲染回放，以及**训练前的演示预热**（把有奖励的轨迹塞进
replay buffer）。后者是 `dreamer.py` 干的活，不该反过来依赖 `scripts/`。

================================================================================
为什么需要演示预热
================================================================================
强制协作厨房是硬探索任务。实测：随机策略在稀疏奖励下跑 200 局，**没有一局拿到
任何奖励**；用 shaped 也只有 3/200 局拿到一次取盘子的 +3。于是训练时 replay 里
一个正样本都没有 —— 奖励头学成"恒等于 0"，价值函数恒等于 0，actor 拿不到任何
梯度。跑了一万两千步之后 actor 熵仍然是 6 个动作的均匀上限（ln 6 = 1.79）。

这不是超参问题，加步数也没用：目标里就没有非零值可学。先把带奖励的轨迹放进
buffer，是这类任务最常见的破法（Overcooked-AI 原论文直接用人类示范）。

================================================================================
`HandoffPolicy` 是什么
================================================================================
强制协作厨房里的交接协议，三个区域各一个角色::

    1 号 按**当前订单**取洋葱或番茄 -> 放上左交接台面
    2 号 从左台面取料 -> 下锅 -> 装满后开火
    3 号 取盘子 -> 放上右交接台面
    2 号 从右台面取盘子 -> 等汤好 -> 盛汤 -> 放回右台面
    3 号 从右台面取汤 -> 上菜

它同时是**这个布局可解的证据**：地形上"能隔台面递东西"和"不能"长得一模一样，
差别只在"放上台面、对面再拿起来"这条协议跑不跑得通。实测三人协作 220~340 分，
任一人单独行动 0 分 —— 后者是布局成立的证据，不是它坏了。
"""
from __future__ import annotations

from collections import deque

import numpy as np


def bfs_step(mdp, start, goal, blocked):
    """格子上走一步（绕开台面与给定的阻挡格）。返回 Action，走不到返回 None。"""
    from overcooked_ai_py.mdp.actions import Direction

    grid = mdp.terrain_mtx
    h, w = len(grid), len(grid[0])
    if start == goal:
        return None
    prev = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        for d in Direction.ALL_DIRECTIONS:
            nxt = (cur[0] + d[0], cur[1] + d[1])
            if not (0 <= nxt[0] < w and 0 <= nxt[1] < h):
                continue
            if grid[nxt[1]][nxt[0]] != " " or nxt in blocked or nxt in prev:
                continue
            prev[nxt] = (cur, d)
            q.append(nxt)
    if goal not in prev:
        return None
    node = goal
    while prev[node] is not None and prev[node][0] != start:
        node = prev[node][0]
    return prev[node][1] if prev[node] is not None else None


class HandoffPolicy:
    """三个区域各一个角色的交接协议。`__call__` 返回 (N,) 的动作下标。"""

    def __init__(self, env):
        self.env, self.mdp = env, env._mdp
        g = self.mdp.terrain_mtx
        self.h, self.w = len(g), len(g[0])
        self.pots = self._cells("P")
        self.onions = self._cells("O")
        self.tomatoes = self._cells("T")
        self.dishes = self._cells("D")
        self.serve = self._cells("S")[0]
        # 交接墙上的每一格都是槽位。只留中间一格会在"端着汤要放下、槽位却被多余
        # 盘子占着"时死锁 —— 三方互等的那个版本就是这么卡住的。
        self.left_slots = [(2, 1), (2, 2), (2, 3)]
        self.right_slots = [(5, 1), (5, 2), (5, 3)]

    def _order(self):
        """(当前订单的配料, 该去哪个分发器)。订单每交一锅就换一次。"""
        ing = self.env.current_order[0]
        return ing, (self.onions[0] if ing == "onion" else self.tomatoes[0])

    def _cells(self, ch):
        return [(x, y) for y in range(self.h) for x in range(self.w)
                if self.mdp.terrain_mtx[y][x] == ch]

    def _free(self, i):
        return {p.position for j, p in enumerate(self.env._state.players) if j != i}

    def _approach(self, i, target, face):
        from overcooked_ai_py.mdp.actions import Action

        me = self.env._state.players[i]
        if (me.position[0] + face[0], me.position[1] + face[1]) == target:
            if tuple(me.orientation) != face:
                return face, False
            return Action.INTERACT, True
        spots = [(target[0] - face[0], target[1] - face[1])]
        for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            c = (target[0] + d[0], target[1] + d[1])
            if 0 <= c[1] < self.h and 0 <= c[0] < self.w \
                    and self.mdp.terrain_mtx[c[1]][c[0]] == " ":
                spots.append(c)
        for walls in (self._free(i), set()):
            best, bestd = None, None
            for c in spots:
                if c in walls or not (0 <= c[1] < self.h and 0 <= c[0] < self.w):
                    continue
                if self.mdp.terrain_mtx[c[1]][c[0]] != " ":
                    continue
                act = bfs_step(self.mdp, me.position, c, walls)
                if act is not None:
                    d = abs(c[0] - me.position[0]) + abs(c[1] - me.position[1])
                    if bestd is None or d < bestd:
                        best, bestd = act, d
            if best is not None:
                return best, False
        return Action.STAY, False

    def _empty_slot(self, slots):
        return [s for s in slots if s not in self.env._state.objects]

    def _slot_with(self, slots, pred):
        for s in slots:
            o = self.env._state.objects.get(s)
            if o is not None and pred(o):
                return s
        return None

    def __call__(self):
        from overcooked_ai_py.mdp.actions import Action

        st = self.env._state
        held = [p.held_object.name if p.held_object else None for p in st.players]
        pot = st.objects.get(self.pots[0])
        # 订单每交一锅就换，所以每步都重新读，不能缓存
        ingredient, dispensers = self._order()
        recipe_len = len(self.env.current_order)
        n_in = 0 if pot is None else len(pot.ingredients)
        full = n_in >= recipe_len
        cooking = pot is not None and not pot.is_idle
        ready = pot is not None and pot.is_ready
        # ⚠️ 只要还有一锅汤"在路上"，配料区就**不要**开始下一轮。
        #
        # 少了这条会有个很隐蔽的竞态：2 号把汤从锅里取出来的那一刻锅就空了，
        # 配料区于是按**旧订单**去取菜；几步之后 3 号上菜、订单换新，配料区手上
        # 那棵菜就成了错的，煮出来永远交不出去 —— 整局卡死。实测三个种子从
        # 340 掉到 260/20/20，后两个种子 300 步只交付了 1 锅。
        soup_in_flight = (
            pot is not None
            or held[1] == "soup" or held[2] == "soup"
            or any(o.name.startswith("soup") for o in st.objects.values())
        )
        can_take_more = not cooking and not full and not soup_in_flight

        def on_left():
            return self._slot_with(
                self.left_slots, lambda o: o.name == ingredient)

        acts = []
        for i in range(self.env.num_agents):
            if i == 0:                                    # 配料区：按订单取菜
                if held[0] == ingredient:
                    free = self._empty_slot(self.left_slots)
                    a = self._approach(0, free[0], (1, 0))[0] if free else Action.STAY
                elif can_take_more and on_left() is None:
                    a, _ = self._approach(0, dispensers, (-1, 0))
                else:
                    a = Action.STAY
            elif i == 1:                                  # 锅区
                if held[1] == ingredient:
                    a, _ = self._approach(1, self.pots[0], (0, -1))
                elif held[1] == "dish":
                    a = self._approach(1, self.pots[0], (0, -1))[0] if ready else Action.STAY
                elif held[1] == "soup":
                    free = self._empty_slot(self.right_slots)
                    a = self._approach(1, free[0], (1, 0))[0] if free else Action.STAY
                elif can_take_more and on_left() is not None:
                    a, _ = self._approach(1, on_left(), (-1, 0))
                elif full and pot.is_idle:
                    a, _ = self._approach(1, self.pots[0], (0, -1))     # 开火
                elif (cooking or ready) and self._slot_with(
                        self.right_slots, lambda o: o.name == "dish"):
                    a, _ = self._approach(
                        1, self._slot_with(self.right_slots, lambda o: o.name == "dish"),
                        (1, 0))
                else:
                    a = Action.STAY
            else:                                         # 盘+上菜区
                soup_slot = self._slot_with(self.right_slots,
                                            lambda o: o.name.startswith("soup"))
                if held[2] == "soup":
                    a, _ = self._approach(2, self.serve, (1, 0))
                elif held[2] == "dish":
                    free = self._empty_slot(self.right_slots)
                    a = self._approach(2, free[0], (-1, 0))[0] if free else Action.STAY
                elif soup_slot is not None:
                    a, _ = self._approach(2, soup_slot, (-1, 0))
                elif (cooking or ready) and held[1] is None and not self._slot_with(
                        self.right_slots, lambda o: o.name == "dish"):
                    # 只在"锅在煮、且锅区手上还没有盘子"时补盘子。少了 `held[1] is None`
                    # 这一条，它会不停取盘子，把交接槽位占满。
                    a, _ = self._approach(2, self.dishes[0], (1, 0))
                else:
                    a = Action.STAY
            acts.append(Action.ACTION_TO_INDEX[a])
        return np.asarray(acts)


class DemoAgent:
    """把 `HandoffPolicy` 包成 `simulate_marl` 认识的策略接口。

    `simulate_marl` 每轮把所有队伍观测拼成一个 batch 交给策略，而演示策略是
    **有状态的**（它直接读 env 的当前局面），所以这里按队伍逐个调用，再把结果拼回去。
    批内顺序是 team-major（`np.concatenate([o for o in obs], 0)`），与这里一致。
    """

    def __init__(self, envs, num_actions: int):
        self.envs = list(envs)
        self.num_actions = int(num_actions)
        self.policies = [HandoffPolicy(e) for e in self.envs]

    def __call__(self, obs, done, state=None, **kwargs):
        idx = np.concatenate([p() for p in self.policies]).astype(np.int64)
        onehot = np.zeros((idx.size, self.num_actions), dtype=np.float32)
        onehot[np.arange(idx.size), idx] = 1.0
        # logprob 只用于写 replay；演示数据不参与策略梯度，给 0 即可
        return {"action": onehot, "logprob": np.zeros(idx.size, dtype=np.float32)}, None
