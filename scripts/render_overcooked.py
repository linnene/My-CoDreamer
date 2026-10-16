"""把 Overcooked 的一局渲染成视频 —— 演示用，不依赖显示服务器。

================================================================================
为什么自己写而不是用官方的可视化
================================================================================
官方的 `overcooked_ai_py.visualization` 基于 **pygame**，要一个能用的显示后端；
本机无 sudo，装不了 xvfb 之类的东西。matplotlib 用 Agg 后端纯离屏渲染，
**不碰屏幕**，也不需要额外系统包。

================================================================================
它画什么（"直观"就落在这里）
================================================================================
  - 台面 / 锅 / 洋葱 / 盘子 / 上菜口，用固定配色，一眼分得出
  - 锅上标烹饪进度（tick / cook_time），煮好了变绿
  - 每个 agent 一个圆 + 朝向箭头；手持物用一个同心小圆表示
  - **每个 agent 的视野窗口**用半透明方框画出来 —— 这一层是重点：
    它让"谁看得见什么"直接可见，而那正是通信要解决的问题
  - 标题栏给 step / 累计奖励

用法::

    python scripts/render_overcooked.py --out /tmp/oc.gif
    python scripts/render_overcooked.py --episodes 8 --pick 3 --out /tmp/oc.mp4
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
D3T = REPO / "third_party" / "dreamerv3-torch"
sys.path.insert(0, str(D3T))

# 地形 -> 颜色（与 Overcooked 的习惯一致：地板白、台面深、锅黑、洋葱黄绿、盘子蓝、上菜口灰）
TERRAIN_COLOR = {
    " ": "#ffffff",   # 地板
    "X": "#4a4a4a",   # 台面
    "P": "#1a1a1a",   # 锅
    "O": "#c8b900",   # 洋葱分发器
    "T": "#cc4444",   # 番茄分发器
    "D": "#3b7dd8",   # 盘子分发器
    "S": "#8a8a8a",   # 上菜口
}
AGENT_COLOR = ["#e8552d", "#2d7de8", "#2db356", "#b32db3", "#e8a02d"]
HELD_COLOR = {"onion": "#c8b900", "tomato": "#cc4444", "dish": "#3b7dd8", "soup": "#2db356"}


def draw_state(ax, env, title: str = "", show_view: bool = True):
    import matplotlib.patches as mpatches

    mdp = env._mdp
    state = env._state
    grid = mdp.terrain_mtx
    h, w = len(grid), len(grid[0])

    ax.clear()
    for y in range(h):
        for x in range(w):
            ax.add_patch(
                mpatches.Rectangle(
                    (x, y), 1, 1, facecolor=TERRAIN_COLOR.get(grid[y][x], "#ffffff"),
                    edgecolor="#cccccc", lw=0.5,
                )
            )

    # 台面上的东西
    for (x, y), obj in state.objects.items():
        name = obj.name
        if name.startswith("soup"):
            # 锅里：按烹饪进度填充。
            # ⚠️ `obj.cook_time` 在汤**还没开火**时会抛 ValueError
            # （"Recipe is not determined until soup begins cooking"）——
            # 未开火时配方还没定下来。所以只有非 idle 才去读它。
            ct = obj._cooking_tick
            if ct < 0:
                total, frac = 0, 0.0
            else:
                total = max(obj.cook_time, 1)
                frac = min(1.0, (ct + 1) / total)
            ax.add_patch(
                mpatches.Rectangle(
                    (x + 0.15, y + 0.15), 0.7, 0.7,
                    facecolor="#2db356" if obj.is_ready else "#9ad3ae", lw=0,
                )
            )
            # 显示钳到 total：ct 会推到 cook_time 就停，+1 只是为了让"刚开火"不显示 0
            ax.text(x + 0.5, y + 0.5, f"{min(ct + 1, total)}/{total}",
                    ha="center", va="center", fontsize=7, color="#0a3a1a")
        else:
            ax.add_patch(
                mpatches.Circle(
                    (x + 0.5, y + 0.5), 0.22,
                    facecolor=HELD_COLOR.get(name, "#888888"), lw=0,
                )
            )

    # agent
    for i, p in enumerate(state.players):
        x, y = p.position
        c = AGENT_COLOR[i % len(AGENT_COLOR)]
        if show_view:
            s = env._sight
            ax.add_patch(
                mpatches.Rectangle(
                    (x - s, y - s), 2 * s + 1, 2 * s + 1,
                    facecolor=c, alpha=0.10, edgecolor=c, lw=1.2, ls="--",
                )
            )
        ax.add_patch(
            mpatches.Circle(
                (x + 0.5, y + 0.5), 0.34, facecolor=c, lw=1.0, edgecolor="white",
            )
        )
        dx, dy = p.orientation
        ax.annotate("", xy=(x + 0.5 + 0.36 * dx, y + 0.5 + 0.36 * dy),
                    xytext=(x + 0.5, y + 0.5),
                    arrowprops=dict(arrowstyle="-|>", color="white", lw=1.6))
        ax.text(x + 0.5, y + 0.12, str(i), ha="center", va="center",
                fontsize=8, color="white", fontweight="bold")
        if p.held_object is not None:
            ax.add_patch(
                mpatches.Circle(
                    (x + 0.5, y + 0.5), 0.15,
                    facecolor=HELD_COLOR.get(p.held_object.name, "#888888"),
                    lw=1.0, edgecolor="white",
                )
            )

    ax.set_xlim(-0.1, w + 0.1)
    ax.set_ylim(h + 0.1, -0.1)          # y 轴朝下，与 terrain_mtx 的行序一致
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)


def render(states_log, env, out: pathlib.Path, fps: int = 4):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 3.6), dpi=110)

    def frame(k):
        draw_state(ax, env, title=states_log[k]["title"])
        return (ax,)

    # 逐帧重建：draw_state 直接读 env._state，所以先把状态回放回去
    frames = []
    for rec in states_log:
        env._state = rec["state"]
        draw_state(ax, env, title=rec["title"])
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())

    plt.close(fig)

    if out.suffix.lower() == ".gif":
        import imageio.v2 as imageio

        imageio.mimsave(out, frames, duration=1.0 / fps, loop=0)
    else:
        import imageio.v2 as imageio

        imageio.mimsave(out, frames, fps=fps)
    return len(frames)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--layout", default="three_player")
    p.add_argument("--episodes", type=int, default=6, help="先跑几局挑一局")
    p.add_argument("--pick", type=int, default=-1,
                   help="挑哪一局渲染（默认挑奖励最高的那局）")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=int, default=4)
    p.add_argument("--policy", default="scripted", choices=["scripted", "random"],
                   help="random 一局下来一锅都煮不出来（奖励 0），只用于验证渲染器")
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("/tmp/overcooked.gif"))
    args = p.parse_args()

    from envs.overcooked_env import OvercookedTeamEnv

    env = OvercookedTeamEnv(layout=args.layout, seed=args.seed)
    n = env.num_agents
    rng = np.random.default_rng(args.seed)

    best = None
    for ep in range(args.episodes):
        env.seed(args.seed + ep)
        env.reset()
        policy = ScriptedPolicy(env) if args.policy == "scripted" else None
        log = [{"state": _copy_state(env), "title": f"ep{ep} step 0"}]
        while True:
            # 接上训练好的 policy 时，把这里换成 agent._policy 给出的动作即可
            # （接口一样是 (N,) 的动作下标）。
            a = (policy() if policy is not None
                 else rng.integers(0, env.action_space.n, size=n))
            _, r, done, _ = env.step(a)
            log.append({
                "state": _copy_state(env),
                "title": f"ep{ep} step {env._state.timestep}  "
                         f"reward {env.total_reward:.0f}",
            })
            if done:
                break
        if best is None or env.total_reward > best[0]:
            best = (env.total_reward, log)
    env.close()

    if args.pick >= 0:
        # 重新跑指定那一局（简单起见只支持默认挑选，指定局时提示）
        print(f"（--pick 暂未接入，仍用奖励最高的那一局，奖励 {best[0]:.0f}）")
    k = render(best[1], env, args.out, fps=args.fps)
    print(f"写出 {k} 帧 -> {args.out}（该局奖励 {best[0]:.0f}）")
    return 0


def _copy_state(env):
    """深拷贝一份状态 —— render 是回放，不能边走边画。"""
    import copy

    return copy.deepcopy(env._state)


# ============================================================================
# 脚本策略 —— 让回放里真的有人在做饭。
#
# 随机策略一局下来一锅都煮不出来（实测奖励 0），拿它做演示没有意义。
# 这个策略不需要多聪明，只要能把"取洋葱 → 下锅 → 开火 → 装盘 → 上菜"这条链
# 走通，回放里就看得见协作。
# ============================================================================
def _bfs_step(mdp, start, goal, blocked):
    """格子上走一步（绕开台面与别的 agent）。返回 Action，走不到返回 None。"""
    from collections import deque

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


class ScriptedPolicy:
    """只够把一条交付链走通的规则策略，用于演示回放。"""

    def __init__(self, env):
        self.env = env
        self.mdp = env._mdp
        grid = self.mdp.terrain_mtx
        h, w = len(grid), len(grid[0])
        self.pots = [(x, y) for y in range(h) for x in range(w) if grid[y][x] == "P"]
        self.onions = [(x, y) for y in range(h) for x in range(w) if grid[y][x] == "O"]
        self.dishes = [(x, y) for y in range(h) for x in range(w) if grid[y][x] == "D"]
        self.serve = [(x, y) for y in range(h) for x in range(w) if grid[y][x] == "S"]

    def _adjacent_free(self, target):
        """站在哪一格才能对 target 交互（四邻中的可走格）。"""
        mdp, grid = self.mdp, self.mdp.terrain_mtx
        out = []
        for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            c = (target[0] + d[0], target[1] + d[1])
            if 0 <= c[1] < len(grid) and 0 <= c[0] < len(grid[0]) and grid[c[1]][c[0]] == " ":
                out.append(c)
        return out

    def _go_and_interact(self, i, target):
        """朝 target 走；贴上了就转朝向并 INTERACT。返回 (动作, 是否已交互)。"""
        from overcooked_ai_py.mdp.actions import Action, Direction

        state = self.env._state
        me = state.players[i]
        blocked = {p.position for j, p in enumerate(state.players) if j != i}
        cands = self._adjacent_free(target)
        # 已经贴上：转身 → 交互
        for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if (me.position[0] + d[0], me.position[1] + d[1]) == target:
                if tuple(me.orientation) != d:
                    return d, False
                return Action.INTERACT, True
        # 否则走到最近的一个可交互格
        best, bestd = None, None
        for c in cands:
            if c in blocked:
                continue
            act = _bfs_step(self.mdp, me.position, c, blocked)
            if act is not None:
                n = abs(c[0] - me.position[0]) + abs(c[1] - me.position[1])
                if bestd is None or n < bestd:
                    best, bestd = act, n
        if best is None:
            return Action.STAY, False
        return best, False

    def __call__(self):
        from overcooked_ai_py.mdp.actions import Action

        env = self.env
        state = env._state
        pots = {p: state.objects.get(p) for p in self.pots}
        ready = [p for p, o in pots.items() if o is not None and o.name.startswith("soup") and o.is_ready]
        idle_with_stuff = [
            p for p, o in pots.items()
            if o is not None and o.name.startswith("soup") and o.is_idle and len(o.ingredients) > 0
        ]
        empty_pots = [p for p, o in pots.items() if o is None]

        acts = []
        for i, me in enumerate(state.players):
            held = me.held_object.name if me.held_object else None
            if held == "soup":
                a, _ = self._go_and_interact(i, self.serve[0])
            elif held == "dish":
                tgt = ready[0] if ready else (self.pots[i % len(self.pots)])
                a, _ = self._go_and_interact(i, tgt)
            elif held == "onion":
                tgt = idle_with_stuff[0] if idle_with_stuff else (
                    empty_pots[0] if empty_pots else self.pots[i % len(self.pots)]
                )
                a, _ = self._go_and_interact(i, tgt)
            elif ready:
                a, _ = self._go_and_interact(i, self.dishes[i % len(self.dishes)])
            elif idle_with_stuff:
                a, _ = self._go_and_interact(i, idle_with_stuff[0])
            else:
                # 分工：偶数号取洋葱，奇数号去锅边等着/开火
                if i % 2 == 0:
                    a, _ = self._go_and_interact(i, self.onions[i % len(self.onions)])
                else:
                    a, _ = self._go_and_interact(i, self.pots[i % len(self.pots)])
            acts.append(Action.ACTION_TO_INDEX[a])
        return np.asarray(acts)


if __name__ == "__main__":
    raise SystemExit(main())
