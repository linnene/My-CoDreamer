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
# ---- 配色（暖色木地板 + 深色台面，靠明暗做出一点立体感）----
FLOOR_A, FLOOR_B = "#efe0c8", "#e6d4b6"      # 交错的地板砖
COUNTER_TOP, COUNTER_BODY = "#b98d5e", "#8a6039"
SERVE_TOP, SERVE_BODY = "#cfd4d8", "#9aa2a8"
POT_BODY, POT_RIM = "#3b3b3b", "#636363"
ONION, ONION_RING = "#e9c34a", "#c9a02e"
TOMATO = "#d24a3d"
DISH, DISH_RIM = "#fbfbfb", "#c9c9c9"
SOUP_RAW, SOUP_READY = "#d9a24a", "#57b85a"
AGENT_COLOR = ["#e8552d", "#2f7fd8", "#2fae5a", "#a44fc4", "#e8a02d"]
SKIN = "#f2c9a0"
HAT = "#fdfdfd"


def _round(ax, x, y, w, h, fc, ec="#00000022", lw=1.0, r=0.12, z=2):
    import matplotlib.patches as mp

    ax.add_patch(mp.FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z))


def _counter(ax, x, y, top=COUNTER_TOP, body=COUNTER_BODY, z=2):
    """台面：主体 + 上方一条亮边，看起来像有厚度。"""
    _round(ax, x + 0.04, y + 0.04, 0.92, 0.92, body, z=z)
    _round(ax, x + 0.04, y + 0.04, 0.92, 0.30, top, z=z + 0.1)


def _onion(ax, cx, cy, s=0.26, z=6):
    import matplotlib.patches as mp

    ax.add_patch(mp.Circle((cx, cy), s, facecolor=ONION, edgecolor=ONION_RING,
                           lw=1.2, zorder=z))
    for k in (0.45, 0.75):
        ax.add_patch(mp.Arc((cx, cy), 2 * s * k, 2 * s * k, theta1=200, theta2=340,
                            edgecolor=ONION_RING, lw=0.9, zorder=z + 0.1))
    ax.plot([cx, cx], [cy + s * 0.9, cy + s * 1.5], color="#4f8a3a", lw=1.6,
            solid_capstyle="round", zorder=z + 0.2)


def _dish(ax, cx, cy, s=0.26, z=6):
    import matplotlib.patches as mp

    ax.add_patch(mp.Circle((cx, cy), s, facecolor=DISH, edgecolor=DISH_RIM,
                           lw=1.4, zorder=z))
    ax.add_patch(mp.Circle((cx, cy), s * 0.55, facecolor="none",
                           edgecolor=DISH_RIM, lw=0.8, zorder=z + 0.1))


def _pot(ax, x, y, soup, z=6):
    """锅：深色锅身 + 亮色锅沿；有汤时按进度着色，煮好冒泡。"""
    import matplotlib.patches as mp

    _round(ax, x + 0.10, y + 0.14, 0.80, 0.72, POT_BODY, "#1e1e1e", 1.2, r=0.10, z=z)
    if soup is not None:
        ct = soup._cooking_tick
        ready = ct >= 0 and soup.is_ready
        frac = 0.0 if ct < 0 else min(1.0, (ct + 1) / max(soup.cook_time, 1))
        col = SOUP_READY if ready else SOUP_RAW
        ax.add_patch(mp.Ellipse((x + 0.5, y + 0.52), 0.60, 0.34, facecolor=col,
                                alpha=0.55 + 0.45 * frac, zorder=z + 0.2))
        if ready:
            for dx in (-0.16, 0.02, 0.18):
                ax.add_patch(mp.Circle((x + 0.5 + dx, y + 0.72), 0.045,
                                       facecolor="white", alpha=0.85,
                                       edgecolor="none", zorder=z + 0.3))
    ax.add_patch(mp.Ellipse((x + 0.5, y + 0.30), 0.86, 0.26, facecolor=POT_RIM,
                            edgecolor="#2a2a2a", lw=1.0, zorder=z + 0.4))


def _chef(ax, x, y, color, facing, z=8):
    """厨师：身体 + 头 + 白帽子 + 朝向。"""
    import matplotlib.patches as mp

    ax.add_patch(mp.Circle((x + 0.5, y + 0.58), 0.30, facecolor=color,
                           edgecolor="#00000033", lw=1.0, zorder=z))
    ax.add_patch(mp.Circle((x + 0.5, y + 0.52), 0.20, facecolor=SKIN,
                           edgecolor="#00000022", lw=0.8, zorder=z + 0.1))
    ax.add_patch(mp.Ellipse((x + 0.5, y + 0.70), 0.34, 0.20, facecolor=HAT,
                            edgecolor="#00000022", lw=0.8, zorder=z + 0.2))
    ax.add_patch(mp.Circle((x + 0.5, y + 0.78), 0.13, facecolor=HAT,
                           edgecolor="#00000022", lw=0.8, zorder=z + 0.2))
    dx, dy = facing
    ax.plot([x + 0.5 + 0.24 * dx, x + 0.5 + 0.42 * dx],
            [y + 0.50 + 0.24 * dy, y + 0.50 + 0.42 * dy],
            color="#2b2b2b", lw=2.6, solid_capstyle="round", zorder=z + 0.3)


def draw_state(ax, env, title: str = "", show_view: bool = True):
    import matplotlib.patches as mp

    mdp = env._mdp
    state = env._state
    grid = mdp.terrain_mtx
    h, w = len(grid), len(grid[0])

    ax.clear()
    ax.set_facecolor("#f7f2e7")

    for y in range(h):
        for x in range(w):
            ch = grid[y][x]
            if ch == " " or ch.isdigit():
                # 地板：交错砖
                _round(ax, x + 0.02, y + 0.02, 0.96, 0.96,
                       FLOOR_A if (x + y) % 2 == 0 else FLOOR_B, "none", 0, r=0.06, z=1)
            elif ch == "S":
                _counter(ax, x, y, SERVE_TOP, SERVE_BODY)
                for k in range(3):                    # 上菜口的条纹
                    ax.plot([x + 0.16, x + 0.84], [y + 0.34 + 0.16 * k] * 2,
                            color="#ffffff88", lw=1.2, zorder=3)
            elif ch == "O":
                _counter(ax, x, y)
                _onion(ax, x + 0.5, y + 0.58, 0.24, z=4)
            elif ch == "T":
                _counter(ax, x, y)
                ax.add_patch(mp.Circle((x + 0.5, y + 0.58), 0.24, facecolor=TOMATO,
                                       edgecolor="#a8352a", lw=1.2, zorder=4))
            elif ch == "D":
                _counter(ax, x, y)
                _dish(ax, x + 0.5, y + 0.58, 0.24, z=4)
            elif ch == "P":
                _counter(ax, x, y)
                _pot(ax, x, y, state.objects.get((x, y)))
            else:
                _counter(ax, x, y)

    # 台面上散落的物体（不在锅/分发器上的那些）
    for (x, y), obj in state.objects.items():
        if grid[y][x] in "OPDS":
            continue
        if obj.name.startswith("soup"):
            _pot(ax, x, y, obj)
        elif obj.name == "onion":
            _onion(ax, x + 0.5, y + 0.5, 0.22)
        elif obj.name == "tomato":
            ax.add_patch(mp.Circle((x + 0.5, y + 0.5), 0.22, facecolor=TOMATO,
                                   edgecolor="#a8352a", lw=1.2, zorder=6))
        elif obj.name == "dish":
            _dish(ax, x + 0.5, y + 0.5, 0.22)

    for i, p in enumerate(state.players):
        x, y = p.position
        c = AGENT_COLOR[i % len(AGENT_COLOR)]
        if show_view:
            s = env._sight
            ax.add_patch(mp.FancyBboxPatch(
                (x - s + 0.06, y - s + 0.06), 2 * s + 0.88, 2 * s + 0.88,
                boxstyle="round,pad=0,rounding_size=0.1",
                facecolor=c, alpha=0.10, edgecolor=c, lw=1.6, ls=(0, (4, 3)),
                zorder=5))
        _chef(ax, x, y, c, p.orientation)
        if p.held_object is not None:
            hx, hy = x + 0.5 + 0.46 * p.orientation[0], y + 0.46 + 0.46 * p.orientation[1]
            n = p.held_object.name
            if n == "onion":
                _onion(ax, hx, hy, 0.17)
            elif n == "tomato":
                ax.add_patch(mp.Circle((hx, hy), 0.17, facecolor=TOMATO,
                                       edgecolor="#a8352a", lw=1.0, zorder=9))
            elif n == "dish":
                _dish(ax, hx, hy, 0.17)
            else:
                ax.add_patch(mp.Circle((hx, hy), 0.17, facecolor=SOUP_READY,
                                       edgecolor="#2f7a33", lw=1.2, zorder=9))

    ax.set_xlim(-0.15, w + 0.15)
    ax.set_ylim(h + 0.15, -0.15)          # y 轴朝下，与 terrain_mtx 的行序一致
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
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
