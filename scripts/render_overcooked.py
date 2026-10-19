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
def draw_state(ax, env, title: str = "", show_view: bool = True):
    """把当前状态画成一帧像素画。

    实际的绘制在 `scripts/_oc_pixel.py`：那里把精灵烘成一张 RGB 画布，这里只负责
    `imshow(interpolation="nearest")` —— **最近邻是刻意的**，它保证放大后是硬边
    像素；用默认插值会把像素糊成渐变，像素画的味道就没了。
    """
    from _oc_pixel import canvas_from_state

    canvas = canvas_from_state(env, show_view=show_view)
    h, w = canvas.shape[:2]
    ax.clear()
    ax.imshow(canvas, interpolation="nearest", zorder=1)
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
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
    p.add_argument("--layout", default="forced_coordination_3p",
                   help="默认是强制协作厨房；three_player 那种开放厨房单人就能通关")
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

    # 强制协作厨房的交付链实测 27 步才出第一锅，35 步一锅都交不出去
    #
    # `random_start_pos=False` 是**必须**的：交接策略把角色按编号写死（1 号在配料区），
    # 而随机起点会把 1 号扔到别的区，三个人的活全错位，整局 0 分。训练时无所谓
    # （策略是学出来的，不是写死的），演示回放不行。
    env = OvercookedTeamEnv(layout=args.layout, max_steps=60, cook_time=5,
                            orders=(("onion",), ("tomato",)),
                            random_start_pos=False, seed=args.seed)
    n = env.num_agents
    rng = np.random.default_rng(args.seed)

    best = None
    for ep in range(args.episodes):
        env.seed(args.seed + ep)
        env.reset()
        # 强制协作厨房要交接协议；开放厨房用通用策略
        if args.policy == "scripted":
            policy = (HandoffPolicy(env) if args.layout == "forced_coordination_3p"
                      else ScriptedPolicy(env))
        else:
            policy = None
        log = [{"state": _copy_state(env), "title": f"ep{ep} step 0"}]
        while True:
            # 接上训练好的 policy 时，把这里换成 agent._policy 给出的动作即可
            # （接口一样是 (N,) 的动作下标）。
            a = (policy() if policy is not None
                 else rng.integers(0, env.action_space.n, size=n))
            _, r, done, _ = env.step(a)
            # 订单写进标题：它决定了配料区该拿什么，不显示就看不懂回放在干什么。
            # 用英文是因为 matplotlib 的默认字体没有中文，会渲染成方框。
            order = "+".join(env.current_order).upper()
            log.append({
                "state": _copy_state(env),
                "title": f"step {env._state.timestep}   ORDER: {order}   "
                         f"score {env.total_reward:.0f}",
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
# 脚本策略
#
# 交接协议与 BFS 走位放在 `envs/overcooked_demos.py`：训练前的**演示预热**
# 也要用同一份策略，而 dreamer.py 不该反过来依赖 scripts/。
# ============================================================================
from envs.overcooked_demos import HandoffPolicy  # noqa: E402,F401
from envs.overcooked_demos import bfs_step as _bfs_step  # noqa: E402


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
        # 先把别的 agent 当墙找路；找不到就**无视它们再找一次**。
        # 这一档很重要：窄厨房里队友经常正好堵在唯一通路上，此时严格绕行会算出
        # "无路可走"，策略就原地罚站到天荒地老（实测能把成功率压到 1/10）。
        # 无视别人至少会朝目标挪，堵的人自己会走开。
        for walls in (blocked, set()):
            best, bestd = None, None
            for c in cands:
                if c in walls:
                    continue
                act = _bfs_step(self.mdp, me.position, c, walls)
                if act is not None:
                    n = abs(c[0] - me.position[0]) + abs(c[1] - me.position[1])
                    if bestd is None or n < bestd:
                        best, bestd = act, n
            if best is not None:
                return best, False
        return Action.STAY, False

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

        # ── 该有几个人在搬洋葱 ──────────────────────────────────────────────
        # 这是个**配额**，不是"谁有空谁去"。没有配额的版本（全员无脑搬）会死锁：
        # 锅满之后每个 agent 都抱着洋葱不放，而手里有洋葱就腾不出手去拿盘子，
        # 于是没人装盘上菜，整局 0 分。反过来，以前的"奇数号去锅边等"又是人形
        # 路障。两个极端都试过，所以这里按锅的剩余容量算。
        #
        # 只用**可观测量**：锅里的配料数、谁手上拿着洋葱 —— 都是 state 里读得到的。
        cap = len(getattr(env, "_recipe", ("onion",)))
        room = sum(cap - len(o.ingredients) for p, o in pots.items()
                   if o is not None and o.name.startswith("soup") and o.is_idle)
        room += cap * len(empty_pots)
        carrying = sum(1 for p in state.players
                       if p.held_object is not None and p.held_object.name == "onion")
        need_onion = room - carrying
        # 能接洋葱的锅：空的，或还没开火且没装满的
        sockets = [p for p, o in pots.items()
                   if o is None or (o.name.startswith("soup") and o.is_idle
                                    and len(o.ingredients) < cap)]

        acts = []
        for i, me in enumerate(state.players):
            held = me.held_object.name if me.held_object else None
            if held == "soup":
                a, _ = self._go_and_interact(i, self.serve[0])
            elif held == "dish":
                # 端着盘子但没锅可盛 —— **原地等**，别去锅边挤。
                # 原来是无脑走到锅边，在 cramped_room（2 人、通道只有一格宽）里
                # 会把去洋葱的唯一通路堵死，整局 0 分。
                if ready:
                    a, _ = self._go_and_interact(i, ready[0])
                else:
                    a = Action.STAY
            elif held == "onion":
                # 锅接得下就送过去；接不下就**原地等**，别抱着洋葱去锅边杵着。
                # 开火不需要专人：锅里有料之后，空手的 agent 会在下面的
                # `idle_with_stuff` 分支顺手把它点着。
                if sockets:
                    a, _ = self._go_and_interact(i, sockets[0])
                else:
                    a = Action.STAY
            elif ready:
                a, _ = self._go_and_interact(i, self.dishes[i % len(self.dishes)])
            elif idle_with_stuff:
                a, _ = self._go_and_interact(i, idle_with_stuff[0])
            elif need_onion > 0:
                a, _ = self._go_and_interact(i, self.onions[i % len(self.onions)])
            else:
                a = Action.STAY
            acts.append(Action.ACTION_TO_INDEX[a])
        return np.asarray(acts)


if __name__ == "__main__":
    raise SystemExit(main())
