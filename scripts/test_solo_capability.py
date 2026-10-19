"""一个布局到底需不需要多个 agent —— 冻住其余的，只让一个跑。

================================================================================
这个测试在回答什么
================================================================================
"多人协作任务"最容易被忽略的一种失败是**它根本不需要多人**。原来的开放厨房就是：
三个人各干各的，每个都能单独把配料→下锅→装盘→上菜走完，另外两个只是并行加速。

这件事**从代码和地形上都看不出来** —— 地图是连通的，谁都能走到任何站点，看起来
完全正常。只有冻住其余 agent、只让一个跑，才量得出来：

    全员得分 == 单人得分   ->  这个布局不需要协作（不管它看起来多像协作）

================================================================================
它不能回答什么
================================================================================
用的是通用脚本策略（`render_overcooked.ScriptedPolicy`），它**不会交接**。所以对
走道被切开的布局（比如 `forced_coordination_3p`），单人也拿 0 —— 但这个 0 不能
证明"需要协作"，只能说明这个策略在这个布局上无从下手。

所以脚本会先算**走道连通分量**：分量 > 1 的布局直接标"不适用"并跳过。真正强制
交接的布局请用 `scripts/test_forced_coordination.py`（它带交接协议，正反两面都验）。

用法::

    python scripts/test_solo_capability.py                    # 全部自带 + 自写布局
    python scripts/test_solo_capability.py three_player ...   # 指定
"""
from __future__ import annotations

import pathlib
import sys
from collections import deque

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "third_party" / "dreamerv3-torch"))
sys.path.insert(0, str(REPO / "scripts"))

LAYOUT_DIR = pathlib.Path(
    "/home/wyb/anaconda3/envs/codreamer/lib/python3.10/site-packages"
    "/overcooked_ai_py/data/layouts"
)
SEEDS = 6
MAX_STEPS = 200


def components(mdp) -> int:
    """走道（空格）的 4 邻连通分量数。=1 表示谁都能走到任何地方。"""
    g = mdp.terrain_mtx
    h, w = len(g), len(g[0])
    seen, comp = set(), 0
    for y in range(h):
        for x in range(w):
            if g[y][x] != " " or (x, y) in seen:
                continue
            comp += 1
            q = deque([(x, y)])
            seen.add((x, y))
            while q:
                cx, cy = q.popleft()
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < w and 0 <= ny < h and g[ny][nx] == " " \
                            and (nx, ny) not in seen:
                        seen.add((nx, ny))
                        q.append((nx, ny))
    return comp


def run(env, policy, solo=None):
    from overcooked_ai_py.mdp.actions import Action

    stall = Action.ACTION_TO_INDEX[Action.STAY]
    for _ in range(MAX_STEPS):
        a = np.asarray(policy()).copy()
        if solo is not None:
            a[[j for j in range(env.num_agents) if j != solo]] = stall
        _, _, done, _ = env.step(a)
        if done:
            break
    return env.total_reward


def check_layout(name: str) -> str:
    from envs.overcooked_env import OvercookedTeamEnv
    from render_overcooked import ScriptedPolicy

    env = OvercookedTeamEnv(layout=name, max_steps=MAX_STEPS, seed=0)
    n = env.num_agents
    ncomp = components(env._mdp)
    if ncomp > 1:
        env.close()
        return f"{name:<32}{n:<5}{ncomp:<6}不适用（走道被切开，通用策略不会交接）"

    best_all, best_solo = 0.0, []
    for ep in range(SEEDS):
        env.seed(ep)
        env.reset()
        best_all = max(best_all, run(env, ScriptedPolicy(env)))
        for i in range(env.num_agents):
            env.seed(ep)
            env.reset()
            v = run(env, ScriptedPolicy(env), solo=i)
            while len(best_solo) <= i:
                best_solo.append(0.0)
            best_solo[i] = max(best_solo[i], v)
    env.close()

    solo_txt = "/".join(f"{v:.0f}" for v in best_solo)
    if best_all <= 0:
        # 一个都拿不到分：这条不能用来判断"需不需要协作"，通用策略在这个布局上
        # 就没走通过，单人也 0 分说明不了任何事。
        verdict = "都拿不到分（策略不适用，判定无效）"
    elif max(best_solo) >= best_all:
        verdict = "**单人就能拿满分 -> 不需要协作**"
    else:
        verdict = "单人明显落后 -> 有协作价值"
    return f"{name:<32}{n:<5}{ncomp:<6}{best_all:>5.0f}       {solo_txt:<14}{verdict}"


def main() -> int:
    names = sys.argv[1:]
    if not names:
        names = ["three_player", "cramped_room_1onion"]
        if LAYOUT_DIR.exists():
            names += [p.stem for p in sorted(LAYOUT_DIR.glob("*.layout"))]

    print(f"{'layout':<32}{'人数':<5}{'分量':<6}{'全员':<8}{'各 agent 单独':<14}判定")
    print("-" * 110)
    for n in names:
        try:
            print(check_layout(n))
        except Exception as e:                                    # noqa: BLE001
            print(f"{n:<32}失败 {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
