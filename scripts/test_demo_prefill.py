"""演示预热的**飞行前检查** —— 在训练条件下，奖励真的会进 replay 吗？

================================================================================
为什么需要这个测试
================================================================================
演示预热出过一次事故：预热跑满 1200 步，`train_return` 全程 0，replay 里
16836 条转移**没有一条带奖励**。而同一份策略在单独测试里每局 40 分。

差别只有一个 —— **随机出生点**。训练环境默认 `random_start_pos=True`，而官方的
`get_random_start_state_fn` 在所有走道格里均匀抽，会把 agent 扔到不属于它的区域，
三个角色全错位，整局结构上无解。单环境、固定起点的测试**碰不到这条路径**。

所以这个测试按**训练时的真实条件**跑：多个环境（对应 `--envs`）、随机出生点、
和 `simulate_marl` 相同的批处理顺序。检查的是"写进 replay 的那些转移里有没有
正奖励"，这正是训练侧真正依赖的东西。

纯 numpy，不碰 torch / GPU，几秒跑完。

用法::

    python scripts/test_demo_prefill.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "third_party" / "dreamerv3-torch"))

N_ENVS = 4          # 与 pilot 的 --envs 一致
STEPS = 1200        # 与 overcooked_demo_steps 一致


def main() -> int:
    from envs.overcooked_demos import DemoAgent
    from envs.overcooked_env import OvercookedTeamEnv

    envs = [
        OvercookedTeamEnv(
            layout="forced_coordination_3p",
            orders=(("onion",), ("tomato",)),
            cook_time=5, max_steps=60, sight=1, comm_radius=3.0,
            # ★ 这一条是关键：训练默认就是 True，而它会走完全不同的一条路径
            random_start_pos=True, seed=i,
        )
        for i in range(N_ENVS)
    ]
    for e in envs:
        e.reset()
    demo = DemoAgent(envs, envs[0].action_space.n)

    n_agents = envs[0].num_agents
    rewards: list[float] = []
    done = np.ones(N_ENVS, dtype=bool)
    for _ in range(STEPS):
        out, _ = demo(None, None)
        idx = out["action"].argmax(-1).reshape(N_ENVS, n_agents)
        for i, e in enumerate(envs):
            if done[i]:
                e.reset()
            _, r, d, _ = e.step(idx[i])
            rewards.extend(np.asarray(r).reshape(-1).tolist())
            done[i] = bool(d)

    arr = np.asarray(rewards, dtype=np.float32)
    nonzero = int((arr != 0).sum())
    total = float(arr.sum())
    print(f"{N_ENVS} 个环境 × {STEPS} 步，随机出生点")
    print(f"  写进 replay 的转移 {arr.size} 条，带奖励的 {nonzero} 条，奖励总和 {total:.0f}")

    ok = nonzero > 0 and total > 0
    print(f"  [{'ok  ' if ok else 'FAIL'}] 演示预热能产生正样本"
          + ("" if ok else "   —— 检查 random_start_pos 与角色分配"))
    for e in envs:
        e.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
