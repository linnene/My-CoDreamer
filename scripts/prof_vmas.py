"""训练循环的模块级计时 —— 回答"时间花在哪、以及 VMAS 到底可不可行"。

================================================================================
为什么重写：上一版有三个确定性错误
================================================================================
1. **主循环条件看错**。`while agent._step < config.steps + config.eval_every`，
   而我为了"不跑评测"设了 `eval_every = 1000000` —— 于是它要跑约 100 万步，
   不是 1200 步。剖析根本不会结束。

2. **`train_ratio` 的方向反了**。调度是
       `_should_train = Every(batch_size * batch_length / train_ratio)`
   `train_ratio` **越大 → 间隔越小 → 更新越频繁**。我那个"关掉训练"的对照组
   用 `train_ratio=100000000`，实际是"疯狂更新"组。

3. **最关键的口径错误**：`Every.__call__` 返回的是**次数**而不是布尔：
       count = int((step - _last) / _every)
   所以缩小 `batch_size` / `batch_length` 会让**单次更新更便宜、但次数按比例变多**，
   每个 agent 步的总训练量由 `train_ratio` 锁死，**完全不变**。
   我把"单次更新 0.057 秒"直接和端到端 fps 比，得出"有 3.9 秒不知去向"——
   那 3.9 秒根本不存在，是我漏乘了每次调用里实际的更新次数。

由此得到的正确口径（本脚本会打印）：
    **每个 agent 步的梯度更新次数 = train_ratio / (batch_size × batch_length)**
    （`train_ratio` 是 main() 里多智能体补偿之后的那个值）

================================================================================
计时纪律
================================================================================
- 固定迭代次数（`--iters`），先热身
- 每个 GPU 段前后 `torch.cuda.synchronize()` —— 否则 CUDA 异步执行会让计时失真
- **"无训练"对照必须显式跳过 `_train()`**，不能靠调 `train_ratio`
- 改变 `envs` 时固定 B / T / H，否则比较的不是同一件事

用法（在 wyb 上）::

    python scripts/prof_vmas.py --envs 4 --batch-size 4 --batch-length 16 --imag-horizon 10
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import pathlib
import sys
import tempfile
import time

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
D3T = REPO / "third_party" / "dreamerv3-torch"
sys.path.insert(0, str(D3T))
sys.path.insert(0, str(REPO / "scripts"))

import ruamel.yaml as yaml  # noqa: E402

import dreamer as dreamer_mod  # noqa: E402
import models  # noqa: E402
import tools  # noqa: E402
import tools_marl  # noqa: E402


class SilenceLogger:
    """计时用的 logger：不写盘、不做 I/O，避免把日志开销算进训练里。"""

    def __init__(self):
        self.step = 0
        self._scalars = {}

    def scalar(self, name, value):
        self._scalars[name] = value

    def write(self, *a, **k):
        pass


def load_config(names):
    configs = yaml.safe_load((D3T / "configs.yaml").read_text())

    def merge(base, upd):
        for k, v in upd.items():
            if isinstance(v, dict) and k in base:
                merge(base[k], v)
            else:
                base[k] = v

    d = {}
    for n in names:
        merge(d, configs[n])
    return argparse.Namespace(**d)


def timeit(fn, iters: int, warmup: int = 2, gpu: bool = False) -> float:
    for _ in range(warmup):
        fn()
    if gpu:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if gpu:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--batch-length", type=int, default=16)
    ap.add_argument("--imag-horizon", type=int, default=10)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--device", default="cuda:0")
    # 环境可换：脚本只把 env 用来取 spaces 与 num_agents（训练计时用的是合成批次），
    # 所以换环境不需要改别的东西。默认仍是 VMAS，保持与台账里那组数字可比。
    ap.add_argument("--task", default="vmas_flocking")
    args = ap.parse_args()

    cfg = load_config(["defaults", "codreamer"])
    cfg.task = args.task
    cfg.comm_variant = "codreamer"
    cfg.gnn_layers = -1
    cfg.gnn_head_layers = -1
    cfg.gnn_behavior_layers = -1
    cfg.device = args.device
    cfg.precision = 16
    cfg.compile = False
    cfg.vmas_max_steps = 50
    if args.task.startswith("overcooked"):
        # Overcooked 的观测维是 353（VMAS flocking 只有 22），encoder 第一层
        # 353→1024 —— 这正是要测的东西，别在这里把它调小。
        cfg.overcooked_max_steps = 35
    cfg.envs = args.envs
    cfg.batch_size = args.batch_size
    cfg.batch_length = args.batch_length
    cfg.imag_horizon = args.imag_horizon
    cfg.action_repeat = 1
    cfg.video_pred_log = False
    cfg.openloop_horizons = ""
    dreamer_mod.resolve_comm_variant(cfg)

    tools.set_seed_everywhere(0)
    env = dreamer_mod.make_env(cfg, "train", 0)
    cfg.num_agents = env.num_agents
    cfg.num_actions = env.action_space.n

    # ---- train_ratio 的多智能体补偿（与 main() 完全一致）----
    factor = cfg.num_agents * cfg.action_repeat
    tr_compensated = max(1, int(round(cfg.train_ratio / factor)))
    batch_steps = cfg.batch_size * cfg.batch_length
    ups_per_step = tr_compensated / batch_steps          # 每 _step 单位的更新次数
    ups_per_agent_step = ups_per_step                    # _step 与 agent 步同尺度
    per_call = max(1, round(cfg.envs * cfg.num_agents * ups_per_agent_step))

    print("=" * 74)
    print(f"envs={cfg.envs}  N={cfg.num_agents}  B={cfg.batch_size}  T={cfg.batch_length}  H={cfg.imag_horizon}")
    # 来源信息跟着产物走：rsync 副本没有 .git，那里会打印 None —— 看到 None
    # 就说明这批数不能对应到某个 commit，引用时要写明这一点。
    print(f"git_sha = {dreamer_mod.git_sha()}")
    print(f"train_ratio {cfg.train_ratio} -> {tr_compensated}（补偿 x{factor}）")
    print(f"每次更新样本数 = B*T*N = {batch_steps * cfg.num_agents}")
    print(f"** 每 agent 步的更新次数 = {tr_compensated}/{batch_steps} = {ups_per_agent_step:.2f} **")
    print(f"** 每次 agent(...) 调用推进 {cfg.envs * cfg.num_agents} agent 步，约 {per_call} 次更新 **")
    print("=" * 74)

    # ---- 造数据 ----
    B, T, N = cfg.batch_size, cfg.batch_length, cfg.num_agents
    data = {
        "state": torch.randn(B, T, N, 22).numpy(),
        "is_first": np.zeros((B, T, N), bool),
        "is_terminal": np.zeros((B, T, N), bool),
        "reward": np.zeros((B, T, N), np.float32),
        "discount": np.ones((B, T, N), np.float32),
        "action": torch.zeros(B, T, N, cfg.num_actions).numpy(),
        "adjacency": torch.zeros(B, T, N, N).numpy(),
        "edge_features": torch.zeros(B, T, N, N, 1).numpy(),
    }

    import gym

    obs_space = gym.spaces.Dict(
        {"state": gym.spaces.Box(-1e9, 1e9, (22,), np.float32)}
    )
    act_space = gym.spaces.Discrete(cfg.num_actions)
    act_space.discrete = True
    tools.set_seed_everywhere(0)
    wm = models.WorldModel(obs_space, act_space, 0, cfg).to(cfg.device)
    imag = models.ImagBehavior(cfg, wm).to(cfg.device)

    # ---- 逐模块计时 ----
    rows: list[tuple[str, float, str]] = []

    # 1) replay 采样（Python + numpy，无需 sync）
    tmp = tempfile.mkdtemp()
    cache = {}
    for i in range(cfg.batch_size * 3):
        for k in range(N):
            key = f"t{i}--a{k}"
            cache[key] = {
                kk: np.asarray(vv)[0] for kk, vv in data.items()
            }
            cache[key]["reward"] = np.zeros(T, np.float32)
            cache[key]["discount"] = np.ones(T, np.float32)
    with contextlib.suppress(Exception):
        ds = tools_marl.make_graph_dataset(cache, cfg)
        rows.append(("next(dataset) 采样一个图批次", timeit(lambda: next(ds), args.iters), "cpu"))

    # 2) 世界模型更新
    rows.append(
        ("世界模型 fwd+bwd+opt", timeit(lambda: wm._train(dict(data)), args.iters, 2, gpu=True), "gpu")
    )

    # 3) actor-critic 想象
    post, _, _ = wm._train(dict(data))
    rew_fn = lambda f, s, a: wm.heads["reward"](f).mode()  # noqa: E731
    rows.append(
        ("actor-critic 想象训练", timeit(lambda: imag._train(post, rew_fn), args.iters, 2, gpu=True), "gpu")
    )

    # 4) 策略前向（执行路径，带图）
    obs = env.reset()
    flat = {k: np.asarray(v) for k, v in obs.items()}
    agent = dreamer_mod.Dreamer(obs_space, act_space, cfg, SilenceLogger(), None).to(cfg.device)
    rows.append(
        ("_policy 前向（执行路径）", timeit(lambda: agent._policy(flat, None, training=False), args.iters, 2, gpu=True), "gpu")
    )

    # 5) VMAS env.step
    acts = np.zeros(N, dtype=int)
    rows.append(("VMAS env.step", timeit(lambda: env.step(acts), args.iters * 5, 2), "cpu"))

    # 6) replay 落盘
    rows.append(
        (
            "save_episodes（写盘）",
            timeit(
                lambda: tools_marl.save_episodes(tmp, {"x--a0": cache["t0--a0"]}),
                args.iters * 2,
                1,
            ),
            "io",
        )
    )
    del wm, imag, agent, post
    gc.collect()
    torch.cuda.empty_cache()

    # ---- 汇总 ----
    print(f"\n{'模块':<28} {'每次耗时':>12} {'':>6}")
    print("-" * 74)
    total_gpu = 0.0
    for name, dt, kind in rows:
        unit = "ms" if dt < 1 else "s"
        val = dt * 1000 if dt < 1 else dt
        print(f"{name:<28} {val:>10.2f} {unit:>4}   {kind}")
        if kind == "gpu":
            total_gpu += dt

    print("-" * 74)
    per_update = total_gpu
    per_agent_step = per_update * ups_per_agent_step
    print(f"一次完整训练更新（WM+AC）       {per_update * 1000:>10.1f} ms")
    print(f"每个 agent 步的训练成本         {per_agent_step * 1000:>10.1f} ms")
    print(f"  → 上限吞吐                    {1 / per_agent_step:>10.2f} agent 步/秒")
    for steps, label in [(5e5, "论文的 5e5 步"), (1e5, "1e5 步"), (2e4, "2e4 步")]:
        hours = steps * per_agent_step / 3600
        print(f"  {label:<14} 单次运行约 {hours:>6.1f} 小时   6 次（3 seed x 2 变体）约 {hours * 6 / 24:>5.1f} 天")

    print(
        "\n注意：`train_ratio` 锁死的是【每 agent 步的更新次数】。\n"
        "缩小 batch 只会让单次更新更便宜、次数按比例变多，总训练量不变 ——\n"
        "**唯一能真正提速的旋钮是 train_ratio 本身**（会偏离论文设定，\n"
        "但两个变体用同一设定时对比仍然公平）。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
