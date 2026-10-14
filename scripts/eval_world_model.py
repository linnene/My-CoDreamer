"""Held-out 上的**多步 open-loop** 预测误差 —— 判断世界模型"想没想对"。

================================================================================
为什么不能用 replay 上的训练损失
================================================================================
`plot_phase2.py` 的第三张图画的是 `metrics.jsonl` 里的 `reward_loss` / `state_loss`，
并据此判断"策略有没有在利用模型误差"。这个推论**不成立**，因为：

  1. 那是**训练损失** —— 同一份 replay 被反复优化，损失下降可以只是记住了它；
  2. 它用的是 **posterior** 隐状态，而 posterior 每一步都**看过当步观测**
     （见 WorldModel._train：`feat = get_feat(post)`）。
     所以它衡量的是"给定当前帧，重建当前帧"，**不是**对未来开环预测的能力。

真正的问题——"世界模型想象出来的轨迹准不准"——只有 open-loop 能回答：

    用 posterior 吃 context 步（允许看观测）→ 取某一时刻的状态
    → 之后**只喂真实动作、不再看任何观测**，让 prior 往前滚 K 步
    → 把第 k 步的预测与真值比

而且必须用**没参与训练**的数据（`evaldir` 的 rollout 与 `traindir` 是两批，
天然 held-out）。

图也用**冻结的起点图**（论文要求想象阶段复用起点图），所以这个误差反映的正是
想象轨迹所处的条件。

================================================================================
四个量
================================================================================
    reward_mae    预测奖励 vs 真实奖励（世界模型最直接有用的东西）
    cont_mae      预测继续概率 vs 真实（1 - is_terminal）
    kl            rollout 出来的 prior 与真实 posterior 的 KL —— 隐状态漂移
    state_mse    用 prior 特征重建观测的 MSE（decoder）—— 状态漂移的另一面

用法::

    python scripts/eval_world_model.py \
        --checkpoint "IDreamer=/tmp/runs/idreamer_s0_base/latest.pt" \
        --checkpoint "CoDreamer=/tmp/runs/codreamer_s0_base/latest.pt" \
        --horizons 1 2 4 8 16
"""

from __future__ import annotations

import argparse
import functools
import pathlib
import sys
import tempfile

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
D3T = REPO / "third_party" / "dreamerv3-torch"
sys.path.insert(0, str(D3T))

import dreamer as dreamer_mod  # noqa: E402
import models  # noqa: E402
import tools  # noqa: E402
import tools_marl  # noqa: E402


def load_checkpoint(path: pathlib.Path, device: str) -> dict:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    for key in ("agent_state_dict", "config"):
        if key not in ckpt:
            raise SystemExit(f"{path} 里没有 {key}; 是不是旧格式的 checkpoint？")
    return ckpt


def build_agent(config, obs_space, act_space):
    agent = dreamer_mod.Dreamer(obs_space, act_space, config, _SilentLogger(), None)
    return agent.to(config.device)


class _SilentLogger:
    """simulate_marl 需要一个 logger，但这里不需要写盘。"""

    step = 0

    def scalar(self, *a, **k):
        pass

    def write(self, *a, **k):
        pass


def heldout_batch(config, agent, repeats: int = 20, teams: int = 40):
    """用**策略自己**跑一批全新的 episode，作为 held-out。

    为什么不能用 `evaldir`：`tools_marl.simulate_marl` 在 `is_eval=True` 时
    **不落盘**（那个分支里没有 save_episodes），所以 evaldir 是空的。

    为什么不能用训练 replay：它的每一帧都被梯度更新看过，拿它算预测误差等于
    在训练集上报数。

    为什么不用随机策略采：那会得到与训练分布不同的状态，测的就不是"世界模型
    在自己见过的状态附近能不能预测"。用策略自己跑，数据既在分布内、又从未参与
    任何一次梯度更新。
    """
    with tempfile.TemporaryDirectory() as tmp:
        envs = [dreamer_mod.make_env(config, "eval", i) for i in range(config.envs)]
        try:
            tools_marl.simulate_marl(
                functools.partial(agent, training=False),
                envs,
                {},
                tmp,
                _SilentLogger(),
                is_eval=False,          # False 才会落盘，之后按与训练**完全相同**的格式读回来
                limit=10**6,
                episodes=teams * config.num_agents,
            )
        finally:
            for e in envs:
                try:
                    e.close()
                except Exception:
                    pass
        eps = tools.load_episodes(tmp, limit=10**6)
        if not eps:
            raise SystemExit("held-out rollout 没写出任何 episode")
        dataset = tools_marl.make_graph_dataset(eps, config)
        return [next(dataset) for _ in range(repeats)]


@torch.no_grad()
def openloop(wm, data, config, horizons, max_horizon: int):
    """返回 {horizon: {reward_mae, cont_mae, kl, state_mse, n}}。"""
    obs = wm.preprocess(data)
    graph = models.graph_from_data(obs, config.device)
    embed = wm.encoder(obs)
    post, _ = wm.dynamics.observe(embed, obs["action"], obs["is_first"], graph=graph)

    T = obs["action"].shape[1]
    H = max_horizon
    if T < H + 2:
        raise SystemExit(f"batch_length={T} 太短，open-loop 至少要 {H + 2}")

    t0 = T - H - 1
    state = {k: v[:, t0] for k, v in post.items()}
    # 冻结起点图 —— 与想象阶段一致（论文：整段轨迹复用起点图）
    if graph is None:
        g0 = None
    else:
        g0 = (graph[0][:, t0], None if graph[1] is None else graph[1][:, t0])

    out = {}
    for k in range(1, H + 1):
        state = wm.dynamics.img_step(state, obs["action"][:, t0 + k - 1], sample=False, graph=g0)
        feat = wm.dynamics.get_feat(state)

        rew_pred = wm.heads["reward"](feat, graph=g0).mean()
        cont_pred = wm.heads["cont"](feat, graph=g0).mean()
        rew_true = obs["reward"][:, t0 + k]
        cont_true = obs["cont"][:, t0 + k]

        # 隐状态漂移：rollout 出的 prior 与真实 posterior 的 KL
        post_k = {key: v[:, t0 + k] for key, v in post.items()}
        _, kl_value, _, _ = wm.dynamics.kl_loss(post_k, state, 1.0, 0.5, 1.0)

        # 状态漂移的另一面：用 prior 特征重建观测
        recon = wm.heads["decoder"](feat)["state"].mean()
        state_mse = ((recon - obs["state"][:, t0 + k]) ** 2).mean()

        if k in horizons:
            out[k] = {
                "reward_mae": float((rew_pred - rew_true).abs().mean()),
                "cont_mae": float((cont_pred - cont_true).abs().mean()),
                "kl": float(kl_value.mean()),
                "state_mse": float(state_mse),
                "n": int(rew_true.numel()),
            }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", action="append", required=True, help="名称=路径")
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--batches", type=int, default=20, help="重复采样几批")
    ap.add_argument("--teams", type=int, default=40, help="held-out rollout 采几支队")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    ckpts = []
    for item in args.checkpoint:
        name, _, path = item.partition("=")
        ckpts.append((name or pathlib.Path(path).stem, pathlib.Path(path).expanduser()))

    horizons = sorted(set(args.horizons))
    H = max(horizons)

    results = {}
    for name, path in ckpts:
        ckpt = load_checkpoint(path, args.device)
        cfg = argparse.Namespace(**ckpt["config"])
        cfg.device = args.device
        print(
            f"\n=== {name}  ({path})\n"
            f"    {cfg.comm_variant}  gnn: rssm={cfg.gnn_layers} "
            f"head={cfg.gnn_head_layers} behavior={cfg.gnn_behavior_layers}"
        )
        with tempfile.TemporaryDirectory() as tmp:
            env = dreamer_mod.make_env(cfg, "eval", 0)
            cfg.num_agents = getattr(env, "num_agents", 1)
            acts = env.action_space
            cfg.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
            cfg.evaldir = cfg.evaldir or str(pathlib.Path(tmp) / "eval_eps")
            agent = build_agent(cfg, env.observation_space, env.action_space)
            inc = agent.load_state_dict(ckpt["agent_state_dict"], strict=False)
            if inc.missing_keys or inc.unexpected_keys:
                raise SystemExit(
                    f"{path} 结构不匹配：缺 {len(inc.missing_keys)} 多 "
                    f"{len(inc.unexpected_keys)} —— 拒绝用随机权重评测"
                )
            agent.eval()
            wm = agent._wm

            batches = heldout_batch(cfg, agent, args.batches, args.teams)
            acc: dict[int, list[dict]] = {h: [] for h in horizons}
            for data in batches:
                got = openloop(wm, data, cfg, set(horizons), H)
                for h in horizons:
                    acc[h].append(got[h])
        env.close()

        print(f"    {'horizon':>8}  {'reward_mae':>11}  {'cont_mae':>9}  {'kl':>8}  {'state_mse':>10}")
        rows = {}
        for h in horizons:
            m = {
                key: float(np.mean([b[key] for b in acc[h]]))
                for key in ("reward_mae", "cont_mae", "kl", "state_mse")
            }
            rows[h] = m
            print(
                f"    {h:>8}  {m['reward_mae']:>11.4f}  {m['cont_mae']:>9.4f}  "
                f"{m['kl']:>8.4f}  {m['state_mse']:>10.4f}"
            )
        results[name] = rows

    if args.out_json:
        import json

        p = pathlib.Path(args.out_json).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(results, ensure_ascii=False, indent=1))
        print(f"\n  已保存 {p}")

    print(
        "\n  说明：全部是 held-out（evaldir）上的**开环**误差 —— 只喂真实动作、"
        "\n  不再看观测。曲线随 horizon 上升多少，才是世界模型\"想得准不准\"。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
