"""E0：用**同一个探针**测量两批数据，把"测量口径"这个假设排除掉。

================================================================================
为什么要这样设计
================================================================================
现在有两个互相矛盾的数：

    开环（训练日志，eval 录制局）     奖励误差 ≈ 0.000
    利用探针（新 rollout，策略动作）  奖励误差 ≈ 0.151

两者用的是**同一个统计量**（`_dist_mean`），而且在 k=1 时
"从后验走一步 + 奖励头" 与开环的 h1 完全是同一个计算。所以差异**不可能**
来自公式，只可能来自：

    (a) 喂进去的数据不同（状态分布）
    (b) 我漏掉了某个计算细节

这个脚本把 openloop_error 的 h1 计算原样复刻到 eval episode 上，再把同一份
代码跑在新 rollout 上。两批数据、同一个函数：

    两批结果接近  -> 之前的差异是某处细节（继续查计算）
    两批结果不同  -> 差异来自**数据**：模型在一批状态上准、在另一批上偏

================================================================================
用法
================================================================================
    python scripts/e0_probe_on_eval_eps.py \
        --checkpoint oc60k=/tmp/runs/oc_demo_60k/codreamer_shaped_s0/latest.pt \
        --evaldir /tmp/runs/oc_demo_60k/codreamer_shaped_s0/eval_eps \
        --episodes 8 --steps 60
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "third_party" / "dreamerv3-torch"))

import dreamer as dreamer_mod  # noqa: E402
import models  # noqa: E402
import tools  # noqa: E402
import tools_marl  # noqa: E402


class _SilentLogger:
    step = 0

    def scalar(self, *a, **k):
        pass

    def write(self, *a, **k):
        pass


def _load_agent(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = argparse.Namespace(**ckpt["config"])
    cfg.device = device
    cfg.envs = 1
    cfg.evaldir = cfg.evaldir or ""
    env = dreamer_mod.make_env(cfg, "eval", 0)
    cfg.num_agents = getattr(env, "num_agents", 1)
    acts = env.action_space
    cfg.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
    agent = dreamer_mod.Dreamer(
        env.observation_space, env.action_space, cfg, _SilentLogger(), None
    ).to(device)
    agent.load_state_dict(ckpt["agent_state_dict"], strict=False)
    agent.eval()
    return agent, cfg, env


def _stats(pred, real, state_mse):
    P, R = np.asarray(pred), np.asarray(real)
    return {
        "n": int(P.size),
        "pred_mean": float(P.mean()),
        "pred_std": float(P.std()),
        "real_mean": float(R.mean()),
        "real_std": float(R.std()),
        "bias": float(P.mean() - R.mean()),
        "mae": float(np.abs(P - R).mean()),
        "state_mse": float(np.mean(state_mse)) if state_mse else float("nan"),
    }


def probe_fixed_data(agent, cfg, data, dev):
    """openloop_error 的 h1 原样复刻：奖励头看**一步**之后的先验隐状态。"""
    wm, dyn = agent._wm, agent._wm.dynamics
    obs = wm.preprocess(data)
    graph = models.graph_from_data(obs, dev)
    with torch.no_grad():
        embed = wm.encoder(obs)
        post, _ = dyn.observe(embed, obs["action"], obs["is_first"], graph=graph)
        # 用 t 时刻的后验 + t 时刻的动作，预测 t+1
        state = {k: v[:, :-1] for k, v in post.items()}
        act = obs["action"][:, :-1]
        g_prev = _slice_graph(graph, slice(None, -1))
        nxt = dyn.img_step(state, act, sample=False, graph=g_prev)
        feat = dyn.get_feat(nxt)
        head = wm.heads["reward"]
        rp = head(feat, graph=g_prev) if "reward" in wm._comm_heads else head(feat)
        pred = models._dist_mean(rp).squeeze(-1)
        rew = obs["reward"][:, 1:]
        dec = wm.heads["decoder"](feat)["state"]
        recon = models._dist_mean(dec)
        target = obs["state"][:, 1:]
        smse = float(((recon - target) ** 2).mean())
    return pred.reshape(-1).cpu().numpy(), rew.reshape(-1).cpu().numpy(), smse


def _slice_graph(graph, sl):
    if graph is None:
        return None
    return tuple(g[:, sl] if g is not None else None for g in graph)


def probe_fresh_rollout(agent, cfg, env, dev, steps, policy="trained"):
    """同一条计算，跑在**新 rollout** 上。

    `policy` 是 E1 的自变量，三臂只换这一个东西：
      trained  当前 checkpoint 的 actor（mode）
      demo     脚本交接策略（真实奖励非零）
      random   均匀随机（真实奖励恒 0，且不依赖任何学习）
    """
    wm, dyn, behavior = agent._wm, agent._wm.dynamics, agent._task_behavior
    n_agents = max(1, int(cfg.num_agents))
    rng = np.random.default_rng(0)
    demo = None
    if policy == "demo":
        from envs.overcooked_demos import HandoffPolicy
        demo = HandoffPolicy(env)
    preds, reals, smses = [], [], []
    with torch.no_grad():
        out = env.reset()
        obs = out[0] if isinstance(out, tuple) else out
        latent, prev_action = None, None
        for _ in range(steps):
            # 环境给的是 (N,·)，执行路径要 (1,N,·)：补一个 batch 轴
            o = {
                k: torch.as_tensor(np.asarray(v), device=dev, dtype=torch.float32)[None]
                for k, v in obs.items() if "log_" not in k
            }
            graph = models.execution_graph(o, n_agents, dev)
            embed = wm.encoder(o)
            latent, _ = dyn.obs_step(latent, prev_action, embed, o["is_first"], graph=graph)
            feat = dyn.get_feat(latent)
            # 三臂的唯一区别就在这一行
            if policy == "trained":
                action = behavior.actor(feat, graph=graph).mode()
                idx = action.argmax(-1)[0].cpu().numpy().astype(np.int64)
            elif policy == "demo":
                idx = np.asarray(demo()).reshape(-1).astype(np.int64)
                action = torch.zeros(
                    (1, n_agents, cfg.num_actions), device=dev, dtype=torch.float32
                )
                action[0, torch.arange(n_agents), torch.as_tensor(idx, device=dev)] = 1.0
            else:                                            # random
                idx = rng.integers(0, cfg.num_actions, size=n_agents).astype(np.int64)
                action = torch.zeros(
                    (1, n_agents, cfg.num_actions), device=dev, dtype=torch.float32
                )
                action[0, torch.arange(n_agents), torch.as_tensor(idx, device=dev)] = 1.0

            nxt = dyn.img_step(latent, action, sample=False, graph=graph)
            nfeat = dyn.get_feat(nxt)
            head = wm.heads["reward"]
            rp = head(nfeat, graph=graph) if "reward" in wm._comm_heads else head(nfeat)
            preds.append(models._dist_mean(rp).reshape(-1).cpu())

            out = env.step(idx.reshape(1, -1))
            obs, rew = out[0], out[1]
            # 状态头：模型认为的下一步观测 vs 真实下一步观测
            if isinstance(obs, dict) and "state" in obs:
                rn = torch.as_tensor(
                    np.asarray(obs["state"]), device=dev, dtype=torch.float32
                )[None].reshape(-1)
                recon = models._dist_mean(wm.heads["decoder"](nfeat)["state"]).reshape(-1)
                smses.append(float(((recon - rn) ** 2).mean()))
            reals.append(torch.as_tensor(np.asarray(rew), device=dev, dtype=torch.float32).reshape(-1).cpu())
            prev_action = action
            if bool(np.asarray(out[2]).all()):
                break
    return (torch.cat(preds).numpy(), torch.cat(reals).numpy(),
            float(np.mean(smses)) if smses else float("nan"))


def probe_prior_vs_post(agent, cfg, env, dev, steps):
    """E4：同一时刻、同一状态，奖励头分别看**后验**和**先验**特征。

    奖励头在 `WorldModel._train` 里是对着**后验**特征训的（观测校正过），
    而 actor 的想象里查询的是**先验**特征（模型自己滚出来的）。两者在数据熟悉的
    状态上接近，在策略探索到的不确定状态上会分开。

    如果偏差只出现在 prior 那一列，就说明问题不是"奖励头不准"，而是
    **训练和推理喂给它的特征不是同一个分布** —— 这决定了修法。
    """
    wm, dyn, behavior = agent._wm, agent._wm.dynamics, agent._task_behavior
    n_agents = max(1, int(cfg.num_agents))
    head = wm.heads["reward"]
    out = env.reset()
    obs = out[0] if isinstance(out, tuple) else out
    latent, prev_action = None, None
    rows = []
    # ⚠️ 对齐：奖励 r_t 是"从 s_t 走到 s_{t+1}"拿到的那一份。所以要比的是
    #   · 先验：s_t 时刻的隐状态 + 动作 a_t 推出来的下一步（想象时的条件）
    #   · 后验：**下一轮** obs_step 得到的 s_{t+1}（训练时的条件）
    # 早先版本拿 s_t 的预测去比 r_t，差了一步，两个偏差都算成 0 —— 是测量错误，
    # 不是模型没问题。
    pending = None          # (r_prior_for_next, real_reward)
    with torch.no_grad():
        for t in range(steps):
            o = {
                k: torch.as_tensor(np.asarray(v), device=dev, dtype=torch.float32)[None]
                for k, v in obs.items() if "log_" not in k
            }
            graph = models.execution_graph(o, n_agents, dev)
            embed = wm.encoder(o)
            latent, _ = dyn.obs_step(
                latent, prev_action, embed, o["is_first"], graph=graph
            )
            post_feat = dyn.get_feat(latent)

            def rew(f):
                # ⚠️ 对**所有 agent** 取平均。早先只取 [0]（配料区那个），
                # 而它的状态最简单、本来就不偏，于是把三个 agent 的偏差平均掉了，
                # 和 B 组（全体平均 +0.096）直接矛盾。
                rp = head(f, graph=graph) if "reward" in wm._comm_heads else head(f)
                return float(models._dist_mean(rp).mean())

            # 后验这一列属于**上一步的转移**（s_t 的后验 vs r_{t-1}）
            if pending is not None:
                rows.append((rew(post_feat), pending[0], pending[1]))

            action = behavior.actor(post_feat, graph=graph).mode()
            # 先验这一列：同一个状态、同一个动作，用模型自己的下一步
            nxt = dyn.img_step(latent, action, sample=False, graph=graph)
            r_prior_next = rew(dyn.get_feat(nxt))

            idx = action.argmax(-1)[0].cpu().numpy().astype(np.int64)
            out = env.step(idx.reshape(1, -1))
            obs, rew_v = out[0], out[1]
            pending = (r_prior_next, float(np.asarray(rew_v).reshape(-1).mean()))
            prev_action = action
            if bool(np.asarray(out[2]).all()):
                break
    return np.asarray(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--evaldir", required=True)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    name, path = args.checkpoint.split("=", 1)
    agent, cfg, env = _load_agent(path, args.device)
    dev = args.device

    eps = tools.load_episodes(args.evaldir, limit=int(cfg.dataset_size))
    print(f"{name}: eval 目录里 {len(eps)} 条 agent episode")
    if not eps:
        print("没有 eval 数据，跳过 A 组")
    else:
        data = next(tools_marl.make_graph_dataset(eps, cfg))
        p, r, smse = probe_fixed_data(agent, cfg, data, dev)
        a = _stats(p, r, [smse])
        print("A 组（eval 录制局，录制动作）")
        print(f"   n={a['n']}  pred={a['pred_mean']:+.4f}  real={a['real_mean']:+.4f}  "
              f"bias={a['bias']:+.4f}  mae={a['mae']:.4f}  state_mse={a['state_mse']:.6f}")

    print("\nB 组（新 rollout，三个策略臂；模型、环境、动作以外全部相同）")
    arms = {}
    for pol in ("trained", "demo", "random"):
        p2, r2, smse2 = probe_fresh_rollout(
            agent, cfg, env, dev, args.steps * args.episodes, policy=pol
        )
        arms[pol] = _stats(p2, r2, [smse2])
        b = arms[pol]
        print(f"   {pol:<8} n={b['n']:<5} pred={b['pred_mean']:+.4f}  real={b['real_mean']:+.4f}  "
              f"bias={b['bias']:+.4f}  mae={b['mae']:.4f}  state_mse={b['state_mse']:.6f}")

    if len(eps):
        print("\n判读")
        print(f"  A（eval 录制局）      bias={a['bias']:+.4f}")
        for pol, b in arms.items():
            print(f"  B-{pol:<8}          bias={b['bias']:+.4f}")
        print("  demo 臂真实奖励非零，是唯一能检验'有奖励时模型准不准'的对照。")

    # ── E4：后验 vs 先验 ────────────────────────────────────────────────
    print("\nE4（同一时刻，奖励头分别吃后验特征 / 先验特征）")
    for pol in ("trained", "demo"):
        rows = probe_prior_vs_post(agent, cfg, env, dev, args.steps)
        if not len(rows):
            continue
        post_bias = float(np.nanmean(rows[:, 0] - rows[:, 2]))
        prior_bias = float(np.nanmean(rows[:, 1] - rows[:, 2]))
        print(f"   {pol:<8} n={len(rows):<4} 后验偏差={post_bias:+.4f}   "
              f"先验偏差={prior_bias:+.4f}   真实均值={rows[:, 2].mean():+.4f}")
    print("   判读：后验≈0 而先验显著为正 -> 奖励头在训练分布上是准的，")
    print("         问题出在想象时喂的是先验特征（训练/推理特征不同分布）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
