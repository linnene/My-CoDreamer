"""想象轨迹是否塌缩？—— 直接测 prior 往前滚时各头还有没有变化。

================================================================================
为什么
================================================================================
训练日志里有一组自相矛盾的读数：

    reward_loss   0.3     ← 真实数据（posterior）上奖励头**学得挺好**
    imag_reward_mean 0.0  ← 想象出来的奖励均值是 0
    imag_reward_std  0.0  ← 而且是**常数**，完全没有变化

如果奖励头在真实特征上准、在想象特征上恒为常数，那问题不在头，在**想象出来的
特征**：prior 往前滚几步之后塌到了同一个点。

这解释了 critic 校准脚本测到的东西 —— λ-return 的自举项吃掉了一切，目标退化成
"critic 自己乘一个折扣"，任何常数都是它的不动点，于是 V 与真实回报相关系数 ≈ 0
（IDreamer +0.036、CoDreamer +0.127）。

================================================================================
测什么
================================================================================
用 posterior 吃完前段序列取一个真实状态，然后**只喂真实动作、不看观测**让 prior
往前滚 H 步，逐 k 记录：

    rew_std(k)     想象奖励在 (batch, agent) 上的标准差 —— 塌缩则趋于 0
    cont_std(k)    同理
    feat_delta(k)  第 k 步特征与第 k-1 步的距离 —— 塌缩则趋于 0
    feat_spread(k) 特征在 (batch, agent) 上的散度 —— 塌缩则趋于 0

对照：真实奖励在同一步上的标准差。若 imag 的 std 比真实小几个数量级，
就是塌缩，不是"奖励本来就小"。

用法::

    python scripts/check_imagination.py --checkpoint NAME=PATH --episodes DIR
"""

from __future__ import annotations

import argparse
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


class _SilentLogger:
    step = 0

    def scalar(self, *a, **k):
        pass

    def write(self, *a, **k):
        pass


def probe(path: pathlib.Path, args) -> dict:
    ckpt = torch.load(path, map_location=args.device, weights_only=False)
    cfg = argparse.Namespace(**ckpt["config"])
    cfg.device = args.device

    with tempfile.TemporaryDirectory() as tmp:
        env = dreamer_mod.make_env(cfg, "eval", 0)
        cfg.num_agents = getattr(env, "num_agents", 1)
        acts = env.action_space
        cfg.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
        cfg.evaldir = cfg.evaldir or str(pathlib.Path(tmp) / "eval_eps")
        agent = dreamer_mod.Dreamer(
            env.observation_space, env.action_space, cfg, _SilentLogger(), None
        ).to(cfg.device)
        inc = agent.load_state_dict(ckpt["agent_state_dict"], strict=False)
        if inc.missing_keys or inc.unexpected_keys:
            raise SystemExit(f"{path} 结构不匹配")
        agent.eval()
        eps = tools.load_episodes(pathlib.Path(args.episodes), limit=10**6)
        dataset = tools_marl.make_graph_dataset(eps, cfg)
        batches = [next(dataset) for _ in range(args.batches)]
        env.close()

    wm = agent._wm
    H = int(args.horizon)

    rows = []
    with torch.no_grad():
        for data in batches:
            obs = wm.preprocess(data)
            graph = models.graph_from_data(obs, cfg.device)
            embed = wm.encoder(obs)
            post, _ = wm.dynamics.observe(
                embed, obs["action"], obs["is_first"], graph=graph
            )
            T = obs["action"].shape[1]
            t0 = T - H - 1
            state = {k: v[:, t0] for k, v in post.items()}
            g0 = None
            if graph is not None:
                g0 = (graph[0][:, t0], None if graph[1] is None else graph[1][:, t0])

            prev = wm.dynamics.get_feat(state)
            rec = []
            for k in range(1, H + 1):
                state = wm.dynamics.img_step(
                    state, obs["action"][:, t0 + k - 1], sample=False, graph=g0
                )
                feat = wm.dynamics.get_feat(state)
                rew = models._dist_mean(
                    wm.heads["reward"](feat, graph=g0)
                    if "reward" in wm._comm_heads else wm.heads["reward"](feat)
                ).reshape(feat.shape[:-1])
                ch = wm.heads["cont"]
                cont_pred = models._dist_mean(
                    ch(feat, graph=g0) if "cont" in wm._comm_heads else ch(feat)
                ).reshape(feat.shape[:-1])
                rec.append({
                    "k": k,
                    "cont_mean": float(cont_pred.mean()),
                    "cont_std": float(cont_pred.std()),
                    "rew_std": float(rew.std()),
                    "rew_absmean": float(rew.abs().mean()),
                    "feat_delta": float((feat - prev).norm(dim=-1).mean()),
                    "feat_spread": float(feat.std(dim=1).mean()),
                    "true_rew_std": float(obs["reward"][:, t0 + k].std()),
                    "true_rew_absmean": float(obs["reward"][:, t0 + k].abs().mean()),
                })
                prev = feat
            rows.append(rec)

    keys = rows[0][0].keys()
    out = {k: {} for k in keys if k != "k"}
    for r in rows:
        for step in r:
            for key in out:
                out[key].setdefault(step["k"], []).append(step[key])
    res = {k: {kk: float(np.mean(vv)) for kk, vv in v.items()} for k, v in out.items()}
    res["_gamma"] = float(cfg.discount)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", action="append", required=True, help="名称=路径")
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    for item in args.checkpoint:
        name, _, path = item.partition("=")
        print(f"\n=== {name} ===")
        r = probe(pathlib.Path(path).expanduser(), args)
        ks = sorted(k for k in r["rew_std"])
        print(f"  {'k':>3}{'imag_rew_std':>15}{'true_rew_std':>15}{'imag|rew|':>12}"
              f"{'feat_delta':>13}{'feat_spread':>13}{'cont_mean':>12}")
        for k in ks:
            print(f"  {k:>3}{r['rew_std'][k]:>15.3e}{r['true_rew_std'][k]:>15.3e}"
                  f"{r['rew_absmean'][k]:>12.3e}{r['feat_delta'][k]:>13.3e}"
                  f"{r['feat_spread'][k]:>13.3e}{r['cont_mean'][k]:>12.4f}")

        first, last = ks[0], ks[-1]
        ratio = r["rew_std"][last] / max(r["rew_std"][first], 1e-12)
        tratio = r["rew_std"][last] / max(r["true_rew_std"][last], 1e-12)
        print(f"  → 想象奖励的散度 k=1→{last}: {ratio:.3f}×   "
              f"末步 imag/true 散度比: {tratio:.3e}")
        print("     塌缩的判据：散度随 k 单调趋零，且比真实散度小几个数量级")
        cm = r["cont_mean"][last]
        g = float(r["_gamma"])
        print(f"  → cont(k={last}) = {cm:.4f}  → 折扣 γ·cont = {g*cm:.5f}  "
              f"→ 累积视界 1/(1-γ·cont) ≈ {1.0/max(1e-9, 1-g*cm):.1f} 步")

    print("\n" + "=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
