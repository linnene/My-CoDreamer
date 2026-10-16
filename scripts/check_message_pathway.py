"""通信通路是不是真的在传信息？—— 只改图，别的一律不动，看预测和动作动不动。

================================================================================
为什么需要这个
================================================================================
台账目前对"通信在起作用"只有**间接**证据：

  - 参数量恒等式（打开 GNN 确实多了参数）
  - 四臂消融里 wm_comm 与其余三臂的 argmax 不同（差 0.739）

但参数量证明的是"参数在那儿"，不是"信息流过"。而四臂消融那条测的是 **SEG**，
SEG 的有效累积视界只有 1.4 步，策略看一步就够 —— 在那个环境上"通信配置之间
动作相同"是**预期的**，不能当作"通路是死的"。

本脚本直接测：**固定一切，只换图**，看世界模型预测与 actor 决策变不变。

================================================================================
四种图（唯一的自变量）
================================================================================
  real      真实的 adjacency + edge_features（基线）
  empty     邻接全零。GATv2 层内部加自环，所以每个节点只看得到自己
            —— 这就是"消息被清零"的等价形态，且不需要改任何模型代码
  full      完全图（去掉对角），边特征用真实边的均值填充
  shuffled  保持每个 (b,t) 图的**边数不变**，但把边随机重连
            —— 密度一样、拓扑打乱，用于区分"有邻居"与"是这些邻居"

================================================================================
读什么数
================================================================================
  rew / cont   世界模型两个头的输出，与 real 的相对差（只换图，特征不变）
  logit_d      actor 的 logits 与 real 的 L2 距离
  argmax_agr   actor 贪心动作与 real 的一致率
  hard_frac    四张图里出现过 >1 种 argmax 的 (样本, agent) 比例

================================================================================
★ 承重的对照 ★
================================================================================
**必须同时跑一个 `idreamer` checkpoint。** 它的三处 GNN 层数都是 0，图**从不被消费**，
所以四种干预下输出必须**逐位相同**。这是证明"本脚本的干预真的只动了图、
没有侧漏到别处"的唯一判据。没有这一栏，codreamer 上的正结果不可解读。

用法::

    python scripts/check_message_pathway.py \\
      --checkpoint idreamer=/tmp/runs/vmas_rep_idreamer_s1/latest.pt \\
      --checkpoint codreamer=/tmp/runs/vmas_rep_codreamer_s1/latest.pt \\
      --episodes /tmp/runs/vmas_rep_codreamer_s1/eval_eps
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

MODES = ("real", "empty", "full", "shuffled")


class _SilentLogger:
    step = 0

    def scalar(self, *a, **k):
        pass

    def write(self, *a, **k):
        pass


# --------------------------------------------------------------------- 图的构造


def _intervene(adj, edge, mode, rng):
    """adj (B,T,N,N)、edge (B,T,N,N,F) 或 None → 同形状的新图。

    内部先摊平成 (B*T,N,N) 逐图处理，再还原形状 —— `graph_from_data` 给的是
    四维（含时间轴），而"每个时刻一张图"才是干预的单位。
    """
    if mode == "real":
        return adj, edge

    shape = adj.shape
    n = shape[-1]
    adj = adj.reshape(-1, n, n)
    eshape = None if edge is None else edge.shape
    edge = None if edge is None else edge.reshape(-1, n, n, eshape[-1])
    eye = torch.eye(n, dtype=torch.bool, device=adj.device)
    off = ~eye

    def _restore(a, e):
        # ⚠️ 不能用 adj.shape 还原 edge：F=1 时 (B*T,N,N,1) → (B,T,N,N) 会**静默**
        # 吃掉特征维，GNN 那边立刻变成 (N,N) 求 Linear，报 mat1 16x4。
        return a.reshape(shape), (None if e is None else e.reshape(eshape))

    if mode == "empty":
        a = torch.zeros_like(adj)
        e = None if edge is None else torch.zeros_like(edge)
        return _restore(a, e)

    if mode == "full":
        a = off[None].expand_as(adj).to(adj.dtype).clone()
        if edge is None:
            return _restore(a, None)
        # 新边的特征用"真实边的均值"填，避免引入一个 0（0 在真实图里表示"无边"）
        present = adj.bool()
        fill = edge[present].mean() if present.any() else torch.ones(
            (), dtype=edge.dtype, device=edge.device)
        e = torch.where(present.unsqueeze(-1), edge, fill.expand_as(edge))
        return _restore(a, e)

    if mode == "shuffled":
        # 保持每个 (b,t) 的边数，但把边打乱重连（不碰对角）
        a = torch.zeros_like(adj)
        off_pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
        counts = adj.bool().sum(dim=(1, 2)).tolist()
        for k in range(a.shape[0]):
            c = min(int(counts[k]), len(off_pairs))
            if c <= 0:
                continue
            for p in rng.permutation(len(off_pairs))[:c]:
                i, j = off_pairs[int(p)]
                a[k, i, j] = 1.0
        if edge is None:
            return _restore(a, None)
        present = adj.bool()
        fill = edge[present].mean() if present.any() else torch.ones(
            (), dtype=edge.dtype, device=edge.device)
        e = torch.where(a.bool().unsqueeze(-1), fill.expand_as(edge),
                        torch.zeros_like(edge))
        return _restore(a, e)

    raise ValueError(mode)


# ------------------------------------------------------------------------ 探测


def probe(path: pathlib.Path, episodes: pathlib.Path, args, rng) -> dict:
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
        eps = tools.load_episodes(episodes, limit=10**6)
        dataset = tools_marl.make_graph_dataset(eps, cfg)
        n_agents = cfg.num_agents
        env.close()

    wm = agent._wm
    acc = {m: {"rew": [], "cont": [], "logit": [], "argmax": []} for m in MODES}

    with torch.no_grad():
        for it in range(args.batches):
            data = next(dataset)
            obs = wm.preprocess(data)
            embed = wm.encoder(obs)
            real_g = models.graph_from_data(obs, cfg.device)
            if real_g is None:
                raise SystemExit("这个 checkpoint 的观测里没有图，无法做干预测试")

            for mode in MODES:
                # ★ 承重的一行 ★ RSSM 的 stoch 是**采样**出来的，同一个输入连跑两次
                # 结果不同。不固定种子的话，"四种图的差异"里混着采样噪声 ——
                # 实测过：idreamer（图从不被消费）也会显示出 logit Δ 3.1 的假差异。
                # 四次前向用同一个种子 → 采样流一致，剩下的差异只可能来自图。
                torch.manual_seed(args.seed * 100003 + it)
                torch.cuda.manual_seed_all(args.seed * 100003 + it)

                adj, edge = _intervene(real_g[0], real_g[1], mode, rng)
                g = (adj, edge)
                post, _ = wm.dynamics.observe(
                    embed, obs["action"], obs["is_first"], graph=g
                )
                feat = wm.dynamics.get_feat(post)

                rew = models._dist_mean(
                    wm.heads["reward"](feat, graph=g)
                    if "reward" in wm._comm_heads else wm.heads["reward"](feat)
                ).reshape(feat.shape[:-1])
                ch = wm.heads["cont"]
                cont = models._dist_mean(
                    ch(feat, graph=g)
                    if "cont" in wm._comm_heads else ch(feat)
                ).reshape(feat.shape[:-1])

                # actor 要的是 (B*T, N, ·)：post 的布局正是这个
                flat_feat = feat.reshape(-1, n_agents, feat.shape[-1])
                flat_g = (adj.reshape(-1, n_agents, n_agents),
                          None if edge is None else
                          edge.reshape(-1, n_agents, n_agents, edge.shape[-1]))
                logits = agent._task_behavior.actor(flat_feat, graph=flat_g).logits

                rec = {"rew": rew, "cont": cont, "logit": logits,
                       "argmax": logits.argmax(-1)}
                acc[mode]["rew"].append(rec["rew"].reshape(-1))
                acc[mode]["cont"].append(rec["cont"].reshape(-1))
                acc[mode]["logit"].append(rec["logit"].reshape(-1, logits.shape[-1]))
                acc[mode]["argmax"].append(rec["argmax"].reshape(-1))

    out = {}
    for m in MODES:
        rew = torch.cat(acc[m]["rew"])
        cont = torch.cat(acc[m]["cont"])
        logit = torch.cat(acc[m]["logit"])
        argm = torch.cat(acc[m]["argmax"])
        out[m] = {"rew": rew, "cont": cont, "logit": logit, "argmax": argm}

    b = out["real"]
    rows = {}
    for m in MODES:
        v = out[m]
        rows[m] = {
            "rew_absdiff": float((v["rew"] - b["rew"]).abs().mean()),
            "cont_absdiff": float((v["cont"] - b["cont"]).abs().mean()),
            "rew_scale": float(b["rew"].abs().mean()),
            "logit_l2": float((v["logit"] - b["logit"]).norm(dim=-1).mean()),
            "argmax_agr": float((v["argmax"] == b["argmax"]).float().mean()),
            "bitwise": bool(torch.equal(v["logit"], b["logit"])),
        }
    multi = torch.stack([out[m]["argmax"] for m in MODES])
    rows["_hard_frac"] = float((multi.max(0).values != multi.min(0).values).float().mean())
    rows["_n"] = int(b["logit"].shape[0])
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", action="append", required=True, help="名称=路径")
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    for item in args.checkpoint:
        name, _, path = item.partition("=")
        print(f"\n=== {name} ===")
        r = probe(pathlib.Path(path).expanduser(), pathlib.Path(args.episodes), args, rng)
        print(f"  {'图':<10}{'rew|Δ|':>12}{'cont|Δ|':>12}{'logit Δ':>12}"
              f"{'argmax 一致':>13}{'逐位相同':>10}")
        for m in MODES:
            v = r[m]
            print(f"  {m:<10}{v['rew_absdiff']:>12.3e}{v['cont_absdiff']:>12.3e}"
                  f"{v['logit_l2']:>12.3e}{v['argmax_agr']:>13.4f}"
                  f"{'是' if v['bitwise'] else '否':>10}")
        print(f"  → 参考量级：|rew| 均值 {r['real']['rew_scale']:.3e}，"
              f"样本数 {r['_n']}")
        print(f"  → 四种图里出现过 >1 种贪心动作的 (样本,agent) 比例："
              f"{r['_hard_frac']:.4f}")

    print("\n" + "=" * 74)
    print("读法：")
    print("  · idreamer 那一栏四行必须**逐位相同** —— 图从不被消费。若不同，")
    print("    说明本脚本的干预侧漏了，下面的数全部作废。")
    print("  · codreamer 上 'empty' 若与 'real' 有可见差，则通信通路确实在传信息。")
    print("  · 'shuffled' 与 'real' 的差 ＞ 'empty' 与 'real' 的差，说明模型用的")
    print("    不只是'有几个邻居'，而是'是哪些邻居'。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
