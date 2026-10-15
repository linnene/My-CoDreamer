import numpy as np
import copy
import torch
from torch import nn

import networks
import tools

to_np = lambda x: x.detach().cpu().numpy()


# [CoDreamer] 图批次 (B, T, N, ·) -> 展平批次 (B*N, T, ·)
def fold_agent_axis(data):
    """把样本内部的 agent 轴折进 batch 维。

    数据管线（tools_marl.sample_episode_graphs）产出的是 (B, T, N, ·)：N 是**样本
    内部**的一个轴，因为它同一时刻的 N 个 agent 必须能互相看见（GNN 要在它上面聚合）。

    但 IDreamer 的语义是"每个 agent 各训各的"，agent 之间没有信息流动，所以把 N
    折进 batch 与分开训练**完全等价** —— 前向是逐样本独立的，也没有 BatchNorm。
    折完之后 encoder / RSSM / 各预测头 / actor-critic 一行都不用改。

    CoDreamer 则相反：GNN 正是要在 agent 轴上聚合，所以不能折。
    验收标准：GNN 层数为 0 时，折平的损失必须与不折的同一份数据完全一致。
    """
    out = {}
    for key, value in data.items():
        value = np.asarray(value)
        assert value.ndim >= 3, (
            f"expected (B, T, N, ...) for folding, got {key}={value.shape}"
        )
        b, t, n = value.shape[:3]
        # 必须先转置再 reshape。(B,T,N,·) 在内存里 T 与 N 是交错的，直接 reshape
        # 成 (B*N,T,·) 会得到错位的垃圾。先换成 (B,N,T,·) 再合并前两维才是对的。
        # （这一步曾经写错，被 scripts/test_graph_pipeline.py 的顺序断言抓到。）
        perm = (0, 2, 1) + tuple(range(3, value.ndim))
        out[key] = value.transpose(perm).reshape((b * n, t) + value.shape[3:])
    return out


# [CoDreamer] 从批次里取出图拓扑。没有图就返回 None（IDreamer 路径）。
def graph_from_data(data, device=None, dtype=torch.float32):
    """返回 (adjacency, edge_features) 或 None。

    形状：adjacency (B, T, N, N)，edge_features (B, T, N, N, F)。
    agent 轴是倒数第二个轴，与数据管线一致。

    没看到 `adjacency` 就返回 None —— 这是一个干净的开关：环境不开图（
    `seg_graph_obs: false`）时整条链路自动退化成 IDreamer。
    """
    if "adjacency" not in data:
        return None
    def as_tensor(v):
        if torch.is_tensor(v):
            return v.to(device=device, dtype=dtype) if device is not None else v
        return torch.as_tensor(v, dtype=dtype, device=device)
    adjacency = as_tensor(data["adjacency"])
    edge = as_tensor(data["edge_features"]) if "edge_features" in data else None
    return adjacency, edge


# [CoDreamer] 执行路径的图：把铺平的 (batch, N) 还原成 GNN 要的 (batch, N, N)
#
# 这里必须显式还原，因为**环境与数据管线使用的是两种不同的约定**：
#
#   环境          每个 agent 的观测里带的是**整张** (N, N) 邻接矩阵（同一份给所有
#                 agent）—— 见 envs/seg_gym.py 的 _make_obs。
#   数据管线      `_graph_chunk` 把 N 个 agent 各自的那一行堆起来，得到 (T, N, N)。
#
# 两者恰好自洽：agent k 存的那一行就是矩阵的第 k 行，所以堆起来正好还原矩阵。
# 但中间要经过 simulate_marl 的 `np.concatenate([o[k] for o in obs], 0)`：
# 每份 (N, N) 沿 dim 0 拼起来得到 (n_teams*N, N) —— **每一行恰好是某支队伍
# 邻接矩阵的某一行**。GNN 要的是 (..., N, N)，直接把这堆行喂进去是错的。
#
# 所以这里把它读作 (n_teams, N, N) 再按 agent 广播回去。
#
# ⚠️ 依赖 simulate_marl 的一个不变量：batch 里 agent 是按队伍连续、且队伍内按
#    agent 下标 0..N-1 排列的。这不是偶然 —— `np.concatenate([o[k] for o in obs], 0)`
#    与 `acts.reshape(n_teams, n_agents, ·)` 用的就是同一个顺序，所以是真实约束。
#    下面用 assert 把它变成会报错的东西，而不是注释里的默契。
def execution_graph(obs, num_agents, device=None, dtype=torch.float32):
    """执行路径的 (adjacency, edge_features)，要求形状 (n_teams, N, N)。

    ⚠️ **调用前必须已经补上 agent 轴**（见 Dreamer._policy 里的 add_agent_axis）。
    这个函数不是多余的包装：这个形状已经错过三次了 ——
        ① 忘了传图；
        ② 传了但没把 (batch, N) 还原成矩阵；
        ③ 还原了矩阵但 batch 里没有 agent 轴。
    三次都表现为 GNN 内部的广播 / reshape 报错，而不是在调用点报错。
    所以这里把"agent 轴存在"变成显式断言。

    环境给每个 agent 的是**整张** (N, N) 邻接矩阵；simulate_marl 把每队铺平成
    (n_teams*N, ·) 后又铺回来，于是队伍 t 的 N 行恰好构成它自己的矩阵。
    """
    if "adjacency" not in obs:
        return None
    n = int(num_agents)

    def as_tensor(v):
        if torch.is_tensor(v):
            return v.to(device=device, dtype=dtype) if device is not None else v
        return torch.as_tensor(v, dtype=dtype, device=device)

    adjacency = as_tensor(obs["adjacency"])
    if n > 1:
        if adjacency.dim() != 3 or adjacency.shape[1] != n or adjacency.shape[2] != n:
            raise ValueError(
                f"执行路径的 adjacency 应该是 (n_teams, N={n}, N={n})，"
                f"得到 {tuple(adjacency.shape)}。"
                "是否忘了在 _policy 里补 agent 轴？"
            )
    else:
        if adjacency.dim() != 2:
            raise ValueError(
                f"单智能体下 adjacency 应该是 (n_teams, N)，得到 {tuple(adjacency.shape)}"
            )

    edge = None
    if "edge_features" in obs:
        edge = as_tensor(obs["edge_features"])
        want = (n, n) if n > 1 else (n,)
        if tuple(edge.shape[1:3]) != want or edge.shape[0] != adjacency.shape[0]:
            raise ValueError(
                f"执行路径的 edge_features 应该是 (n_teams, {n}, {n}, F)，"
                f"得到 {tuple(edge.shape)}"
            )
    return adjacency, edge


def _dist_mean(dist):
    """取分布的均值。

    不同分布类的接口**不一致**：奖励头（symlog_disc）把 `mean` 定为**属性**，
    继续头（binary）定为**方法**。直接 `.mean()` 会在前者上报
    `'Tensor' object is not callable`。这个坑踩过三次，所以统一走这里。
    """
    m = dist.mean
    return m() if callable(m) else m


@torch.no_grad()
def openloop_error(wm, data, horizons):
    """held-out 上的**多步开环**预测误差。返回 {horizon: {...}}。

    流程：posterior 吃完整个序列（允许看观测）→ 取某个时刻的状态 → 之后**只喂
    真实动作、不再看任何观测**，让 prior 往前走 → 在每个 horizon 比较预测与真值。

    为什么不能用 `_train` 返回的 `reward_loss` / `state_loss`：
      1. 那是**训练损失** —— 同一份 replay 被反复优化，损失下降可以只是记住了它；
      2. 它用的是 **posterior** 隐状态（`feat = get_feat(post)`），而 posterior 每步
         都**看过当步观测**。它衡量的是"给定当前帧重建当前帧"，不是预测未来。

    图用**冻结的起点图**（论文要求想象阶段复用起点图），所以这个误差反映的正是
    想象轨迹所处的条件。

    四个量：
        reward_mae   预测奖励 vs 真实奖励（最直接有用的东西）
        cont_mae     预测继续概率 vs 真实（1 - is_terminal）
        kl           rollout 出的 prior 与真实 posterior 的 KL —— 隐状态漂移
        state_mse    用 prior 特征重建观测的 MSE（decoder）

    这个函数是**训练日志与事后脚本共用的唯一实现**。复制一份出去就等于让两边的
    口径慢慢分家。
    """
    horizons = sorted({int(h) for h in horizons})
    H = max(horizons)
    obs = wm.preprocess(data)
    graph = graph_from_data(obs, wm._config.device)
    embed = wm.encoder(obs)
    post, _ = wm.dynamics.observe(embed, obs["action"], obs["is_first"], graph=graph)

    T = obs["action"].shape[1]
    if T < H + 2:
        raise ValueError(f"batch_length={T} 太短，open-loop 至少要 {H + 2}")
    t0 = T - H - 1
    state = {k: v[:, t0] for k, v in post.items()}
    if graph is None:
        g0 = None
    else:
        g0 = (graph[0][:, t0], None if graph[1] is None else graph[1][:, t0])

    def head(name, feat):
        # 与 _train 一致：只有 reward / cont 头接收图，decoder 永远独立。
        if name in wm._comm_heads:
            return wm.heads[name](feat, graph=g0)
        return wm.heads[name](feat)

    out: dict[int, dict[str, float]] = {}

    def align(pred, target):
        """预测头返回 (..., 1)，真值是 (...,)。

        不 squeeze 的话广播会**从右边对齐**： (2,4,1) 对 (2,4) 被读成
        (2,4,1) 对 (1,2,4)，dim1 上 4 vs 2 直接报错。
        """
        return pred.squeeze(-1) if pred.dim() == target.dim() + 1 else pred

    for k in range(1, H + 1):
        state = wm.dynamics.img_step(
            state, obs["action"][:, t0 + k - 1], sample=False, graph=g0
        )
        feat = wm.dynamics.get_feat(state)
        if k not in horizons:
            continue
        rew_true = obs["reward"][:, t0 + k]
        cont_true = obs["cont"][:, t0 + k]
        rew_pred = align(_dist_mean(head("reward", feat)), rew_true)
        cont_pred = align(_dist_mean(head("cont", feat)), cont_true)
        post_k = {key: v[:, t0 + k] for key, v in post.items()}
        _, kl_value, _, _ = wm.dynamics.kl_loss(post_k, state, 1.0, 0.5, 1.0)
        recon = _dist_mean(wm.heads["decoder"](feat)["state"])
        out[k] = {
            "reward_mae": float((rew_pred - rew_true).abs().mean()),
            "cont_mae": float((cont_pred - cont_true).abs().mean()),
            "kl": float(kl_value.mean()),
            "state_mse": float(
                ((recon - obs["state"][:, t0 + k]) ** 2).mean()
            ),
        }
    return out


# [CoDreamer] 图批次 (B, T, N, N) -> (B*T, N, N)
def flatten_graph(graph):
    """把图的时间维与 batch 维合并，与 `ImagBehavior._imagine` 对隐状态做的 flatten
    保持一致。

    B 与 T 是相邻的前置维，所以直接 reshape 就是对的（不需要转置）。
    想象阶段用**起点那张图**并用到底（论文："the adjacency matrix of the starting
    graph is used for all subsequent graphs"），所以只需要一份扁平化的图。
    """
    if graph is None:
        return None
    return tuple(
        None if g is None else g.reshape([-1] + list(g.shape[2:])) for g in graph
    )


class RewardEMA:
    """running mean and std"""

    def __init__(self, device, alpha=1e-2):
        self.device = device
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95], device=device)

    def __call__(self, x, ema_vals):
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)
        # this should be in-place operation
        ema_vals[:] = self.alpha * x_quantile + (1 - self.alpha) * ema_vals
        scale = torch.clip(ema_vals[1] - ema_vals[0], min=1.0)
        offset = ema_vals[0]
        return offset.detach(), scale.detach()


class WorldModel(nn.Module):
    def __init__(self, obs_space, act_space, step, config):
        super(WorldModel, self).__init__()
        self._step = step
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
        self.encoder = networks.MultiEncoder(shapes, **config.encoder)
        self.embed_size = self.encoder.outdim
        # [CoDreamer] 边特征维度由环境声明（`edge_features` 的最后一维）。
        # 没开图时就为 0，RSSM 里不建边投影。
        edge_dim = shapes.get("edge_features", (0,))[-1]
        # [CoDreamer] 留给 ImagBehavior：actor/critic 的 GNN 需要同样的边维度
        self.edge_dim = edge_dim
        self.dynamics = networks.RSSM(
            config.dyn_stoch,
            config.dyn_deter,
            config.dyn_hidden,
            config.dyn_rec_depth,
            config.dyn_discrete,
            config.act,
            config.norm,
            config.dyn_mean_act,
            config.dyn_std_act,
            config.dyn_min_std,
            config.unimix_ratio,
            config.initial,
            config.num_actions,
            self.embed_size,
            config.device,
            # [CoDreamer] 论文 Table 2：RSSM GNN Layers = 1。
            # 0 层时 GraphAggregator 是严格恒等，整条链路退化成 IDreamer。
            gnn_layers=config.gnn_layers,
            gnn_edge_dim=edge_dim,
            gnn_heads=config.gnn_heads,
        )
        self.heads = nn.ModuleDict()
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        self.heads["decoder"] = networks.MultiDecoder(
            feat_size, shapes, **config.decoder
        )
        self.heads["reward"] = networks.MLP(
            feat_size,
            (255,) if config.reward_head["dist"] == "symlog_disc" else (),
            config.reward_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist=config.reward_head["dist"],
            outscale=config.reward_head["outscale"],
            device=config.device,
            name="Reward",
            # [CoDreamer] 论文 Table 2：Reward & Cont GNN Layers = 1
            gnn_layers=config.gnn_head_layers,
            gnn_edge_dim=edge_dim,
            gnn_heads=config.gnn_heads,
        )
        self.heads["cont"] = networks.MLP(
            feat_size,
            (),
            config.cont_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist="binary",
            outscale=config.cont_head["outscale"],
            device=config.device,
            name="Cont",
            gnn_layers=config.gnn_head_layers,
            gnn_edge_dim=edge_dim,
            gnn_heads=config.gnn_heads,
        )
        # [CoDreamer] 论文："only the reward and terminal state prediction heads use
        # communication, **while the decoder head remains independent**"。
        # decoder 重建的是自己的观测，本来就是局部信息，所以不插 GNN。
        self._comm_heads = ("reward", "cont")
        for name in config.grad_heads:
            assert name in self.heads, name
        self._model_opt = tools.Optimizer(
            "model",
            self.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            config.weight_decay,
            opt=config.opt,
            use_amp=self._use_amp,
        )
        print(
            f"Optimizer model_opt has {sum(param.numel() for param in self.parameters())} variables."
        )
        # other losses are scaled by 1.0.
        self._scales = dict(
            reward=config.reward_head["loss_scale"],
            cont=config.cont_head["loss_scale"],
        )

    def _train(self, data, update=True):
        # action (batch_size, batch_length, act_dim)
        # image (batch_size, batch_length, h, w, ch)
        # reward (batch_size, batch_length)
        # discount (batch_size, batch_length)
        #
        # [CoDreamer] `update=False` 时只做前向、不跑优化器（供 Phase 2 的
        # "冻结世界模型"设定使用）。前向仍然必须做：actor-critic 的想象需要
        # post 作为起点，而且预测损失还要记录下来用于诊断。
        #
        # [CoDreamer] 多智能体时 data 是 (B, T, N, ·)：agent 轴是**样本内部**的一维，
        # 因为 GNN 要在同一时刻对 N 个 agent 做邻域聚合。
        #
        # 整条链路**不需要折平**：encoder / MLP 头 / decoder / RSSM 的 GRUCell 全部
        # 只对最后一维做操作，天然支持任意前置维度。折平反而会让 agent 彼此不可见。
        # （早期版本曾在这里折平，已废弃；fold_agent_axis 现在只给 actor-critic 用。）
        data = self.preprocess(data)
        # [CoDreamer] 图拓扑走旁路：不进 encoder（见 configs.yaml 的 mlp_keys），
        # 只喂给 RSSM 的 GNN。没有这个 key 就传 None → IDreamer 路径。
        graph = graph_from_data(data, self._config.device)
        with tools.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                embed = self.encoder(data)
                post, prior = self.dynamics.observe(
                    embed, data["action"], data["is_first"], graph=graph
                )
                kl_free = self._config.kl_free
                dyn_scale = self._config.dyn_scale
                rep_scale = self._config.rep_scale
                kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                    post, prior, kl_free, dyn_scale, rep_scale
                )
                # [CoDreamer] 原来写死 embed.shape[:2]；多智能体下 loss 比 embed 多
                # 一维（agent 轴），改成按自身维数对齐。
                assert kl_loss.shape == embed.shape[: kl_loss.dim()], kl_loss.shape
                preds = {}
                for name, head in self.heads.items():
                    grad_head = name in self._config.grad_heads
                    feat = self.dynamics.get_feat(post)
                    feat = feat if grad_head else feat.detach()
                    # [CoDreamer] 只有 reward / cont 头接收图（论文明确 decoder 独立）。
                    # graph 为 None（未开图）或无 GNN 层时这里是恒等。
                    if name in self._comm_heads:
                        pred = head(feat, graph=graph)
                    else:
                        pred = head(feat)
                    if type(pred) is dict:
                        preds.update(pred)
                    else:
                        preds[name] = pred
                losses = {}
                for name, pred in preds.items():
                    loss = -pred.log_prob(data[name])
                    # [CoDreamer] 同上：多智能体时比 embed 多一维 agent 轴
                    assert loss.shape == embed.shape[: loss.dim()], (name, loss.shape)
                    losses[name] = loss
                scaled = {
                    key: value * self._scales.get(key, 1.0)
                    for key, value in losses.items()
                }
                model_loss = sum(scaled.values()) + kl_loss
            if update:
                metrics = self._model_opt(torch.mean(model_loss), self.parameters())
            else:
                # [CoDreamer] 冻结：只记损失，不更新参数。用 detach 避免白算梯度。
                metrics = {
                    "model_loss": to_np(torch.mean(model_loss).detach()),
                    "model_grad_norm": 0.0,
                }

        # Store scalar metrics to avoid keeping (batch,time) arrays until the next log step.
        metrics.update(
            {f"{name}_loss": to_np(torch.mean(loss)) for name, loss in losses.items()}
        )
        metrics["kl_free"] = kl_free
        metrics["dyn_scale"] = dyn_scale
        metrics["rep_scale"] = rep_scale
        metrics["dyn_loss"] = to_np(torch.mean(dyn_loss))
        metrics["rep_loss"] = to_np(torch.mean(rep_loss))
        metrics["kl"] = to_np(torch.mean(kl_value))
        with torch.cuda.amp.autocast(self._use_amp):
            metrics["prior_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(prior).entropy())
            )
            metrics["post_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(post).entropy())
            )
            context = dict(
                embed=embed,
                feat=self.dynamics.get_feat(post),
                kl=kl_value,
                postent=self.dynamics.get_dist(post).entropy(),
            )
        post = {k: v.detach() for k, v in post.items()}
        return post, context, metrics

    # this function is called during both rollout and training
    def preprocess(self, obs):
        # [CoDreamer] 注意：图批次 (B,T,N,·) 的折平**不在这里**做。
        # `_policy` 也调 preprocess，而它拿到的是 simulate_marl 铺平的 (B*N, ·)，
        # 那里没有 agent 轴。折平只属于 _train（见那里的 fold_agent_axis）。
        obs = {
            k: torch.tensor(v, device=self._config.device, dtype=torch.float32)
            for k, v in obs.items()
        }
        # [CoDreamer] 上游无条件做 obs["image"] /= 255；向量观测环境（如我们的
        # Sequential Estimate Game）根本不存在 "image" 这个 key，会 KeyError。
        # 改为条件执行。登记于 docs/reproduction-plan.md「改动清单」。
        # 保留原行为：有 image 就照旧归一化。
        if "image" in obs:
            obs["image"] = obs["image"] / 255.0
        if "discount" in obs:
            obs["discount"] *= self._config.discount
            # (batch_size, batch_length) -> (batch_size, batch_length, 1)
            obs["discount"] = obs["discount"].unsqueeze(-1)
        # 'is_first' is necesarry to initialize hidden state at training
        assert "is_first" in obs
        # 'is_terminal' is necesarry to train cont_head
        assert "is_terminal" in obs
        obs["cont"] = (1.0 - obs["is_terminal"]).unsqueeze(-1)
        return obs

    def video_pred(self, data):
        data = self.preprocess(data)
        embed = self.encoder(data)

        states, _ = self.dynamics.observe(
            embed[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5]
        )
        recon = self.heads["decoder"](self.dynamics.get_feat(states))["image"].mode()[
            :6
        ]
        reward_post = self.heads["reward"](self.dynamics.get_feat(states)).mode()[:6]
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.dynamics.imagine_with_action(data["action"][:6, 5:], init)
        openl = self.heads["decoder"](self.dynamics.get_feat(prior))["image"].mode()
        reward_prior = self.heads["reward"](self.dynamics.get_feat(prior)).mode()
        # observed image is given until 5 steps
        model = torch.cat([recon[:, :5], openl], 1)
        truth = data["image"][:6]
        model = model
        error = (model - truth + 1.0) / 2.0

        return torch.cat([truth, model, error], 2)


class ImagBehavior(nn.Module):
    def __init__(self, config, world_model):
        super(ImagBehavior, self).__init__()
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        self._world_model = world_model
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        self.actor = networks.MLP(
            feat_size,
            (config.num_actions,),
            config.actor["layers"],
            config.units,
            config.act,
            config.norm,
            config.actor["dist"],
            config.actor["std"],
            config.actor["min_std"],
            config.actor["max_std"],
            absmax=1.0,
            temp=config.actor["temp"],
            unimix_ratio=config.actor["unimix_ratio"],
            outscale=config.actor["outscale"],
            name="Actor",
            # [CoDreamer] 论文 Table 3：Actor-Critic GNN Layers = 1。
            # 两个网络各自一层，与 reward/cont 头同样的位置（作用在 units 维隐层上）。
            gnn_layers=config.gnn_behavior_layers,
            gnn_edge_dim=getattr(world_model, "edge_dim", 0),
            gnn_heads=config.gnn_heads,
        )
        self.value = networks.MLP(
            feat_size,
            (255,) if config.critic["dist"] == "symlog_disc" else (),
            config.critic["layers"],
            config.units,
            config.act,
            config.norm,
            config.critic["dist"],
            outscale=config.critic["outscale"],
            device=config.device,
            name="Value",
            gnn_layers=config.gnn_behavior_layers,
            gnn_edge_dim=getattr(world_model, "edge_dim", 0),
            gnn_heads=config.gnn_heads,
        )
        if config.critic["slow_target"]:
            self._slow_value = copy.deepcopy(self.value)
            self._updates = 0
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self._actor_opt = tools.Optimizer(
            "actor",
            self.actor.parameters(),
            config.actor["lr"],
            config.actor["eps"],
            config.actor["grad_clip"],
            **kw,
        )
        print(
            f"Optimizer actor_opt has {sum(param.numel() for param in self.actor.parameters())} variables."
        )
        self._value_opt = tools.Optimizer(
            "value",
            self.value.parameters(),
            config.critic["lr"],
            config.critic["eps"],
            config.critic["grad_clip"],
            **kw,
        )
        print(
            f"Optimizer value_opt has {sum(param.numel() for param in self.value.parameters())} variables."
        )
        if self._config.reward_EMA:
            # register ema_vals to nn.Module for enabling torch.save and torch.load
            self.register_buffer(
                "ema_vals", torch.zeros((2,), device=self._config.device)
            )
            self.reward_ema = RewardEMA(device=self._config.device)

    def _train(
        self,
        start,
        objective,
        graph=None,
    ):
        """
        graph: None 或 (adjacency, edge_features)，形状 (B*T, N, N) / (B*T, N, N, F)。
               **已经扁平化**（与 start 同形），且是**想象轨迹起点那张图**，
               整段轨迹都用它 —— 论文："CoDreamer does not predict new adjacency
               matrices over the imagined trajectories, the adjacency matrix of the
               starting graph is used for all subsequent graphs"。
        """
        self._update_slow_target()
        metrics = {}

        with tools.RequiresGrad(self.actor):
            with torch.cuda.amp.autocast(self._use_amp):
                imag_feat, imag_state, imag_action = self._imagine(
                    start, self.actor, self._config.imag_horizon, graph=graph
                )
                reward = objective(imag_feat, imag_state, imag_action)
                actor_ent = self.actor(imag_feat, graph=graph).entropy()
                state_ent = self._world_model.dynamics.get_dist(imag_state).entropy()
                # this target is not scaled by ema or sym_log.
                target, weights, base = self._compute_target(
                    imag_feat, imag_state, reward, graph=graph
                )
                actor_loss, mets = self._compute_actor_loss(
                    imag_feat,
                    imag_action,
                    target,
                    weights,
                    base,
                    graph=graph,
                )
                actor_loss -= self._config.actor["entropy"] * actor_ent[:-1, ..., None]
                actor_loss = torch.mean(actor_loss)
                metrics.update(mets)
                value_input = imag_feat

        with tools.RequiresGrad(self.value):
            with torch.cuda.amp.autocast(self._use_amp):
                value = self.value(value_input[:-1].detach(), graph=graph)
                # [CoDreamer] 上游这里还会再 `target = torch.stack(target, dim=1)`：
                # 旧的 lambda_return 返回元组，而 `_compute_actor_loss` 里的 stack
                # 是**局部变量**，所以外层这个 target 仍是元组，需要再展开一次。
                # 现在 lambda_return 直接返回张量 (time, *batch, 1)，两处 stack 都去掉。
                # （漏掉这一处会把 ImagBehavior 直接跑崩；
                #  test_graph_pipeline 只跑 WorldModel._train，盖不到这条路径，
                #  是靠真实训练发现的。）
                # (time, batch, 1), (time, batch, 1) -> (time, batch)
                value_loss = -value.log_prob(target.detach())
                slow_target = self._slow_value(value_input[:-1].detach(), graph=graph)
                if self._config.critic["slow_target"]:
                    value_loss -= value.log_prob(slow_target.mode().detach())
                # (time, batch, 1), (time, batch, 1) -> (1,)
                # [CoDreamer] 同上：`[:, :, None]` 写死两维，改为轴无关写法
                value_loss = torch.mean(weights[:-1] * value_loss[..., None])

        metrics.update(tools.tensorstats(value.mode(), "value"))
        metrics.update(tools.tensorstats(target, "target"))
        metrics.update(tools.tensorstats(reward, "imag_reward"))
        if self._config.actor["dist"] in ["onehot"]:
            metrics.update(
                tools.tensorstats(
                    torch.argmax(imag_action, dim=-1).float(), "imag_action"
                )
            )
        else:
            metrics.update(tools.tensorstats(imag_action, "imag_action"))
        metrics["actor_entropy"] = to_np(torch.mean(actor_ent))
        with tools.RequiresGrad(self):
            metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
            metrics.update(self._value_opt(value_loss, self.value.parameters()))
        return imag_feat, imag_state, imag_action, weights, metrics

    def _imagine(self, start, policy, horizon, graph=None):
        dynamics = self._world_model.dynamics
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        # start 是 (B, T, N, ·)；(B,T) 相邻，合并成轨迹维后是 (B*T, N, ·)。
        # 图由调用方用 models.flatten_graph 做过同样的扁平化，两者对齐。
        start = {k: flatten(v) for k, v in start.items()}

        def step(prev, _):
            state, _, _ = prev
            feat = dynamics.get_feat(state)
            inp = feat.detach()
            # [CoDreamer] ★ 想象阶段的两个通信点，都用**同一张冻结的起点图** ★
            #
            # ① actor 自己的 GNN（论文 Table 3：Actor-Critic GNN Layers = 1）
            # ② RSSM 的 GNN。这一条尤其重要：世界模型是**带着图**训练的，若想象时
            #    不喂图，它看到的隐状态就处于分布外，整条想象轨迹系统性偏差。
            #    （缺了它时，CoDreamer 的 Phase 2 会完全涨不动 —— 因为它的世界模型
            #     学到的东西在想象里被抽掉了。）
            action = policy(inp, graph=graph).sample()
            succ = dynamics.img_step(state, action, graph=graph)
            return succ, feat, action

        succ, feats, actions = tools.static_scan(
            step, [torch.arange(horizon)], (start, None, None)
        )
        states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}

        return feats, states, actions

    def _compute_target(self, imag_feat, imag_state, reward, graph=None):
        if "cont" in self._world_model.heads:
            inp = self._world_model.dynamics.get_feat(imag_state)
            discount = (
                self._config.discount
                * self._world_model.heads["cont"](inp, graph=graph).mean
            )
        else:
            discount = self._config.discount * torch.ones_like(reward)
        value = self.value(imag_feat, graph=graph).mode()
        # [CoDreamer] ★ 自举项用哪个 critic ★
        #
        # 上游这里把**在线** critic 同时用作 λ-return 的中间混合项与自举项：
        #
        #     target = lambda_return(reward[1:], value[:-1], disc, bootstrap=value[-1])
        #
        # 而被训练的就是 `self.value`，于是目标里含着它自己。慢目标 `_slow_value`
        # 确实存在、也确实在做 Polyak 更新，但**只用在了损失的正则项里**
        # （见 `_train` 的 `value_loss -= value.log_prob(slow_target...)`），
        # 从不参与目标计算 —— 低通滤波器不是锚。
        #
        # 后果在 2026-10-15 实测到过：target_mean 与 value_mean 逐点相等、
        # V 与真实折扣回报的相关系数只有 0.03（重测 0.13），价值单调膨胀到
        # 每步奖励的 ~1000 倍且不收敛，而 actor 的 advantage 随之归零。
        #
        # `value_bootstrap: 'slow'` 把自举换成慢目标，用来判定上述诊断是否成立。
        # 默认 'online' = 与上游逐位一致，不改任何既有结果。
        if getattr(self._config, "value_bootstrap", "online") == "slow":
            bootstrap = self._slow_value(imag_feat, graph=graph).mode()
        else:
            bootstrap = value
        target = tools.lambda_return(
            reward[1:],
            bootstrap[:-1],
            discount[1:],
            bootstrap=bootstrap[-1],
            lambda_=self._config.discount_lambda,
            axis=0,
        )
        weights = torch.cumprod(
            torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
        ).detach()
        return target, weights, value[:-1]

    def _compute_actor_loss(
        self,
        imag_feat,
        imag_action,
        target,
        weights,
        base,
        graph=None,
    ):
        metrics = {}
        inp = imag_feat.detach()
        policy = self.actor(inp, graph=graph)
        # Q-val for actor is not transformed using symlog
        # [CoDreamer] 上游这里是 `target = torch.stack(target, dim=1)`：因为旧的
        # lambda_return 返回的是沿 batch 维 unbind 出的元组。现在它直接返回带时间轴
        # 的张量 (time, *batch, 1)，不再需要 stack。
        if self._config.reward_EMA:
            offset, scale = self.reward_ema(target, self.ema_vals)
            normed_target = (target - offset) / scale
            normed_base = (base - offset) / scale
            adv = normed_target - normed_base
            metrics.update(tools.tensorstats(normed_target, "normed_target"))
            metrics["EMA_005"] = to_np(self.ema_vals[0])
            metrics["EMA_095"] = to_np(self.ema_vals[1])
        else:
            adv = target - base

        if self._config.imag_gradient == "dynamics":
            actor_target = adv
        elif self._config.imag_gradient == "reinforce":
            # [CoDreamer] `[:, :, None]` 写死两维；带 agent 轴时会把轴插错位置
            # （(T,B,N) -> (T,B,1,N)）。`[..., None]` 对三维输入完全等价。
            actor_target = (
                policy.log_prob(imag_action)[:-1][..., None]
                * (target - self.value(imag_feat[:-1], graph=graph).mode()).detach()
            )
        elif self._config.imag_gradient == "both":
            actor_target = (
                policy.log_prob(imag_action)[:-1][..., None]
                * (target - self.value(imag_feat[:-1], graph=graph).mode()).detach()
            )
            mix = self._config.imag_gradient_mix
            actor_target = mix * target + (1 - mix) * actor_target
            metrics["imag_gradient_mix"] = mix
        else:
            raise NotImplementedError(self._config.imag_gradient)
        actor_loss = -weights[:-1] * actor_target
        return actor_loss, metrics

    def _update_slow_target(self):
        if self._config.critic["slow_target"]:
            if self._updates % self._config.critic["slow_target_update"] == 0:
                mix = self._config.critic["slow_target_fraction"]
                for s, d in zip(self.value.parameters(), self._slow_value.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1
