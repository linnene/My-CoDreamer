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

    def _train(self, data):
        # action (batch_size, batch_length, act_dim)
        # image (batch_size, batch_length, h, w, ch)
        # reward (batch_size, batch_length)
        # discount (batch_size, batch_length)
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
            metrics = self._model_opt(torch.mean(model_loss), self.parameters())

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
    ):
        self._update_slow_target()
        metrics = {}

        with tools.RequiresGrad(self.actor):
            with torch.cuda.amp.autocast(self._use_amp):
                imag_feat, imag_state, imag_action = self._imagine(
                    start, self.actor, self._config.imag_horizon
                )
                reward = objective(imag_feat, imag_state, imag_action)
                actor_ent = self.actor(imag_feat).entropy()
                state_ent = self._world_model.dynamics.get_dist(imag_state).entropy()
                # this target is not scaled by ema or sym_log.
                target, weights, base = self._compute_target(
                    imag_feat, imag_state, reward
                )
                actor_loss, mets = self._compute_actor_loss(
                    imag_feat,
                    imag_action,
                    target,
                    weights,
                    base,
                )
                actor_loss -= self._config.actor["entropy"] * actor_ent[:-1, ..., None]
                actor_loss = torch.mean(actor_loss)
                metrics.update(mets)
                value_input = imag_feat

        with tools.RequiresGrad(self.value):
            with torch.cuda.amp.autocast(self._use_amp):
                value = self.value(value_input[:-1].detach())
                # [CoDreamer] 上游这里还会再 `target = torch.stack(target, dim=1)`：
                # 旧的 lambda_return 返回元组，而 `_compute_actor_loss` 里的 stack
                # 是**局部变量**，所以外层这个 target 仍是元组，需要再展开一次。
                # 现在 lambda_return 直接返回张量 (time, *batch, 1)，两处 stack 都去掉。
                # （漏掉这一处会把 ImagBehavior 直接跑崩；
                #  test_graph_pipeline 只跑 WorldModel._train，盖不到这条路径，
                #  是靠真实训练发现的。）
                # (time, batch, 1), (time, batch, 1) -> (time, batch)
                value_loss = -value.log_prob(target.detach())
                slow_target = self._slow_value(value_input[:-1].detach())
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

    def _imagine(self, start, policy, horizon):
        dynamics = self._world_model.dynamics
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start.items()}
        if self._config.num_agents > 1:
            # [CoDreamer] ★ 步骤 3 的临时状态 ★
            #
            # start 现在是 (B*T, N, ·)：(B,T) 已合成轨迹维，agent 轴保留。
            # 但 actor-critic 还没接自己的 GNN（论文 Table 3 的 GNN Layers=1），
            # 而 upstream 的 lambda_return 末尾硬编码了
            #   reshape([shape[0], shape[1], 1])
            # 对形状有强假设，同时带 agent 轴会很难改动。
            #
            # 所以这里把 agent 轴**也**折进 batch，让 AC 完全走 upstream 的路径
            # （等价于 IDreamer 的 AC）。世界模型的 GNN 仍然生效 —— 它作用在
            # img_step 里，不受这个折平影响。
            #
            # 步骤 5 会：① 去掉这次折平；② 给 actor/critic 加 GNN；
            # ③ 把**冻结的起点图**喂给想象轨迹（论文要求，见 networks_imagine）。
            start = {k: flatten(v) for k, v in start.items()}

        def step(prev, _):
            state, _, _ = prev
            feat = dynamics.get_feat(state)
            inp = feat.detach()
            action = policy(inp).sample()
            succ = dynamics.img_step(state, action)
            return succ, feat, action

        succ, feats, actions = tools.static_scan(
            step, [torch.arange(horizon)], (start, None, None)
        )
        states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}

        return feats, states, actions

    def _compute_target(self, imag_feat, imag_state, reward):
        if "cont" in self._world_model.heads:
            inp = self._world_model.dynamics.get_feat(imag_state)
            discount = self._config.discount * self._world_model.heads["cont"](inp).mean
        else:
            discount = self._config.discount * torch.ones_like(reward)
        value = self.value(imag_feat).mode()
        target = tools.lambda_return(
            reward[1:],
            value[:-1],
            discount[1:],
            bootstrap=value[-1],
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
    ):
        metrics = {}
        inp = imag_feat.detach()
        policy = self.actor(inp)
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
                * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
        elif self._config.imag_gradient == "both":
            actor_target = (
                policy.log_prob(imag_action)[:-1][..., None]
                * (target - self.value(imag_feat[:-1]).mode()).detach()
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
