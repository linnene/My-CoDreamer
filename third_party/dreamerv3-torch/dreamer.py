import argparse
import functools
import os
import pathlib
import sys

# [CoDreamer] 上游无条件写死 osmesa；Windows 上没有 OSMesa，dm_control 会 ImportError。
# [CoDreamer] 改为默认值不变、允许外部覆盖（远端 Windows 用 MUJOCO_GL=glfw）。
# [CoDreamer] 登记于 docs/reproduction-plan.md「改动清单」。
os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np
import ruamel.yaml as yaml

sys.path.append(str(pathlib.Path(__file__).parent))

import exploration as expl
import models
import tools
import tools_marl
import envs.wrappers as wrappers
from parallel import Parallel, Damy

import torch
from torch import nn
from torch import distributions as torchd


to_np = lambda x: x.detach().cpu().numpy()


class Dreamer(nn.Module):
    def __init__(self, obs_space, act_space, config, logger, dataset):
        super(Dreamer, self).__init__()
        self._config = config
        self._logger = logger
        self._should_log = tools.Every(config.log_every)
        batch_steps = config.batch_size * config.batch_length
        self._should_train = tools.Every(batch_steps / config.train_ratio)
        self._should_pretrain = tools.Once()
        self._should_reset = tools.Every(config.reset_every)
        self._should_expl = tools.Until(int(config.expl_until / config.action_repeat))
        self._metrics = {}
        # this is update step
        self._step = logger.step // config.action_repeat
        self._update_count = 0
        self._dataset = dataset
        # [CoDreamer] Phase 2 的变体 A：世界模型冻结，只有策略在想象里训。
        # 由 main() 在穿过 env 切断点时置为 True（见 Dreamer.set_world_model_frozen）。
        self._world_model_frozen = False
        self._wm = models.WorldModel(obs_space, act_space, self._step, config)
        self._task_behavior = models.ImagBehavior(config, self._wm)
        if (
            config.compile and os.name != "nt"
        ):  # compilation is not supported on windows
            self._wm = torch.compile(self._wm)
            self._task_behavior = torch.compile(self._task_behavior)
        reward = lambda f, s, a: self._wm.heads["reward"](f).mean()
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: expl.Random(config, act_space),
            plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
        )[config.expl_behavior]().to(self._config.device)

    def __call__(self, obs, reset, state=None, training=True, collect_policy=True):
        step = self._step
        if training:
            pretraining = bool(self._should_pretrain())
            steps = (
                self._config.pretrain
                if pretraining
                else self._should_train(step)
            )
            for _ in range(steps):
                # [CoDreamer] 预训练阶段只更新世界模型（见 _train 的说明）。
                #
                # ⚠️ 这里曾经写成 `not (pretraining or self._world_model_frozen)`。
                # 预训练时两个开关于是**同时**为 False —— 那 500 步变成 500 次不更新
                # 任何参数的前向，日志却照旧打印"预训练完成"。冻结世界模型是 Phase 2
                # 的独立开关，与预训练毫无关系，不该混进同一个表达式。
                # 由 scripts/test_training_loop.py 盯死。
                self._train(
                    next(self._dataset),
                    update_behavior=not pretraining,
                    update_world_model=not self._world_model_frozen,
                )
                self._update_count += 1
                self._metrics["update_count"] = self._update_count
            if self._should_log(step):
                for name, values in self._metrics.items():
                    self._logger.scalar(name, float(np.mean(values)))
                    self._metrics[name] = []
                if self._config.video_pred_log:
                    openl = self._wm.video_pred(next(self._dataset))
                    self._logger.video("train_openl", to_np(openl))
                self._logger.write(fps=True)

        if collect_policy:
            policy_output, state = self._policy(obs, state, training)
        else:
            # [CoDreamer] Phase 2（想象阶段）：环境已切断，不需要动作。
            #
            # 为什么仍然要调 __call__：训练节奏由 self._step 与 _should_train 驱动，
            # 而 _step 只在这里按 batch 推进（`self._step += len(reset)`）。若绕开它
            # 直接调 _train，梯度更新频率就会变，Phase 2 与 Phase 1 不再可比。
            # 这里只是把策略前向跳掉，训练部分原样保留。
            policy_output = {}

        if training:
            self._step += len(reset)
            self._logger.step = self._config.action_repeat * self._step
        return policy_output, state

    def _policy(self, obs, state, training):
        # [CoDreamer] ★ 执行路径的 agent 轴 ★
        #
        # 训练时数据是 (B, T, N, ·)，agent 轴显式存在；但 simulate_marl 把每个队伍
        # 铺平成 (n_teams*N, ·) 之后才交给策略，agent 轴就**没有了**。
        # GNN 必须在同一时刻对 N 个 agent 做邻域聚合，没有这个轴，它会把 batch 维
        # 当成邻居维 —— 实测报错是 `agg.reshape(*node.shape[:-1], ...)` 拿到 [4]。
        #
        # 所以这里把布局还原成 (n_teams, N, ·)，算完再铺平回去。这同时解决了两个
        # 形状问题：图是 (batch, N)（一堆行）而不是 (batch, N, N)，隐状态也没有
        # agent 轴。两个问题同源：**执行路径缺的是 agent 轴**。
        #
        # n_agents == 1 时不动任何形状 —— 单智能体路径必须与上游逐位一致。
        n_agents = max(1, int(getattr(self._config, "num_agents", 1)))
        reshape = n_agents > 1

        def add_agent_axis(x):
            return x.reshape((-1, n_agents) + tuple(x.shape[1:]))

        def drop_agent_axis(x):
            return x.reshape((-1,) + tuple(x.shape[2:]))

        if state is None:
            latent = action = None
        else:
            latent, action = state
            if reshape:
                latent = {k: add_agent_axis(v) for k, v in latent.items()}
                action = add_agent_axis(action)
        obs = self._wm.preprocess(obs)
        if reshape:
            obs = {k: add_agent_axis(v) for k, v in obs.items()}
        embed = self._wm.encoder(obs)
        # [CoDreamer] ★ 执行路径也必须拿到图 ★
        #
        # 曾经只有训练路径（_train -> _imagine）传图，_policy 这里一个字都没提。
        # 后果是：收集数据用的动作来自一个看不见图的 actor，而世界模型是带着图训练
        # 的；部署评测也评的是看不见图的策略。CoDreamer 于是从未在真实执行中使用通信。
        # 由 scripts/test_training_loop.py 盯死。
        graph = models.execution_graph(obs, n_agents, self._config.device)
        latent, _ = self._wm.dynamics.obs_step(
            latent, action, embed, obs["is_first"], graph=graph
        )
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = self._wm.dynamics.get_feat(latent)
        if not training:
            actor = self._task_behavior.actor(feat, graph=graph)
            action = actor.mode()
        elif self._should_expl(self._step):
            # 探索策略（Random / Plan2Explore）与任务策略不是同一个网络，它们的
            # actor 不接受 graph 参数。`greedy` 时 _expl_behavior 就是 _task_behavior，
            # 必须走带图的那一条，否则同一个网络会有两种输入分布。
            if self._config.expl_behavior == "greedy":
                actor = self._task_behavior.actor(feat, graph=graph)
            else:
                actor = self._expl_behavior.actor(feat)
            action = actor.sample()
        else:
            actor = self._task_behavior.actor(feat, graph=graph)
            action = actor.sample()
        logprob = actor.log_prob(action)
        if reshape:
            # state 是跨步传递的，必须回到与传入时相同的铺平布局
            latent = {k: drop_agent_axis(v) for k, v in latent.items()}
            action = drop_agent_axis(action)
            logprob = drop_agent_axis(logprob)
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()
        if self._config.actor["dist"] == "onehot_gumble":
            action = torch.one_hot(
                torch.argmax(action, dim=-1), self._config.num_actions
            )
        policy_output = {"action": action, "logprob": logprob}
        state = (latent, action)
        return policy_output, state

    def set_world_model_frozen(self, frozen: bool) -> None:
        """[CoDreamer] 冻结/解冻世界模型（只影响是否做梯度更新，不影响前向）。

        Phase 2 用它区分两种设定：
            冻结  —— 测"这个世界模型够不够好"
            不冻  —— 允许它继续在**冻结的 replay** 上收敛（仍然不接触环境）
        """
        self._world_model_frozen = bool(frozen)

    def _train(self, data, update_behavior=True, update_world_model=True):
        metrics = {}
        post, context, mets = self._wm._train(data, update=update_world_model)
        metrics.update(mets)
        # [CoDreamer] 预训练阶段跳过 actor-critic 与探索策略。
        #
        # 论文（附录 D.3）：
        #   "we perform 500 training steps of the world model **before any
        #    behaviour is learnt**. This form of pre-training is performed in
        #    order to avoid negative performance effects due to a primacy bias"
        #
        # 上游这里无条件训 actor-critic，于是 config.pretrain 只相当于
        # "开局先连做 N 次训练"，而不是论文说的"策略晚 N 步再开始学"。
        # 两者的区别在于：预训练期间世界模型还不准，它想象的轨迹是假的，
        # 策略在这堆假数据上训就会被带偏（primacy bias）。
        #
        # update_behavior=False 时只有世界模型在更新 → 与论文一致。
        # 登记于 docs/reproduction-plan.md「改动清单」。
        if update_behavior:
            start = post
            # [CoDreamer] 取出图并扁平化。`_imagine` 会把隐状态的 (B,T) 合并成轨迹维，
            # 所以图的 (B,T) 必须做同样处理，两者才对得上。
            # 同一份扁平图既给 reward 头（它也有 GNN），也给 ImagBehavior。
            graph = models.flatten_graph(
                models.graph_from_data(data, self._config.device)
            )
            reward = lambda f, s, a: self._wm.heads["reward"](
                self._wm.dynamics.get_feat(s), graph=graph
            ).mode()
            metrics.update(self._task_behavior._train(start, reward, graph=graph)[-1])
            if self._config.expl_behavior != "greedy":
                mets = self._expl_behavior.train(start, context, data)[-1]
                metrics.update({"expl_" + key: value for key, value in mets.items()})
        for name, value in metrics.items():
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)


def _log_openloop(wm, config, logger):
    """[CoDreamer] 用 evaldir 里的 held-out 局算多步开环误差，写进训练日志。

    为什么必须进日志、而不是只看事后脚本：事后只能看几个 checkpoint，
    而"世界模型的预测能力随训练怎么变"是一条**曲线**，和回报曲线要能并排看。

    数据是刚才那次评测写进 evaldir 的 —— 由**当前策略**现跑，从不参与任何
    梯度更新（见 tools_marl.simulate_marl 里 is_eval 分支的说明）。

    失败一律降级为跳过 + 打印，绝不让诊断指标把训练搞挂。
    """
    spec = config.openloop_horizons
    if not spec:
        return
    if isinstance(spec, str):
        horizons = [int(x) for x in spec.replace(",", " ").split() if x]
    else:
        horizons = [int(x) for x in spec]
    if not horizons:
        return
    if config.batch_length < max(horizons) + 2:
        print(
            f"[CoDreamer] 跳过开环误差：batch_length={config.batch_length} 装不下 "
            f"horizon={max(horizons)}（需要 >= {max(horizons) + 2}）"
        )
        return

    eps = tools.load_episodes(config.evaldir, limit=config.dataset_size)
    if not eps:
        return
    try:
        data = next(tools_marl.make_graph_dataset(eps, config))
        errs = models.openloop_error(wm, data, horizons)
    except (ValueError, KeyError) as exc:
        # ValueError：队伍不完整 / 序列太短 —— 刚开始训练时很正常
        print(f"[CoDreamer] 跳过开环误差：{exc}")
        return
    for h, metrics in errs.items():
        for key, value in metrics.items():
            logger.scalar(f"openloop_h{h}_{key}", value)
    print(
        "[CoDreamer] held-out 开环误差 "
        + "  ".join(
            f"h{h}: rew={m['reward_mae']:.3f} cont={m['cont_mae']:.3f} "
            f"kl={m['kl']:.3f}"
            for h, m in sorted(errs.items())
        )
    )


def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def make_dataset(episodes, config):
    generator = tools.sample_episodes(episodes, config.batch_length)
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset


def make_env(config, mode, id):
    suite, task = config.task.split("_", 1)
    # [CoDreamer] 多智能体环境：队伍级，自己处理 id / 时间限制 / 动作选择，
    # 因此直接返回，不接下面那条 gym wrapper 链（TimeLimit / SelectAction / UUID
    # 都需要 env 是 gym.Env）。收集由 tools_marl.simulate_marl 负责。
    if suite == "seg":
        import envs.seg_gym as seg_gym
        from envs.seg import Spec

        # [CoDreamer] 环境规格由 config 驱动，便于做横向对比实验（见
        # docs/reproduction-plan.md）。默认值就是论文 Table 1。
        spec = Spec(
            num_agents=config.seg_num_agents,
            horizon=config.seg_horizon,
            num_actions=config.seg_num_actions,
            edge_density=config.seg_edge_density,
            local_state_ratio=config.seg_local_state_ratio,
            target_transform=config.seg_target_transform,
        )
        return seg_gym.SequentialEstimateGameEnv(
            spec,
            seed=config.seed + id,
            obs_mode=config.seg_obs_mode,
            graph_obs=config.seg_graph_obs,
        )
    if suite == "vmas":
        import envs.vmas_env as vmas_env

        # [CoDreamer] VMAS 也是队伍级环境，同样直接返回。
        #
        # `task` 来自 `config.task.split("_", 1)`，所以 `vmas_buzz_wire` 得到的
        # 是 `buzz_wire`（带下划线也正确）—— 与 VMASEnv.TASKS 的键一致。
        # 场景参数（agent 数、目标数、障碍数）写死在 envs/vmas_env.py 的 TASKS 里，
        # 取值来自论文 Table 8/9/10；只有 C 需要在这里可覆盖，因为论文没给。
        return vmas_env.VMASEnv(
            task=task,
            seed=config.seed + id,
            add_agent_index=config.vmas_add_agent_index,
            graph_obs=config.vmas_graph_obs,
            # <= 0 表示“用任务默认值”（见 TASKS）；正值才覆盖
            comm_radius=(
                None if config.vmas_comm_radius <= 0 else config.vmas_comm_radius
            ),
            max_steps=config.vmas_max_steps,
            device=config.vmas_device,
        )
    if suite == "dmc":
        import envs.dmc as dmc

        env = dmc.DeepMindControl(
            task, config.action_repeat, config.size, seed=config.seed + id
        )
        env = wrappers.NormalizeActions(env)
    elif suite == "atari":
        import envs.atari as atari

        env = atari.Atari(
            task,
            config.action_repeat,
            config.size,
            gray=config.grayscale,
            noops=config.noops,
            lives=config.lives,
            sticky=config.stickey,
            actions=config.actions,
            resize=config.resize,
            seed=config.seed + id,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "dmlab":
        import envs.dmlab as dmlab

        env = dmlab.DeepMindLabyrinth(
            task,
            mode if "train" in mode else "test",
            config.action_repeat,
            seed=config.seed + id,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "memorymaze":
        from envs.memorymaze import MemoryMaze

        env = MemoryMaze(task, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "crafter":
        import envs.crafter as crafter

        env = crafter.Crafter(task, config.size, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "minecraft":
        import envs.minecraft as minecraft

        env = minecraft.make_env(task, size=config.size, break_speed=config.break_speed)
        env = wrappers.OneHotAction(env)
    else:
        raise NotImplementedError(suite)
    env = wrappers.TimeLimit(env, config.time_limit)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    if suite == "minecraft":
        env = wrappers.RewardObs(env)
    return env


def resolve_comm_variant(config):
    """[CoDreamer] 把单一开关 `comm_variant` 展开到三处 GNN 的层数上。

    CoDreamer 与 IDreamer 的差别散布在**三处**独立的 GNN 上（RSSM / 预测头 /
    actor-critic），而它们各自默认关闭。于是 `--gnn_layers 1` 只能得到"半个
    CoDreamer"——README 曾经把它写成"两个变体之间的唯一开关"，是错的。

         idreamer  -> 三处全 0（严格恒等）
         codreamer -> 三处全 1（论文 Table 2/3 全为 1 层）

    单独设 gnn_layers / gnn_head_layers / gnn_behavior_layers 可以覆盖它（做消融
    用）；它们的 yaml 默认值是 -1，意思是"跟随 comm_variant"。本函数**必须**在
    建模型之前调用，否则 checkpoint 里存的 config 是没展开的。

    抽成独立函数是为了让 scripts/test_training_loop.py 能调同一份逻辑，
    而不是在测试里复制一遍（复制出来的东西会和真代码逐渐分家）。
    """
    if config.comm_variant not in ("idreamer", "codreamer"):
        raise SystemExit(
            f"comm_variant 只能是 'idreamer' 或 'codreamer'，收到 {config.comm_variant!r}"
        )
    n = {"idreamer": 0, "codreamer": 1}[config.comm_variant]
    for key in ("gnn_layers", "gnn_head_layers", "gnn_behavior_layers"):
        if getattr(config, key) < 0:
            setattr(config, key, n)
    state = (config.gnn_layers, config.gnn_head_layers, config.gnn_behavior_layers)
    if len(set(state)) != 1:
        print(
            "[CoDreamer][警告] 三处 GNN 层数不一致 "
            f"(rssm={state[0]}, head={state[1]}, behavior={state[2]}) —— "
            "这不是标准的 CoDreamer/IDreamer 对照，请确认这是你想要的消融。"
        )
    print(
        f"[CoDreamer] comm_variant={config.comm_variant}  "
        f"gnn: rssm={config.gnn_layers} head={config.gnn_head_layers} "
        f"behavior={config.gnn_behavior_layers}"
    )
    return config


def main(config):
    tools.set_seed_everywhere(config.seed)
    # [CoDreamer] ★ 单一变体开关 ★ —— 见 resolve_comm_variant 的说明。
    resolve_comm_variant(config)

    if config.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.logdir).expanduser()
    config.traindir = config.traindir or logdir / "train_eps"
    config.evaldir = config.evaldir or logdir / "eval_eps"
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat

    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    step = count_steps(config.traindir)
    # step in logger is environmental step
    logger = tools.Logger(logdir, config.action_repeat * step)

    print("Create envs.")
    if config.offline_traindir:
        directory = config.offline_traindir.format(**vars(config))
    else:
        directory = config.traindir
    train_eps = tools.load_episodes(directory, limit=config.dataset_size)
    if config.offline_evaldir:
        directory = config.offline_evaldir.format(**vars(config))
    else:
        directory = config.evaldir
    eval_eps = tools.load_episodes(directory, limit=1)
    make = lambda mode, id: make_env(config, mode, id)
    train_envs = [make("train", i) for i in range(config.envs)]
    eval_envs = [make("eval", i) for i in range(config.envs)]
    # [CoDreamer] 多智能体：队伍级环境自带 num_agents。
    # 单个 agent 的环境没有这个属性，回退到 1。
    config.num_agents = getattr(train_envs[0], "num_agents", 1)
    multi_agent = config.num_agents > 1
    if multi_agent:
        # 队伍级环境不能套 Damy/Parallel：
        #   - Damy 的 step/reset 返回 lambda（延迟调用），而 simulate_marl 直接要元组
        #   - 一个队伍内部已经含 N 个 agent，再并行会破坏同步推进
        print(f"Multi-agent: num_agents={config.num_agents}, teams={config.envs}")
        # [CoDreamer] 训练强度补偿。
        #
        # Dreamer.__call__ 结尾是 `self._step += len(reset)`，而多智能体下 reset 长度
        # 是 队伍数×agent数，所以 _step 每队伍步增加 num_agents（单智能体时只增加
        # 1/action_repeat）。于是 `_should_train` 会比预期频繁 num_agents*action_repeat 倍。
        #
        # 这里把 interval 放大同样倍数（即把 train_ratio 缩小），使"每个队伍步做多少次
        # 梯度更新"与单智能体设定一致。推导见 tools_marl.py 模块 docstring。
        factor = config.num_agents * config.action_repeat
        before = config.train_ratio
        config.train_ratio = max(1, int(round(config.train_ratio / factor)))
        print(f"train_ratio {before} -> {config.train_ratio} (compensating x{factor})")
        collect = tools_marl.simulate_marl
    else:
        collect = tools.simulate
        if config.parallel:
            train_envs = [Parallel(env, "process") for env in train_envs]
            eval_envs = [Parallel(env, "process") for env in eval_envs]
        else:
            train_envs = [Damy(env) for env in train_envs]
            eval_envs = [Damy(env) for env in eval_envs]
    acts = train_envs[0].action_space
    print("Action Space", acts)
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    state = None
    if not config.offline_traindir:
        prefill = max(0, config.prefill - count_steps(config.traindir))
        print(f"Prefill dataset ({prefill} steps).")
        # [CoDreamer] 多智能体下策略看到的 batch 是 队伍数 × 每队 agent 数
        batch = config.envs * config.num_agents
        if hasattr(acts, "discrete"):
            random_actor = tools.OneHotDist(
                torch.zeros(config.num_actions).repeat(batch, 1)
            )
        else:
            random_actor = torchd.independent.Independent(
                torchd.uniform.Uniform(
                    torch.tensor(acts.low).repeat(batch, 1),
                    torch.tensor(acts.high).repeat(batch, 1),
                ),
                1,
            )

        def random_agent(o, d, s):
            action = random_actor.sample()
            logprob = random_actor.log_prob(action)
            return {"action": action, "logprob": logprob}, None

        state = collect(
            random_agent,
            train_envs,
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=prefill,
        )
        logger.step += prefill * config.action_repeat
        print(f"Logger: ({logger.step} steps).")

    print("Simulate agent.")
    # [CoDreamer] 多智能体时用【图采样器】：每次吐一张图 (T, N, ·)，
    # B 张堆起来即 (B, T, N, ·)。与上游 sample_episodes 的唯一区别是同一时刻的
    # N 个 agent 切在同一份样本里 —— 这是 GNN 能在 agent 轴上聚合的前提。
    # config.batch_size 在这里的单位是【队伍数】，不是序列数（见 tools_marl.make_graph_dataset）。
    #
    # eval_dataset 仍用上游采样器：它只服务于 video_pred，而 video_pred 走的是
    # 单序列形状。config.video_pred_log 为 true 时多智能体路径需要另行适配。
    if multi_agent:
        if config.video_pred_log:
            raise NotImplementedError(
                "multi-agent + video_pred_log 尚未适配：(B,T,N,·) 需要图版本的 video_pred"
            )
        train_dataset = tools_marl.make_graph_dataset(train_eps, config)
    else:
        train_dataset = make_dataset(train_eps, config)
    eval_dataset = make_dataset(eval_eps, config)
    agent = Dreamer(
        train_envs[0].observation_space,
        train_envs[0].action_space,
        config,
        logger,
        train_dataset,
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)
    if (logdir / "latest.pt").exists():
        checkpoint = torch.load(logdir / "latest.pt")
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        agent._should_pretrain._once = False

    # make sure eval will be executed once after config.steps
    #
    # [CoDreamer] Phase 2（环境切断）的阀值。
    #
    # imagination_after_frac 是占总步数的比例：0.5 = 过一半就切断环境，
    # 之后策略只能靠世界模型的想象轨迹继续提升。
    # 注意这是按 agent._step 算的，与 config.steps 同一量纲。
    imag_threshold = (
        int(config.steps * config.imagination_after_frac)
        if config.imagination_only
        else None
    )
    env_cutoff = False
    dataset_at_cutoff = None
    # [CoDreamer] 检查点粒度受**循环粒度**限制，这个耦合不明显，会静默吞掉检查点。
    #
    # 每个循环迭代推进 `eval_every × num_agents` 个 agent 步（多智能体下 batch =
    # envs × num_agents）。而 ckpt 的落点是 `(logger.step // ckpt_every) * ckpt_every`，
    # 只在迭代末尾算一次 —— 所以**一次迭代最多留一个检查点**。若
    # ckpt_every < eval_every × num_agents，实际粒度就是后者，其余落点被跳过。
    #
    # 2026-10-17 实测：eval_every=3000、num_agents=4 → 每次迭代 12000 步，
    # ckpt_every=1200 本想要约 20 个点，实际只得到 2 个（12000、24000）。
    # 当时唯一的表现是"文件比预期少"，没有任何报错。
    _iter_agent_steps = int(config.eval_every) * int(getattr(config, "num_agents", 1))
    if int(getattr(config, "ckpt_every", 0) or 0) > 0 and _iter_agent_steps > config.ckpt_every:
        print(
            f"[CoDreamer] ⚠️ 检查点粒度受限：每次迭代推进 {_iter_agent_steps} agent 步"
            f"（eval_every={config.eval_every} × num_agents={config.num_agents}），"
            f"大于 ckpt_every={config.ckpt_every} —— 实际粒度将是 "
            f"{_iter_agent_steps}，不是 {config.ckpt_every}。"
            f" 想更密就把 eval_every 调小。"
        )
    while agent._step < config.steps + config.eval_every:
        logger.write()
        if config.eval_episode_num > 0:
            # 评测仍然进行：它只读策略、不产生训练数据，不违背"切断环境"。
            print("Start evaluation.")
            eval_policy = functools.partial(agent, training=False)
            collect(
                eval_policy,
                eval_envs,
                eval_eps,
                config.evaldir,
                logger,
                is_eval=True,
                episodes=config.eval_episode_num,
            )
            if config.video_pred_log:
                video_pred = agent._wm.video_pred(next(eval_dataset))
                logger.video("eval_openl", to_np(video_pred))
            # [CoDreamer] 世界模型"想得准不准"的唯一口径。
            # 紧跟在评测后面：这一轮评测刚把 held-out 局写进 evaldir。
            _log_openloop(agent._wm, config, logger)

        # [CoDreamer] 过了阀值就把环境切断。
        # logger.scalar("env_cutoff") 写成 0/1，画图时可以直接用它定位竖线。
        if imag_threshold is not None and not env_cutoff and agent._step >= imag_threshold:
            env_cutoff = True
            dataset_at_cutoff = tools_marl.count_dataset_size(train_eps)
            agent.set_world_model_frozen(config.imagination_freeze_world_model)
            print(
                f"[CoDreamer] 环境已切断于 step={agent._step} "
                f"(阈值 {imag_threshold}，占比 {config.imagination_after_frac}). "
                f"dataset_size={dataset_at_cutoff}；"
                f"世界模型{'冻结' if config.imagination_freeze_world_model else '继续在冻结的 replay 上训练'}。"
                f"此后不再有新的环境数据。"
            )
            # 存下切断时刻的策略。否则 latest.pt 每轮被覆盖，事后无法区分
            # "切断时的策略"与"最终策略" —— 而那正是本实验要比较的一对。
            torch.save(
                {
                    "agent_state_dict": agent.state_dict(),
                    "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
                    # [CoDreamer] 把解析后的 config 也存进去。
                    # 没有它，事后无法重建模型 —— 光看 metrics.jsonl 推不出
                    # gnn_layers / 环境规格这些关键超参，只好靠人手抄命令行，
                    # 极易与训练时不一致。scripts/eval_policy.py 依赖这一项。
                    "config": vars(config),
                },
                logdir / "cutoff.pt",
            )
            print(f"[CoDreamer] 已保存切断时刻的 checkpoint: {logdir / 'cutoff.pt'}")
        logger.scalar("env_cutoff", 1.0 if env_cutoff else 0.0)

        if env_cutoff:
            # Phase 2：只推进训练节奏，不碰环境
            state = tools_marl.simulate_imagination(
                agent,
                train_envs,
                train_eps,
                config.traindir,
                logger,
                limit=config.dataset_size,
                steps=config.eval_every,
                state=state,
            )
        else:
            print("Start training.")
            state = collect(
                agent,
                train_envs,
                train_eps,
                config.traindir,
                logger,
                limit=config.dataset_size,
                steps=config.eval_every,
                state=state,
            )

        # [CoDreamer] 硬证据：Phase 2 里 dataset_size 必须一直等于切断时的值。
        # 一旦有空数据进来，这里会立刻把偏差写成标量，画图时一眼能看出来。
        if dataset_at_cutoff is not None:
            logger.scalar(
                "dataset_size_drift",
                float(tools_marl.count_dataset_size(train_eps) - dataset_at_cutoff),
            )

        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
            # [CoDreamer] 同上：存下 config 以便事后重建模型做部署评测
            "config": vars(config),
        }
        torch.save(items_to_save, logdir / "latest.pt")
        # [CoDreamer] 按间隔另存一份**带步号**的 checkpoint。
        #
        # 上游只留 latest.pt，每一轮覆盖 —— 事后手里永远只有最终点。于是"某个量随
        # 训练怎么变"这条曲线拿不到，而那恰恰是判据：模型利用是随训练收紧（自增强
        # 回路）还是保持不动（静态偏差）、价值膨胀从第几步开始，都需要不止一个点。
        # 2026-10-15 的诊断就卡在这里，只能拿最终 checkpoint 说话。
        #
        # 默认 ckpt_every=0 = 不存，与上游行为逐位一致。按块存，同一块重复保存时
        # 文件名相同（幂等），不会写重。注意体积：单份约 0.2~0.3 GB。
        ckpt_every = int(getattr(config, "ckpt_every", 0) or 0)
        if ckpt_every > 0:
            mark = (logger.step // ckpt_every) * ckpt_every
            dst = logdir / f"ckpt_{mark:09d}.pt"
            if mark > 0 and not dst.exists():
                torch.save(items_to_save, dst)

    # [CoDreamer] 循环结束后再评测一次。
    #
    # 上游靠循环条件 `_step < steps + eval_every` 多跑一轮来保证收尾评测：
    # 单智能体时每轮 `_step += eval_every`，最后一轮正好是"训完再评"。
    #
    # 多智能体下每轮推进的是 `batch × eval_every`（batch = envs × num_agents），
    # 迭代次数因此少好几倍，很容易刚做完最后一次训练就退出 —— 于是**最后一整段
    # 训练（往往就是 Phase 2 全部）没有任何评测**。
    # 这个 bug 的表现很隐：训练日志正常、checkpoint 正常，只是曲线少了一截。
    # 是靠 plot_phase2.py 的面板 ② 画出来是空的才发现的。
    if config.eval_episode_num > 0:
        print("Final evaluation (after the last training block).")
        eval_policy = functools.partial(agent, training=False)
        collect(
            eval_policy,
            eval_envs,
            eval_eps,
            config.evaldir,
            logger,
            is_eval=True,
            episodes=config.eval_episode_num,
        )

    # [CoDreamer] 尾巴上再写一次。
    #
    # 上游把 logger.write() 放在循环**顶部**，于是最后一轮迭代产生的一切
    # （eval_return、dataset_size、env_cutoff、dataset_size_drift）都写不出去。
    # 本实验恰好最关心尾部（Phase 2 整段都在后半），所以必须补这一次刷新。
    logger.write()

    for env in train_envs + eval_envs:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    args, remaining = parser.parse_known_args()
    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
    )

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    main(parser.parse_args(remaining))
