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
                # [CoDreamer] 预训练阶段只更新世界模型（见 _train 的说明）
                self._train(
                    next(self._dataset),
                    update_behavior=not pretraining,
                    update_world_model=not (pretraining or self._world_model_frozen),
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
        if state is None:
            latent = action = None
        else:
            latent, action = state
        obs = self._wm.preprocess(obs)
        embed = self._wm.encoder(obs)
        latent, _ = self._wm.dynamics.obs_step(latent, action, embed, obs["is_first"])
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = self._wm.dynamics.get_feat(latent)
        if not training:
            actor = self._task_behavior.actor(feat)
            action = actor.mode()
        elif self._should_expl(self._step):
            actor = self._expl_behavior.actor(feat)
            action = actor.sample()
        else:
            actor = self._task_behavior.actor(feat)
            action = actor.sample()
        logprob = actor.log_prob(action)
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
            reward = lambda f, s, a: self._wm.heads["reward"](
                self._wm.dynamics.get_feat(s)
            ).mode()
            metrics.update(self._task_behavior._train(start, reward)[-1])
            if self._config.expl_behavior != "greedy":
                mets = self._expl_behavior.train(start, context, data)[-1]
                metrics.update({"expl_" + key: value for key, value in mets.items()})
        for name, value in metrics.items():
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)


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


def main(config):
    tools.set_seed_everywhere(config.seed)
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
        }
        torch.save(items_to_save, logdir / "latest.pt")
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
