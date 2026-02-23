import argparse
import functools
import os
import pathlib
import sys
from utils_.exit_program import sp
if os.name == "nt":
    os.environ["MUJOCO_GL"] = "glfw"
else:
    os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import ruamel.yaml as yaml

sys.path.append(str(pathlib.Path(__file__).parent))

import exploration as expl
import models
import tools
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
        # this is update step 环境步数计数器
        self._step = logger.step // config.action_repeat
        self._update_count = 0
        self._dataset = dataset
        self._wm = models.WorldModel(obs_space, act_space, self._step, config)
        self._task_behavior = models.ImagBehavior(config, self._wm)
        # 在想象空间（world model里roll out的latent）上训练
        if (
            config.compile and os.name != "nt"
        ):  # compilation is not supported on windows
            self._wm = torch.compile(self._wm)
            self._task_behavior = torch.compile(self._task_behavior)
        reward = lambda f, s, a: self._wm.heads["reward"](f).mean()
        # 探索策略
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: expl.Random(config, act_space),
            plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
        )[config.expl_behavior]().to(self._config.device)

    def __call__(self, obs, reset, state=None, training=True):
        # print("64"*50)
        # for i in obs.keys():
        #     print(i, obs[i].shape)
        # print(reset)
        # print(state)
        #
        # sp()
        step = self._step
        if training:
            steps = (
                self._config.pretrain
                if self._should_pretrain()
                else self._should_train(step)
            )
            for _ in range(steps):
                self._train(next(self._dataset))
                self._update_count += 1
                #记录更新次数
                self._metrics["update_count"] = self._update_count
            if self._should_log(step):
                for name, values in self._metrics.items():
                    self._logger.scalar(name, float(np.mean(values)))
                    self._metrics[name] = []
                if self._config.video_pred_log:
                    openl = self._wm.video_pred(next(self._dataset))
                    self._logger.video("train_openl", to_np(openl))
                self._logger.write(fps=True)
        # print(obs, state, training) None False
        policy_output, state = self._policy(obs, state, training)
        # sp()
        # policy_output = {"action": action, "logprob": logprob}
        # state = (latent, action)
        if training:
            self._step += len(reset)
            self._logger.step = self._config.action_repeat * self._step
        return policy_output, state

    #输入：原始观测与上一步的隐状态 (latent, action)。输出：当前动作与新的隐状态。
    def _policy(self, obs, state, training):
        if state is None:
            # RSSM的信念状态
            latent = action = None
        else:

            latent, action = state
        obs = self._wm.preprocess(obs)    #xt
        embed = self._wm.encoder(obs)      #et
        #latent ht

        # post["deter"] = ℎ𝑡
        # post["stoch"] = 𝑧𝑡
        # post = {"stoch": stoch, "deter": prior["deter"], **stats}
        latent, _ = self._wm.dynamics.obs_step(latent, action, embed, obs["is_first"])
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = self._wm.dynamics.get_feat(latent)  # st=(ht,zt）
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

    def _train(self, data):
        metrics = {}
        # data = {
        #     'image': [16, 50, H, W, C],  # 图像序列
        #     'action': [16, 50, act_dim],  # 动作序列
        #     'reward': [16, 50],  # 回报序列
        #     'discount': [16, 50],  # 折扣因子
        #     'is_first': [16, 50],  # 标记序列起点
        #     ...
        # }
        post, context, mets = self._wm._train(data)
        #{stoch： 【】， deter：【】}
        metrics.update(mets)
        start = post

        reward = lambda f, s, a: self._wm.heads["reward"](
            self._wm.dynamics.get_feat(s)
        ).mode()
        print(158, "*"*50)
        #训练策略，也就是a, c
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
    #dmc walker_walk
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
    #为np.cuda设置相同的随机种子
    tools.set_seed_everywhere(config.seed)

    if config.deterministic_run:
        tools.enable_deterministic_run()

    print(config.logdir)

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

    #确定训练从哪一步开始
    step = count_steps(config.traindir)
    # step in logger is environmental step
    logger = tools.Logger(logdir, config.action_repeat * step)

    print("Create envs.")
    if config.offline_traindir:
        directory = config.offline_traindir.format(**vars(config))
    else:
        directory = config.traindir
    train_eps = tools.load_episodes(directory, limit=config.dataset_size)

    # print(244, train_eps["20251006T142028-f223bc1914b840328ed40622551f6a4c-501"].keys())
    # sys.exit()
    if config.offline_evaldir:
        directory = config.offline_evaldir.format(**vars(config))
    else:
        directory = config.evaldir
    #或取过去训练的结果
    eval_eps = tools.load_episodes(directory, limit=1)
    # print(250, eval_eps['20251006T141549-e7a0ba23eeff43b8849fcbb3982a8d19-501'])
    # dict_keys(['orientations', 'height', 'velocity', 'image', 'is_terminal', 'is_first', 'reward', 'discount', 'action',
    #            'logprob'])
    # sys.exit()
    # print(eval_eps)
    make = lambda mode, id: make_env(config, mode, id)
    # obs.shape: (64, 64, 3)
    #
    # Step0:
    # action = [-0.32  0.77 - 0.21  0.56 - 0.10  0.89]
    # reward = 0.04
    # obs.shape = (64, 64, 3)

    train_envs = [make("train", i) for i in range(config.envs)]
    eval_envs = [make("eval", i) for i in range(config.envs)]
    if config.parallel:
        train_envs = [Parallel(env, "process") for env in train_envs]
        eval_envs = [Parallel(env, "process") for env in eval_envs]
    else:
        train_envs = [Damy(env) for env in train_envs]
        eval_envs = [Damy(env) for env in eval_envs]
    acts = train_envs[0].action_space
    #[-1,1] * 6 , 一个6维的图像，连续空间
    print("Action Space", acts)

    #只有离散动作空间才有.n属性
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]  #6
    state = None
    # print(264, config.offline_traindir)


    # if 0:
    # print(306, config.offline_traindir)
    if not config.offline_traindir:

        prefill = max(0, config.prefill - count_steps(config.traindir))
        print(f"Prefill dataset ({prefill} steps).")
        if hasattr(acts, "discrete"):
            random_actor = tools.OneHotDist(
                torch.zeros(config.num_actions).repeat(config.envs, 1)
            )
        else:
            random_actor = torchd.independent.Independent(
                torchd.uniform.Uniform(
                    torch.tensor(acts.low).repeat(config.envs, 1),
                    torch.tensor(acts.high).repeat(config.envs, 1),
                ),
                1,
            )

        # observation(o)：当前观测；
        # done(d)：是否 episode结束；
        # state(s)：agent内部记忆状态（比如RNN隐状态)
        #logprob对应动作概率，用于行为克隆 / IRL等训练（这里可忽略）
        def random_agent(o, d, s):
            action = random_actor.sample()
            logprob = random_actor.log_prob(action)
            return {"action": action, "logprob": logprob}, None

        # print(random_agent(0,0,0))
        # sys.exit()
        #random_agent返回值
        # ({'action': tensor([[-0.0075, 0.5364, -0.8230, -0.7359, -0.3852, 0.2682],
        #                     [-0.0198, 0.7929, -0.0887, 0.2646, -0.3022, -0.1966],
        #                     [-0.9553, -0.6623, -0.4122, 0.0370, 0.3953, 0.6000],
        #                     [-0.6779, -0.4355, 0.3632, 0.8304, -0.2058, 0.7483]]),
        #   'logprob': tensor([-4.1589, -4.1589, -4.1589, -4.1589])}, None)
        #

        state = tools.simulate(
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


    train_dataset = make_dataset(train_eps, config)
    eval_dataset = make_dataset(eval_eps, config)
    # print(360, state)
    # print(301,train_envs[0].observation_space,
    #     train_envs[0].action_space,)

    v_data = next(train_dataset)
    keys = v_data.keys()
    for i in keys:
        print(i, v_data[i].shape)

    # print(next(train_dataset).keys())
    # print(next(train_dataset)["image"].shape)

    # sys.exit()
    # Dict(height: Box(-inf, inf, (1,), float32),
    # image: Box(0, 255, (64, 64, 3), uint8),
    # orientations: Box(-inf, inf, (14,), float32),
    # velocity: Box( -inf, inf, (9,), float32))
    # Box(-1.0, 1.0, (6,), float32)

    # data = {
    #     'image': [16, 50, H, W, C],  # 图像序列
    #     'action': [16, 50, act_dim],  # 动作序列
    #     'reward': [16, 50],  # 回报序列
    #     'discount': [16, 50],  # 折扣因子
    #     'is_first': [16, 50],  # 标记序列起点
    #     ...
    # }
    # print(
    #     301,train_envs[0].observation_space,
    #     train_envs[0].action_space,
    # )
    #
    # Dict(height: Box(-inf, inf, (1,), float32), image: Box(0, 255, (64, 64, 3), uint8), orientations: Box(-inf, inf,(14,), float32), velocity: Box(
    #     -inf, inf, (9,), float32)) Box(-1.0, 1.0, (6,), float32)
    #
    # Dict(height: Box(-inf, inf, (1,), float32), image: Box(0, 255, (64, 64, 3), uint8), orientations: Box(-inf, inf, (14,),float32), velocity: Box(
    #     -inf, inf, (9,), float32)) Box(-1.0, 1.0, (6,), float32)


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
    while agent._step < config.steps + config.eval_every:
        logger.write()
        if config.eval_episode_num > 0:
            print("Start evaluation.")
            eval_policy = functools.partial(agent, training=False)
            tools.simulate(
                eval_policy,
                eval_envs, #环境
                eval_eps,  #历史记录
                config.evaldir,   #要保存的文件夹
                logger,         #logger的信息
                is_eval=True,
                episodes=config.eval_episode_num,
            )
            if config.video_pred_log:
                video_pred = agent._wm.video_pred(next(eval_dataset))
                logger.video("eval_openl", to_np(video_pred))
        print("Start training.")
        state = tools.simulate(
            agent,
            train_envs,
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=config.eval_every,
            state=state,
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
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--configs", nargs="+")
    # args, remaining = parser.parse_known_args()

    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")

    args, remaining = parser.parse_known_args([
        "--configs", "dmc_vision",
        "--task", "dmc_walker_walk",
        "--logdir", "./logdir/dmc_walker_walk"
    ])

    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
    )
    # print(configs)
    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
    # print(469, name_list)
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])


    parser = argparse.ArgumentParser()


    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        # print(arg_type)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))

    main(parser.parse_args(remaining))
