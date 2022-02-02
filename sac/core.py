import torch
from dataclasses import dataclass
from .agent import SACAgent
from .buffer import ReplayBuffer
from torch.utils.data import DataLoader
import numpy as np
from .transformations import Transformations, Next, Transpose, Truncate
from .utils import build_encoder_decoder, make_env
from .runner import Runner
from collections import deque
from dm_control.suite.wrappers import pixels
from .wrappers import depthMapWrapper, FrameSkip, StackFrames, dmWrapper, PixelsToGym
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
import pathlib
from ruamel.yaml import YAML
import datetime
import gc
nn = torch.nn
td = torch.distributions
#torch.autograd.set_detect_anomaly(True)


@dataclass
class Config:
    # agent
    alpha = .1
    hidden = 1024
    mean_scale = 5
    rho_critic = .99
    rho_encoder = .95
    gamma = .99
    actor_lr = 1e-3
    critic_lr = 1e-3
    alpha_lr = 1e-4
    device = 'cuda'
    actor_grad_max = None
    critic_grad_max = None
    frames_stack = 3
    actions_repeat = 2
    # 3 layer with relu 1024 hidden - arxiv.org/abs/1910.01741

    # runner
    nenvs = 10
    max_steps = 30
    n_evals = 1
    buffer_capacity = 5 * 10**5

    # autoencoder
    emb_dim = 50
    encoder = 'MLP' #'PointNet', 'CNN'
    ae_l2 = 1e-7
    ae_lr = 1e-4
    ae_grad_max = None
    ae_latent_reg = 1e-7

    # pointnet
    pn_depth = 120
    pn_number = 800
    pn_layers = 4

    # cnn
    cnn_depth = 32
    cnn_layers = 4

    #train
    batch_size = 128
    steps_per_epoch = 10**3
    evaluation_steps = 10**4
    total_steps = 10**6
    pretrain_steps = 10**3
    save_freq = 4*10**4
    logdir = 'logdir'

    #task
    task = 'cartpole_balance'
    comment = ''
    load = False


config = Config()


class SAC:
    def __init__(self, config=Config()):
        self.c = config
        self._env = self.make_env()

        obs_shape, action_shape, enc_obs_shape = self._build()
        self._task_path = pathlib.Path(self.c.logdir).joinpath(
            f'./{self.c.task}/{self.c.encoder}/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
        if not self.c.load:
            self._task_path.mkdir(parents=True)
        self.callback = SummaryWriter(log_dir=self._task_path)
        self.agent = SACAgent(enc_obs_shape, action_shape, self.c, self.encoder, self.decoder, self.callback)


        @torch.no_grad()
        def policy(obs, training):
            return self.agent.act(obs, training).detach().cpu().numpy()

        self.buffer = ReplayBuffer(obs_shape, action_shape, self.c.buffer_capacity)
        self.runner = Runner(self.make_env, policy, self.buffer, self.c,
                             transformations=Transformations(Next(['observations'], axis=0), Truncate(amount=1, axis=0),
                                                             Transpose(0, 1)))

    def learn(self):
        logs = deque(maxlen=10)
        running_reward = deque(maxlen=5)
        recalc = lambda t: t // self.c.nenvs // self.c.max_steps
        eval_freq = recalc(self.c.evaluation_steps)
        iters = self.c.steps_per_epoch // self.c.batch_size
        save_freq = recalc(self.c.save_freq)

        if self.c.load and True:
            self.load()
        else:
            self._write_hparams()
            (self._task_path / 'buffer').mkdir()
            self.runner.prefill(recalc(min(self.c.pretrain_steps, self.c.buffer_capacity)))

        t = 0
        with trange(self.c.total_steps) as pbar:
            while True:
                t += 1
                tr = next(self.runner)
                self.agent.step = self.runner.interactions_count
                running_reward.append(tr['rewards'].sum().item() / self.runner.nenvs
                                      / self.c.max_steps / self.c.frames_stack)
                self.callback.add_scalar('train/running_reward', np.mean(running_reward), self.runner.interactions_count)
                pbar.update(self.runner.interactions_count - pbar.n)

                dl = iter(DataLoader(self.buffer, batch_size=self.c.batch_size, shuffle=True))
                for _ in range(iters):
                    obs, action, reward, done, next_obs = map(lambda tensor: tensor.to(self.c.device), next(dl))
                    self.agent.learn(obs, action, reward, done, next_obs)

                del dl

                if t % eval_freq == 0:
                    score = self.runner.evaluate()[0]
                    logs.append(score)
                    self.callback.add_scalar('test/reward', score, self.runner.interactions_count)
                    pbar.set_postfix(score=score, mean10=np.mean(logs))

                if t % save_freq == 0:
                    self.save()
                gc.collect()

    def _build(self):
        action_shape = self._env.action_space.shape[0]
        sample = self._env.reset()
        obs_shape = sample.shape
        self.encoder, self.decoder = build_encoder_decoder(self.c, obs_shape)
        with torch.no_grad():
            sample = torch.from_numpy(sample[None])
            sample = self.encoder(sample)
            enc_obs_shape = sample.shape[1]
        return obs_shape, action_shape, enc_obs_shape

    def make_env(self):
        env = make_env(self.c.task)#, task_kwargs={'random': 1})
        if self.c.encoder == 'MLP':
            env = dmWrapper(env)
        elif self.c.encoder == 'PointNet':
            env = depthMapWrapper(env, device=self.c.device, points=self.c.pn_number)
        elif self.c.encoder == 'CNN':
            env = pixels.Wrapper(env, render_kwargs={'camera_id': 1, 'width': 64, 'height': 64})
            env = PixelsToGym(env)
        else:
            raise NotImplementedError
        return StackFrames(FrameSkip(env, self.c.actions_repeat), self.c.frames_stack)

    def save(self):
        #self.buffer.save(self._task_path)
        torch.save({
            'interactions': self.runner.interactions_count,
            'model': self.agent.state_dict(),
            'actor_optim': self.agent.actor_optim.state_dict(),
            'critic_optim': self.agent.critic_optim.state_dict(),
            'ae_optim': self.agent.autoencoder_optim.state_dict(),
            'alpha_optim': self.agent.alpha_optim.state_dict(),
            }, self._task_path / 'checkpoint')

    def load(self):
        y = YAML(typ='unsafe')
        y.register_class(Config)

        with (self._task_path / 'hyperparams').open() as hp:
            conf_dict = y.load(hp)
        for k, v in conf_dict.items():
            setattr(self.c, k, v)

        #self.buffer.load(self._task_path)

        chkp = torch.load(self._task_path / 'checkpoint')
        with torch.no_grad():
            self.agent.load_state_dict(chkp['model'])
            self.agent.actor_optim.load_state_dict(chkp['actor_optim'])
            self.agent.critic_optim.load_state_dict(chkp['critic_optim'])
            self.agent.autoencoder_optim.load_state_dict(chkp['ae_optim'])
            self.agent.requires_grad_(False)
            self.agent.alpha_optim.load_state_dict(chkp['alpha_optim'])
            self.agent.alpha = chkp['alpha']
        self.runner.interactions_count = chkp['interactions']

    def _write_hparams(self):
        y = YAML(typ='unsafe')
        y.register_class(Config)
        conf_dict = {}
        for key in dir(self.c):
            if not key.startswith('_'):
                conf_dict[key] = getattr(self.c, key)
        with (self._task_path / 'hyperparams').open('w') as hp:
            y.dump(conf_dict, hp)





