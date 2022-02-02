from sac import wrappers
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from .utils import make_env, build_encoder_decoder
import pathlib
from ruamel.yaml import YAML
from .core import Config
import ipdb
nn = torch.nn


class PairedEnv:
    def __init__(self, configs):
        self.env = make_env(configs.task, task_kwargs={'random': 0})
        self.c = configs
        self.state_env = wrappers.FrameSkip(wrappers.dmWrapper(self.env), self.c.actions_repeat)
        self.wrapped_env = self.make_env()
        self.action_space = self.state_env.action_space

    def observation(self, timestamp):
        return {
            'state': self.state_env.observation(timestamp),
            'wrapped': self.wrapped_env.observation(timestamp)
        }

    def step(self, action):
        timestamp = self.env.step(action)
        self.state_env.step(action)
        self.wrapped_env.step(action)
        return self.observation(timestamp)

    def reset(self):
        self.state_env.reset()
        self.wrapped_env.reset()
        ts = self.env.reset()
        return self.observation(ts)

    def make_env(self):
        if self.c.encoder == 'MLP':
            env = wrappers.dmWrapper(self.env)

        elif self.c.encoder == 'PointNet':
            env = wrappers.depthMapWrapper(self.env, device=self.c.device, points=self.c.pn_number)
        elif self.c.encoder == 'CNN':
            env = wrappers.pixels.Wrapper(self.env, render_kwargs={'camera_id': 1, 'width': 64, 'height': 64})
            env = wrappers.PixelsToGym(env)
        else:
            raise NotImplementedError
        return wrappers.StackFrames(wrappers.FrameSkip(env, self.c.actions_repeat), self.c.frames_stack)


class PCA:
    def __init__(self, postfix):
        self.env, self.c, self.encoder = self.load(postfix)
        self.proj = self._build()
        self.opt = torch.optim.Adam(self.proj.parameters(), 1e-2)

    def _build(self):
        state, embed = self.sample_dataset(1)
        return nn.Linear(embed.shape[-1], state.shape[-1]).to(self.c.device)

    def step(self, action):
        states = self.env.step(action)
        states['wrapped'] = self.encoder(torch.from_numpy(states['wrapped'][None]).to(self.c.device))
        return states

    def sample_dataset(self, n=1000):
        state, embedding = [], []
        for _ in range(n):
            timestamp = self.step(self.env.action_space.sample())
            state.append(timestamp['state'])
            embedding.append(timestamp['wrapped'])
        return torch.from_numpy(np.stack(state)).to(self.c.device), torch.cat(embedding)

    def load(self, postfix):
        path = pathlib.Path().joinpath(postfix)
        y = YAML()
        with (path / 'hyperparams').open() as hp:
            conf_dict = y.load(hp)
        config = Config()
        for k, v in conf_dict.items():
            setattr(config, k, v)

        env = PairedEnv(config)
        obs = env.reset()['wrapped']
        encoder, _ = build_encoder_decoder(config, obs.shape)
        encoder(torch.from_numpy(obs[None]))
        chkp = torch.load(path / 'checkpoint')
        with torch.no_grad():
            params = chkp['model']
            enc_params = {}
            for k, v in params.items():
                if k.startswith('encoder'):
                    enc_params[k.replace('encoder.', '')] = v

        encoder.load_state_dict(enc_params)
        encoder.to(config.device)
        return env, config, encoder

    def learn(self):
        t = 0
        stats = deque(maxlen=10)
        stats.append(np.inf)
        while np.mean(stats) > .1 and t < 30:
            t += 1
            states, embed = self.sample_dataset(100)
            preds = self.proj(embed)
            loss = (preds-states).pow(2).mean()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            stats.append(loss.item())
            try:
                from IPython.display import clear_output
                clear_output(wait=True)
            except ImportError:
                pass
            for i in range(min(states.shape[-1], 5)):
                plt.plot(states[:, i].detach().cpu(), label='states')
                plt.plot(preds[:, i].detach().cpu(), label='recon')
                plt.legend()
                plt.show()
        return stats
