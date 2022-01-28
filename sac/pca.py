from dm_control import suite
from .wrappers import depthMapWrapper
from .wrappers.dm_control import Wrapper, dmWrapper
from .core import SAC
import torch
import numpy as np
nn = torch.nn
from collections import deque
import matplotlib.pyplot as plt
from IPython.display import clear_output


class PairedEnv(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.wrapped_env = depthMapWrapper(env)
        self.state_env = dmWrapper(env)

    def observation(self, timestamp):
        return {
            'state': self.state_env.observation(timestamp),
            'wrapped': self.wrapped_env.observation(timestamp)
        }

    def step(self, action):
        timestamp = self.env.step(action)
        return self.observation(timestamp)

    def reset(self):
        ts = self.env.reset()
        return self.observation(ts)


class PCA:
    def __init__(self, configs):
        self.c = configs
        self.encoder = self.load_encoder()
        self.env = self._make_env()
        self.proj = self._build()
        self.proj.to(self.c.device)
        self.opt = torch.optim.SGD(self.proj.parameters(), 1e-3)

    def _make_env(self):
        return PairedEnv(suite.load(*self.c.task.split('_', 1)))

    def _build(self):
        state, embed = self.sample_dataset(1)
        return nn.Linear(embed.shape[-1], state.shape[-1])

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

    def load_encoder(self):
        self.c.load = True
        self.c.encoder = 'PointNet'
        self.c.buffer_capacity = 10 ** 2

        s = SAC(self.c)
        return s.agent.encoder

    def learn(self):
        t = 0
        stats = deque(maxlen=10)
        stats.append(np.inf)
        while np.mean(stats) > 1 or t < 100:
            t += 1
            states, embed = self.sample_dataset(1000)
            preds = self.proj(embed)
            loss = (preds-states).pow(2).mean()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            stats.append(loss.item())
            clear_output(wait=True)
            for i in range(max(states.shape[-1], 5)):
                plt.plot(states[:, i].detach().cpu(), label='states')
                plt.plot(preds[:, i].detach().cpu(), label='recon')
                plt.legend()
                plt.show()
        return stats
