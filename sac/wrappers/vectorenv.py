import numpy as np
import gym
from ..utils import extract_shapes
from collections import namedtuple


envinfo = namedtuple('Envinfo', ('n_states', 'n_actions', 'nenvs'))


class VectorEnv(gym.Env):
    """
    Automatically resets envs that are finished
    """
    def __init__(self, make_env, nenvs):
        # make env includes wrappers right now
        self._nenvs = nenvs
        self._envs = [make_env() for _ in range(nenvs)]
        self._states = [self._envs[i].reset() for i in range(self.nenvs)]

    def step(self, actions):
        # should return outputs as batch
        assert len(actions) == self.nenvs
        next_states, rewards, dones, infos = [], [], [], []
        for action, env in zip(actions, self._envs):
            next_state, reward, done, info = env.step(action)
            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        states = self._states
        self._states = [self._envs[i].reset() if dones[i] else next_states[i] for i in range(self.nenvs)]
        return np.stack(states), np.stack(rewards), np.stack(dones), np.stack(infos)

    def reset(self):
        self._states = [self._envs[i].reset() for i in range(self.nenvs)]
        return np.stack(self._states)

    @property
    def nenvs(self):
        return self._nenvs

    @property
    def envs(self):
        return self._envs

    @property
    def action_space(self):
        return self._envs[0].action_space

    @property
    def observation_space(self):
        return self._envs[0].observation_space

    @property
    def info(self):
        n_states, n_actions = extract_shapes(self.envs[0])
        return envinfo(n_states=n_states, n_actions=n_actions, nenvs=self.nenvs)
