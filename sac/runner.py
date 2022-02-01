import numpy as np
from collections import namedtuple
from .transformations import Concatenate, Transformations
from .wrappers import VectorEnv


envinfo = namedtuple('Envinfo', ('n_states', 'n_actions', 'nenvs'))


class Runner:
    def __init__(self, make_env, policy, buffer, configs, transformations=Transformations()):
        self.c = configs
        self._envs = VectorEnv(make_env, self.c.nenvs)
        self.policy = policy
        self.interactions_count = 0
        self._prev_obs = self._envs.reset()
        self.transformations = transformations
        self.buffer = buffer
        self._concatenate = Concatenate(axis=0)

    def simulate(self):
        obs = self._prev_obs
        observations, actions, rewards, dones = [], [], [], []
        for t in range(self.c.max_steps):
            action = self.policy(obs, True)
            new_obs, r, d, _ = self._envs.step(action)
            observations.append(obs)
            actions.append(action)
            rewards.append(r[:, None])
            dones.append(d[:, None])
            obs = new_obs
            self.interactions_count += self.c.nenvs

        tr = dict(
            actions=actions,
            rewards=rewards,
            done_flags=dones,
            observations=observations
        )
        for k, v in tr.items():
            tr[k] = np.stack(v, 0)

        self._prev_obs = obs
        return tr

    def evaluate(self):
        logs = []
        #env = deepcopy(self._envs) fix deepcopy object bug
        env = self._envs
        for _ in range(self.c.n_evals):
            mask = np.zeros(self.nenvs, dtype=np.bool_)
            obs = env.reset()
            Rs = np.zeros(self.nenvs)
            while not np.all(mask):
                actions = self.policy(obs, False)
                obs, rewards, dones, _ = env.step(actions)
                Rs += rewards * (1 - mask)
                mask = np.bitwise_or(mask, dones)
            logs.extend(Rs.tolist())
        self._prev_obs = obs
        return np.mean(logs), np.std(logs)

    def __iter__(self):
        return self

    def __next__(self):
        tr = self.simulate()
        tr = self.transformations(tr)
        self._fill_buffer(tr)
        return tr

    def _fill_buffer(self, tr):
        transitions = self._concatenate(tr.copy())
        observations, actions, rewards, dones, next_observations = \
            map(lambda k: transitions[k], ('observations', 'actions', 'rewards', 'done_flags', 'next_observations'))
        self.buffer.add(observations, actions, rewards, dones, next_observations)

    def prefill(self, steps):
        def rnd_policy(obs, training):
            return np.stack([self._envs.action_space.sample() for _ in range(self.nenvs)])

        p = self.policy
        self.policy = rnd_policy

        for _ in range(steps):
            next(self)

        self.policy = p

    @property
    def nenvs(self):
        return self.c.nenvs
