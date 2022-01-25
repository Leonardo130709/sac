import numpy as np

class ReplayBuffer:
    def __init__(self, obs_shape, action_shape, capacity):
        self.capacity = capacity
        if isinstance(obs_shape, int):
            obs_shape = (obs_shape,)
        if isinstance(action_shape, int):
            action_shape = (action_shape,)

        self.obs = np.zeros((capacity, *obs_shape), dtype=np.float16)
        self.actions = np.zeros((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.float16)

        self._idx = 0
        self.size = 0

    def _add(self, obs, action, reward, done, next_obs):
        np.copyto(self.obs[self._idx], obs)
        np.copyto(self.actions[self._idx], action)
        np.copyto(self.rewards[self._idx], reward)
        np.copyto(self.next_obs[self._idx], next_obs)
        np.copyto(self.dones[self._idx], done)

        self._idx = (self._idx + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)

    def add(self, obs, action, reward, done, next_obs):
        if hasattr(obs, '__len__'):
            for o, a, r, d, no in zip(obs, action, reward, done, next_obs):
                self._add(o, a, r, d, no)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.obs[idx].astype(np.float32), self.actions[idx], self.rewards[idx], self.dones[idx], self.next_obs[idx].astype(np.float32)

    def save(self, path):
        np.save(path / 'buffer/obs', self.obs)
        np.save(path / 'buffer/actions', self.actions)
        np.save(path / 'buffer/rewards', self.rewards)
        np.save(path / 'buffer/dones', self.dones)
        np.save(path / 'buffer/next_obs', self.next_obs)
        np.save(path / 'buffer/idx', self._idx)

    def load(self, path):
        self.obs = np.load(path / 'buffer/obs.npy')
        self.actions = np.load(path / 'buffer/actions.npy')
        self.rewards = np.load(path / 'buffer/rewards.npy')
        self.dones = np.load(path / 'buffer/dones.npy')
        self.next_obs = np.load(path / 'buffer/next_obs.npy')
        self._idx = np.load(path / 'buffer/idx.npy')
        self.size = len(self.rewards)