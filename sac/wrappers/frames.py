import numpy as np
from gym import Wrapper
from gym.spaces import Box
from collections import deque


class FrameSkip(Wrapper):
    def __init__(self, env, frames_number):
        super().__init__(env)
        self.fn = frames_number

    def step(self, action):
        R = 0
        for i in range(self.fn):
            next_obs, reward, done, info = self.env.step(action)
            R += reward
            if done:
                break
        return np.float32(next_obs), np.float32(R), done, info

    def reset(self):
        return np.float32(self.env.reset())


class StackFrames(Wrapper):
    def __init__(self, env, nframes=3):
        self._n = nframes
        self._deq = deque(maxlen=nframes)
        self._env = env

        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        low = np.repeat(self.observation_space.low[np.newaxis, ...], nframes, axis=0)
        high = np.repeat(
            self.observation_space.high[np.newaxis, ...], nframes, axis=0
        )
        self.observation_space = Box(
            low=low, high=high, dtype=self.observation_space.dtype
        )

    def step(self, action):
        new_obs, reward, done, info = self._env.step(action)
        self._deq.append(new_obs)
        return self.observation(None), np.float32(reward), done, info

    def reset(self):
        obs = self._env.reset()
        [self._deq.append(obs) for _ in range(self._n)]
        return self.observation(None)

    def observation(self, observation):
        return np.concatenate(self._deq)
