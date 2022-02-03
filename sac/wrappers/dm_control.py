from torchvision.transforms import Resize
from PIL import Image
import numpy as np
from gym.spaces import Box


class Wrapper:
    """ Partially solves problem with  compatibality"""

    def __init__(self, env):
        self._env = env
        self.observation_space, self.action_space = self._infer_spaces(env)

    def observation(self, timestamp):
        return np.float32(timestamp.observation)

    def reward(self, timestamp):
        return np.float32(timestamp.reward)

    def done(self, timestamp):
        return timestamp.last()

    def step(self, action):
        timestamp = self._env.step(action)
        obs = self.observation(timestamp)
        r = self.reward(timestamp)
        d = self.done(timestamp)
        return obs, r, d, None

    def reset(self):
        return self.observation(self._env.reset())

    @staticmethod
    def _infer_spaces(env):
        lim = 1.
        spec = env.action_spec()
        action_space = Box(low=spec.minimum.astype(np.float32), dtype=np.float32,
                           high=spec.maximum.astype(np.float32), shape=spec.shape)
        ar = list(env.observation_spec().values())[0]
        obs_sample = np.concatenate(list(map(lambda ar: ar.generate_value() if ar.shape != () else [1],
                                             env.observation_spec().values())))
        obs_space = Box(low=-lim, high=lim, shape=obs_sample.shape, dtype=ar.dtype)
        return obs_space, action_space

    @property
    def unwrapped(self):
        if hasattr(self._env, 'unwrapped'):
            return self._env.unwrapped
        return self._env

    def __getattr__(self, item):
        return getattr(self._env, item)

    def __repr__(self):
        return f"{self.__class__.__name__}"


class PixelsToGym(Wrapper):
    def __init__(self, env):
        self._env = env
        self.resize = Resize((64, 64))

    def observation(self, timestamp):
        obs = timestamp.observation['pixels']
        obs = Image.fromarray(obs, 'RGB')
        obs = self.resize(obs)
        obs = np.array(obs) / 255.
        return obs.transpose((2, 1, 0))

    @property
    def observation_space(self):
        # correspondent space have to be extracted from the dm_control API -> gym API
        return Box(low=0., high=1., shape=(3,))

    @property
    def action_space(self):
        return Box(low=-2., high=2., shape=(1,))


class dmWrapper(Wrapper):

    def observation(self, timestamp):
        obs = np.array([])
        for v in timestamp.observation.values():
            if not v.ndim:
                v = v[None]
            obs = np.concatenate((obs, v))
        return obs.astype(np.float32)
