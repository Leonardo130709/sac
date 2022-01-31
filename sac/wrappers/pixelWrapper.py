from .dm_control import Wrapper
from torchvision.transforms import Resize
from PIL import Image
import numpy as np
from gym.spaces import Box


class PixelsToGym(Wrapper):
    def __init__(self, env):
        self._env = env
        #self.resize = Resize((64, 64))

    def observation(self, timestamp):
        obs = timestamp.observation['pixels']
        obs = Image.fromarray(obs, 'RGB')
        #obs = self.resize(obs)
        #obs = np.array(obs) / 255.
        obs = np.array(obs)
        return obs.transpose((2, 1, 0))

    @property
    def observation_space(self):
        # correspondent space have to be extracted from the dm_control API -> gym API
        return Box(low=0., high=1., shape=(3,))

    @property
    def action_space(self):
        return Box(low=-2., high=2., shape=(1,))
