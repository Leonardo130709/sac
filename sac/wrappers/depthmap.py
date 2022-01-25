from .dm_control import Wrapper
from ..utils import PointCloudGenerator
import torch
from dm_control.mujoco.wrapper import MjvOption
import ctypes


class depthMapWrapper(Wrapper):

    def __init__(self, env,
                 camera_id=0,
                 height=240,
                 width=320,
                 device='cpu',
                 return_pos=False,
                 points=1000,
                 ):
        super().__init__(env)
        self._env = env
        self.points = points
        self._depth_kwargs = dict(camera_id=camera_id, height=height, width=width,
                                  depth=True, scene_option=self._prepare_scene())
        self.return_pos = return_pos
        self.pcg = PointCloudGenerator(**self.pc_params, device=device)

    def reset(self):
        return self.observation(self._env.reset())

    def __getattr__(self, name):
        return getattr(self._env, name)

    def observation(self, timestamp):
        depth = self._env.physics.render(**self._depth_kwargs)
        pc = self.pcg.get_PC(depth)
        pc = self._segmentation(pc)
        if self.return_pos:
            pos = self._env.physics.position()
            return pc, pos
        return pc.detach().cpu().numpy()

    def _segmentation(self, pc):
        dist_thresh = 10
        #TODO find a way
        pc = pc[pc[..., 2] < dist_thresh] # smth like infty cutting
        if self.points:
        #ind = randperm(pc.size(-2), device=self.pcg.device)[:self.points]
            amount = pc.size(-2)
            if amount > self.points:
                #ind = randint(high=pc.size(-2), size=(self.points,), device=self.pcg.device)
                ind = torch.randperm(amount, device=self.pcg.device)[:self.points]
                pc = torch.index_select(pc, -2, ind)
            elif amount < self.points:
                zeros = torch.zeros(self.points - amount, *pc.shape[1:], device=self.pcg.device)
                pc = torch.cat([pc, zeros])
        return pc

    def _prepare_scene(self):
        scene = MjvOption()
        scene.ptr.contents.flags = (ctypes.c_uint8*22)(0)

        return scene

    @property
    def pc_params(self):
        # device
        fovy = self._env.physics.model.cam_fovy[0]
        return dict(
            camera_fovy=fovy,
            image_height=self._depth_kwargs.get('height') or 240,
            image_width=self._depth_kwargs.get('width') or 320)
