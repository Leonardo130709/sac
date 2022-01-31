from torch import nn
from .models import PointCloudDecoder, PointCloudEncoder, PixelDecoder, PixelEncoder
import torch
import math
import numpy as np
from gym.spaces import Box, Discrete
import plotly.graph_objects as go
td = torch.distributions


class PseudoTape:
    def __init__(self, params):
        self._params = params if hasattr(params, '__len__') else list(params)
        self._state = len(self._params) * [None]
        for i, p in enumerate(self._params):
            self._state[i] = p.requires_grad

    def __enter__(self):
        for p in self._params:
            p.requires_grad = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        for i, p in enumerate(self._params):
            p.requires_grad = self._state[i]


def grads_sum(model):
    s = 0
    for p in model.parameters():
        s += p.grad.sum()
    return s.item()


def build_mlp(sizes, activation=nn.ELU):
    mlp = []
    for i in range(1, len(sizes)):
        mlp.append(nn.Linear(sizes[i-1], sizes[i]))
        mlp.append(activation())
    return nn.Sequential(*mlp[:-1])


def extract_shapes(env):
    def extract(space):
        shape = None
        if isinstance(space, Box):
            shape = space.shape[0]
        elif isinstance(space, Discrete):
            shape = space.n
        else:
            raise NotImplementedError
        return shape
    return map(lambda x: extract(x), [env.observation_space, env.action_space])


def build_encoder_decoder(configs, obs_shape):
    encoder = []
    decoder = []
    emb_dim = configs.emb_dim
    nf = configs.frames_stack
    if configs.encoder == 'MLP':
        encoder.append(nn.Flatten())
        encoder.append(build_mlp([obs_shape[0]] + 1*[configs.hidden] + [emb_dim]))
        encoder.append(nn.Tanh())
        decoder.append(build_mlp([emb_dim]+1*[configs.hidden]+[obs_shape[0]]))
    elif configs.encoder == 'PointNet':
        encoder.append(PointCloudEncoder(obs_shape[-1], configs.pn_depth, configs.pn_layers))
        decoder.append(PointCloudDecoder(obs_shape[-1], configs.pn_depth, configs.pn_layers, obs_shape[0]))
    elif configs.encoder == 'PointNetMLP':
        encoder.append(nn.Flatten())
        encoder.append(build_mlp([np.prod(obs_shape)] + 3*[configs.hidden] + [emb_dim]))
        encoder.append(nn.Tanh())
        decoder.append(build_mlp([emb_dim] + 3*[configs.hidden] + [np.prod(obs_shape)]))
        decoder.append(nn.Unflatten(1, obs_shape))
        configs.encoder = 'PointNet'
    elif configs.encoder == 'CNN':
        enc = PixelEncoder(obs_shape[0], configs.cnn_depth, configs.cnn_layers)
        _, _, width, height = enc.conv(torch.ones(1, *obs_shape)).shape
        encoder.append(enc)
        decoder.append(PixelDecoder(width, height, obs_shape[0], configs.cnn_depth, configs.cnn_layers))
    else:
        raise NotImplementedError

    #decoder.append(nn.Unflatten(1, (nf, obs_shape // nf)))
    return nn.Sequential(*encoder), nn.Sequential(*decoder)


def soft_update(online, target, rho):
    for po, pt in zip(online.parameters(), target.parameters()):
        pt.copy_(rho*pt + (1-rho)*po)


class PointCloudGenerator:
    def __init__(self, camera_fovy, image_height, image_width, device, cam_matrix=None, rot_matrix=None, position=None):
        super(PointCloudGenerator, self).__init__()

        self.fovy = math.radians(camera_fovy)
        self.height = image_height
        self.width = image_width
        self.device = device

        if rot_matrix != None:
            self.rot_matrix = torch.tensor(rot_matrix, dtype=torch.float32, device=device, requires_grad=False)

        if position != None:
            self.position = torch.tensor(position, dtype=torch.float32, device=device, requires_grad=False)

        if cam_matrix != None:
            self.cam_matrix = cam_matrix
        else:
            self.cam_matrix = self.get_cam_matrix()

        self.fx = self.cam_matrix[0, 0]
        self.fy = self.cam_matrix[1, 1]
        self.cx = self.cam_matrix[0, 2]
        self.cy = self.cam_matrix[1, 2]

        self.uv1 = torch.ones((self.height, self.width, 3), dtype=torch.float32, device=device, requires_grad=False)
        for i in range(self.height):
            for j in range(self.width):
                self.uv1[i][j][0] = ((i + 1) - self.cx) / self.fx
                self.uv1[i][j][1] = ((j + 1) - self.cy) / self.fy
        #print(self.uv1.shape)
        self.uv1 = self.uv1.reshape(-1, 3)
        #   print(self.uv1.shape)

    def get_cam_matrix(self):
        f = self.height / (2 * math.tan(self.fovy / 2))

        return torch.tensor(((f, 0, self.width / 2), (0, f, self.height / 2), (0, 0, 1)),
                            dtype=torch.float32, device=self.device, requires_grad=False)

    def reshape_depth(self, depth):
        depth = torch.tensor(np.flip(depth, axis=0).copy(), dtype=torch.float32, device=self.device,
                             requires_grad=False)
        depth = depth.reshape(-1, 1)
        return torch.cat((depth, depth, depth), dim=-1)

    def get_PC(self, depth):
        depth = self.reshape_depth(depth)
        xyz = depth * self.uv1
        return xyz


def draw_pc(pc_arr):
    x = np.array([pc_arr[i][0] for i in range(pc_arr.shape[0])])
    y = np.array([pc_arr[i][1] for i in range(pc_arr.shape[0])])
    z = np.array([pc_arr[i][2] for i in range(pc_arr.shape[0])])

    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(
        size=2,
        color=z,  # set color to an array/list of desired values
        colorscale='Viridis',  # choose a colorscale
        opacity=0.8
    ))])

    fig.show()


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif any(map(lambda module: isinstance(m, module), (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose1d))):
        m.bias.data.fill_(0.0)
        nn.init.orthogonal_(m.weight.data)


class TanhTransform(td.transforms.TanhTransform):
    lim = 0.99999997
    def _inverse(self, y):
        y = torch.clamp(y, min=-self.lim, max=self.lim)
        return torch.atanh(y)
