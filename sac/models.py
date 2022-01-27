import torch
nn = torch.nn
F = nn.functional
from rltools.common.utils import build_mlp


class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim=10):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs, action):
        x = torch.cat([obs, action], -1)
        return self.fc(x)


class DoubleCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim=10, encoder=nn.Identity()):
        super().__init__()

        self.encoder = encoder
        self.Q1 = Critic(input_dim, hidden_dim)
        self.Q2 = Critic(input_dim, hidden_dim)

    def forward(self, obs, action):
        obs = self.encoder(obs)
        return self.Q1(obs, action), self.Q2(obs, action)


class PointCloudEncoder(nn.Module):
    def __init__(self, in_features, depth):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_features, depth, 1),
            #nn.BatchNorm1d(depth),
            nn.ReLU(),
            nn.Conv1d(depth, 2 * depth, 1),
            #nn.BatchNorm1d(2 * depth),
            nn.ReLU(),
            nn.Conv1d(2 * depth, 4 * depth, 1),
            #nn.BatchNorm1d(4 * depth),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(4*depth, depth),
            nn.LayerNorm(depth),
            nn.Tanh()
        )

    def forward(self, X):
        X = X.transpose(-1, -2)
        X = self.conv(X)
        X = torch.max(X, -1)[0]
        X = self.fc(X)
        return X


class PointCloudDecoder(nn.Module):
    def __init__(self, out_features, depth, n_points):
        super().__init__()

        self.fc = nn.Linear(depth, n_points * 4 * depth)

        self.deconv = nn.Sequential(
            nn.Unflatten(1, (4 * depth, n_points)),
            nn.ConvTranspose1d(4 * depth, 2 * depth, 1),
            #nn.BatchNorm1d(2 * depth),
            nn.ELU(),
            nn.ConvTranspose1d(2 * depth, depth, 1),
            #nn.BatchNorm1d(depth),
            nn.ELU(),
            nn.ConvTranspose1d(depth, out_features, 1)
        )

    def forward(self, X):
        X = self.fc(X)
        X = self.deconv(X)
        return X.transpose(-1, -2)
