import torch
from torchvision import transforms as T
nn = torch.nn


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


class PointCloudEncoderOld(nn.Module):
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


class PointCloudEncoder(nn.Module):
    def __init__(self, in_features, depth, layers):
        super().__init__()

        self.convs = nn.ModuleList([nn.Conv1d(in_features, depth, 1)])
        for _ in range(layers-1):
            self.convs.append(nn.ReLU(inplace=True))
            self.convs.append(nn.Conv1d(depth, depth, 1))

        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(depth, depth),
            nn.LayerNorm(depth),
            nn.Tanh()

        )

    def forward(self, x):
        x = x.transpose(-1, -2)
        for layer in self.convs:
            x = layer(x)
        x = torch.max(x, -1)[0]
        return self.fc(x)


class PointCloudDecoder(nn.Module):
    def __init__(self, out_features, depth, layers, pn_number):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(depth, depth * pn_number),
            nn.ReLU(inplace=True),
        )

        self.deconvs = nn.ModuleList([nn.Unflatten(1, (depth, pn_number))])
        for _ in range(layers-1):
            self.deconvs.append(nn.ConvTranspose1d(depth, depth, 1))
            self.deconvs.append(nn.ReLU(inplace=True))

        self.deconvs.append(nn.ConvTranspose1d(depth, out_features, 1))

    def forward(self, x):
        x = self.fc(x)
        for layer in self.deconvs:
            x = layer(x)
        return x.transpose(-1, -2)


class PointCloudDecoderOld(nn.Module):
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


class PixelEncoder(nn.Module):
    def __init__(self, in_channels=1, depth=32, layers=3):
        super().__init__()

        self.depth = depth
        self.convs = nn.ModuleList([
            T.Normalize(.5, 1),
            nn.Conv2d(in_channels, depth, 3, stride=2)
        ])
        for _ in range(layers-1):
            self.convs.append(nn.ReLU(inplace=True))
            self.convs.append(nn.Conv2d(depth, depth, 3, stride=1))

        self.fc = None

    def conv(self, img):
        img = img / 255.
        for conv in self.convs:
            img = conv(img)
        return img

    def forward(self, img):
        if self.fc is None:
            self._build(img)
        img = self.conv(img)
        img = self.fc(img)
        return img

    def _build(self, img):
        sample = self.conv(img)
        _, _, w, h = sample.size()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(w*h*self.depth, self.depth),
            nn.LayerNorm(self.depth),
            nn.Tanh(),
        )
        return w, h


class PixelDecoder(nn.Module):
    def __init__(self, width, height, out_channels=3, depth=32, layers=2):
        super().__init__()
        # TODO find formulas instead of hardcoding dims
        self.fc = nn.Sequential(
            nn.Linear(depth, depth*width*height),
            nn.ReLU(inplace=True),
        )

        self.deconvs = nn.ModuleList([nn.Unflatten(-1, (depth, width, height))])

        for _ in range(layers-1):
            self.deconvs.append(nn.ConvTranspose2d(depth, depth, 3, stride=1))
            self.deconvs.append(nn.ReLU(inplace=True))

        self.deconvs.append(nn.ConvTranspose2d(depth, out_channels, 3, stride=2, output_padding=1))
        self.deconvs.append(T.Normalize(-0.5, 1.))

    def forward(self, x):
        x = self.fc(x)
        for deconv in self.deconvs:
            x = deconv(x)
        x *= 255
        return x
