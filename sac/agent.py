import torch
nn = torch.nn
td = torch.distributions
F = nn.functional
from torch.nn.utils import clip_grad_norm_
from copy import deepcopy
from .models import DoubleCritic
from .utils import soft_update, grads_sum, PseudoTape, weight_init
from pytorch3d.loss import chamfer_distance
import numpy as np


class SACAgent(nn.Module):
    def __init__(self, obs_emb_dim, action_dim, config, encoder, decoder, callback):
        super().__init__()

        self.c = config

        self.encoder = encoder
        self.decoder = decoder

        self.actor = nn.Sequential(
            nn.Linear(obs_emb_dim, config.hidden),
            nn.ReLU(),
            nn.Linear(config.hidden, config.hidden),
            nn.ReLU(),
            nn.Linear(config.hidden, 2 * action_dim)
        )

        self.critic = DoubleCritic(obs_emb_dim + action_dim, config.hidden, self.encoder)
        self.target_critic = deepcopy(self.critic)
        self.alpha = nn.Parameter(torch.tensor(np.log(self.c.alpha)))
        self.target_entropy = np.prod(action_dim)
        self.compile()

        self.requires_grad_(False)

        self.callback = callback
        self.step = 0

    def policy(self, obs):
        obs = self.encoder(obs)
        mu, std = self.actor(obs).chunk(2, -1)
        std = torch.clamp(std, -5, 2)
        std = F.softplus(std)
        mu = self.c.mean_scale * torch.tanh(mu / self.c.mean_scale)
        dist = td.Normal(mu, std)
        dist = td.TransformedDistribution(dist, td.transforms.TanhTransform())
        return dist

    @torch.no_grad()
    def act(self, obs, training):
        if not torch.is_tensor(obs):
            obs = torch.from_numpy(obs).to(device=self.c.device)

        dist = self.policy(obs)
        if training:
            action = dist.sample()
        else:
            action = torch.mean(dist.sample(sample_shape=[1000]), 0)  # only works for normal
        return action

    @torch.no_grad()
    def _target_update(self):
        soft_update(self.critic.Q1, self.target_critic.Q1, self.c.rho_critic)
        soft_update(self.critic.Q2, self.target_critic.Q2, self.c.rho_critic)
        soft_update(self.critic.encoder, self.target_critic.encoder, self.c.rho_encoder)

    def learn(self, obs, actions, rewards, dones, next_obs):
        with PseudoTape(self.critic.parameters()):
            cl = self._critic_loss(obs, actions, rewards, dones, next_obs)
            self.critic_optim.zero_grad()
            cl.backward()
            self.callback.add_scalar('train/critic_grads', grads_sum(self.critic), self.step)
            if self.c.critic_grad_max:
                clip_grad_norm_(self.critic.parameters(), self.c.critic_grad_max)
            self.critic_optim.step()

        with PseudoTape(self.actor.parameters()):
            pl = self._policy_loss(obs)
            self.actor_optim.zero_grad()
            pl.backward()
            self.callback.add_scalar('train/actor_grads', grads_sum(self.actor), self.step)
            if self.c.actor_grad_max:
                clip_grad_norm_(self.actor.parameters(), self.c.actor_grad_max)
            self.actor_optim.step()

        with PseudoTape([self.alpha]):
            al = self._alpha_loss(obs)
            self.alpha_optim.zero_grad()
            al.backward()
            self.alpha_optim.step()
            self.callback.add_scalar('train/alpha', self.alpha.exp().item(), self.step)

        with PseudoTape(self.ae_params):
            ael = self._ae_loss(obs)
            self.autoencoder_optim.zero_grad()
            ael.backward()
            self.callback.add_scalar('train/encoder_grads', grads_sum(self.encoder), self.step)
            self.callback.add_scalar('train/decoder_grads', grads_sum(self.decoder), self.step)
            if self.c.ae_grad_max:
                clip_grad_norm_(self.ae_params, self.c.ae_grad_max)
            self.autoencoder_optim.step()

        self._target_update()
        loss = pl + cl + ael
        self.callback.add_scalar('train/actor_loss', pl.item(), self.step)
        self.callback.add_scalar('train/critic_loss', cl.item(), self.step)
        self.callback.add_scalar('train/autoencoder_loss', ael.item(), self.step)
        return loss.item()

    def _policy_loss(self, observations):
        dist = self.policy(observations)
        actions = dist.rsample()
        log_prob = dist.log_prob(actions)
        v1, v2 = self.critic(observations, actions)
        assert v1.shape == v2.shape == log_prob.shape
        loss = torch.minimum(v1, v2) - self.alpha.exp() * log_prob
        assert loss.shape == log_prob.shape
        return - loss.mean()

    def _critic_loss(self, obs, actions, rewards, dones, next_obs):
        with torch.no_grad():
            dist = self.policy(next_obs)
            self.callback.add_scalar('train/entropy', dist.base_dist.entropy().mean().item(), self.step)
            next_actions = dist.sample()
            log_prob = dist.log_prob(next_actions)
            tv1, tv2 = self.target_critic(next_obs, next_actions)
            target_values = torch.minimum(tv1, tv2) - self.alpha.exp() * log_prob
            self.callback.add_scalar('train/mean_val', target_values.mean().item(), self.step)
            target_values = rewards + self.c.gamma * (1 - dones) * target_values

        v1, v2 = self.critic(obs, actions)
        return (v1 - target_values).pow(2).mean() + (v2 - target_values).pow(2).mean()

    def _ae_loss(self, obs):
        latent = self.encoder(obs)
        reconstruction = self.decoder(latent)
        # questionable
        if self.c.encoder == 'PointNet':
            loss = chamfer_distance(obs, reconstruction)[0]
        elif self.c.encoder == 'MLP':
            loss = (obs - reconstruction).pow(2).mean()
        return loss + self.c.ae_latent_reg * latent.pow(2).sum(-1).mean()

    def _alpha_loss(self, obs):
        with torch.no_grad():
            dist = self.policy(obs)
            actions = dist.sample()
            log_prob = dist.log_prob(actions)
        return  - self.alpha.exp() * (log_prob - self.target_entropy).mean()

    def compile(self):
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), self.c.actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), self.c.critic_lr)
        self.ae_params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.autoencoder_optim = torch.optim.Adam(self.ae_params,
                                                  self.c.ae_lr, weight_decay=self.c.ae_l2)
        self.alpha_optim = torch.optim.Adam([self.alpha], self.c.alpha_lr)
        self.to(self.c.device)
        self.apply(weight_init)
