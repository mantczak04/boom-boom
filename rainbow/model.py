# -*- coding: utf-8 -*-
# Adapted from Kaixhin/Rainbow model.py (MIT).
# Replaced the Atari-sized CNN with a board-sized dueling distributional network.
#
# State contract: input (batch, C, H, W) with history_length = 1 (C = obs_channels).

from __future__ import division

import math

import torch
from torch import nn
from torch.nn import functional as F


class NoisyLinear(nn.Module):
    """Factorised NoisyLinear layer with bias (unchanged from Rainbow)."""

    def __init__(self, in_features, out_features, std_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input):
        if self.training:
            return F.linear(
                input,
                self.weight_mu + self.weight_sigma * self.weight_epsilon,
                self.bias_mu + self.bias_sigma * self.bias_epsilon,
            )
        return F.linear(input, self.weight_mu, self.bias_mu)


class DQN(nn.Module):
    """C51 dueling network for (C, H, W) Minesweeper observations."""

    def __init__(self, args, action_space: int):
        super().__init__()
        self.atoms = args.atoms
        self.action_space = action_space
        in_channels = args.obs_channels

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        h = args.board_height
        w = args.board_width
        self.conv_output_size = 64 * h * w

        self.fc_h_v = NoisyLinear(
            self.conv_output_size, args.hidden_size, std_init=args.noisy_std
        )
        self.fc_h_a = NoisyLinear(
            self.conv_output_size, args.hidden_size, std_init=args.noisy_std
        )
        self.fc_z_v = NoisyLinear(args.hidden_size, self.atoms, std_init=args.noisy_std)
        self.fc_z_a = NoisyLinear(
            args.hidden_size, action_space * self.atoms, std_init=args.noisy_std
        )

    def forward(self, x, log=False):
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        v = self.fc_z_v(F.relu(self.fc_h_v(x)))
        a = self.fc_z_a(F.relu(self.fc_h_a(x)))
        v = v.view(-1, 1, self.atoms)
        a = a.view(-1, self.action_space, self.atoms)
        q = v + a - a.mean(1, keepdim=True)
        if log:
            q = F.log_softmax(q, dim=2)
        else:
            q = F.softmax(q, dim=2)
        return q

    def reset_noise(self):
        for name, module in self.named_children():
            if name.startswith("fc"):
                module.reset_noise()
