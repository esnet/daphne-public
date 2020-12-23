import math

import cherry as ch
import torch
from torch import nn
from torch.distributions import Categorical


def linear_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.zero_()
    return module


class Policy(nn.Module):

    def __init__(self, input_size, output_size, hidden_size, hidden_layers, device='cpu'):
        super(Policy, self).__init__()
        self.device = device
        self.input_size = input_size 
        self.output_size = output_size
        hiddens = []
        for _ in range(hidden_layers):
            hiddens.append(hidden_size) 
        layers = [linear_init(nn.Linear(self.input_size, hiddens[0])), nn.ReLU()]
        for i, o in zip(hiddens[:-1], hiddens[1:]):
            layers.append(linear_init(nn.Linear(i, o)))
            layers.append(nn.ReLU())
        layers.append(linear_init(nn.Linear(hiddens[-1], self.output_size)))
        self.mean = nn.Sequential(*layers)
        self.input_size = input_size
    def density(self, state):
        # state = ch.onehot(state, dim=self.input_size)
        state = state.to(self.device, non_blocking=True)
        loc = self.mean(state)
        return Categorical(logits=loc)

    def log_prob(self, state, action):
        density = self.density(state)
        action = action.to(self.device, non_blocking=True)
        return density.log_prob(action).mean(dim=1, keepdim=True)
        # return density.log_prob(action).mean().view(-1, 1).detach()

    def forward(self, state):
        density = self.density(state)
        # print(density)
        action = density.sample()
        return action