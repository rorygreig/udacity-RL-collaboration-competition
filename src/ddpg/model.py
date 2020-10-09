import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return -lim, lim


class Actor(nn.Module):
    """Shared network Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=128, dropout_p=0.1, activation=nn.ReLU()):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        # self.fc1 = nn.Linear(state_size, fc1_units)
        # self.fc2 = nn.Linear(fc1_units, fc2_units)
        # self.fc3 = nn.Linear(fc2_units, action_size)
        # self.dropout = nn.Dropout(p=dropout_p)
        # self.reset_parameters()
        self.m = nn.Sequential(
            nn.Linear(state_size, fc1_units),
            activation,
            nn.Dropout(p=dropout_p),
            nn.Linear(fc1_units, fc2_units),
            activation,
            nn.Dropout(p=dropout_p),
            nn.Linear(fc2_units, action_size),
            nn.Tanh()
        )

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        output_lim = self.fc3.weight.data.abs().mean() / 50
        self.fc3.weight.data.uniform_(-output_lim, output_lim)

    def add_noise(self, sigma=0.1):
        for param in self.parameters():
            param.add_(torch.randn(param.size()) * sigma)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        # x = F.relu(self.fc1(state))
        # x = self.dropout(x)
        # x = F.relu(self.fc2(x))
        # x = self.dropout(x)
        # return F.tanh(self.fc3(x))
        return self.m(state)


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, input_size, output_size, seed, fc1_units=256, fc2_units=512, dropout_p=0.1, activation=nn.ReLU()):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        # self.fcs1 = nn.Linear(state_size, fcs1_units)
        # self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        # self.fc3 = nn.Linear(fc2_units, 1)
        # self.dropout = nn.Dropout(p=dropout_p)
        # self.reset_parameters()
        self.m = nn.Sequential(
            nn.Linear(input_size, fc1_units),
            activation,
            nn.Dropout(p=dropout_p),
            nn.Linear(fc1_units, fc2_units),
            activation,
            nn.Dropout(p=dropout_p),
            nn.Linear(fc2_units, output_size),
            nn.Identity()
        )

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # xs = F.relu(self.fcs1(state))
        # x = torch.cat((xs, action), dim=1)
        # x = F.relu(self.fc2(x))
        # x = self.dropout(x)
        # return self.fc3(x)
        x = torch.cat((state, action), dim=1)
        self.m(x)
