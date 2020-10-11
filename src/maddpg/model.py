import torch
import torch.nn as nn


class Actor(nn.Module):
    """Shared network Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64, dropout_p=0.1, activation=nn.ReLU):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.m = nn.Sequential(
            nn.Linear(state_size, fc1_units),
            activation(),
            nn.Dropout(p=dropout_p),
            nn.Linear(fc1_units, fc2_units),
            activation(),
            nn.Dropout(p=dropout_p),
            nn.Linear(fc2_units, action_size),
            nn.Tanh()
        )

    def add_noise(self, sigma=0.1):
        for param in self.parameters():
            param.add_(torch.randn(param.size()) * sigma)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        return self.m(state)


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, input_size, output_size, seed, fc1_units=64, fc2_units=64, dropout_p=0.1, activation=nn.ReLU):
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
        self.m = nn.Sequential(
            nn.Linear(input_size, fc1_units),
            activation(),
            nn.Dropout(p=dropout_p),
            nn.Linear(fc1_units, fc2_units),
            activation(),
            nn.Dropout(p=dropout_p),
            nn.Linear(fc2_units, output_size),
            nn.Identity()
        )

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = torch.cat((state, action), dim=1)
        return self.m(x)
