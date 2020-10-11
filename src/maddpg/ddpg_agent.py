import random

import torch
import torch.nn.functional as F
import torch.optim as optim

from src.maddpg.model import Actor, Critic
from src.maddpg.noise import OUNoise
from src.maddpg.util import soft_update

LR_ACTOR = 1e-3  # learning rate of the actor
LR_CRITIC = 2e-3  # learning rate of the critic
WEIGHT_DECAY = 0.0  # L2 weight decay
GAMMA = 0.95  # discount factor
TAU = 1e-2  # for soft update of target parameters


class Agent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, num_agents, random_seed, dropout_p=0.1):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed, dropout_p=dropout_p)
        self.actor_target = Actor(state_size, action_size, random_seed, dropout_p=dropout_p)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        critic_state_size = num_agents * state_size
        critic_action_size = num_agents * action_size
        critic_input_size = critic_state_size + critic_action_size
        critic_output_size = 1
        self.critic_local = Critic(critic_input_size, critic_output_size, random_seed, dropout_p=dropout_p)
        self.critic_target = Critic(critic_input_size, critic_output_size, random_seed, dropout_p=dropout_p)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process

        self.weight_noise_sigma = 0.0
        self.action_noise_sigma = 0.3

        self.noise = OUNoise(action_size, scale=1.0, sigma=self.action_noise_sigma)

    def act(self, state, noise_coefficient=0.0):
        """Returns actions for given state as per current policy."""
        self.actor_local.eval()
        with torch.no_grad():
            # add noise to parameter weights
            self.actor_local.add_noise(noise_coefficient * self.weight_noise_sigma)
            action = self.actor_local(state) + noise_coefficient * self.noise.sample()
        self.actor_local.train()

        return torch.clamp(action, -1, 1)

    def act_target(self, state):
        self.actor_target.eval()
        with torch.no_grad():
            action = self.actor_target(state)
        self.actor_target.train()

        return torch.clamp(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def update_critic(self, reward, combined_state, combined_next_state, combined_actions, combined_next_actions, dones):
        """Update local critic network"""

        with torch.no_grad():
            Q_targets_next = self.critic_target(combined_next_state, combined_next_actions).squeeze(-1)
        # Compute Q targets for current states
        Q_targets = reward + (GAMMA * Q_targets_next * (1 - dones))

        # Compute critic loss
        Q_expected = self.critic_local(combined_state, combined_actions).squeeze(-1)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # clip gradient to improve stability
        torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1.0)
        self.critic_optimizer.step()

    def update_actor(self, combined_state, combined_actions_pred):
        """Update local actor network"""

        actor_loss = -self.critic_local(combined_state, combined_actions_pred).mean()

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def update_targets(self):
        """Soft update target networks"""
        soft_update(self.critic_local, self.critic_target, TAU)
        soft_update(self.actor_local, self.actor_target, TAU)


