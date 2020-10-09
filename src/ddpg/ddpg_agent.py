import numpy as np
import random
import copy

from src.ddpg.model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

LR_ACTOR = 1e-5  # learning rate of the actor
LR_CRITIC = 1e-5  # learning rate of the critic
WEIGHT_DECAY = 0.0  # L2 weight decay
GAMMA = 0.98  # discount factor
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
        self.noise = OUNoise(action_size, random_seed)

        self.weight_noise_sigma = 0.1
        self.action_noise_sigma = 0.3

    def act(self, state, noise_coefficient=0.0):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float()
        self.actor_local.eval()
        with torch.no_grad():
            # add noise to parameter weights
            self.actor_local.add_noise(noise_coefficient * self.weight_noise_sigma)
            action = self.actor_local(state).data.numpy()
        self.actor_local.train()

        action += noise_coefficient * self.noise.sample()
        return np.clip(action, -1, 1)

    def update_critic(self, reward, combined_state, combined_next_state, combined_actions, combined_next_actions, dones):
        Q_targets_next = self.critic_target(combined_next_state, combined_next_actions)
        # Compute Q targets for current states
        Q_targets = reward + (GAMMA * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(combined_state, combined_actions)
        # critic_loss = F.mse_loss(Q_expected, Q_targets)
        # TODO: does Q_targets need to be detached?
        critic_loss = F.mse_loss(Q_expected, Q_targets.detach())

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # clip gradient to improve stability
        torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

    def update_actor(self, combined_state, combined_actions_pred):
        actor_loss = -self.critic_local(combined_state, combined_actions_pred).mean(dim=0)

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def update_targets(self):
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def sample_gaussian_noise(self, sigma=0.3):
        return np.random.normal(0.0, sigma, self.action_size)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.1):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        np.random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        # dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        dx = self.theta * (self.mu - x) + self.sigma * np.random.normal(0.0, self.sigma, len(x))
        self.state = x + dx
        return self.state
