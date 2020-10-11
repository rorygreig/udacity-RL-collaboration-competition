import torch
from collections import deque

from src.maddpg.ddpg_agent import Agent
from src.maddpg.replay_buffer import ReplayBuffer
from src.plotting import *

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 1024  # minibatch size


class MADDPG:
    def __init__(self, env, target_average_score=0.5, seed=1):
        """Initialize an Agent object.

        Params
        ======
            env: OpenAI VectorEnv environment
            target_average_score: the average score to stop training at
        """

        self.env = env
        self.target_average_score = target_average_score

        agent_a = Agent(self.env.state_size, self.env.action_size, self.env.num_agents, random_seed=1)
        agent_b = Agent(self.env.state_size, self.env.action_size, self.env.num_agents, random_seed=2)
        self.agents = [agent_a, agent_b]

        self.network_update_period = 1
        self.num_network_updates = 5

        self.checkpoint_period = 200
        self.noise_end_episode = 600
        self.noise_coefficient = 5.0
        self.noise_delta = 1.0 / self.noise_end_episode
        self.min_noise = 0.1

        # Replay memory
        self.memory = ReplayBuffer(self.env.action_size, BUFFER_SIZE, BATCH_SIZE, seed)

    def train(self, n_episodes=30000, max_t=500):
        """Run all episodes and periodically update network weights"""

        print("Training DDPG on continuous control")

        recent_scores = deque(maxlen=100)
        scores = []
        average_scores = []

        # run for all episodes
        for i_episode in range(1, n_episodes + 1):
            # Run a single episode
            episode_scores = self.run_episode(max_t)

            # Keep track of scores
            score = np.max(episode_scores) / self.env.reward_scaling  # adjust for reward scaling
            scores.append(score)
            recent_scores.append(score)
            average_score = np.mean(recent_scores)
            average_scores.append(average_score)

            # Periodically update agent network weights
            if i_episode % self.network_update_period == 0:
                self.update_agent_networks()

            print(f"\rEpisode {i_episode}\tAverage Score: {average_score:.6f}\tScore: {score:.6f}", end="")
            if i_episode % self.checkpoint_period == 0:
                self.store_weights('checkpoint')
                plot_scores_with_average(scores, average_scores)
                print('\nEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(recent_scores)))

            if average_score > self.target_average_score:
                print("Reached target average score, finishing training")
                break

        plot_scores_with_average(scores, average_scores)
        self.store_weights('final')
        return scores

    def run_episode(self, max_t):
        """
        Run a single episode of the environment, then update exploration noise
        """
        states = self.env.reset()
        episode_scores = np.zeros(self.env.num_agents)

        # reset noise for all agents
        for agent in self.agents:
            agent.reset()

        for t in range(max_t):
            # get next actions from actor network
            actions = []
            for state, agent in zip(states, self.agents):
                agent_actions = agent.act(torch.from_numpy(state).float(), noise_coefficient=self.noise_coefficient)
                actions.append(agent_actions.data.numpy())

            actions = np.array(actions)
            next_states, rewards, dones, _ = self.env.step(actions)

            self.memory.add(states, actions, rewards, next_states, dones)

            states = next_states
            episode_scores += rewards
            if np.any(dones):
                break

        # decrease noise
        self.noise_coefficient = max(self.noise_coefficient - self.noise_delta, self.min_noise)

        return episode_scores

    def update_agent_networks(self):
        """Learn by updating actor and critic networks for all agents, if enough samples are available in memory"""
        if len(self.memory) > BATCH_SIZE:
            for i in range(self.num_network_updates):
                for (i_agent, agent) in enumerate(self.agents):
                    # sample an experience from the replay buffer
                    experiences = self.memory.sample()
                    self.update_single_agent(experiences, agent, i_agent)

    def update_single_agent(self, experiences, agent_to_update, agent_index):
        """Update actor and critic values for an individual agent"""
        states, actions, rewards, next_states, dones = experiences

        combined_state = torch.flatten(states, start_dim=1, end_dim=-1)
        combined_next_state = torch.flatten(next_states, start_dim=1, end_dim=-1)
        combined_actions = torch.flatten(actions, start_dim=1, end_dim=-1)

        # Update critic
        # Get predicted next-state actions from target actor networks of both agents
        next_target_actions = self.get_next_target_actions(states)

        agent_rewards = rewards[:, agent_index]
        agent_dones = dones[:, agent_index]

        agent_to_update.update_critic(agent_rewards, combined_state, combined_next_state, combined_actions,
                                      next_target_actions, agent_dones)

        # Update actor
        # Calculate actor local predictions using local actor networks of both agents
        next_actions = self.get_next_actions(states, agent_index)
        agent_to_update.update_actor(combined_state, next_actions)

        # update target networks
        agent_to_update.update_targets()

    def get_next_target_actions(self, states):
        """Get target actions from all agents target actors"""
        target_actions = []
        for (i, agent) in enumerate(self.agents):
            agent_states = states[:, i]
            target_actions.append(agent.actor_target(agent_states))

        return torch.cat(target_actions, dim=1)

    def get_next_actions(self, states, agent_index):
        """Get actions from all agents local actors"""
        next_actions = []
        for (i, agent) in enumerate(self.agents):
            agent_states = states[:, i]
            if i == agent_index:
                next_actions.append(agent.actor_local(agent_states))
            else:
                next_actions.append(agent.actor_local(agent_states).detach())

        return torch.cat(next_actions, dim=1)

    def store_weights(self, filename_prefix='checkpoint'):
        print("\nStoring weights")
        for agent in self.agents:
            torch.save(agent.actor_local.state_dict(), "weights/" + filename_prefix + '_actor.pth')
            torch.save(agent.critic_local.state_dict(), "weights/" + filename_prefix + '_critic.pth')

    def run_with_stored_weights(self):
        # load stored weights from training
        for agent in self.agents:
            agent.actor_local.load_state_dict(torch.load("weights/final_actor.pth"))

        # run multiple episodes since they can be quite fast
        num_episodes = 20

        for i_episode in range(num_episodes):
            states = self.env.reset(train_mode=False)
            while True:
                with torch.no_grad():
                    actions = []
                    for state, agent in zip(states, self.agents):
                        agent_actions = agent.act(torch.from_numpy(state).float(), noise_coefficient=self.noise_coefficient)
                        actions.append(agent_actions.data.numpy())

                    actions = np.array(actions)
                    next_states, rewards, dones, _ = self.env.step(actions)

                    states = next_states

                if np.any(dones):
                    break
