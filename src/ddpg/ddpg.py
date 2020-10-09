import torch
from collections import deque

from src.ddpg.ddpg_agent import Agent
from src.plotting import *


class DDPG:
    def __init__(self, env, target_average_score=0.5):
        """Initialize an Agent object.

        Params
        ======
            env: OpenAI VectorEnv environment
        """

        self.env = env
        self.target_average_score = target_average_score

        agent_a = Agent(self.env.state_size, self.env.action_size, self.env.num_agents, random_seed=1)
        agent_b = Agent(self.env.state_size, self.env.action_size, self.env.num_agents, random_seed=2)
        self.agents = [agent_a, agent_b]

        self.network_update_period = 20
        self.num_network_updates = 4

        self.checkpoint_period = 500

        # factor by which each agent takes account of the other agents reward
        self.reward_share_factor = 0.8

        self.reward_scaling = 10.0

    def train(self, n_episodes=30000, max_t=500):
        print("Training DDPG on continuous control")

        recent_scores = deque(maxlen=100)
        scores = []
        average_scores = []

        # run for all episodes
        for i_episode in range(1, n_episodes + 1):
            states = self.env.reset()
            episode_scores = np.zeros(self.env.num_agents)

            for t in range(max_t):
                # get next actions from actor network
                actions = []
                for state, agent in zip(states, self.agents):
                    actions.append(agent.act(state, add_noise=True))

                next_states, individual_rewards, dones, _ = self.env.step(actions)

                collab_rewards = self.calculate_collab_rewards(individual_rewards)

                combined_state = np.concatenate((states[0], states[1]))
                combined_next_state = np.concatenate((next_states[0], next_states[1]))

                # store experience separately for each agent
                for agent, s, a, r, s_next, d in zip(self.agents, states, actions, collab_rewards, next_states, dones):
                    agent.store_experience(s, a, r, s_next, combined_state, combined_next_state, d)

                states = next_states
                episode_scores += individual_rewards
                if np.any(dones):
                    break

            # periodically update actor and critic network weights
            if i_episode % self.network_update_period == 0:
                for i in range(self.num_network_updates):
                    for agent in self.agents:
                        agent.update_networks()

            for agent in self.agents:
                agent.update_noise()

            score = np.max(episode_scores)
            scores.append(score)
            recent_scores.append(score)
            average_score = np.mean(recent_scores)
            average_scores.append(average_score)

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

    def store_weights(self, filename_prefix='checkpoint'):
        print("Storing weights")
        for agent in self.agents:
            torch.save(agent.actor_local.state_dict(), "weights/" + filename_prefix + '_actor.pth')
            torch.save(agent.critic_local.state_dict(), "weights/" + filename_prefix + '_critic.pth')

    def run_with_stored_weights(self):
        # load stored weights from training
        for agent in self.agents:
            agent.actor_local.load_state_dict(torch.load("weights/final_actor.pth"))
            agent.critic_local.load_state_dict(torch.load("weights/final_critic.pth"))

        states = self.env.reset(train_mode=False)
        scores = np.zeros(self.env.num_agents)

        i = 0
        while True:
            i += 1
            with torch.no_grad():
                # actions = np.array([self.agent.act(state, add_noise=False) for state in states])
                actions = []
                for state, agent in zip(states, self.agents):
                    actions.append(agent.act(state, add_noise=False))
                next_states, rewards, dones, _ = self.env.step(actions)
                scores += rewards
                states = next_states

            if np.any(dones):
                break
        print(f'Ran for {i} episodes. Final score (averaged over agents): {np.mean(scores)}')

    def calculate_collab_rewards(self, individual_rewards):
        shared_rewards = np.array(individual_rewards)
        total_reward_to_share = np.sum(shared_rewards) * self.reward_share_factor
        for i in range(shared_rewards.shape[0]):
            shared_rewards[i] = (1 - self.reward_share_factor) * shared_rewards[i] + total_reward_to_share

        return shared_rewards * self.reward_scaling
