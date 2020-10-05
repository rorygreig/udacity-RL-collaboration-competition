import torch
import numpy as np
from collections import deque
from tqdm import tqdm

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

        self.agent = Agent(self.env.state_size, self.env.action_size, random_seed=10)

        self.network_update_period = 1
        self.num_network_updates = 1

        self.checkpoint_period = 150

    def train(self, n_episodes=3000, max_t=500):
        print("Training DDPG on continuous control")

        recent_scores = deque(maxlen=100)
        scores = []
        average_scores = []

        # run for all episodes
        for i_episode in range(1, n_episodes+1):
            states = self.env.reset()
            episode_scores = np.zeros(self.env.num_agents)

            for t in range(max_t):
                # get next actions from actor network
                actions = np.array([self.agent.act(state, add_noise=True) for state in states])
                # actions = np.random.randn(self.env.num_agents, self.env.action_size)
                next_states, rewards, dones, _ = self.env.step(actions)

                # store experience separately for each agent
                for s, a, r, s_next, d in zip(states, actions, rewards, next_states, dones):
                    self.agent.store_experience(s, a, r, s_next, d)

                # periodically update actor and critic network weights
                if t % self.network_update_period == 0:
                    for i in range(self.num_network_updates):
                        self.agent.update_networks()

                states = next_states
                episode_scores += rewards
                if np.any(dones):
                    break

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
        self.env.close(terminate=True)
        return scores

    def store_weights(self, filename_prefix='checkpoint'):
        print("Storing weights")
        torch.save(self.agent.actor_local.state_dict(), "weights/" + filename_prefix + '_actor.pth')
        torch.save(self.agent.critic_local.state_dict(), "weights/" + filename_prefix + '_critic.pth')

    def run_with_stored_weights(self):
        # load stored weights from training
        self.agent.actor_local.load_state_dict(torch.load("weights/final_actor.pth"))
        self.agent.critic_local.load_state_dict(torch.load("weights/final_critic.pth"))

        states = self.env.reset(train_mode=False)
        scores = np.zeros(self.env.num_agents)

        i = 0
        while True:
            i += 1
            with torch.no_grad():
                actions = np.array([self.agent.act(state, add_noise=False) for state in states])
                next_states, rewards, dones, _ = self.env.step(actions)
                scores += rewards
                states = next_states

            if np.any(dones):
                break
        print(f'Ran for {i} episodes. Final score (averaged over agents): {np.mean(scores)}')

        self.env.close(terminate=True)

