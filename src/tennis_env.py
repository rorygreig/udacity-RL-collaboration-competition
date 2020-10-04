import gym
import numpy as np
from gym import spaces
from gym.vector import VectorEnv
from unityagents import UnityEnvironment

fixed_reward = 1.0

class TennisMultiAgentEnv(VectorEnv):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, env_filepath):
        self.unity_env = UnityEnvironment(file_name=env_filepath)
        # get the default brain
        self.brain_name = self.unity_env.brain_names[0]
        brain = self.unity_env.brains[self.brain_name]

        env_info = self.unity_env.reset(train_mode=True)[self.brain_name]

        states = env_info.vector_observations
        self.state_size = states.shape[1]
        self.action_size = brain.vector_action_space_size
        self.num_agents = len(env_info.agents)

        print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], self.state_size))
        print('The state for the first agent looks like:', states[0])

        high = np.full(self.action_size, 30.0)
        self.action_space = spaces.Box(low=-high, high=high, dtype=np.float)

        high = np.ones(self.state_size)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float)

        print(f"\nState size: {self.state_size}, action size: {self.action_size}, number of agents: {self.num_agents}")

        super(TennisMultiAgentEnv, self).__init__(self.num_agents, self.observation_space, self.action_space)

    def step(self, actions):
        env_info = self.unity_env.step(actions)[self.brain_name]

        rewards = [fixed_reward if rew > 0.0 else 0.0 for rew in env_info.rewards]
        dones = env_info.local_done
        next_states = env_info.vector_observations

        return next_states, rewards, dones, {}

    def reset(self, train_mode=True):
        env_info = self.unity_env.reset(train_mode=train_mode)[self.brain_name]
        return env_info.vector_observations

    def render(self, mode='human'):
        pass

    def close(self, terminate):
        self.unity_env.close()
