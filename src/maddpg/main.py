# main function that sets up environments
# perform training loop

from src.tennis_env import TennisMultiAgentEnv
from src.maddpg.buffer import ReplayBuffer
from src.maddpg.maddpg import MADDPG
import torch
import numpy as np
from tensorboardX import SummaryWriter
import os


def seeding(seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    seeding()

    # number of training episodes.
    # change this to higher number to experiment. say 30000.
    number_of_episodes = 5000
    episode_length = 80
    batchsize = 512
    # how many episodes to save policy and gif
    save_interval = 100
    t = 0
    
    # amplitude of OU noise
    # this slowly decreases to 0
    noise = 2
    noise_reduction = 0.9999

    # how many episodes before update
    episode_per_update = 2

    log_path = os.getcwd()+"/log"
    model_dir = os.getcwd()+"/model_dir"
    
    os.makedirs(model_dir, exist_ok=True)

    env = TennisMultiAgentEnv("./Tennis_Linux/Tennis.x86_64")
    
    # keep 5000 episodes worth of replay
    buffer_size = 1e6
    buffer = ReplayBuffer(buffer_size, seed=1)
    
    # initialize policy and critic
    maddpg = MADDPG(env.state_size, env.action_size, env.num_agents)
    logger = SummaryWriter(log_dir=log_path)
    agent0_reward = []
    agent1_reward = []

    # training loop
    # show progressbar
    import progressbar as pb
    widget = ['episode: ', pb.Counter(),'/',str(number_of_episodes),' ', 
              pb.Percentage(), ' ', pb.ETA(), ' ', pb.Bar(marker=pb.RotatingMarker()), ' ' ]
    
    timer = pb.ProgressBar(widgets=widget, maxval=number_of_episodes).start()

    for episode in range(0, number_of_episodes):

        timer.update(episode)

        reward_this_episode = np.zeros(2)
        obs = env.reset()
        obs = torch.from_numpy(obs).float()

        # save info or not
        save_info = (episode % save_interval == 0)

        for episode_t in range(episode_length):

            # explore = only explore for a certain number of episodes
            # action input needs to be transposed
            actions = maddpg.act(obs.unsqueeze(0), noise=noise)
            noise *= noise_reduction
            
            actions_for_env = torch.stack(actions).detach().numpy()
            
            # step forward one frame
            next_obs, rewards, dones, info = env.step(actions_for_env)

            next_obs = torch.from_numpy(next_obs).float()

            obs_full = torch.flatten(obs)
            next_obs_full = torch.flatten(next_obs)
            
            buffer.add(obs, obs_full, actions_for_env, rewards, next_obs, next_obs_full, dones)
            
            reward_this_episode += rewards

            obs, obs_full = next_obs, next_obs_full
        
        # update once after every episode_per_update
        if len(buffer) > batchsize and episode % episode_per_update == 0:
            for a_i in range(env.num_agents):
                samples = buffer.sample(batchsize)
                maddpg.update(samples, a_i, logger)
            maddpg.update_targets()  # soft update the target network towards the actual networks

        agent0_reward.append(reward_this_episode[0])
        agent1_reward.append(reward_this_episode[1])

        if episode % 100 == 0 or episode == number_of_episodes-1:
            avg_rewards = [np.mean(agent0_reward), np.mean(agent1_reward)]
            agent0_reward = []
            agent1_reward = []
            for a_i, avg_rew in enumerate(avg_rewards):
                logger.add_scalar('agent%i/mean_episode_rewards' % a_i, avg_rew, episode)

        # saving model
        save_dict_list = []
        if save_info:
            for i in range(env.num_agents):

                save_dict = {'actor_params': maddpg.maddpg_agent[i].actor.state_dict(),
                             'actor_optim_params': maddpg.maddpg_agent[i].actor_optimizer.state_dict(),
                             'critic_params': maddpg.maddpg_agent[i].critic.state_dict(),
                             'critic_optim_params': maddpg.maddpg_agent[i].critic_optimizer.state_dict()}
                save_dict_list.append(save_dict)

                torch.save(save_dict_list, 
                           os.path.join(model_dir, 'episode-{}.pt'.format(episode)))

    logger.close()
    timer.finish()


if __name__ == '__main__':
    main()
