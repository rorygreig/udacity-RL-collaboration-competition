# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from src.maddpg.ddpg import DDPGAgent
import torch
from src.maddpg.utilities import soft_update, transpose_to_tensor

device = 'cpu'


class MADDPG:
    def __init__(self, state_size, action_size, num_agents, discount_factor=0.95, tau=0.02):
        super(MADDPG, self).__init__()

        in_actor = state_size
        out_actor = action_size
        hidden_in_actor = 16
        hidden_out_actor = 8

        in_critic = num_agents * state_size + num_agents * action_size
        hidden_in_critic = 32
        hidden_out_critic = 16

        self.maddpg_agent = [DDPGAgent(in_actor, hidden_in_actor, hidden_out_actor, out_actor, in_critic,
                                       hidden_in_critic, hidden_out_critic),
                             DDPGAgent(in_actor, hidden_in_actor, hidden_out_actor, out_actor, in_critic,
                                       hidden_in_critic, hidden_out_critic)]
        
        self.discount_factor = discount_factor
        self.tau = tau
        self.iter = 0

    def get_actors(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [ddpg_agent.actor for ddpg_agent in self.maddpg_agent]
        return actors

    def get_target_actors(self):
        """get target_actors of all the agents in the MADDPG object"""
        target_actors = [ddpg_agent.target_actor for ddpg_agent in self.maddpg_agent]
        return target_actors

    def act(self, obs_all_agents, noise=0.0):
        """get actions from all agents in the MADDPG object"""
        # actions = [agent.act(obs, noise) for agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        actions = []
        for (i, agent) in enumerate(self.maddpg_agent):
            agent_obs = obs_all_agents[:, i]
            actions.append(agent.act(agent_obs, noise).squeeze(0))
        return actions

    def target_act(self, obs_all_agents, noise=0.0):
        """get target network actions from all the agents in the MADDPG object """
        # target_actions = [ddpg_agent.target_act(obs, noise) for ddpg_agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        target_actions = []
        for (i, agent) in enumerate(self.maddpg_agent):
            agent_obs = obs_all_agents[:, i]
            target_actions.append(agent.target_act(agent_obs, noise))
        return target_actions

    def update(self, samples, agent_number, logger):
        """update the critics and actors of all the agents """

        obs, obs_full, action, reward, next_obs, next_obs_full, done = samples
        
        agent = self.maddpg_agent[agent_number]
        agent.critic_optimizer.zero_grad()

        #critic loss = batch mean of (y- Q(s,a) from target network)^2
        #y = reward of this timestep + discount * Q(st+1,at+1) from target network
        target_actions = self.target_act(next_obs)
        target_actions = torch.cat(target_actions, dim=1)
        
        target_critic_input = torch.cat((next_obs_full, target_actions), dim=1).to(device)
        
        with torch.no_grad():
            q_next = agent.target_critic(target_critic_input)
        
        y = reward[:, agent_number] + self.discount_factor * q_next.squeeze(1) * (1 - done[:, agent_number])
        action = torch.flatten(action, start_dim=1, end_dim=-1)
        critic_input = torch.cat((obs_full, action), dim=1).to(device)
        q = agent.critic(critic_input)

        huber_loss = torch.nn.SmoothL1Loss()
        critic_loss = huber_loss(q.squeeze(0), y.detach())
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
        agent.critic_optimizer.step()

        # update actor network using policy gradient
        agent.actor_optimizer.zero_grad()
        # make input to agent
        # detach the other agents to save computation
        # saves some time for computing derivative
        q_input = []
        for i_agent, agent in enumerate(self.maddpg_agent):
            if i_agent == agent_number:
                q_input.append(agent.actor(obs[:, agent_number]))
            else:
                q_input.append(agent.actor(obs[:, agent_number]).detach())
                
        q_input = torch.cat(q_input, dim=1)
        # combine all the actions and observations for input to critic
        # many of the obs are redundant, and obs[1] contains all useful information already
        q_input2 = torch.cat((obs_full, q_input), dim=1)
        
        # get the policy gradient
        actor_loss = -agent.critic(q_input2).mean()
        actor_loss.backward()
        #torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),0.5)
        agent.actor_optimizer.step()

        al = actor_loss.cpu().detach().item()
        cl = critic_loss.cpu().detach().item()
        logger.add_scalars('agent%i/losses' % agent_number,
                           {'critic loss': cl,
                            'actor_loss': al},
                           self.iter)

    def update_targets(self):
        """soft update targets"""
        self.iter += 1
        for ddpg_agent in self.maddpg_agent:
            soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.tau)
            soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.tau)
            
            
            




