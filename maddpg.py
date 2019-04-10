# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from ddpg_agent import Agent
import torch
import random
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MADDPG:
    def __init__(self, state_size, action_size, n_agents, fc1, fc2, random_seed, 
                 buffer_size=int(1e5), batch_size=512, tau=1e-3, gamma=0.99,
                 lr_actor=1e-4,lr_critic=1e-4, weight_decay=0):
        
        self.in_actor = state_size
        self.out_actor = action_size
        self.in_critic = (state_size + action_size)*n_agents
        self.n_agents = n_agents
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.weight_decay = weight_decay
        
        self.maddpg_agents = [Agent(self.in_actor, fc1, fc2, self.out_actor, self.in_critic, random_seed,
                                    self.buffer_size, self.batch_size, self.tau, self.gamma,
                                    self.lr_actor, self.lr_critic, self.weight_decay) \
                              for i in range(n_agents)]
       
    def step(self, states, actions, rewards, next_states, dones):
        all_infos = zip(self.maddpg_agents, states, actions, rewards, next_states, dones)
        for id, infos in enumerate(all_infos):
            agent, state, action, reward, next_state, done = infos
            if id == 0:
                opponent_states = states[1]
                opponent_actions = actions[1]
                opponent_next_states = next_states[1]
                agent.step(state, action, reward, next_state, done, opponent_states, opponent_actions, opponent_next_states)
            elif id == 1:
                opponent_states = states[0]
                opponent_actions = actions[0]
                opponent_next_states = next_states[0]
                agent.step(state, action, reward, next_state, done, opponent_states, opponent_actions, opponent_next_states)
                
    def act(self, states, add_noise=True):
        actions = np.zeros([self.n_agents, self.out_actor])
        for id, agent in enumerate(self.maddpg_agents):
            actions[id, :] = agent.act(states[id], add_noise)
        return actions

    def reset(self):
        for agent in self.maddpg_agents:
            agent.reset()
            
    
    def save_state_dict(self, file_prefix):
        for i in range(len(self.maddpg_agents)):
            agent_label = file_prefix + str(i)
            agent_actor_file = agent_label + '_actor.pth'
            agent_critic_file = agent_label + '_critic.pth'
            torch.save(self.maddpg_agents[i].actor_local.state_dict(), agent_actor_file)
            print('Saved actor checkpoint of agent {:d} to {}'.format(i, agent_actor_file))
            torch.save(self.maddpg_agents[i].critic_local.state_dict(), agent_critic_file)
            print('Saved critic checkpoint of agent {:d} to {}'.format(i, agent_actor_file))
            i+=1
        
        
    def load_state_dict(self, file_prefix):
        for i in range(len(self.maddpg_agents)):
            agent_label = file_prefix + str(i)
            agent_actor_file = agent_label + '_actor.pth'
            agent_critic_file = agent_label + '_critic.pth'
            self.maddpg_agents[i].actor_local.load_state_dict(torch.load(agent_actor_file, map_location='cpu'))
            print('Loaded actor checkpoint to agent {:d} from {}'.format(i, agent_actor_file))
            self.maddpg_agents[i].critic_local.load_state_dict(torch.load(agent_critic_file, map_location='cpu'))
            print('Loaded critic checkpoint to agent {:d} from {}'.format(i, agent_actor_file))
            i+=1
            
    def print_agent_network(self):
        for i in range(len(self.maddpg_agents)):
            print('agent %i:' % (i + 1))
            print(self.maddpg_agents[i].actor_local)
            print(self.maddpg_agents[i].critic_local)
            print('\n')
            
