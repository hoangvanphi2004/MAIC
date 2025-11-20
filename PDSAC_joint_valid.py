import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque
import random
from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper

# SAC Networks
class Actor(nn.Module):
    def __init__(self, num_agents, obs_dim, action_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super(Actor, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.fc1 = nn.Linear(num_agents * obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, obs):
        obs = obs.view(obs.size(0), -1)  # Flatten for joint input
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std
    
    def sample(self, obs, epsilon=1e-6):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        
        log_prob = normal.log_prob(z)
        log_prob -= torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        
        mean = torch.tanh(mean)
        return action, log_prob, mean

class Critic(nn.Module):
    def __init__(self, num_agents, obs_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        # Q1
        self.fc1 = nn.Linear(num_agents * obs_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        # Q2
        self.fc4 = nn.Linear(num_agents * obs_dim + action_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, 1)
        
    def forward(self, obs, action):
        obs = obs.view(obs.size(0), -1)  # Flatten for joint input
        sa = torch.cat([obs, action], 1)
        
        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)
        
        q2 = F.relu(self.fc4(sa))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)
        
        return q1, q2

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))
    
    def sample(self, batch_size):
        obs, action, reward, next_obs, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(obs), np.array(action), np.array(reward), np.array(next_obs), np.array(done)
    
    def __len__(self):
        return len(self.buffer)

class PDSAC:
    def __init__(self, num_agents, obs_dim, action_dim, hidden_dim=256, lr=3e-3, gamma=0.99, tau=0.01, alpha=0.2, auto_entropy_tuning=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Store number of discrete actions, but use continuous action_dim=1
        self.n_actions = action_dim * num_agents
        self.n_agent_actions = action_dim
        self.num_agents = num_agents
        continuous_action_dim = 1
        
        self.actor = Actor(num_agents, obs_dim, continuous_action_dim, hidden_dim).to(self.device)
        self.critic = Critic(num_agents, obs_dim, continuous_action_dim, hidden_dim).to(self.device)
        self.critic_target = Critic(num_agents, obs_dim, continuous_action_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.gamma = gamma
        self.tau = tau
        
        # Automatic entropy tuning
        self.auto_entropy_tuning = auto_entropy_tuning
        if self.auto_entropy_tuning:
            self.target_entropy = -continuous_action_dim  # Heuristic value
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = alpha
        
        self.action_dim = continuous_action_dim
        
    def select_action(self, obs, evaluate=False):
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        if evaluate:
            _, _, continuous_action = self.actor.sample(obs)
        else:
            continuous_action, _, _ = self.actor.sample(obs)
        # Convert continuous action to discrete
        continuous_action = continuous_action.detach().cpu().numpy()[0]
        discrete_action = int((continuous_action[0] + 1) / 2 * (self.n_actions - 1))
        discrete_action = np.clip(discrete_action, 0, self.n_actions - 1)
        discrete_actions = []
        for i in range(self.num_agents):
            agent_action = discrete_action % self.n_agent_actions
            discrete_actions.append(agent_action)
            discrete_action = discrete_action // self.n_agent_actions
        return discrete_actions
    
    def update(self, replay_buffer, batch_size):
        obs, action, reward, next_obs, done = replay_buffer.sample(batch_size)
        
        obs = torch.FloatTensor(obs).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        # Convert discrete actions to continuous for the actor-critic
        # action_continuous = torch.FloatTensor([[(a / (self.n_actions - 1)) * 2 - 1] for a in action]).to(self.device)
        action_combined = torch.ones(batch_size).to(self.device)
        for i in range(self.num_agents):
            action_combined *= (action[:, i] * (self.n_agent_actions ** i))
        action_continuous = torch.FloatTensor([[(a / (self.n_actions - 1)) * 2 - 1] for a in action_combined]).to(self.device)

        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_obs = torch.FloatTensor(next_obs).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)
        
        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_obs)
            target_q1, target_q2 = self.critic_target(next_obs, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = reward + (1 - done) * self.gamma * target_q
        
        current_q1, current_q2 = self.critic(obs, action_continuous)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        new_action, log_prob, _ = self.actor.sample(obs)
        q1_new, q2_new = self.critic(obs, new_action)
        q_new = torch.min(q1_new, q2_new)
        
        # Update alpha if automatic entropy tuning is enabled
        if self.auto_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp().item()
            alpha_loss_value = alpha_loss.item()
        else:
            alpha_loss_value = 0
        
        actor_loss = (self.alpha * log_prob - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        # Calculate average Q values for monitoring
        avg_q1 = current_q1.mean().item()
        avg_q2 = current_q2.mean().item()
        avg_q = (avg_q1 + avg_q2) / 2
        
        # Calculate entropy
        entropy = -log_prob.mean().item()
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'avg_q': avg_q,
            'avg_q1': avg_q1,
            'avg_q2': avg_q2,
            'alpha': self.alpha,
            'alpha_loss': alpha_loss_value,
            'entropy': entropy,
            'reward': reward.mean().item()
        }