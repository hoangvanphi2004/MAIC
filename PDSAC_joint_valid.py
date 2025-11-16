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
        self.num_agents = num_agents
        self.action_dim = action_dim
        
        # Input: all agents' observations concatenated
        self.fc1 = nn.Linear(num_agents * obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Output: single continuous action for joint action space
        self.mean = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Linear(hidden_dim, 1)
        
    def forward(self, obs):
        # obs shape: (batch, num_agents, obs_dim)
        # Flatten to (batch, num_agents * obs_dim)
        if obs.dim() == 3:
            obs = obs.reshape(obs.size(0), -1)
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)  # (batch, 1)
        log_std = self.log_std(x)  # (batch, 1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std
    
    def sample(self, obs, epsilon=1e-6):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)  # (batch, 1)
        
        log_prob = normal.log_prob(z)
        log_prob -= torch.log(1 - action.pow(2) + epsilon)
        # Already (batch, 1)
        
        mean = torch.tanh(mean)
        return action, log_prob, mean

class Critic(nn.Module):
    def __init__(self, num_agents, obs_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.num_agents = num_agents
        # Q1 - Input: all agents' observations + single joint action
        self.fc1 = nn.Linear(num_agents * obs_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        # Q2
        self.fc4 = nn.Linear(num_agents * obs_dim + action_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, 1)
        
    def forward(self, obs, action):
        # obs shape: (batch, num_agents, obs_dim)
        # action shape: (batch, 1) - single joint action
        # Flatten obs to (batch, num_agents * obs_dim)
        if obs.dim() == 3:
            obs = obs.reshape(obs.size(0), -1)
        
        sa = torch.cat([obs, action], 1)
        
        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)
        
        q2 = F.relu(self.fc4(sa))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)
        
        return q1, q2

class ReplayBufferPDSAC:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, obs, action, reward, next_obs, done):
        """Store multi-agent transition
        obs: (num_agents, obs_dim)
        action: list of num_agents actions or joint action index
        reward: scalar (team reward)
        next_obs: (num_agents, obs_dim)
        done: bool
        """
        self.buffer.append((obs, action, reward, next_obs, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, action, reward, next_obs, done = zip(*batch)
        # Convert to numpy arrays with proper shapes
        obs = np.array(obs)  # (batch, num_agents, obs_dim)
        # action is list of individual actions, convert to joint action indices
        action = np.array(action)  # (batch, num_agents) or (batch,)
        reward = np.array(reward)  # (batch,)
        next_obs = np.array(next_obs)  # (batch, num_agents, obs_dim)
        done = np.array(done)  # (batch,)
        return obs, action, reward, next_obs, done
    
    def __len__(self):
        return len(self.buffer)

class PDSACReinforceAgent:
    def __init__(self, num_agents, obs_dim, action_dim, hidden_dim=128, lr=3e-3, gamma=0.99, tau=0.01, alpha=0.2, auto_entropy_tuning=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Store number of discrete actions, but use continuous action_dim=1 per agent
        self.n_actions = action_dim
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
        """Select actions for all agents using joint action space
        obs: (num_agents, obs_dim)
        returns: list of discrete actions for each agent
        """
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)  # (1, num_agents, obs_dim)
        if evaluate:
            _, _, continuous_action = self.actor.sample(obs)
        else:
            continuous_action, _, _ = self.actor.sample(obs)
        # Convert single continuous action to joint discrete action index
        # continuous_action shape: (1, 1)
        continuous_action = continuous_action.detach().cpu().numpy()[0, 0]  # scalar
        
        # Map continuous action [-1, 1] to joint discrete action space [0, action_dim^num_agents - 1]
        joint_action_space_size = self.n_actions ** self.num_agents
        joint_action_idx = int((continuous_action + 1) / 2 * (joint_action_space_size - 1))
        joint_action_idx = np.clip(joint_action_idx, 0, joint_action_space_size - 1)
        
        # Decompose joint action index to individual agent actions
        discrete_actions = []
        base = joint_action_idx
        for _ in range(self.num_agents):
            discrete_actions.append(base % self.n_actions)
            base //= self.n_actions
        
        return discrete_actions
    
    def update(self, replay_buffer, batch_size):
        obs, action, reward, next_obs, done = replay_buffer.sample(batch_size)
        
        # obs: (batch, num_agents, obs_dim)
        # action: (batch, num_agents) - individual actions
        obs = torch.FloatTensor(obs).to(self.device)
        
        # Convert individual actions to joint action indices, then to continuous
        joint_action_space_size = self.n_actions ** self.num_agents
        joint_action_indices = []
        for agent_actions in action:
            # Convert list of agent actions to joint action index
            joint_idx = 0
            for i, a in enumerate(agent_actions):
                joint_idx += a * (self.n_actions ** i)
            joint_action_indices.append(joint_idx)
        
        # Convert joint action indices to continuous [-1, 1]
        action_continuous = torch.FloatTensor(
            [[(idx / (joint_action_space_size - 1)) * 2 - 1] for idx in joint_action_indices]
        ).to(self.device)  # (batch, 1)
        
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
        
        # Calculate entropy from log_prob
        entropy = -log_prob.mean().item()
        
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
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'avg_q': avg_q,
            'avg_q1': avg_q1,
            'avg_q2': avg_q2,
            'q_new': q_new.mean().item(),
            'alpha': self.alpha,
            'alpha_loss': alpha_loss_value,
            'target_q': target_q.mean().item(),
            'entropy': entropy,
            'log_prob': log_prob.mean().item(),
            'reward': reward.mean().item()
        }