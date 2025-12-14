import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque


class DQN(nn.Module):
    """Simple DQN for each agent using global state."""
    def __init__(self, input_dim, hidden_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class SimpleMixingNetwork(nn.Module):
    """Simple mixing network with monotonic constraint."""
    def __init__(self, num_agents, state_dim, hidden_dim=64):
        super(SimpleMixingNetwork, self).__init__()
        self.num_agents = num_agents
        self.hidden_dim = hidden_dim
        
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_agents * hidden_dim)
        )
        
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.hyper_b1 = nn.Linear(state_dim, hidden_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, q_values, state):
        """
        Args:
            q_values: [batch, num_agents] or [num_agents]
            state: [batch, state_dim] or [state_dim]
        Returns:
            mixed_q: [batch] or scalar
        """
        if q_values.dim() == 1:
            q_values = q_values.unsqueeze(0)
            state = state.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size = q_values.shape[0]
        
        w1 = torch.abs(self.hyper_w1(state))
        w1 = w1.view(batch_size, self.num_agents, self.hidden_dim)
        b1 = self.hyper_b1(state).view(batch_size, 1, self.hidden_dim)
        
        hidden = torch.relu(torch.bmm(q_values.unsqueeze(1), w1) + b1)
        
        w2 = torch.abs(self.hyper_w2(state))
        w2 = w2.view(batch_size, self.hidden_dim, 1)
        b2 = self.hyper_b2(state).view(batch_size, 1, 1)
        
        q_tot = torch.bmm(hidden, w2) + b2
        q_tot = q_tot.view(batch_size)
        
        if squeeze_output:
            q_tot = q_tot.squeeze(0)
        
        return q_tot


class SimpleQMIXNetwork(nn.Module):
    def __init__(self, num_agents, state_dim, observation_dim, action_dim):
        super(SimpleQMIXNetwork, self).__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.observation_dim = observation_dim
        
        self.agent_nets = nn.ModuleList([
            DQN(observation_dim, 64, action_dim) for _ in range(num_agents)
        ])
        
        self.mixing_net = SimpleMixingNetwork(num_agents, state_dim, hidden_dim=64)

    def forward(self, state, actions, observations, batch_size):
        """
        Args:
            state: [batch, state_dim] - global state for mixing
            actions: [batch, num_agents] - actions taken
            observations: [batch, num_agents, obs_dim] - individual observations
            batch_size: int
        Returns:
            q_tot: [batch]
        """
        q_values_list = []
        for i in range(self.num_agents):
            agent_obs = observations[:, i, :]
            q_vals = self.agent_nets[i](agent_obs)
            q_values_list.append(q_vals)
        
        all_q_values = torch.stack(q_values_list, dim=1)
        
        actions = actions.unsqueeze(-1)
        selected_q_values = all_q_values.gather(2, actions.long()).squeeze(-1)
        
        q_tot = self.mixing_net(selected_q_values, state)
        
        return q_tot

    def max_q_value(self, state, observations, batch_size):
        """Get maximum Q-values for target network."""
        q_values_list = []
        for i in range(self.num_agents):
            agent_obs = observations[:, i, :]
            q_vals = self.agent_nets[i](agent_obs)
            q_values_list.append(q_vals)
        
        all_q_values = torch.stack(q_values_list, dim=1)
        max_q_values, _ = all_q_values.max(dim=2)
        
        q_tot = self.mixing_net(max_q_values, state)
        
        return q_tot


class SimpleReplayBuffer:
    """Simple replay buffer for transitions (step-based, not episode-based)."""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Push a single transition."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample transitions and return batched tensors."""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        states = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32)
        actions = torch.tensor(np.array([b[1] for b in batch]), dtype=torch.int64)
        rewards = torch.tensor(np.array([b[2] for b in batch]), dtype=torch.float32)
        next_states = torch.tensor(np.array([b[3] for b in batch]), dtype=torch.float32)
        dones = torch.tensor(np.array([b[4] for b in batch]), dtype=torch.float32)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class QMIXAgent:
    def __init__(self, num_agents, state_dim, action_dim, observation_dim, lr=3e-4, num_episodes=50000):
        self.q_net = SimpleQMIXNetwork(num_agents, state_dim, observation_dim, action_dim)
        self.target_net = SimpleQMIXNetwork(num_agents, state_dim, observation_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = SimpleReplayBuffer(capacity=int(5e5))

        self.epsilon = 1.0
        self.epsilon_decay = (1 - 0.05) / (num_episodes * 0.9)
        self.epsilon_min = 0.05
        self.gamma = 0.99
        self.action_dim = action_dim
        self.batch_size = 32
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.observation_dim = observation_dim
        self.tau = 0.005

    def select_action(self, available_actions, observations):
        """
        Args:
            available_actions: [num_agents, action_dim]
            observations: [num_agents, obs_dim]
        Returns:
            actions: [num_agents]
        """
        with torch.no_grad():
            actions = []
            for i in range(self.num_agents):
                obs_tensor = torch.tensor(observations[i], dtype=torch.float32).unsqueeze(0)
                q_values = self.q_net.agent_nets[i](obs_tensor)
                
                avail_indices = [a for a, valid in enumerate(available_actions[i]) if valid]
                if random.random() < self.epsilon:
                    actions.append(int(random.choice(avail_indices)))
                else:
                    q_values[0, np.array(available_actions[i]) == 0] = float('-inf')
                    action = q_values.argmax().item()
                    actions.append(action)
            
            return actions

    def train(self):
        if len(self.buffer) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        single_obs_dim = self.state_dim // self.num_agents
        obs_reshaped = states.reshape(self.batch_size, self.num_agents, single_obs_dim)
        next_obs_reshaped = next_states.reshape(self.batch_size, self.num_agents, single_obs_dim)
        
        agent_ids = torch.eye(self.num_agents).unsqueeze(0).repeat(self.batch_size, 1, 1)
        observations = torch.cat([obs_reshaped, agent_ids], dim=-1)
        next_observations = torch.cat([next_obs_reshaped, agent_ids], dim=-1)
        
        q_tot = self.q_net(states, actions, observations, self.batch_size)
        
        with torch.no_grad():
            max_next_q = self.target_net.max_q_value(next_states, next_observations, self.batch_size)
            targets = rewards + self.gamma * (1 - dones) * max_next_q
        
        loss = torch.nn.functional.mse_loss(q_tot, targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return {'loss': loss.item(), 'q_value': q_tot.mean().item()}

    def update_target(self, soft=False):
        if soft:
            for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        else:
            self.target_net.load_state_dict(self.q_net.state_dict())
