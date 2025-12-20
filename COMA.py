import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import gymnasium as gym
from collections import deque
import random
import matplotlib.pyplot as plt


class ReplayBuffer:
    """Stores transitions for batch training"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (np.array(state), np.array(action), np.array(reward),
                np.array(next_state), np.array(done))

    def __len__(self):
        return len(self.buffer)


class EpisodeMemory:
    """Stores trajectory for COMA policy gradient"""
    def __init__(self, num_agents=1):
        self.num_agents = num_agents
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []

    def push(self, state, action, reward, log_prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(np.array(reward))
        self.log_probs.append(np.array(log_prob))

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []

    def __len__(self):
        return len(self.rewards)


class PolicyNet(nn.Module):
    """Decentralized actor network for each agent"""
    def __init__(self, state_shape, action_dim, hidden_dim=128):
        super(PolicyNet, self).__init__()
        self.conv1 = nn.Conv2d(state_shape[2], 16, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)

        def conv2d_size_out(size, kernel_size=4, stride=2, padding=1):
            return (size + 2 * padding - kernel_size) // stride + 1

        h = conv2d_size_out(conv2d_size_out(state_shape[0]))
        w = conv2d_size_out(conv2d_size_out(state_shape[1]))
        linear_input_size = h * w * 32

        self.fc1 = nn.Linear(linear_input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = state / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        action_probs = F.softmax(self.fc2(x), dim=-1)
        return action_probs

    def sample_action(self, state):
        action_probs = self.forward(state)
        action_probs = action_probs + 1e-8
        action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, action_probs


class ValueNet(nn.Module):
    """Centralized critic network for COMA"""
    def __init__(self, num_agents, state_shape, action_dim_per_agent, hidden_dim=256):
        super(ValueNet, self).__init__()
        self.num_agents = num_agents
        self.action_dim = action_dim_per_agent
        
        # Conv layers to encode state
        self.conv1 = nn.Conv2d(state_shape[2], 16, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)

        def conv2d_size_out(size, kernel_size=4, stride=2, padding=1):
            return (size + 2 * padding - kernel_size) // stride + 1

        h = conv2d_size_out(conv2d_size_out(state_shape[0]))
        w = conv2d_size_out(conv2d_size_out(state_shape[1]))
        embed_size = h * w * 32

        # Joint input: all agent embeddings + all agent actions (one-hot)
        joint_input_size = num_agents * (embed_size + action_dim_per_agent)

        self.fc1 = nn.Linear(joint_input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def encode_agent_state(self, agent_state):
        """Encode single agent's state observation"""
        x = agent_state / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        return x

    def forward(self, states, actions):
        """
        states: (batch_size, num_agents, C, H, W)
        actions: (batch_size, num_agents) or one-hot (batch_size, num_agents, action_dim)
        """
        batch_size = states.size(0)
        
        # Encode all agent states
        embeds = []
        for i in range(self.num_agents):
            agent_state = states[:, i]
            embed = self.encode_agent_state(agent_state)
            embeds.append(embed)
        
        embeds = torch.cat(embeds, dim=1)
        
        # Convert actions to one-hot if needed
        if actions.dim() == 2:
            actions_oh = F.one_hot(actions.long(), num_classes=self.action_dim).float()
        else:
            actions_oh = actions.float()
        
        actions_flat = actions_oh.view(batch_size, -1)
        joint = torch.cat([embeds, actions_flat], dim=1)
        
        x = F.relu(self.fc1(joint))
        x = F.relu(self.fc2(x))
        q_value = self.fc_out(x)
        
        return q_value



class COMA:
    """Counterfactual Multi-Agent Policy Gradient"""
    def __init__(self, state_shape, action_dim, num_agents=2, hidden_dim=256, lr=3e-4, 
                 gamma=0.99, tau=0.01):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.gamma = gamma
        self.tau = tau
        self.n_actions = action_dim
        self.num_agents = num_agents
        
        # Decentralized policy networks for each agent
        self.policies = [PolicyNet(state_shape, action_dim, hidden_dim).to(self.device)
                        for _ in range(num_agents)]
        self.policy_optimizer = optim.Adam(
            [p for policy in self.policies for p in policy.parameters()], 
            lr=lr
        )
        
        # Centralized critic (value network) - dùng chung cho cả Q và counterfactual baseline
        self.value_net = ValueNet(num_agents, state_shape, action_dim, hidden_dim).to(self.device)
        self.value_net_target = ValueNet(num_agents, state_shape, action_dim, hidden_dim).to(self.device)
        self.value_net_target.load_state_dict(self.value_net.state_dict())
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)

    def select_action(self, state, evaluate=False):
        """Select actions for all agents"""
        with torch.no_grad():
            state_arr = np.array(state)
            if state_arr.ndim == 3:
                state_arr = state_arr[np.newaxis, ...]
            
            actions = []
            log_probs = []
            
            for i, policy in enumerate(self.policies):
                s = torch.FloatTensor(state_arr[i]).permute(2, 0, 1).unsqueeze(0).to(self.device)
                if evaluate:
                    action_probs = policy(s)
                    action = torch.argmax(action_probs, dim=-1)
                    actions.append(int(action.item()))
                    log_probs.append(None)
                else:
                    action, log_prob, _ = policy.sample_action(s)
                    actions.append(int(action.item()))
                    log_probs.append(float(log_prob.item()))
            
            return actions, log_probs

    def update_value_net(self, replay_buffer, batch_size=64):
        """Update centralized value network using TD"""
        if len(replay_buffer) < batch_size:
            return {}
        
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        
        state = torch.FloatTensor(state).permute(0, 1, 4, 2, 3).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(next_state).permute(0, 1, 4, 2, 3).to(self.device)
        done = torch.FloatTensor(done).to(self.device)
        
        # Get mean reward across agents
        if reward.dim() == 2:
            reward_scalar = reward.mean(dim=1, keepdim=True)
        else:
            reward_scalar = reward.unsqueeze(1)
        
        # Target Q-value using next actions
        with torch.no_grad():
            next_actions = []
            for i, policy in enumerate(self.policies):
                ns_i = next_state[:, i]
                action_i, _, _ = policy.sample_action(ns_i)
                next_actions.append(action_i.unsqueeze(1))
            
            next_actions_cat = torch.cat(next_actions, dim=1)
            next_q = self.value_net_target(next_state, next_actions_cat)
            target_q = reward_scalar + (1 - done.unsqueeze(1)) * self.gamma * next_q
        
        # Compute current Q-value
        current_q = self.value_net(state, action)
        value_loss = F.mse_loss(current_q, target_q)
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0)
        self.value_optimizer.step()
        
        # Soft update target network
        for target_param, param in zip(self.value_net_target.parameters(), self.value_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {'value_loss': value_loss.item()}

    def compute_counterfactual_baseline(self, state, action, agent_i):
        """
        Compute counterfactual baseline: Σ_{a'_i} π_i(a'_i | o_i) * Q(s, u^{a'_i})
        Marginalize over all possible actions of agent i weighted by policy
        """
        batch_size = state.size(0)
        state_i = state[:, agent_i]
        
        # Get policy distribution for agent i
        action_probs = self.policies[agent_i](state_i)  # (batch, n_actions)
        
        # Compute Q for all possible actions of agent i
        q_values = []
        for a_prime in range(self.n_actions):
            # Create modified action with a_prime for agent i
            actions_counterfactual = action.clone()
            actions_counterfactual[:, agent_i] = a_prime
            
            # Compute Q(s, u^{a_prime})
            q_a_prime = self.value_net(state, actions_counterfactual)
            q_values.append(q_a_prime)
        
        # Stack Q-values: (batch, n_actions, 1)
        q_values = torch.stack(q_values, dim=1)
        
        # Weighted sum: Σ π(a') * Q(s, u^{a'})
        # action_probs: (batch, n_actions), q_values: (batch, n_actions, 1)
        baseline = (action_probs.unsqueeze(-1) * q_values).sum(dim=1)  # (batch, 1)
        
        return baseline

    def update_policies(self, episode_memory):
        """Update policies using COMA counterfactual advantage"""
        if len(episode_memory) == 0:
            return {}
        
        states_np = np.array(episode_memory.states)
        actions_np = np.array(episode_memory.actions)
        # Removed unused returns computation and related code
        
        states_tensor = torch.FloatTensor(states_np).permute(0, 1, 4, 2, 3).to(self.device)
        actions_tensor = torch.LongTensor(actions_np).to(self.device)
        
        total_policy_loss = 0.0
        
        # Update each agent's policy using counterfactual advantage
        for agent_i in range(self.num_agents):
            policy = self.policies[agent_i]
            state_i = states_tensor[:, agent_i]
            action_i = actions_tensor[:, agent_i]
            
            # Get action probabilities and log probs
            action_probs = policy(state_i)
            dist = Categorical(action_probs)
            log_probs_i = dist.log_prob(action_i)
            
            # Compute Q-value for actual trajectory
            q_actual = self.value_net(states_tensor, actions_tensor)
            q_actual = q_actual.squeeze(-1)
            
            # Compute counterfactual baseline Q-value
            with torch.no_grad():
                q_counterfactual = self.compute_counterfactual_baseline(
                    states_tensor, actions_tensor, agent_i
                )
                q_counterfactual = q_counterfactual.squeeze(-1)
            
            # Counterfactual advantage (COMA paper): A_i = Q(s,u) - sum_{a'} pi(a'|o_i) Q(s, u^{-i}, a')
            advantage = q_actual - q_counterfactual
            
            # Policy gradient
            # Policy gradient with counterfactual advantage only (COMA)
            policy_loss_i = -(log_probs_i * advantage.detach()).mean()
            total_policy_loss += policy_loss_i
        
        self.policy_optimizer.zero_grad()
        total_policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for policy in self.policies for p in policy.parameters()], 1.0
        )
        self.policy_optimizer.step()
        
        return {'policy_loss': total_policy_loss.item()}

    def save(self, filename):
        torch.save({
            'policies': [p.state_dict() for p in self.policies],
            'value_net': self.value_net.state_dict(),
            'value_net_target': self.value_net_target.state_dict(),
        }, filename)

    def load(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        
        policies_state = checkpoint.get('policies', None)
        if policies_state is not None:
            for p, st in zip(self.policies, policies_state):
                p.load_state_dict(st)
        
        if 'value_net' in checkpoint:
            self.value_net.load_state_dict(checkpoint['value_net'])
        if 'value_net_target' in checkpoint:
            self.value_net_target.load_state_dict(checkpoint['value_net_target'])
