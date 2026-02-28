import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class Actor(nn.Module):
    def __init__(self, state_shape, action_dim, hidden_dim=128):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
        def conv2d_size_out(size, kernel_size=4, stride=2, padding=1):
            return (size + 2 * padding - kernel_size) // stride + 1
        h = conv2d_size_out(conv2d_size_out(state_shape[0]))
        w = conv2d_size_out(conv2d_size_out(state_shape[1]))
        linear_input_size = h * w * 32
        self.fc1 = nn.Linear(linear_input_size, action_dim)
        
    def forward(self, state):
        x = state / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.reshape(x.size(0), -1)
        action_probs = F.softmax(self.fc1(x), dim=-1)
        return action_probs
        
    def sample(self, state):
        action_probs = self.forward(state)
        action_probs = action_probs + 1e-8
        action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = torch.log(action_probs.gather(1, action.unsqueeze(-1)).squeeze(-1))
        return action, log_prob, action_probs

class Critic(nn.Module):
    def __init__(self, num_agents, state_shape, action_dim_per_agent=1, hidden_dim=256):
        super(Critic, self).__init__()
        self.num_agents = num_agents
        self._action_dim_per_agent = action_dim_per_agent
        self.conv1 = nn.Conv2d(state_shape[2], 16, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
        def conv2d_size_out(size, kernel_size=4, stride=2, padding=1):
            return (size + 2 * padding - kernel_size) // stride + 1
        h = conv2d_size_out(conv2d_size_out(state_shape[0]))
        w = conv2d_size_out(conv2d_size_out(state_shape[1]))
        embed_size = h * w * 32
        joint_input_size = embed_size + num_agents * action_dim_per_agent
        
        self.q1_fc1 = nn.Linear(joint_input_size, hidden_dim)
        self.q1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_out = nn.Linear(hidden_dim, 1)
        
        self.q2_fc1 = nn.Linear(joint_input_size, hidden_dim)
        self.q2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_out = nn.Linear(hidden_dim, 1)
        
    def encode_state(self, x):
        x = x / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.reshape(x.size(0), -1)
        return x
        
    def forward(self, state, actions):
        B = state.size(0)
        embed = self.encode_state(state)
        
        if actions.dim() == 2:
            A = self._action_dim_per_agent
            actions_oh = F.one_hot(actions.long(), num_classes=A).float()
        elif actions.dim() == 3:
            actions_oh = actions.float()
        else:
            raise ValueError(f"Unexpected actions shape: {actions.shape}")
            
        actions_flat = actions_oh.view(B, -1)
        joint = torch.cat([embed, actions_flat], dim=1)
        
        x1 = F.relu(self.q1_fc1(joint))
        x1 = F.relu(self.q1_fc2(x1))
        q1 = self.q1_out(x1)
        
        x2 = F.relu(self.q2_fc1(joint))
        x2 = F.relu(self.q2_fc2(x2))
        q2 = self.q2_out(x2)
        return q1, q2
