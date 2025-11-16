import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
from collections import deque, namedtuple
import random
import matplotlib.pyplot as plt


# Simple replay buffer
Transition = namedtuple('Transition', ('obs', 'action', 'reward', 'next_obs', 'done'))

class ReplayBufferPDSAC:
    def __init__(self, capacity=1_000_000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs = torch.tensor(np.stack([b.obs for b in batch]), dtype=torch.float32)
        actions = torch.tensor(np.array([b.action for b in batch]), dtype=torch.long)
        rewards = torch.tensor([b.reward for b in batch], dtype=torch.float32).unsqueeze(1)
        next_obs = torch.tensor(np.stack([b.next_obs for b in batch]), dtype=torch.float32)
        dones = torch.tensor([b.done for b in batch], dtype=torch.float32).unsqueeze(1)
        return obs, actions, rewards, next_obs, dones

    def __len__(self):
        return len(self.buffer)


# Actor network: outputs logits for discrete actions
class DiscreteActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.logits = nn.Linear(hidden, action_dim)

    def forward(self, obs):
        x = self.net(obs)
        return self.logits(x)

    def get_dist(self, obs):
        logits = self.forward(obs)
        probs = F.softmax(logits, dim=-1)
        return torch.distributions.Categorical(probs)

class JointActor(nn.Module):
    def __init__(self, num_agents, obs_dim, action_dim, hidden=128):
        super().__init__()
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(num_agents * obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.logits = nn.Linear(hidden, action_dim ** num_agents)

    def forward(self, obs):
        # obs shape: (batch_size, num_agents, obs_dim)
        if(len(obs.shape) == 2):
            obs = obs.view(1, self.num_agents, self.obs_dim)
        batch_size = obs.size(0)
        # print("obs shape in JointActor:", obs.shape)
        x = obs.view(batch_size, -1)  # flatten to (batch_size, num_agents * obs_dim)
        x = self.net(x)
        logits = self.logits(x)
        logits = logits.view(batch_size, self.action_dim ** self.num_agents)  # reshape to (batch_size, action_dim ** num_agents)
        return logits

    def get_action(self, obs, device=None, return_probs=False):
        if not torch.is_tensor(obs):
            obs = torch.tensor(obs, dtype=torch.float32, device=device)
        else:
            obs = obs.to(device)

        logits = self.forward(obs)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        base = dist.sample()
        log_prob = dist.log_prob(base).unsqueeze(1)
        base_copy = base.clone()
        actions = []
        for _ in range(self.num_agents):
            actions.append(base_copy % self.action_dim)
            base_copy //= self.action_dim
        actions = torch.stack(actions, dim=1).long().to(obs.device)
        
        if return_probs:
            return actions, log_prob, probs
        return actions, log_prob

# Critic that takes (obs, action_onehot) and returns scalar Q
class DiscreteCritic(nn.Module):
    def __init__(self, num_agents, obs_dim, action_dim, hidden=128):
        super().__init__()
        inp = (obs_dim + action_dim) * num_agents
        self.q1 = nn.Sequential(nn.Linear(inp, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.q2 = nn.Sequential(nn.Linear(inp, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, obs, action_onehots):
        action_onehots = action_onehots.view(action_onehots.size(0), -1)
        obs = obs.view(obs.size(0), -1)
        x = torch.cat([obs, action_onehots], dim=1)
        return self.q1(x), self.q2(x)


# Value baseline (obs -> scalar)
class ValueNet(nn.Module):
    def __init__(self, num_agents, obs_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(num_agents * obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, obs):
        obs = obs.view(obs.size(0), -1)
        return self.net(obs)


# Agent combining REINFORCE actor + double-Q critics + value baseline
class PDSACReinforceAgent:
    def __init__(self, num_agents, obs_dim, action_dim, device='cpu'):
        self.device = device
        self.action_dim = action_dim
        self.num_agents = num_agents

        self.actor = JointActor(num_agents, obs_dim, action_dim).to(device)
        self.critic = DiscreteCritic(num_agents, obs_dim, action_dim).to(device)
        self.critic_target = DiscreteCritic(num_agents, obs_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.value = ValueNet(num_agents, obs_dim).to(device)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=3e-3)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=3e-3)
        self.value_opt = optim.Adam(self.value.parameters(), lr=3e-3)

        # entropy temperature alpha (learned)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=3e-3)
        # target entropy for the joint action distribution (num joint actions = action_dim ** num_agents)
        self.target_entropy = -np.log(action_dim ** num_agents) 

        self.gamma = 0.99
        self.tau = 0.01

    def select_action(self, obs, eval_mode=False):
        st = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        logits = self.actor(st)
        probs = F.softmax(logits, dim=-1)
        if eval_mode:
            a = probs.argmax(dim=-1)
        else:
            a = torch.distributions.Categorical(probs).sample()
        return int(a.cpu().numpy()[0])

    def update(self, buffer: ReplayBufferPDSAC, batch_size=64):
        obs, action, reward, next_obs, done = buffer.sample(batch_size)
        obs = obs.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_obs = next_obs.to(self.device)
        done = done.to(self.device)

        # --- critic target (sample next action) ---
        with torch.no_grad():
            next_actions, next_logp = self.actor.get_action(next_obs)
            next_a_onehot = F.one_hot(next_actions, num_classes=self.action_dim).float().to(self.device)
            tq1, tq2 = self.critic_target(next_obs, next_a_onehot)
            tq = torch.min(tq1, tq2) - torch.exp(self.log_alpha) * next_logp
            target_q = reward + (1 - done) * self.gamma * tq

        a_int = action.squeeze(1).long()
        a_onehot = F.one_hot(a_int, num_classes=self.action_dim).float().to(self.device)
        q1, q2 = self.critic(obs, a_onehot)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_opt.zero_grad(); critic_loss.backward(); self.critic_opt.step()

        sampled_a, sampled_logp, sampled_probs = self.actor.get_action(obs, return_probs=True)
        sampled_onehot = F.one_hot(sampled_a, num_classes=self.action_dim).float().to(self.device)

        # alpha update (use sampled entropy)
        alpha_loss = -(self.log_alpha * (-sampled_logp + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        q1_sa, q2_sa = self.critic(obs, sampled_onehot)
        q_sa = torch.min(q1_sa, q2_sa)

        # --- single-sample value baseline: use the sampled action's Q(s,a) as target
        # Use the same sampled action we drew for the actor update (sampled_a)
        with torch.no_grad():
            V_target = q_sa.detach()

        # train value baseline
        v_pred = self.value(obs)
        value_loss = F.mse_loss(v_pred, V_target)
        self.value_opt.zero_grad(); value_loss.backward(); self.value_opt.step()

        advantage = q_sa - v_pred.detach()

        # actor loss (REINFORCE with baseline) + entropy regularization
        # Use a sampled estimate for the entropy (high-variance but true sampling)
        actor_loss = -(sampled_logp * advantage.detach()).mean() - torch.exp(self.log_alpha) * -sampled_logp.mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # soft update critic target
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        # Calculate entropy and max probability for diagnostics
        with torch.no_grad():
            entropy = -(sampled_probs * torch.log(sampled_probs + 1e-8)).sum(dim=-1).mean()
            max_prob = sampled_probs.max(dim=-1)[0].mean()

        return {
            'critic_loss': critic_loss.item(),
            'value_loss': value_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.log_alpha.exp().item(),
            'entropy': entropy.item(),
            'max_prob': max_prob.item()
        }