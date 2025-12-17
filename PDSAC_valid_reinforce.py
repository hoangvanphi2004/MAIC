import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import gymnasium as gym
from collections import deque
import random
import minigrid
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
import matplotlib.pyplot as plt


class ReplayBuffer:
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
        log_prob = dist.log_prob(action)
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

        joint_input_size = num_agents * (embed_size + action_dim_per_agent)

        self.q1_fc1 = nn.Linear(joint_input_size, hidden_dim)
        self.q1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_out = nn.Linear(hidden_dim, 1)

        self.q2_fc1 = nn.Linear(joint_input_size, hidden_dim)
        self.q2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_out = nn.Linear(hidden_dim, 1)

    def encode_agent(self, x):
        x = x / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        return x

    def forward(self, state, actions):
        B = state.size(0)
        embeds = []
        for i in range(self.num_agents):
            agent_img = state[:, i]
            emb = self.encode_agent(agent_img)
            embeds.append(emb)
        embeds = torch.cat(embeds, dim=1)

        if actions.dim() == 2:
            A = self._action_dim_per_agent
            actions_oh = F.one_hot(actions.long(), num_classes=A).float()
        elif actions.dim() == 3:
            actions_oh = actions.float()
        else:
            raise ValueError(f"Unexpected actions shape: {actions.shape}")

        actions_flat = actions_oh.view(B, -1)
        joint = torch.cat([embeds, actions_flat], dim=1)

        x1 = F.relu(self.q1_fc1(joint))
        x1 = F.relu(self.q1_fc2(x1))
        q1 = self.q1_out(x1)

        x2 = F.relu(self.q2_fc1(joint))
        x2 = F.relu(self.q2_fc2(x2))
        q2 = self.q2_out(x2)

        return q1, q2


class SAC_REINFORCE:
    def __init__(self, state_shape, action_dim, num_agents=2, hidden_dim=128, lr=3e-4, gamma=0.99,
                 tau=0.01, alpha=0.01, auto_entropy_tuning=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gamma = gamma
        self.tau = tau
        self.n_actions = action_dim
        self.num_agents = num_agents

        self.actors = [Actor(state_shape, action_dim, hidden_dim).to(self.device)
                       for _ in range(self.num_agents)]
        self.actor_optimizer = optim.Adam([p for a in self.actors for p in a.parameters()], lr=lr)

        self.critic = Critic(self.num_agents, state_shape, action_dim_per_agent=self.n_actions, hidden_dim=256).to(self.device)
        self.critic_target = Critic(self.num_agents, state_shape, action_dim_per_agent=self.n_actions, hidden_dim=256).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.auto_entropy_tuning = auto_entropy_tuning
        if self.auto_entropy_tuning:
            self.target_entropy = -float(self.num_agents * action_dim)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = alpha

    def select_action(self, state, evaluate=False):
        with torch.no_grad():
            state_arr = np.array(state)
            if state_arr.ndim == 3:
                state_arr = state_arr[np.newaxis, ...]
            actions = []
            log_probs = []
            for i, actor in enumerate(self.actors):
                s = torch.FloatTensor(state_arr[i]).permute(2, 0, 1).unsqueeze(0).to(self.device)
                if evaluate:
                    action_probs = actor(s)
                    action = torch.argmax(action_probs, dim=-1)
                    actions.append(int(action.item()))
                    log_probs.append(None)
                else:
                    action, log_prob, _ = actor.sample(s)
                    actions.append(int(action.item()))
                    log_probs.append(float(log_prob.item()))
            return actions, log_probs

    def compute_returns(self, rewards, gamma=0.99):
        arr = np.array(rewards)
        if arr.ndim == 1:
            R = 0.0
            returns = []
            for r in reversed(arr.tolist()):
                R = r + gamma * R
                returns.insert(0, R)
            returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
            if len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            return returns

        T, NA = arr.shape
        returns = np.zeros_like(arr, dtype=np.float32)
        for a in range(NA):
            R = 0.0
            col = arr[:, a].tolist()
            col_ret = []
            for r in reversed(col):
                R = r + gamma * R
                col_ret.insert(0, R)
            returns[:, a] = np.array(col_ret, dtype=np.float32)

        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        if T > 1:
            mean = returns.mean(dim=0, keepdim=True)
            std = returns.std(dim=0, keepdim=True) + 1e-8
            returns = (returns - mean) / std
        return returns

    def update_reinforce(self, episode_memory):
        if len(episode_memory) == 0:
            return {}
        rewards_seq = np.array(episode_memory.rewards)
        returns = self.compute_returns(rewards_seq, gamma=self.gamma)

        states_np = np.array(episode_memory.states)
        actions_np = np.array(episode_memory.actions)

        total_policy_loss = 0.0
        total_entropy = 0.0
        for a_i, actor in enumerate(self.actors):
            states_a = torch.FloatTensor(states_np[:, a_i]).permute(0, 3, 1, 2).to(self.device)
            actions_a = torch.LongTensor(actions_np[:, a_i]).to(self.device)
            action_probs = actor(states_a)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions_a)
            entropy = dist.entropy().mean()

            if returns.dim() == 1:
                returns_a = returns.to(self.device)
            else:
                returns_a = returns[:, a_i].to(self.device)
            alpha = self.alpha.detach() if isinstance(self.alpha, torch.Tensor) else self.alpha
            policy_loss_a = -(new_log_probs * returns_a).mean() - alpha * entropy
            total_policy_loss += policy_loss_a
            total_entropy += entropy.item()

        self.actor_optimizer.zero_grad()
        total_policy_loss.backward()
        torch.nn.utils.clip_grad_norm_([p for a in self.actors for p in a.parameters()], 1.0)
        self.actor_optimizer.step()

        rewards_arr = np.array(episode_memory.rewards)
        try:
            avg_return = float(np.sum(rewards_arr))
        except Exception:
            avg_return = float(rewards_arr)

        return {
            'policy_loss': total_policy_loss.item(),
            'entropy': total_entropy / float(self.num_agents),
            'avg_return': avg_return
        }

    def update_sac(self, replay_buffer, batch_size=64):
        if len(replay_buffer) < batch_size:
            return {}

        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).permute(0, 1, 4, 2, 3).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(next_state).permute(0, 1, 4, 2, 3).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        if reward.dim() == 2:
            reward_for_target = reward.mean(dim=1, keepdim=True)
        else:
            reward_for_target = reward.unsqueeze(1)

        with torch.no_grad():
            next_actions = []
            next_log_prob = 0
            for i, actor in enumerate(self.actors):
                ns_i = next_state[:, i]
                a_i, logp_i, _ = actor.sample(ns_i)
                next_actions.append(a_i.unsqueeze(1).long())
                next_log_prob = next_log_prob + logp_i

            next_actions_cat = torch.cat(next_actions, dim=1)
            next_actions_oh = F.one_hot(next_actions_cat.long(), num_classes=self.n_actions).float()

            next_q1_target, next_q2_target = self.critic_target(next_state, next_actions_oh)
            next_q_target = torch.min(next_q1_target, next_q2_target)

            target_q = next_q_target - self.alpha * next_log_prob.unsqueeze(1)
            target_q_value = reward_for_target + (1 - done.unsqueeze(1)) * self.gamma * target_q

        action_oh = F.one_hot(action.long(), num_classes=self.n_actions).float()

        q1, q2 = self.critic(state, action_oh)
        critic_loss = F.mse_loss(q1, target_q_value) + F.mse_loss(q2, target_q_value)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        alpha_info = {}
        if self.auto_entropy_tuning:
            with torch.no_grad():
                sampled_logp = 0
                for i, actor in enumerate(self.actors):
                    s_i = state[:, i]
                    probs_i = actor(s_i)
                    dist_i = Categorical(probs_i)
                    sampled_a_i = dist_i.sample()
                    sampled_logp = sampled_logp + dist_i.log_prob(sampled_a_i)

            alpha_loss = -(self.log_alpha * (sampled_logp + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp()
            alpha_info = {
                'alpha_loss': alpha_loss.item(),
                'alpha_value': self.alpha.item()
            }

        result = {
            'critic_loss': critic_loss.item(),
            'q_value': q1.mean().item()
        }
        result.update(alpha_info)
        return result

    def save(self, filename):
        torch.save({
            'actors': [a.state_dict() for a in self.actors],
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
        }, filename)

    def load(self, filename):
        checkpoint = torch.load(filename)
        actor_states = checkpoint.get('actors', None)
        if actor_states is not None:
            for a, st in zip(self.actors, actor_states):
                a.load_state_dict(st)
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])