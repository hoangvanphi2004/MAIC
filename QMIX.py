import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random


def conv2d_size_out(size, kernel_size=4, stride=2, padding=1):
    return (size + 2 * padding - kernel_size) // stride + 1


class AgentQNet(nn.Module):
    def __init__(self, obs_shape, action_dim, hidden_dim=128):
        super().__init__()
        c = obs_shape[2]
        h = obs_shape[0]
        w = obs_shape[1]
        self.conv1 = nn.Conv2d(c, 16, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
        h_out = conv2d_size_out(conv2d_size_out(h))
        w_out = conv2d_size_out(conv2d_size_out(w))
        linear_input_size = h_out * w_out * 32
        self.fc1 = nn.Linear(linear_input_size, hidden_dim)
        self.q_out = nn.Linear(hidden_dim, action_dim)

    def encode(self, x):
        x = x / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return x

    def forward(self, obs):
        # obs: (B, C, H, W)
        emb = self.encode(obs)
        q = self.q_out(emb)
        return q, emb


class MixingNetwork(nn.Module):
    def __init__(self, num_agents, state_embed_dim, monotonic=True):
        super().__init__()
        self.num_agents = num_agents
        self.monotonic = monotonic
        self.hyper_w = nn.Sequential(
            nn.Linear(state_embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_agents)
        )
        self.hyper_b = nn.Sequential(
            nn.Linear(state_embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, agent_qs, state_emb):
        # agent_qs: (B, N) selected Q-values per agent
        # state_emb: (B, D)
        w = self.hyper_w(state_emb)
        if self.monotonic:
            w = torch.nn.functional.softplus(w)
        b = self.hyper_b(state_emb)
        q_tot = (w * agent_qs).sum(dim=1, keepdim=True) + b
        return q_tot


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, actions, reward, next_obs, done):
        # obs: (N, H, W, C)
        # actions: list/np of length N (ints)
        self.buffer.append((np.array(obs), np.array(actions), float(reward), np.array(next_obs), bool(done)))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, reward, next_obs, done = zip(*batch)
        obs = np.array(obs)            # (B, N, H, W, C)
        actions = np.array(actions)    # (B, N)
        reward = np.array(reward)      # (B,)
        next_obs = np.array(next_obs)  # (B, N, H, W, C)
        done = np.array(done)          # (B,)
        return obs, actions, reward, next_obs, done

    def __len__(self):
        return len(self.buffer)


class QMIX:
    def __init__(self, obs_shape, action_dim, num_agents, hidden_dim=128, lr=1e-3, gamma=0.99, tau=0.02, share_agent=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_agents = num_agents
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.epsilon = 1.0
        self.eps_min = 0.05
        self.eps_decay = 3e-4
        self.share_agent = share_agent
        self._agent_hidden = hidden_dim

        # Agent Q networks (shared or per-agent)
        if share_agent:
            self.agent_net = AgentQNet(obs_shape, action_dim, hidden_dim).to(self.device)
            self.agent_target = AgentQNet(obs_shape, action_dim, hidden_dim).to(self.device)
            self.agent_target.load_state_dict(self.agent_net.state_dict())
            agent_params = list(self.agent_net.parameters())
        else:
            self.agent_nets = nn.ModuleList([AgentQNet(obs_shape, action_dim, hidden_dim).to(self.device) for _ in range(num_agents)])
            self.agent_targets = nn.ModuleList([AgentQNet(obs_shape, action_dim, hidden_dim).to(self.device) for _ in range(num_agents)])
            for t, n in zip(self.agent_targets, self.agent_nets):
                t.load_state_dict(n.state_dict())
            agent_params = [p for n in self.agent_nets for p in n.parameters()]

        # State embedding: concatenate agent embeddings
        state_embed_dim = self.num_agents * self._agent_hidden

        self.mixer = MixingNetwork(num_agents, state_embed_dim, monotonic=True).to(self.device)
        self.mixer_target = MixingNetwork(num_agents, state_embed_dim, monotonic=True).to(self.device)
        self.mixer_target.load_state_dict(self.mixer.state_dict())

        self.optimizer = torch.optim.Adam(agent_params + list(self.mixer.parameters()), lr=lr)

    def select_action(self, obs):
        # obs: (N, H, W, C)
        actions = []
        with torch.no_grad():
            for i in range(self.num_agents):
                o_i = torch.FloatTensor(obs[i]).permute(2, 0, 1).unsqueeze(0).to(self.device)
                if self.share_agent:
                    q_i, _ = self.agent_net(o_i)
                else:
                    q_i, _ = self.agent_nets[i](o_i)
                if random.random() < self.epsilon:
                    a = random.randrange(self.action_dim)
                else:
                    a = int(torch.argmax(q_i, dim=-1).item())
                actions.append(a)
        return actions

    def _state_embed(self, obs_tensor):
        # obs_tensor: (B, N, C, H, W)
        embs = []
        for i in range(self.num_agents):
            o_i = obs_tensor[:, i]
            if self.share_agent:
                _, emb_i = self.agent_net(o_i)
            else:
                _, emb_i = self.agent_nets[i](o_i)
            embs.append(emb_i)
        state_emb = torch.cat(embs, dim=1)
        return state_emb

    def update(self, replay_buffer, batch_size=64):
        if len(replay_buffer) < batch_size:
            return {}
        obs, actions, rewards, next_obs, dones = replay_buffer.sample(batch_size)

        obs_t = torch.FloatTensor(obs).permute(0, 1, 4, 2, 3).to(self.device)     # (B,N,C,H,W)
        next_obs_t = torch.FloatTensor(next_obs).permute(0, 1, 4, 2, 3).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)                      # (B,N)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)        # (B,1)
        dones_t = torch.FloatTensor(dones.astype(np.float32)).unsqueeze(1).to(self.device)

        # Current Q_tot
        agent_q_selected = []
        for i in range(self.num_agents):
            o_i = obs_t[:, i]
            if self.share_agent:
                q_i, _ = self.agent_net(o_i)
            else:
                q_i, _ = self.agent_nets[i](o_i)
            a_i = actions_t[:, i]
            q_i_a = q_i.gather(1, a_i.view(-1, 1)).view(-1)
            agent_q_selected.append(q_i_a)
        agent_q_selected = torch.stack(agent_q_selected, dim=1)  # (B,N)
        state_emb = self._state_embed(obs_t)
        q_tot = self.mixer(agent_q_selected, state_emb)          # (B,1)

        # Target Q_tot
        with torch.no_grad():
            agent_q_next_max = []
            for i in range(self.num_agents):
                no_i = next_obs_t[:, i]
                if self.share_agent:
                    q_next_i, _ = self.agent_target(no_i)
                else:
                    q_next_i, _ = self.agent_targets[i](no_i)
                max_q_i = q_next_i.max(dim=1)[0]
                agent_q_next_max.append(max_q_i)
            agent_q_next_max = torch.stack(agent_q_next_max, dim=1)  # (B,N)
            next_state_emb = self._state_embed(next_obs_t)
            q_tot_target = self.mixer_target(agent_q_next_max, next_state_emb)     # (B,1)
            y = rewards_t + (1.0 - dones_t) * self.gamma * q_tot_target

        loss = F.mse_loss(q_tot, y)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.mixer.parameters()), 10.0)
        if self.share_agent:
            torch.nn.utils.clip_grad_norm_(list(self.agent_net.parameters()), 10.0)
        else:
            torch.nn.utils.clip_grad_norm_([p for n in self.agent_nets for p in n.parameters()], 10.0)
        self.optimizer.step()

        # Soft update targets
        if self.share_agent:
            for tp, p in zip(self.agent_target.parameters(), self.agent_net.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        else:
            for tnet, net in zip(self.agent_targets, self.agent_nets):
                for tp, p in zip(tnet.parameters(), net.parameters()):
                    tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        for tp, p in zip(self.mixer_target.parameters(), self.mixer.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        # Epsilon decay
        self.epsilon = max(self.eps_min, self.epsilon - self.eps_decay)

        return {
            'loss': float(loss.item()),
            'epsilon': float(self.epsilon),
            'q_tot': float(q_tot.mean().item())
        }

    def save(self, filename):
        state = {
            'share_agent': self.share_agent,
            'num_agents': self.num_agents,
            'action_dim': self.action_dim,
            'agent_net': (self.agent_net.state_dict() if self.share_agent else None),
            'agent_target': (self.agent_target.state_dict() if self.share_agent else None),
            'agent_nets': ([n.state_dict() for n in self.agent_nets] if not self.share_agent else None),
            'agent_targets': ([t.state_dict() for t in self.agent_targets] if not self.share_agent else None),
            'mixer': self.mixer.state_dict(),
            'mixer_target': self.mixer_target.state_dict()
        }
        torch.save(state, filename)

    def load(self, filename):
        chk = torch.load(filename)
        if chk.get('share_agent', True):
            self.agent_net.load_state_dict(chk['agent_net'])
            self.agent_target.load_state_dict(chk['agent_target'])
        else:
            for n, sd in zip(self.agent_nets, chk['agent_nets']):
                n.load_state_dict(sd)
            for t, sd in zip(self.agent_targets, chk['agent_targets']):
                t.load_state_dict(sd)
        self.mixer.load_state_dict(chk['mixer'])
        self.mixer_target.load_state_dict(chk['mixer_target'])
