import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def small_initialize_module(module, std=1e-8):
	for sub_module in module.modules():
		if isinstance(sub_module, (nn.Linear, nn.Conv2d)):
			nn.init.normal_(sub_module.weight, mean=0.0, std=std)
			if sub_module.bias is not None:
				nn.init.zeros_(sub_module.bias)


class AgentQNetwork(nn.Module):
	def __init__(self, obs_shape, action_dim, hidden_dim=128):
		super().__init__()
		self.is_image_obs = len(obs_shape) == 3
		self.action_dim = action_dim
		if self.is_image_obs:
			self.conv1 = nn.Conv2d(obs_shape[2], 16, kernel_size=4, stride=2, padding=1)
			self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)

			def conv2d_size_out(size, kernel_size=4, stride=2, padding=1):
				return (size + 2 * padding - kernel_size) // stride + 1

			h = conv2d_size_out(conv2d_size_out(obs_shape[0]))
			w = conv2d_size_out(conv2d_size_out(obs_shape[1]))
			linear_input_size = h * w * 32
			self.fc1 = nn.Linear(linear_input_size, hidden_dim)
			self.fc2 = nn.Linear(hidden_dim, action_dim)
		else:
			obs_dim = int(np.prod(obs_shape))
			self.fc1 = nn.Linear(obs_dim, hidden_dim)
			self.fc2 = nn.Linear(hidden_dim, hidden_dim)
			self.out = nn.Linear(hidden_dim, action_dim)

	def forward(self, obs):
		if self.is_image_obs:
			x = F.relu(self.conv1(obs))
			x = F.relu(self.conv2(x))
			x = x.reshape(x.size(0), -1)
			x = F.relu(self.fc1(x))
			return self.fc2(x)
		x = obs.reshape(obs.size(0), -1)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		return self.out(x)


class MixingNetwork(nn.Module):
	def __init__(self, num_agents, state_shape, embed_dim=128):
		super().__init__()
		self.num_agents = num_agents
		self.state_dim = int(np.prod(state_shape))
		self.embed_dim = embed_dim

		self.hyper_w1 = nn.Linear(self.state_dim, num_agents * embed_dim)
		self.hyper_b1 = nn.Linear(self.state_dim, embed_dim)
		self.hyper_w2 = nn.Linear(self.state_dim, embed_dim)
		self.hyper_b2 = nn.Sequential(
			nn.Linear(self.state_dim, embed_dim),
			nn.ReLU(),
			nn.Linear(embed_dim, 1),
		)

	def forward(self, agent_qs, states):
		batch_size = agent_qs.size(0)
		states_flat = states.reshape(batch_size, -1)
		w1 = torch.abs(self.hyper_w1(states_flat)).view(batch_size, self.num_agents, self.embed_dim)
		b1 = self.hyper_b1(states_flat).view(batch_size, 1, self.embed_dim)
		hidden = F.elu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)
		w2 = torch.abs(self.hyper_w2(states_flat)).view(batch_size, self.embed_dim, 1)
		b2 = self.hyper_b2(states_flat).view(batch_size, 1, 1)
		q_tot = torch.bmm(hidden, w2) + b2
		return q_tot.squeeze(-1)


class QMIX:
	def __init__(self, obs_shape, state_shape, action_dim, num_agents=2, hidden_dim=128, mixer_hidden_dim=128,
				 lr=3e-4, gamma=0.99, tau=0.01, epsilon_start=1.0, epsilon_end=0.05,
				 epsilon_decay_steps=50000, grad_clip=10.0):
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.num_agents = int(num_agents)
		self.action_dim = int(action_dim)
		self.gamma = float(gamma)
		self.tau = float(tau)
		self.epsilon_start = float(epsilon_start)
		self.epsilon_end = float(epsilon_end)
		self.epsilon_decay_steps = max(1, int(epsilon_decay_steps))
		self.grad_clip = float(grad_clip)
		self.obs_ndim = len(obs_shape)
		self.obs_is_image = len(obs_shape) == 3
		self.state_is_image = len(state_shape) == 3

		self.agent_networks = [
			AgentQNetwork(obs_shape, action_dim, hidden_dim=hidden_dim).to(self.device)
			for _ in range(self.num_agents)
		]
		for network in self.agent_networks:
			small_initialize_module(network)
		self.target_agent_networks = [copy.deepcopy(net).to(self.device) for net in self.agent_networks]
		self.mixer = MixingNetwork(self.num_agents, state_shape, embed_dim=mixer_hidden_dim).to(self.device)
		small_initialize_module(self.mixer)
		self.target_mixer = copy.deepcopy(self.mixer).to(self.device)

		params = [p for net in self.agent_networks for p in net.parameters()]
		params.extend(self.mixer.parameters())
		self.optimizer = optim.Adam(params, lr=lr)

	def _epsilon_by_step(self, total_steps):
		if total_steps is None:
			return self.epsilon_end
		progress = min(float(total_steps) / float(self.epsilon_decay_steps), 1.0)
		return self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)

	def _prepare_obs_tensor(self, obs):
		obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
		if obs_tensor.dim() == self.obs_ndim + 1:
			obs_tensor = obs_tensor.unsqueeze(0)
		if self.obs_is_image:
			obs_tensor = obs_tensor.permute(0, 1, 4, 2, 3)
		return obs_tensor

	def select_action(self, obs, evaluate=False, total_steps=None):
		obs_tensor = self._prepare_obs_tensor(obs)
		epsilon = 0.0 if evaluate else self._epsilon_by_step(total_steps)
		actions = []
		with torch.no_grad():
			for agent_idx, network in enumerate(self.agent_networks):
				q_values = network(obs_tensor[:, agent_idx])
				greedy_action = int(torch.argmax(q_values, dim=-1).item())
				if (not evaluate) and np.random.rand() < epsilon:
					actions.append(int(np.random.randint(self.action_dim)))
				else:
					actions.append(greedy_action)
		return actions, [None] * self.num_agents

	def _soft_update_targets(self):
		for target_net, source_net in zip(self.target_agent_networks, self.agent_networks):
			for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
				target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data)
		for target_param, source_param in zip(self.target_mixer.parameters(), self.mixer.parameters()):
			target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data)

	def update(self, replay_buffer, batch_size=64):
		if len(replay_buffer) < batch_size:
			return {}

		obs, state, action, reward, next_obs, next_state, done = replay_buffer.sample(batch_size)
		obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
		next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
		state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
		next_state = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)
		action = torch.as_tensor(action, dtype=torch.long, device=self.device)
		reward = torch.as_tensor(reward, dtype=torch.float32, device=self.device).view(-1, 1)
		done = torch.as_tensor(done, dtype=torch.float32, device=self.device).view(-1, 1)

		if self.obs_is_image:
			obs = obs.permute(0, 1, 4, 2, 3)
			next_obs = next_obs.permute(0, 1, 4, 2, 3)

		chosen_agent_qs = []
		for agent_idx, network in enumerate(self.agent_networks):
			q_values = network(obs[:, agent_idx])
			chosen_q = q_values.gather(1, action[:, agent_idx].unsqueeze(1)).squeeze(1)
			chosen_agent_qs.append(chosen_q)
		chosen_agent_qs = torch.stack(chosen_agent_qs, dim=1)
		q_total = self.mixer(chosen_agent_qs, state)

		with torch.no_grad():
			target_agent_qs = []
			for agent_idx, network in enumerate(self.agent_networks):
				online_next_q = network(next_obs[:, agent_idx])
				next_action = torch.argmax(online_next_q, dim=1)
				target_next_q = self.target_agent_networks[agent_idx](next_obs[:, agent_idx])
				target_selected_q = target_next_q.gather(1, next_action.unsqueeze(1)).squeeze(1)
				target_agent_qs.append(target_selected_q)
			target_agent_qs = torch.stack(target_agent_qs, dim=1)
			target_total_q = self.target_mixer(target_agent_qs, next_state)
			td_target = reward + (1.0 - done) * self.gamma * target_total_q

		loss = F.mse_loss(q_total, td_target)
		self.optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_([p for net in self.agent_networks for p in net.parameters()], self.grad_clip)
		torch.nn.utils.clip_grad_norm_(self.mixer.parameters(), self.grad_clip)
		self.optimizer.step()
		self._soft_update_targets()

		return {
			'critic_loss': float(loss.item()),
			'q_value': float(q_total.mean().item()),
		}

	def save(self, filename):
		torch.save({
			'agent_networks': [net.state_dict() for net in self.agent_networks],
			'target_agent_networks': [net.state_dict() for net in self.target_agent_networks],
			'mixer': self.mixer.state_dict(),
			'target_mixer': self.target_mixer.state_dict(),
		}, filename)

	def load(self, filename):
		checkpoint = torch.load(filename, map_location=self.device)
		for net, state_dict in zip(self.agent_networks, checkpoint['agent_networks']):
			net.load_state_dict(state_dict)
		for net, state_dict in zip(self.target_agent_networks, checkpoint['target_agent_networks']):
			net.load_state_dict(state_dict)
		self.mixer.load_state_dict(checkpoint['mixer'])
		self.target_mixer.load_state_dict(checkpoint['target_mixer'])