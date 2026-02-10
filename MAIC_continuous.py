import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import gymnasium as gym
from collections import deque
import random
import minigrid
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
import matplotlib.pyplot as plt

from bayesian.Bayesian_sampling import EnsembleRegressor

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
	def __init__(self, state_shape, action_dim, hidden_dim=128, log_std_min=-20, log_std_max=2):
		super(Actor, self).__init__()
		self.log_std_min = log_std_min
		self.log_std_max = log_std_max
		self.conv1 = nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
		def conv2d_size_out(size, kernel_size=4, stride=2, padding=1):
			return (size + 2 * padding - kernel_size) // stride + 1
		h = conv2d_size_out(conv2d_size_out(state_shape[0]))
		w = conv2d_size_out(conv2d_size_out(state_shape[1]))
		linear_input_size = h * w * 32
		self.fc1 = nn.Linear(linear_input_size, hidden_dim)
		self.fc2 = nn.Linear(hidden_dim, hidden_dim)
		self.mean = nn.Linear(hidden_dim, action_dim)
		self.log_std = nn.Linear(hidden_dim, action_dim)

	def forward(self, state):
		x = state / 255.0
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = x.reshape(x.size(0), -1)
		x = self.fc1(x)
		x = self.fc2(x)
		mean = self.mean(x)
		log_std = self.log_std(x)
		log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
		return mean, log_std
	
	def sample(self, state, epsilon=1e-6):
		mean, log_std = self.forward(state)
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
	def forward(self, state, actions_flat):
		B = state.size(0)
		embeds = []
		for i in range(self.num_agents):
			agent_img = state[:, i]
			emb = self.encode_agent(agent_img)
			embeds.append(emb)
		embeds = torch.cat(embeds, dim=1)
		joint = torch.cat([embeds, actions_flat], dim=1)
		x1 = F.relu(self.q1_fc1(joint))
		x1 = F.relu(self.q1_fc2(x1))
		q1 = self.q1_out(x1)
		x2 = F.relu(self.q2_fc1(joint))
		x2 = F.relu(self.q2_fc2(x2))
		q2 = self.q2_out(x2)
		return q1, q2
		
class Normalizer:
	def __init__(self, size, eps=1e-8):
		self.size = size
		self.eps = eps
		self.mean = np.zeros(size, dtype=np.float32)
		self.var = np.ones(size, dtype=np.float32)
		self.count = 0

	def update(self, x):
		batch_mean = np.mean(x, axis=0)
		batch_var = np.var(x, axis=0)
		batch_count = x.shape[0]

		delta = batch_mean - self.mean
		total_count = self.count + batch_count

		new_mean = self.mean + delta * batch_count / total_count
		m_a = self.var * self.count
		m_b = batch_var * batch_count
		M2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
		new_var = M2 / total_count

		self.mean = new_mean
		self.var = new_var
		self.count = total_count

	def normalize(self, x):
		return (x - self.mean) / (np.sqrt(self.var) + self.eps)
		
	def denormalize(self, x):
		return x * (np.sqrt(self.var) + self.eps) + self.mean
		
class MAIC:
	def __init__(self, state_shape, action_dim, num_agents=2, hidden_dim=128, lr=3e-4, gamma=0.99,
				 tau=0.01, alpha=0.01, auto_entropy_tuning=False):
		self.action_dim = action_dim
		self.continuous_action_dim = 1
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.gamma = gamma
		self.tau = tau
		self.num_agents = num_agents
		self.actors = [Actor(state_shape, self.continuous_action_dim, hidden_dim).to(self.device)
					   for _ in range(self.num_agents)]
		self.actors_target = [Actor(state_shape, self.continuous_action_dim, hidden_dim).to(self.device)
							   for _ in range(self.num_agents)]
		
		for i in range(self.num_agents):
			self.actors_target[i].load_state_dict(self.actors[i].state_dict())

		self.actor_optimizer = optim.Adam([p for a in self.actors for p in a.parameters()], lr=lr)
		self.critic = Critic(self.num_agents, state_shape, action_dim_per_agent=self.continuous_action_dim, hidden_dim=256).to(self.device)
		self.critic_target = Critic(self.num_agents, state_shape, action_dim_per_agent=self.continuous_action_dim, hidden_dim=256).to(self.device)
		self.critic_target.load_state_dict(self.critic.state_dict())
		self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
		self.auto_entropy_tuning = auto_entropy_tuning
		
		self.input_ensemble_shape = (state_shape[0] * state_shape[1] * state_shape[2] + self.continuous_action_dim) * num_agents
		self.output_ensemble_shape = state_shape[0] * state_shape[1] * state_shape[2] * num_agents
		self.ensemble_regressor = EnsembleRegressor(in_dim=self.input_ensemble_shape, out_dim=self.output_ensemble_shape, M=5, hidden=256, device=self.device)
		self.ensemble_regressor.setup_optimizers(3e-4)

		self.input_normalizer = Normalizer(self.input_ensemble_shape)
		self.output_normalizer = Normalizer(self.output_ensemble_shape)
		self.information_bonus_normalizer = Normalizer(1)

		if self.auto_entropy_tuning:
			self.target_entropy = -float(self.num_agents * self.continuous_action_dim)
			self.log_alpha1 = torch.zeros(1, requires_grad=True, device=self.device)
			self.alpha1_optimizer = optim.Adam([self.log_alpha1], lr=lr)
			self.alpha1 = self.log_alpha1.exp()

			self.log_alpha2 = torch.zeros(1, requires_grad=True, device=self.device)
			self.alpha2_optimizer = optim.Adam([self.log_alpha2], lr=lr)
			self.alpha2 = self.log_alpha2.exp()
		else:
			self.alpha1 = alpha
			self.alpha2 = alpha
		
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
					_, _, continuous_action = actor.sample(s)
					continuous_action = continuous_action.detach().cpu().numpy()[0]
					action = int((continuous_action[0] + 1) / 2 * (self.action_dim - 1))
					action = np.clip(action, 0, self.action_dim - 1)
					actions.append(int(action.item()))
					log_probs.append(None)
				else:
					# print(f"state: {s.shape}")
					_, log_prob, continuous_action = actor.sample(s)
					continuous_action = continuous_action.detach().cpu().numpy()[0]
					action = int((continuous_action[0] + 1) / 2 * (self.action_dim - 1))
					action = np.clip(action, 0, self.action_dim - 1)
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
		
	def compute_counterfactual_baseline(self, states, actions_cont, agent_idx):
		# print(f"state {states.shape}, action_cont {actions_cont.shape}, agent_idx {agent_idx}")
		with torch.no_grad():
			B = states.size(0)
			actions_cf = actions_cont
			agent_states = states[:, agent_idx]
			mean, log_std = self.actors[agent_idx](agent_states)
			std = log_std.exp()
			dist = torch.distributions.Normal(mean, std)
			K = self.action_dim
			idx_edges = torch.arange(0, K + 1, device=states.device)
			cont_edges = (idx_edges / (K - 1)) * 2 - 1
			eps = 1e-6
			z_edges = torch.atanh(cont_edges.clamp(-1 + eps, 1 - eps))
			cdf = dist.cdf(z_edges.unsqueeze(0))
			action_probs = cdf[:, 1:] - cdf[:, :-1]
			action_probs = action_probs.clamp(1e-8)
			action_probs = action_probs / action_probs.sum(dim=1, keepdim=True)
			# ---- counterfactual Q ----
			q_vals = []
			for a_idx in range(K):
				a_cont = (a_idx / (K - 1)) * 2 - 1
				actions_cf[:, agent_idx] = a_cont
				q1, q2 = self.critic(states, actions_cf)
				q_vals.append(torch.min(q1, q2).squeeze(-1))
			q_stack = torch.stack(q_vals, dim=1)
			baseline = (action_probs * q_stack).sum(dim=1)
		return baseline
		
	def information_bonus(self, states, action_continuous):
		"""Compute information gain using ensemble uncertainty
		I_a(s,a) = sum_j log(1 + σ²_j(s,a) / σ²), where σ² = 1
		"""
		# print(f"states shape: {states.shape}, actions shape: {actions.shape}")
		states = states.reshape(states.size(0), -1)
		action_continuous = action_continuous.reshape(action_continuous.size(0), -1)
		# print(f"states shape: {states.shape}, actions shape: {actions_oh.shape}")
		ensemble_input = torch.cat([states, action_continuous], dim=-1)
		ensemble_input = self.input_normalizer.normalize(ensemble_input.detach().cpu().numpy())
		ensemble_input = torch.FloatTensor(ensemble_input).to(self.device)
		mean_ens, var_total, std_total, std_ale, std_epi = self.ensemble_regressor.mixture_mean_var(ensemble_input, return_decomposed = True)
		info_gain = torch.sum(torch.log(1 + (std_epi ** 2) / (std_ale ** 2)), dim=-1, keepdim=True)
		info_gain = self.information_bonus_normalizer.normalize(info_gain.detach().cpu().numpy())
		info_gain = torch.FloatTensor(info_gain).to(self.device)
		# print(f"Information gain shape: {info_gain.shape}")
		return info_gain

	def train_ensemble_model(self, replay_buffer, batch_size=64, epochs=5):
		"""Train the ensemble dynamics model using collected transitions"""
		if len(replay_buffer) < batch_size:
			return
		states, actions, reward, next_state, done = replay_buffer.sample(batch_size)
		states = torch.FloatTensor(states).to(self.device)
		action_continuous = torch.FloatTensor((actions / (self.action_dim - 1)) * 2 - 1).to(self.device)
		next_state = torch.FloatTensor(next_state).to(self.device)

		states = states.view(states.size(0), -1)
		next_state = next_state.view(next_state.size(0), -1)
		
		action_continuous = action_continuous.view(action_continuous.size(0), -1)

		next_state = self.output_normalizer.normalize(next_state.detach().cpu().numpy())
		next_state = torch.FloatTensor(next_state).to(self.device)
		ensemble_input = torch.cat([states, action_continuous], dim=-1)
		ensemble_input = self.input_normalizer.normalize(ensemble_input.detach().cpu().numpy())
		ensemble_input = torch.FloatTensor(ensemble_input).to(self.device)
		self.ensemble_regressor.train_batch(ensemble_input, next_state)

	def update_sac(self, replay_buffer, batch_size=64):
		if len(replay_buffer) < batch_size:
			return {}
		
		states, actions, reward, next_states, done = replay_buffer.sample(batch_size)
		states = torch.FloatTensor(states).permute(0, 1, 4, 2, 3).to(self.device)
		action_continuous = torch.FloatTensor((actions / (self.action_dim - 1)) * 2 - 1).to(self.device)
		reward = torch.FloatTensor(reward).to(self.device)
		next_states = torch.FloatTensor(next_states).permute(0, 1, 4, 2, 3).to(self.device)
		done = torch.FloatTensor(done).to(self.device)

		# Update Normalizers
		infos = {}
		with torch.no_grad():
			states_flat = states.reshape(states.size(0), -1)
			action_continuous_flat = action_continuous.view(action_continuous.size(0), -1)
			self.information_bonus_normalizer.update(self.information_bonus(states_flat, action_continuous_flat).detach().cpu().numpy())
			self.input_normalizer.update(torch.cat([states_flat, action_continuous_flat], dim=-1).detach().cpu().numpy())
			next_states_flat = next_states.reshape(next_states.size(0), -1)
			self.output_normalizer.update(next_states_flat.detach().cpu().numpy())

		if reward.dim() == 2:
			reward_for_target = reward.mean(dim=1, keepdim=True)
		else:
			reward_for_target = reward.unsqueeze(1)

		with torch.no_grad():
			next_actions = []
			next_log_prob = 0
			for i, actor in enumerate(self.actors):
				ns_i = next_states[:, i]
				a_i, logp_i, _ = actor.sample(ns_i)
				next_actions.append(a_i.unsqueeze(1).long())
				next_log_prob = next_log_prob + logp_i
			next_actions_cat = torch.cat(next_actions, dim=1)
			next_actions_cat_flat = next_actions_cat.view(next_actions_cat.size(0), -1)
			# print(f"next_states size = {next_states.size()}")
			next_states_flat = next_states.reshape(next_states.size(0), -1)
			next_q1_target, next_q2_target = self.critic_target(next_states, next_actions_cat_flat)
			next_q_target = torch.min(next_q1_target, next_q2_target)
			information_bonus = self.information_bonus(next_states_flat, next_actions_cat_flat)
			# print(f"next_q_target {next_q_target.shape}, information_bonus {information_bonus.shape}, next_log_prob {next_log_prob.shape}")
			target_q = next_q_target - self.alpha1 * next_log_prob + self.alpha2 * information_bonus
			target_q_value = reward_for_target + (1 - done.unsqueeze(1)) * self.gamma * target_q

		action_continuous_flat = action_continuous.view(action_continuous.size(0), -1)
		q1, q2 = self.critic(states, action_continuous_flat)
		critic_loss = F.mse_loss(q1, target_q_value) + F.mse_loss(q2, target_q_value)
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
		self.critic_optimizer.step()

		# Update actor
		total_policy_loss = 0
		sampled_logp = []
		sampled_actions = []
		for i, actor in enumerate(self.actors):
			s_i = states[:, i]
			a_i, logp_i, _ = actor.sample(s_i)
			sampled_actions.append(a_i.unsqueeze(1).long())
			sampled_logp.append(logp_i)
		
		with torch.no_grad():
			sampled_actions = torch.stack(sampled_actions, dim = 1);
			sampled_actions_flat = sampled_actions.view(sampled_actions.size(0), -1)
			# print(f"state: {states.shape}, sampled actions flat: {sampled_actions_flat.shape}")
			q1, q2 = self.critic(states, sampled_actions_flat)
			q_values = torch.min(q1, q2).squeeze(-1)

		information_bonus = self.information_bonus(states_flat, sampled_actions_flat).squeeze(-1)
		
		for i, actor in enumerate(self.actors):
			baseline = self.compute_counterfactual_baseline(states, action_continuous, i)
			advantages = (q_values - baseline).detach()
			if len(advantages) > 1:
				advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
			actor_loss = (self.alpha1 * sampled_logp[i] - q_values + self.alpha2 * information_bonus).mean()
			total_policy_loss += actor_loss

		self.actor_optimizer.zero_grad()
		total_policy_loss.backward()
		torch.nn.utils.clip_grad_norm_([p for a in self.actors for p in a.parameters()], 1.0)
		self.actor_optimizer.step()
		
		self.train_ensemble_model(replay_buffer, batch_size=batch_size, epochs=1)

		alpha_info = {}
		if self.auto_entropy_tuning:
			with torch.no_grad():
				sampled_logp = 0
				sampled_actions = []
				for i, actor in enumerate(self.actors):
					s_i = states[:, i]
					a_i, logp_i, _ = actor.sample(s_i)
					sampled_actions.append(a_i.unsqueeze(1).long())
					sampled_logp = sampled_logp + logp_i

				sampled_logp_target = 0
				sampled_actions_target = []
				for i, actor in enumerate(self.actors_target):
					s_i = states[:, i]
					a_i, logp_i, _ = actor.sample(s_i)
					sampled_actions_target.append(a_i.unsqueeze(1).long())
					sampled_logp_target = sampled_logp + logp_i

			alpha_loss = -(self.log_alpha1 * (sampled_logp + self.target_entropy).detach()).mean()
			self.alpha1_optimizer.zero_grad()
			alpha_loss.backward()
			self.alpha1_optimizer.step()
			self.alpha1 = self.log_alpha1.exp()
			
			actions = torch.cat(sampled_actions, dim=1)
			actions_flat = actions.view(actions.size(0), -1)
			actions_target = torch.cat(sampled_actions_target, dim=1)
			actions_target_flat = actions_target.view(actions_target.size(0), -1)
			information_bonus = self.information_bonus(states_flat, actions_flat)
			information_bonus_target = self.information_bonus(states_flat, actions_target_flat)
			#print(f"information_bonus {information_bonus}, information_bonus_target {information_bonus_target}")
			alpha_loss2 = (self.log_alpha2 * (information_bonus - information_bonus_target).detach()).mean()

			self.alpha2_optimizer.zero_grad()
			alpha_loss2.backward()
			self.alpha2_optimizer.step()
			self.alpha2 = self.log_alpha2.exp()

			for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for i in range(self.num_agents):
				for target_param, param in zip(self.actors_target[i].parameters(), self.actors[i].parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			alpha_info = {
				'alpha1_loss': alpha_loss.item(),
				'alpha1_value': self.alpha1.item(),
				'alpha2_loss': alpha_loss2.item(),
				'alpha2_value': self.alpha2.item()
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