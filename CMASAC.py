import time

import mac
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

from bayesian.Bayesian_sampling import EnsembleRegressor

class ReplayBuffer:
	def __init__(self, capacity):
		self.buffer = deque(maxlen=capacity)
	def push(self, obs, state, action, reward, next_obs, next_state, done):
		self.buffer.append((obs, state, action, reward, next_obs, next_state, done))
	def sample(self, batch_size):
		batch = random.sample(self.buffer, batch_size)
		obs, state, action, reward, next_obs, next_state, done = zip(*batch)
		return (np.array(obs), np.array(state), np.array(action), np.array(reward),
				np.array(next_obs), np.array(next_state), np.array(done))
	def __len__(self):
		return len(self.buffer)
	
class Actor(nn.Module):
	def __init__(self, obs_shape, action_dim, hidden_dim=128):
		super(Actor, self).__init__()
		self.conv1 = nn.Conv2d(obs_shape[2], 16, kernel_size=4, stride=2, padding=1)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
		def conv2d_size_out(size, kernel_size=4, stride=2, padding=1):
			return (size + 2 * padding - kernel_size) // stride + 1
		h = conv2d_size_out(conv2d_size_out(obs_shape[0]))
		w = conv2d_size_out(conv2d_size_out(obs_shape[1]))
		linear_input_size = h * w * 32
		self.fc1 = nn.Linear(linear_input_size, action_dim)
	def forward(self, obs):
		x = obs
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = x.reshape(x.size(0), -1)
		action_probs = F.softmax(self.fc1(x), dim=-1)
		return action_probs
	def sample(self, obs):
		action_probs = self.forward(obs)
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
		# Global state input: [batch, channels, height, width]
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
	def encode_global_state(self, state):
		"""Encode global state with CNN.
		Args:
			state: [batch, channels, height, width]
		Returns:
			embedding: [batch, embed_size]
		"""
		x = F.relu(self.conv1(state))
		x = F.relu(self.conv2(x))
		x = x.view(x.size(0), -1)
		return x
	def forward(self, state, actions):
		"""
		Args:
			state: Global state [batch, channels, height, width]
			actions: Agent actions [batch, num_agents] or [batch, num_agents, action_dim]
		Returns:
			q1, q2: Q-values [batch, 1]
		"""
		B = state.size(0)
		# Encode global state once
		embed = self.encode_global_state(state)
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
	
class Normalizer:
    def __init__(self, size, eps=1e-8, device=None):
        self.size = size
        self.eps = eps
        self.device = device if device else torch.device("cpu")
        self.mean = torch.zeros(size, dtype=torch.float32, device=self.device)
        self.var = torch.ones(size, dtype=torch.float32, device=self.device)
        self.count = 0

    def update(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0, unbiased=False)
        batch_count = x.size(0)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + torch.square(delta) * self.count * batch_count / total_count
        new_var = M2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        return (x - self.mean) / (torch.sqrt(self.var) + self.eps)
        
    def denormalize(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        return x * (torch.sqrt(self.var) + self.eps) + self.mean

class SAC_REINFORCE:
	def __init__(self, obs_shape, state_shape, action_dim, num_agents=2, hidden_dim=128, lr=3e-4, gamma=0.99,
				 tau=0.01, alpha1=0.01, alpha2=0.01, auto_entropy_tuning=False):
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.gamma = gamma
		self.tau = tau
		self.n_actions = action_dim
		self.num_agents = num_agents

		self.actors = [Actor(obs_shape, action_dim, hidden_dim).to(self.device)
					   for _ in range(self.num_agents)]
		self.actor_target = [Actor(obs_shape, action_dim, hidden_dim).to(self.device)
							 for _ in range(self.num_agents)]
		for target, source in zip(self.actor_target, self.actors):
			target.load_state_dict(source.state_dict())
		self.actor_optimizer = optim.Adam([p for a in self.actors for p in a.parameters()], lr=lr)
		
		self.critic = Critic(self.num_agents, state_shape, action_dim_per_agent=self.n_actions, hidden_dim=256).to(self.device)
		self.critic_target = Critic(self.num_agents, state_shape, action_dim_per_agent=self.n_actions, hidden_dim=256).to(self.device)
		self.critic_target.load_state_dict(self.critic.state_dict())
		self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
		
		self.auto_entropy_tuning = auto_entropy_tuning
		if self.auto_entropy_tuning:
			self.target_entropy = -float(self.num_agents * action_dim)
			self.log_alpha1 = torch.zeros(1, requires_grad=True, device=self.device)
			self.alpha1_optimizer = optim.Adam([self.log_alpha1], lr=lr)
			self.alpha1 = self.log_alpha1.exp()

			self.log_alpha2 = torch.zeros(1, requires_grad=True, device=self.device)
			self.alpha2_optimizer = optim.Adam([self.log_alpha2], lr=lr)
			self.alpha2 = self.log_alpha2.exp()
		else:
			self.alpha1 = alpha1
			self.alpha2 = alpha2
		
		self.input_ensemble_shape = state_shape[0] * state_shape[1] * state_shape[2] + action_dim * num_agents
		self.output_ensemble_shape = state_shape[0] * state_shape[1] * state_shape[2] * num_agents
		self.ensemble_regressor = EnsembleRegressor(in_dim=self.input_ensemble_shape, out_dim=self.output_ensemble_shape, M=5, hidden=256, device=self.device)
		self.ensemble_regressor.setup_optimizers(3e-4)

		self.input_normalizer = Normalizer(self.input_ensemble_shape, device=self.device)
		self.output_normalizer = Normalizer(self.output_ensemble_shape, device=self.device)
		self.information_bonus_normalizer = Normalizer(1, device=self.device)

	def select_action(self, obs, evaluate=False):
		with torch.no_grad():
			obs_arr = np.array(obs)
			if obs_arr.ndim == 3:
				obs_arr = obs_arr[np.newaxis, ...]
			actions = []
			log_probs = []
			for i, actor in enumerate(self.actors):
				s = torch.FloatTensor(obs_arr[i]).permute(2, 0, 1).unsqueeze(0).to(self.device)
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

	def compute_counterfactual_baseline(self, obs_a_i, states, actions, agent_idx):
		"""Counterfactual baseline b_i(s) = E_{u_i~pi_i}[Q(s, (u_i, u_-i))]."""
		with torch.no_grad():
			actions_oh = F.one_hot(actions.long(), num_classes=self.n_actions).float()
			agent_states = obs_a_i
			action_probs = self.actors[agent_idx](agent_states)
			q_values_per_action = []
			for a in range(self.n_actions):
				candidate_actions = actions_oh.clone()
				candidate_actions[:, agent_idx, :] = 0.0
				candidate_actions[:, agent_idx, a] = 1.0
				q1, q2 = self.critic(states, candidate_actions)
				q_val = torch.min(q1, q2).squeeze(-1)
				q_values_per_action.append(q_val)
			q_stack = torch.stack(q_values_per_action, dim=1)
			baseline = (action_probs * q_stack).sum(dim=1)
		return baseline
	
	def information_bonus(self, states, actions_oh):
		states_flat = states.reshape(states.size(0), -1)
		actions_oh_flat = actions_oh.reshape(actions_oh.shape[0], -1)

		ensemble_input = torch.cat([states_flat, actions_oh_flat], dim=-1)
		ensemble_input_norm = self.input_normalizer.normalize(ensemble_input)
		
		mean_ens, var_total, std_total, std_ale, std_epi = self.ensemble_regressor.mixture_mean_var(ensemble_input_norm, return_decomposed=True)
		info_gain = torch.sum(torch.log(1 + (std_epi ** 2) / (std_ale ** 2 + 1e-8)), dim=-1, keepdim=True) * 0.001
		
		info_gain = self.information_bonus_normalizer.normalize(info_gain) 
		
		return info_gain

	def train_ensemble_model(self, state, actions_oh, next_state):
		states_flat = state.reshape(state.size(0), -1)
		next_state_flat = next_state.reshape(next_state.size(0), -1)
		actions_oh_flat = actions_oh.reshape(actions_oh.shape[0], -1)
		
		ensemble_input = torch.cat([states_flat, actions_oh_flat], dim=-1)
		ensemble_input_norm = self.input_normalizer.normalize(ensemble_input)

		next_state_norm = self.output_normalizer.normalize(next_state_flat)
		
		self.ensemble_regressor.train_batch(ensemble_input_norm, next_state_norm)
		
	def update_sac(self, replay_buffer, batch_size=64):
		if len(replay_buffer) < batch_size:
			return {}
		
		t1 = time.time()
		obs, state, action, reward, next_obs, next_state, done = replay_buffer.sample(batch_size)
		state = torch.FloatTensor(state).permute(0, 3, 1, 2).to(self.device)
		obs = torch.FloatTensor(obs).permute(0, 1, 4, 2, 3).to(self.device)
		action = torch.LongTensor(action).to(self.device)
		reward = torch.FloatTensor(reward).to(self.device)
		next_obs = torch.FloatTensor(next_obs).permute(0, 1, 4, 2, 3).to(self.device)
		next_state = torch.FloatTensor(next_state).permute(0, 3, 1, 2).to(self.device)
		done = torch.FloatTensor(done).to(self.device)
		
		t2 = time.time()
		# Update normalizers
		with torch.no_grad():
			actions_oh = F.one_hot(action.long(), num_classes=self.n_actions).float()

			states_flat = state.reshape(state.size(0), -1)
			actions_oh_flat = actions_oh.reshape(actions_oh.shape[0], -1)
			next_state_flat = next_state.reshape(next_state.size(0), -1)
			ensemble_input = torch.cat([states_flat, actions_oh_flat], dim=-1)
			
			self.input_normalizer.update(ensemble_input)
			self.output_normalizer.update(next_state_flat)

			info_bonus = self.information_bonus(state, actions_oh)
			self.information_bonus_normalizer.update(info_bonus)

		t3 = time.time()
		# Update critic
		if reward.dim() == 2:
			reward_for_target = reward.mean(dim=1, keepdim=True)
		else:
			reward_for_target = reward.unsqueeze(1)
		with torch.no_grad():
			next_actions = []
			next_log_prob = 0
			for i, actor in enumerate(self.actors):
				nobs_i = next_obs[:, i]
				a_i, logp_i, _ = actor.sample(nobs_i)
				next_actions.append(a_i.unsqueeze(1).long())
				next_log_prob = next_log_prob + logp_i
			next_actions_cat = torch.cat(next_actions, dim=1)
			next_actions_oh = F.one_hot(next_actions_cat.long(), num_classes=self.n_actions).float()

			next_q1_target, next_q2_target = self.critic_target(next_state, next_actions_oh)
			next_q_target = torch.min(next_q1_target, next_q2_target)

			info_bonus = self.information_bonus(next_state, next_actions_oh)

			target_q = next_q_target - self.alpha1 * next_log_prob.unsqueeze(1) + self.alpha2 * info_bonus
			# target_q = next_q_target - self.alpha1 * next_log_prob.unsqueeze(1)
			target_q_value = reward_for_target + (1 - done.unsqueeze(1)) * self.gamma * target_q
		action_oh = F.one_hot(action.long(), num_classes=self.n_actions).float()
		q1, q2 = self.critic(state, action_oh)
		critic_loss = F.mse_loss(q1, target_q_value) + F.mse_loss(q2, target_q_value)
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
		self.critic_optimizer.step()
		
		t4 = time.time()
		# Update actor
		with torch.no_grad():
			actions_oh = F.one_hot(action.long(), num_classes=self.n_actions).float()
			q1, q2 = self.critic(state, actions_oh)
			q_values = torch.min(q1, q2).squeeze(-1)
		total_policy_loss = 0.0
		total_entropy = 0.0
		total_advantage = 0.0
		total_info_gain = 0.0
		for a_i, actor in enumerate(self.actors):
			obs_a_i = obs[:, a_i]
			actions_a = action[:, a_i]
			baseline = self.compute_counterfactual_baseline(obs_a_i, state, action, a_i)
			advantages = (q_values - baseline).detach()
			if len(advantages) > 1:
				advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
			action_probs = actor(obs_a_i)
			dist = Categorical(action_probs)
			new_log_probs = dist.log_prob(actions_a)
			entropy = dist.entropy().mean()
			alpha1 = self.alpha1.detach() if isinstance(self.alpha1, torch.Tensor) else self.alpha1
			alpha2 = self.alpha2.detach() if isinstance(self.alpha2, torch.Tensor) else self.alpha2

			info_bonus = self.information_bonus(state, actions_oh)

			policy_loss_a = (((alpha1 * (new_log_probs + 1) - advantages - alpha2 * info_bonus).detach() * new_log_probs).mean())
			# policy_loss_a = ((alpha1 * (new_log_probs + 1) - advantages).detach() * new_log_probs).mean()

			total_policy_loss += policy_loss_a
			total_entropy += entropy.item()
			total_advantage += advantages.mean().item()
			total_info_gain += torch.abs(info_bonus).mean().item()
		self.actor_optimizer.zero_grad()
		total_policy_loss.backward()
		torch.nn.utils.clip_grad_norm_([p for a in self.actors for p in a.parameters()], 1.0)
		self.actor_optimizer.step()
		rewards_arr = np.array(reward.cpu().numpy())

		t5 = time.time()
		# Update ensemble model with current batch
		self.train_ensemble_model(state, actions_oh, next_state)
		t6 = time.time()

		# Update alpha1 and alpha2 if auto-tuning is enabled
		alpha_info = {}
		if self.auto_entropy_tuning:
			# Update alpha1
			with torch.no_grad():
				sampled_logp = 0
				sampled_actions = []
				for i, actor in enumerate(self.actors):
					obs_a_i = obs[:, i]
					probs_i = actor(obs_a_i)
					dist_i = Categorical(probs_i)
					sampled_a_i = dist_i.sample()
					sampled_logp = sampled_logp + dist_i.log_prob(sampled_a_i)
					sampled_actions.append(sampled_a_i.unsqueeze(1).long())
			sampled_actions_cat = torch.cat(sampled_actions, dim=1)
			sampled_actions_oh = F.one_hot(sampled_actions_cat.long(), num_classes=self.n_actions).float()

			alpha1_loss = -(self.log_alpha1.exp() * (sampled_logp + self.target_entropy).detach()).mean()
			self.alpha1_optimizer.zero_grad()
			alpha1_loss.backward()
			self.alpha1_optimizer.step()
			self.alpha1 = self.log_alpha1.exp()

			# Update alpha2
			with torch.no_grad():
				sampled_logp_target = 0
				sampled_actions_target = []
				for i, actor in enumerate(self.actor_target):
					obs_a_i = obs[:, i]
					probs_i = actor(obs_a_i)
					dist_i = Categorical(probs_i)
					sampled_a_i = dist_i.sample()
					sampled_logp_target = sampled_logp_target + dist_i.log_prob(sampled_a_i)
					sampled_actions_target.append(sampled_a_i.unsqueeze(1).long())
			sampled_actions_target_cat = torch.cat(sampled_actions_target, dim=1)
			sampled_actions_target_oh = F.one_hot(sampled_actions_target_cat.long(), num_classes=self.n_actions).float()

			info_bonus_target = self.information_bonus(state, sampled_actions_target_oh)
			info_bonus = self.information_bonus(state, sampled_actions_oh)

			alpha2_loss = (self.log_alpha2.exp() * (info_bonus - info_bonus_target).detach()).mean()
			self.alpha2_optimizer.zero_grad()
			alpha2_loss.backward()
			self.alpha2_optimizer.step()
			# alpha2_loss = torch.tensor(0.0)  # Placeholder since info bonus is not currently used
			self.alpha2 = self.log_alpha2.exp()

			alpha_info = {
				'alpha1_loss': alpha1_loss.item(),
				'alpha1_value': self.alpha1.item(),
				'alpha2_loss': alpha2_loss.item(),
				'alpha2_value': self.alpha2.item()
			}

		t7 = time.time()
		for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		for target_actor, source_actor in zip(self.actor_target, self.actors):
			for target_param, param in zip(target_actor.parameters(), source_actor.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		t8 = time.time()
		# print(f"Times: sample={t2-t1:.3f}s, norm={t3-t2:.3f}s, critic={t4-t3:.3f}s, actor={t5-t4:.3f}s, ensemble={t6-t5:.3f}s, alpha={t7-t6:.3f}s, target_update={t8-t7:.3f}s")
		# print(f"Total update time: {t8-t1:.3f}s")
		try:
			avg_return = float(np.sum(rewards_arr))
		except Exception:
			avg_return = float(rewards_arr)
		policy_return = {
			'policy_loss': total_policy_loss.item(),
			'entropy': total_entropy / float(self.num_agents),
			'avg_return': avg_return,
			'avg_advantage': total_advantage / float(self.num_agents),
			'avg_q_value': q_values.mean().item(),
			'information_gain': total_info_gain / float(self.num_agents)
		}
		result = {
			'critic_loss': critic_loss.item(),
			'q_value': q1.mean().item()
		}
		result.update(alpha_info)
		result.update(policy_return)
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