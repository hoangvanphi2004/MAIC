import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
from torch.distributions import Categorical
from .network import Actor, Critic
from .buffer import ReplayBuffer
from .utils import Normalizer
from .ensemble import EnsembleRegressor
from .ensemble_image import CNNEnsembleRegressor

eps = 1e-8

class SAC_REINFORCE:
	scaled_information_gain_coef = 0.0
	scaled_entropy_coef = 0.0

	def __init__(self, obs_shape, state_shape, action_dim, num_agents=2, hidden_dim=128, lr=3e-4, gamma=0.99,
				 tau=0.01, alpha1=0.01, alpha2=0.01, alpha_kl=0.1, alpha_min=1e-4, policy_update_steps=3, auto_entropy_tuning=False):
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.gamma = gamma
		self.tau = tau
		self.n_actions = action_dim
		self.num_agents = num_agents
		self.alpha_kl = float(alpha_kl)
		self.alpha_min = float(alpha_min)
		self.policy_update_steps = max(1, int(policy_update_steps))
		self.obs_is_image = len(obs_shape) == 3
		self.state_is_image = len(state_shape) == 3

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
			self.target_entropy = float(self.num_agents * action_dim) * 0.2
			self.log_alpha1 = torch.tensor([alpha1], requires_grad=True, device=self.device)
			self.alpha1_optimizer = optim.Adam([self.log_alpha1], lr=lr)
			self.alpha1 = self.log_alpha1.exp()

			self.log_alpha2 = torch.tensor([alpha2], requires_grad=True, device=self.device)
			self.alpha2_optimizer = optim.Adam([self.log_alpha2], lr=lr)
			self.alpha2 = self.log_alpha2.exp()
		else:
			self.alpha1 = alpha1
			self.alpha2 = alpha2
		
		state_dim = int(np.prod(state_shape))
		self.input_ensemble_shape = state_dim + action_dim * num_agents
		self.output_ensemble_shape = state_dim
		if self.state_is_image:
			cnn_state_shape = (state_shape[2], state_shape[0], state_shape[1])
			self.ensemble_regressor = CNNEnsembleRegressor(
				M=5,
				state_shape=cnn_state_shape,
				action_dim=action_dim * num_agents,
				out_dim=self.output_ensemble_shape,
				hidden=256,
				device=self.device,
			)
		else:
			self.ensemble_regressor = EnsembleRegressor(
				in_dim=self.input_ensemble_shape,
				out_dim=self.output_ensemble_shape,
				M=5,
				hidden=256,
				device=self.device,
			)
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
				s = torch.FloatTensor(obs_arr[i]).unsqueeze(0).to(self.device)
				if self.obs_is_image:
					s = s.permute(0, 3, 1, 2)
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
	
	@torch.no_grad()
	def information_bonus_raw(self, states, actions_oh):
		states_flat = states.reshape(states.size(0), -1)
		actions_oh_flat = actions_oh.reshape(actions_oh.shape[0], -1)

		ensemble_input = torch.cat([states_flat, actions_oh_flat], dim=-1)
		ensemble_input_norm = self.input_normalizer.normalize(ensemble_input)
		
		mean_ens, var_total, std_total, std_ale, std_epi = self.ensemble_regressor.mixture_mean_var(ensemble_input_norm, return_decomposed=True)
		info_gain = torch.sum(torch.log(1 + (std_epi ** 2) / eps), dim=-1, keepdim=True)
		return info_gain

	def information_bonus(self, states, actions_oh):
		info_gain = self.information_bonus_raw(states, actions_oh)
		return self.information_bonus_normalizer.normalize(info_gain)

	def train_ensemble_model(self, state, actions_oh, next_state):
		states_flat = state.reshape(state.size(0), -1)
		next_state_flat = next_state.reshape(next_state.size(0), -1)
		actions_oh_flat = actions_oh.reshape(actions_oh.shape[0], -1)
		
		ensemble_input = torch.cat([states_flat, actions_oh_flat], dim=-1)
		ensemble_input_norm = self.input_normalizer.normalize(ensemble_input)
		with torch.no_grad():
			y_mean = self.output_normalizer.mean.detach()
			y_std = torch.sqrt(self.output_normalizer.var.detach() + eps)
		
		self.ensemble_regressor.train_batch(
			ensemble_input_norm,
			next_state_flat,
			y_mean=y_mean,
			y_std=y_std,
			normalize_for_loss=True,
		)
		
	def update_sac(self, replay_buffer, batch_size=64):
		if len(replay_buffer) < batch_size:
			return {}
		
		t1 = time.time()
		obs, state, action, reward, next_obs, next_state, done = replay_buffer.sample(batch_size)
		state = torch.FloatTensor(state).to(self.device)
		if self.state_is_image:
			state = state.permute(0, 3, 1, 2)
		obs = torch.FloatTensor(obs).to(self.device)
		if self.obs_is_image:
			obs = obs.permute(0, 1, 4, 2, 3)
		action = torch.LongTensor(action).to(self.device)
		reward = torch.FloatTensor(reward).to(self.device)
		next_obs = torch.FloatTensor(next_obs).to(self.device)
		if self.obs_is_image:
			next_obs = next_obs.permute(0, 1, 4, 2, 3)
		next_state = torch.FloatTensor(next_state).to(self.device)
		if self.state_is_image:
			next_state = next_state.permute(0, 3, 1, 2)
		done = torch.FloatTensor(done).to(self.device)
		
		t2 = time.time()
		# Update normalizers
		with torch.no_grad():
			actions_oh = F.one_hot(action.long(), num_classes=self.n_actions).float()

			states_flat = state.reshape(state.size(0), -1)
			actions_oh_flat = actions_oh.reshape(actions_oh.shape[0], -1)
			next_state_flat = next_state.reshape(next_state.size(0), -1)
			ensemble_input = torch.cat([states_flat, actions_oh_flat], dim=-1)
			info_bonus_raw = self.information_bonus_raw(state, actions_oh)
			
			self.information_bonus_normalizer.update(info_bonus_raw)
			self.input_normalizer.update(ensemble_input)
			self.output_normalizer.update(next_state_flat)

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
			scaled_info_bonus = self.scaled_information_gain_coef * info_bonus
			scaled_next_log_prob = self.scaled_entropy_coef * next_log_prob
			target_q = next_q_target - self.alpha1 * scaled_next_log_prob.unsqueeze(1) + self.alpha2 * scaled_info_bonus
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
			old_policy_probs = []
			for a_i, actor in enumerate(self.actors):
				obs_a_i = obs[:, a_i]
				probs_i = actor(obs_a_i).clamp_min(eps)
				probs_i = probs_i / probs_i.sum(dim=-1, keepdim=True)
				old_policy_probs.append(probs_i)
		info_bonus = self.information_bonus(state, actions_oh)
		scaled_info_bonus = self.scaled_information_gain_coef * info_bonus.squeeze(-1)
		alpha1 = self.alpha1.detach() if isinstance(self.alpha1, torch.Tensor) else self.alpha1
		alpha2 = self.alpha2.detach() if isinstance(self.alpha2, torch.Tensor) else self.alpha2
		actor_loss_value = 0.0
		total_entropy = 0.0
		total_advantage = 0.0
		total_info_gain = 0.0
		total_kl_divergence = 0.0
		for _ in range(self.policy_update_steps):
			total_policy_loss = 0.0
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
				old_log_probs = torch.log(
					old_policy_probs[a_i].gather(1, actions_a.unsqueeze(1)).clamp_min(eps)
				).squeeze(1)
				entropy = dist.entropy().mean()
				kl_divergence = (
					action_probs * (
						torch.log(action_probs.clamp_min(eps)) - torch.log(old_policy_probs[a_i])
					)
				).sum(dim=-1).mean()

				policy_loss_a = (((
					(alpha1 * self.scaled_entropy_coef + self.alpha_kl) * (new_log_probs + 1)
					- self.alpha_kl * old_log_probs
					- advantages
					- alpha2 * scaled_info_bonus
				).detach() * new_log_probs).mean())

				total_policy_loss += policy_loss_a
				total_entropy += entropy.item()
				total_advantage += advantages.mean().item()
				total_info_gain += info_bonus.mean().item()
				total_kl_divergence += kl_divergence.item()
			self.actor_optimizer.zero_grad()
			total_policy_loss.backward()
			torch.nn.utils.clip_grad_norm_([p for a in self.actors for p in a.parameters()], 1.0)
			self.actor_optimizer.step()
			actor_loss_value += total_policy_loss.item()
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
			min_log_alpha = np.log(self.alpha_min)
			self.log_alpha1.data.clamp_(min=min_log_alpha)
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

			info_bonus_target_raw = self.information_bonus_raw(state, sampled_actions_target_oh)
			info_bonus_raw = self.information_bonus_raw(state, sampled_actions_oh)

			alpha2_loss = (self.log_alpha2.exp() * (info_bonus_raw - info_bonus_target_raw).detach()).mean()
			self.alpha2_optimizer.zero_grad()
			alpha2_loss.backward()
			self.alpha2_optimizer.step()
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
		avg_return = float(np.sum(rewards_arr))
		policy_return = {
			'policy_loss': actor_loss_value / float(self.policy_update_steps),
			'entropy': total_entropy / float(self.num_agents * self.policy_update_steps),
			'avg_return': avg_return,
			'avg_advantage': total_advantage / float(self.num_agents * self.policy_update_steps),
			'avg_q_value': q_values.mean().item(),
			'information_gain': total_info_gain / float(self.num_agents * self.policy_update_steps),
			'kl_divergence': total_kl_divergence / float(self.num_agents * self.policy_update_steps),
			'alpha_kl': self.alpha_kl,
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