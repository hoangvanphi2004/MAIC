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
from Bayesian_sampling import EnsembleRegressor


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
	"""Memory for storing complete episodes for REINFORCE"""
	def __init__(self):
		self.states = []
		self.actions = []
		self.rewards = []
		self.next_states = []
		
	def push(self, state, action, reward, next_state):
		self.states.append(state)
		self.actions.append(action)
		self.rewards.append(reward)
		self.next_states.append(next_state)
		
	def clear(self):
		self.states = []
		self.actions = []
		self.rewards = []
		self.next_states = []
		
	def __len__(self):
		return len(self.rewards)


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_dim=128):
		super(Actor, self).__init__()
		self.state_dim = state_dim
		# Shared backbone
		self.shared = nn.Sequential(
			nn.Linear(state_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU()
		)
		# Discrete action head (action_main)
		self.discrete_head = nn.Linear(hidden_dim, action_dim)
		# Continuous action head (action_halucinate) - mean and log_std, dimension = state_dim
		self.continuous_mean = nn.Linear(hidden_dim, state_dim)
		self.continuous_log_std = nn.Linear(hidden_dim, state_dim)
		self.action_dim = action_dim
		
	def forward(self, state):
		shared_out = self.shared(state)
		
		# Discrete action probabilities
		action_logits = self.discrete_head(shared_out)
		action_probs = F.softmax(action_logits, dim=-1)
		
		# Continuous action mean and std
		mean = self.continuous_mean(shared_out)
		log_std = self.continuous_log_std(shared_out)
		std = torch.exp(log_std.clamp(-20, 2))
		
		return action_probs, mean, std
		
	def sample(self, state):
		"""Sample discrete and continuous actions with gradients"""
		action_probs, mean, std = self.forward(state)
		
		# Discrete action sampling with REINFORCE
		action_probs_normalized = action_probs + 1e-8
		action_probs_normalized = action_probs_normalized / action_probs_normalized.sum(dim=-1, keepdim=True)
		dist_discrete = Categorical(action_probs_normalized)
		action_main = dist_discrete.sample()
		log_prob_discrete = dist_discrete.log_prob(action_main)
		
		# Continuous action sampling with reparameterization trick
		epsilon = torch.randn_like(std)
		action_halucinate = mean + std * epsilon
		action_halucinate_tanh = torch.tanh(action_halucinate)
		
		# Log probability for continuous action (with tanh squashing)
		log_prob_continuous = -0.5 * (epsilon ** 2).sum(dim=-1)
		log_prob_continuous = log_prob_continuous - torch.log(std.sum(dim=-1) + 1e-8)
		# Correction for tanh squashing
		log_prob_continuous = log_prob_continuous - (2 * (torch.log(torch.tensor(2.0)) - action_halucinate - F.softplus(-2 * action_halucinate))).sum(dim=-1)
		
		return action_main, action_halucinate_tanh, log_prob_discrete, log_prob_continuous


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_dim=128):
		super(Critic, self).__init__()
		self.action_dim = action_dim
		# Input: state_dim + action_dim (one-hot) + state_dim (hallucinate action)
		self.net = nn.Sequential(
			nn.Linear(state_dim + action_dim + state_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, 1)
		)
	def forward(self, state, action, hallucinate_action):
		# state: [batch, state_dim], action: [batch] (discrete), hallucinate_action: [batch, state_dim]
		action_one_hot = F.one_hot(action, num_classes=self.action_dim).float()
		x = torch.cat([state, action_one_hot, hallucinate_action], dim=-1)
		q_value = self.net(x)
		return q_value


class Value(nn.Module):
	"""State-value network V(s) -> scalar"""
	def __init__(self, state_dim, hidden_dim=128):
		super(Value, self).__init__()
		self.net = nn.Sequential(
			nn.Linear(state_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, 1)
		)
	def forward(self, state):
		v = self.net(state)
		return v


class HUCRL:
	def __init__(self, state_dim, action_dim, hidden_dim=128, lr=3e-4, gamma=0.99, 
				 tau=0.01, alpha1=0.01, alpha2=0.01, auto_entropy_tuning=True, reward_function=None, done_function=None, num_ensembles=5, beta=1):
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.gamma = gamma
		self.tau = tau
		self.beta = beta
		self.action_dim = action_dim
		# Actor network
		self.actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)
		self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
		self.critic = Critic(state_dim, action_dim, hidden_dim).to(self.device)
		self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(self.device)
		self.critic_target.load_state_dict(self.critic.state_dict())
		self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
		self.value = Value(state_dim, hidden_dim=hidden_dim).to(self.device)
		self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr)
		
		print(f"in_dim for ensemble: {state_dim + action_dim}, out_dim for ensemble: {state_dim}")
		self.ensemble_regressor = EnsembleRegressor(in_dim=state_dim + action_dim, out_dim=state_dim, M=num_ensembles, hidden=hidden_dim, device=self.device)
		self.ensemble_regressor.setup_optimizers(lr)
		self.reward_function = reward_function
		self.done_function = done_function
		self.auto_entropy_tuning = auto_entropy_tuning
		if self.auto_entropy_tuning:
			self.target_entropy = -float(action_dim)
			self.log_alpha1 = torch.zeros(1, requires_grad=True, device=self.device)
			self.alpha_optimizer = optim.Adam([self.log_alpha1], lr=lr)
			self.alpha1 = self.log_alpha1.exp()
		else:
			self.alpha1 = alpha1
		self.alpha2 = alpha2
	def select_action(self, state, evaluate=False):
		with torch.no_grad():
			state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
			if evaluate:
				action_probs, _, _ = self.actor(state)
				action = torch.argmax(action_probs, dim=-1)
				return action.item(), None
			else:
				action_main, action_halucinate, _, _ = self.actor.sample(state)
				return action_main.item(), action_halucinate.detach().cpu().numpy()
			
	def roll_out_hallucinated_next_state(self, initial_state, horizons, hallucinated_memory, hallucinated_buffer):
		"""Roll out hallucinated next states using the ensemble dynamics model"""
		state_tensor = np.array(initial_state, dtype=np.float32)
		for i in range(horizons):
			state_tensor = torch.as_tensor(state_tensor, device=self.device, dtype=torch.float32).unsqueeze(0)
			action_main, action_halucinate, _, _ = self.actor.sample(state_tensor)
			action_main_one_hot = F.one_hot(action_main, num_classes=self.action_dim).float()
			ensemble_input = torch.cat([state_tensor, action_main_one_hot], dim=-1)
			mu, std = self.ensemble_regressor.mixture_mean_var(ensemble_input)
			next_state_tensor = mu + std * self.beta * action_halucinate
			reward = float(self.reward_function(state_tensor.squeeze(0).detach().cpu().numpy(), action_main.item())) if self.reward_function else 0.0
			next_state_np = next_state_tensor.squeeze(0).detach().cpu().numpy()
			done = bool(self.done_function(next_state_np)) if self.done_function else False
			hallucinated_memory.push(
				state_tensor.squeeze(0).detach().cpu().numpy(),
				action_main.item(),
				reward,
				next_state_tensor.squeeze(0).detach().cpu().numpy(),
			)
			hallucinated_buffer.push(
				state_tensor.squeeze(0).detach().cpu().numpy(),
				action_main.item(),
				reward,
				next_state_np,
				float(done)
			)
			if done:
				break
			state_tensor = next_state_np
		return hallucinated_memory
	
	def train_ensemble_model(self, replay_buffer, batch_size=64, epochs=5):
		"""Train the ensemble dynamics model using collected transitions"""
		if len(replay_buffer) < batch_size:
			return
		for epoch in range(epochs):
			for _ in range(len(replay_buffer) // batch_size):
				state, action, reward, next_state, done = replay_buffer.sample(batch_size)
				state = torch.FloatTensor(state).to(self.device)
				action = torch.LongTensor(action).to(self.device)
				action_one_hot = F.one_hot(action, num_classes=self.action_dim).float()
				next_state = torch.FloatTensor(next_state).to(self.device)
				ensemble_input = torch.cat([state, action_one_hot], dim=-1)
				self.ensemble_regressor.train_batch(ensemble_input, next_state)

	def compute_returns(self, rewards, gamma=0.99):
		"""Compute discounted returns for REINFORCE"""
		returns = []
		R = 0
		for r in reversed(rewards):
			R = r + gamma * R
			returns.insert(0, R)
		
		returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
		# Normalize returns for stability
		if len(returns) > 1:
			returns = (returns - returns.mean()) / (returns.std() + 1e-8)
		return returns

	def update_reinforce(self, episode_memory):
		"""REINFORCE policy gradient update using Q-values from critic"""
		if len(episode_memory) == 0:
			return {}
		
		# Convert to tensors
		states = torch.FloatTensor(np.array(episode_memory.states)).to(self.device)
		actions = torch.LongTensor(episode_memory.actions).to(self.device)
		
		# Recompute action probabilities and distributions
		action_probs, mean, std = self.actor(states)
		
		# REINFORCE for discrete action_main
		dist_discrete = Categorical(action_probs)
		new_log_probs_discrete = dist_discrete.log_prob(actions)
		entropy_discrete = dist_discrete.entropy().mean()
		
		# Reparameterization for continuous action_halucinate
		epsilon = torch.randn_like(std)
		action_halucinate = mean + std * epsilon
		action_halucinate_tanh = torch.tanh(action_halucinate)
		log_prob_continuous = -0.5 * (epsilon ** 2).sum(dim=-1)
		log_prob_continuous = log_prob_continuous - torch.log(std.sum(dim=-1) + 1e-8)
		
		# Get Q-values from critic as advantage
		with torch.no_grad():
			q_values = self.critic(states, actions, action_halucinate_tanh).squeeze(1)
			info_bonus = self.information_bonus(states, actions).squeeze(1)
		
		# SAC policy loss with information bonus: E[α1 * log_π - Q + α2 * info_bonus]
		alpha1 = self.alpha1.detach() if isinstance(self.alpha1, torch.Tensor) else self.alpha1
		alpha2 = self.alpha2
		policy_loss_discrete = (alpha1 * new_log_probs_discrete - q_values.detach() + alpha2 * info_bonus.detach()).mean()
		policy_loss_continuous = (alpha1 * log_prob_continuous - q_values.detach() + alpha2 * info_bonus.detach()).mean()
		policy_loss = policy_loss_discrete + policy_loss_continuous
		
		# Update actor
		self.actor_optimizer.zero_grad()
		policy_loss.backward()
		torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
		self.actor_optimizer.step()
		
		return {
			'policy_loss': policy_loss.item(),
			'discrete_loss': policy_loss_discrete.item(),
			'continuous_loss': policy_loss_continuous.item(),
			'entropy': entropy_discrete.item(),
			'avg_q_value': q_values.mean().item(),
		}
		
	def information_bonus(self, state, action):
		"""Compute information gain using ensemble uncertainty
		I_a(s,a) = sum_j log(1 + σ²_j(s,a) / σ²), where σ² = 1
		"""
		action_one_hot = F.one_hot(action, num_classes=self.action_dim).float()
		ensemble_input = torch.cat([state, action_one_hot], dim=-1)
		_, std = self.ensemble_regressor.mixture_mean_var(ensemble_input)
		# std shape: [batch, state_dim]
		sigma_squared = std ** 2
		# Information gain: sum of log(1 + σ²_j) for each dimension
		info_gain = torch.sum(torch.log(1 + sigma_squared), dim=-1, keepdim=True)
		return info_gain
	
	def update_sac(self, replay_buffer, batch_size=64):
		"""SAC critic update using replay buffer with entropy regularization"""
		if len(replay_buffer) < batch_size:
			return {}
		
		# Sample from replay buffer
		state, action, reward, next_state, done = replay_buffer.sample(batch_size)
		
		state = torch.FloatTensor(state).to(self.device)
		action = torch.LongTensor(action).to(self.device)
		reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
		next_state = torch.FloatTensor(next_state).to(self.device)
		done = torch.FloatTensor(done).unsqueeze(1).to(self.device)
		
		# Sample hallucinate actions for current and next states
		with torch.no_grad():
			_, hallucinate_action, _, _ = self.actor.sample(state)
			next_action, next_hallucinate_action, next_log_prob_discrete, next_log_prob_continuous = self.actor.sample(next_state)
			
			# Get alpha values
			alpha1 = self.alpha1.detach() if isinstance(self.alpha1, torch.Tensor) else self.alpha1
			alpha2 = self.alpha2
			
			# Target Q-value with entropy regularization: r + γ * (Q(s', π(s')) - α1 * log π(s') + α2 * information_bonus)
			next_q_target = self.critic_target(next_state, next_action, next_hallucinate_action)
			# Combined entropy from both discrete and continuous actions
			next_entropy = next_log_prob_discrete + next_log_prob_continuous
			info_bonus = self.information_bonus(next_state, next_action)
			target_q_value = reward + (1 - done) * self.gamma * (next_q_target - alpha1 * next_entropy.unsqueeze(1) + alpha2 * info_bonus)
		
		# Current Q-value with state, action (from buffer), and sampled hallucinate_action
		q_value = self.critic(state, action, hallucinate_action)
		critic_loss = F.mse_loss(q_value, target_q_value)
		
		# Update critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
		self.critic_optimizer.step()
		
		# Soft update target networks
		for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
		
		result = {
			'critic_loss': critic_loss.item(),
			'q_value': q_value.mean().item(),
			'target_q_value': target_q_value.mean().item()
		}
		return result
		
	def save(self, filename):
		torch.save({
			'actor': self.actor.state_dict(),
			'critic': self.critic.state_dict(),
			'critic_target': self.critic_target.state_dict(),
		}, filename)
		
	def load(self, filename):
		checkpoint = torch.load(filename)
		self.actor.load_state_dict(checkpoint['actor'])
		self.critic.load_state_dict(checkpoint['critic'])
		self.critic_target.load_state_dict(checkpoint['critic_target'])
