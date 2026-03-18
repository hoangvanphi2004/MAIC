from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F

from network import QMIXModel


class QMIXAgent:
	def __init__(
		self,
		n_agents: int,
		obs_dim: int,
		state_dim: int,
		action_dim: int,
		lr: float = 3e-4,
		gamma: float = 0.99,
		tau: float = 0.01,
		grad_clip_norm: float = 10.0,
		device: str | None = None,
	):
		self.n_agents = n_agents
		self.action_dim = action_dim
		self.gamma = gamma
		self.tau = tau
		self.grad_clip_norm = grad_clip_norm

		if device is None:
			self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		else:
			self.device = torch.device(device)

		self.online = QMIXModel(
			n_agents=n_agents,
			obs_dim=obs_dim,
			state_dim=state_dim,
			action_dim=action_dim,
		).to(self.device)
		self.target = QMIXModel(
			n_agents=n_agents,
			obs_dim=obs_dim,
			state_dim=state_dim,
			action_dim=action_dim,
		).to(self.device)
		self.target.load_state_dict(self.online.state_dict())

		self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)

	@torch.no_grad()
	def select_actions(self, obs: np.ndarray, epsilon: float) -> np.ndarray:
		if random.random() < epsilon:
			return np.array([random.randrange(self.action_dim) for _ in range(self.n_agents)], dtype=np.int64)

		obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
		if obs_t.ndim == 2:
			obs_t = obs_t.unsqueeze(0)  # shape: (1, n_agents, obs_dim)
		# Truyền dummy state nếu không có state
		dummy_state = torch.zeros(obs_t.shape[0], self.online.state_dim, device=self.device)
		agent_qs, _ = self.online(obs=obs_t, state=dummy_state, actions=None)
		greedy = agent_qs.argmax(dim=-1).squeeze(0)
		return greedy.detach().cpu().numpy().astype(np.int64)

	def train_step(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
		obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
		state = torch.as_tensor(batch["state"], dtype=torch.float32, device=self.device)
		actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=self.device)
		rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device).unsqueeze(-1)
		next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
		next_state = torch.as_tensor(batch["next_state"], dtype=torch.float32, device=self.device)
		dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device).unsqueeze(-1)

		agent_qs, q_total = self.online(obs=obs, state=state, actions=actions)

		with torch.no_grad():
			online_next_agent_qs, _ = self.online(obs=next_obs, state=next_state, actions=None)
			next_actions = online_next_agent_qs.argmax(dim=-1)
			_, target_next_q_total = self.target(obs=next_obs, state=next_state, actions=next_actions)
			td_target = rewards + self.gamma * (1.0 - dones) * target_next_q_total

		# print(f"q_total: {q_total[0].mean().item():.3f}, td_target: {td_target[0].item():.3f}, reward: {rewards[0].item():.3f}, done: {dones[0].item()}")
		loss = F.mse_loss(q_total, td_target)

		self.optimizer.zero_grad(set_to_none=True)
		loss.backward()
		torch.nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=self.grad_clip_norm)
		self.optimizer.step()

		self._soft_update_target()

		return {
			"loss": float(loss.item()),
			"q_total_mean": float(q_total.mean().item()),
			"target_mean": float(td_target.mean().item()),
		}

	def _soft_update_target(self) -> None:
		for target_param, online_param in zip(self.target.parameters(), self.online.parameters()):
			target_param.data.mul_(1.0 - self.tau)
			target_param.data.add_(self.tau * online_param.data)

	def save(self, path: Path) -> None:
		path.parent.mkdir(parents=True, exist_ok=True)
		torch.save(
			{
				"online": self.online.state_dict(),
				"target": self.target.state_dict(),
			},
			str(path),
		)
