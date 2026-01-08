from typing import List, Tuple, Optional

import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class BaseNet(nn.Module):

	def __init__(self, in_dim: int, out_dim: int = 1, hidden: int = 128, nlayers: int = 2):
		super().__init__()
		layers = []
		last = in_dim
		for _ in range(nlayers):
			layers.append(nn.Linear(last, hidden))
			layers.append(nn.ReLU())
			last = hidden
		self.feature = nn.Sequential(*layers)
		self.mean_head = nn.Linear(last, out_dim)
		self.logvar_head = nn.Linear(last, out_dim)
		
		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
				nn.init.constant_(m.bias, 0.0)
		nn.init.constant_(self.logvar_head.bias, -1.0)

	def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		h = self.feature(x)
		mu = self.mean_head(h)
		logvar = self.logvar_head(h)
		logvar = torch.clamp(logvar, -10.0, 5.0)
		if mu.shape[-1] == 1:
			mu = mu.squeeze(-1)
			logvar = logvar.squeeze(-1)
		return mu, logvar


def gaussian_nll(mu: torch.Tensor, logvar: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
	var = torch.exp(logvar)
	return 0.5 * (math.log(2 * math.pi) + logvar + (y - mu) ** 2 / var)


def fgsm_attack(model: nn.Module, x: torch.Tensor, y: torch.Tensor, eps: float) -> torch.Tensor:
	x_adv = x.clone().detach().requires_grad_(True)
	mu, logvar = model(x_adv)
	loss = gaussian_nll(mu, logvar, y).mean()
	loss.backward()
	grad = x_adv.grad
	if grad is None:
		return x.detach()
	x_adv = x_adv + eps * grad.sign()
	return x_adv.detach()


class EnsembleRegressor:

	def __init__(self, M: int, in_dim: int, out_dim: int = 1, hidden: int = 128, nlayers: int = 2, device: Optional[torch.device] = None):
		self.M = M
		self.out_dim = out_dim
		self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
		self.models: List[BaseNet] = [BaseNet(in_dim, out_dim, hidden, nlayers).to(self.device) for _ in range(M)]
		self.optimizers: Optional[List[optim.Optimizer]] = None

	def parameters(self):
		params = []
		for m in self.models:
			params += list(m.parameters())
		return params

	def to(self, device: torch.device):
		for m in self.models:
			m.to(device)
		self.device = device

	def eval(self):
		for m in self.models:
			m.eval()

	def train(self):
		for m in self.models:
			m.train()

	def predict_per_model(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		x = x.to(self.device)
		mus = []
		vars_ = []
		for m in self.models:
			mu, logvar = m(x)
			mus.append(mu.unsqueeze(0))
			vars_.append(torch.exp(logvar).unsqueeze(0))
		mus = torch.cat(mus, dim=0)
		vars_ = torch.cat(vars_, dim=0)
		return mus, vars_

	def mixture_mean_var(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		mus, vars_ = self.predict_per_model(x)
		mean_ens = mus.mean(dim=0)
		var_ens = vars_.mean(dim=0) + (mus ** 2).mean(dim=0) - mean_ens ** 2
		var_ens = torch.clamp(var_ens, min=1e-6)
		return mean_ens, var_ens

	def sample_from_mixture(self, x: torch.Tensor, n_samples: int = 1, use_gaussian_approx: bool = False) -> torch.Tensor:
		x = x.to(self.device)
		batch = x.shape[0]
		if use_gaussian_approx:
			mean_ens, var_ens = self.mixture_mean_var(x)
			std = torch.sqrt(var_ens)
			if self.out_dim == 1:
				eps = torch.randn(n_samples, batch, device=self.device)
				return mean_ens.unsqueeze(0) + eps * std.unsqueeze(0)
			else:
				eps = torch.randn(n_samples, batch, self.out_dim, device=self.device)
				return mean_ens.unsqueeze(0) + eps * std.unsqueeze(0)

		mus, vars_ = self.predict_per_model(x)
		comps = torch.randint(0, self.M, size=(n_samples, batch), device=self.device)
		if self.out_dim == 1:
			out = torch.empty(n_samples, batch, device=self.device)
			for i in range(n_samples):
				idx = comps[i]
				mu_i = mus[idx, torch.arange(batch, device=self.device)]
				var_i = vars_[idx, torch.arange(batch, device=self.device)]
				std_i = torch.sqrt(var_i)
				out[i] = mu_i + std_i * torch.randn(batch, device=self.device)
		else:
			out = torch.empty(n_samples, batch, self.out_dim, device=self.device)
			for i in range(n_samples):
				idx = comps[i]
				mu_i = mus[idx, torch.arange(batch, device=self.device)]
				var_i = vars_[idx, torch.arange(batch, device=self.device)]
				std_i = torch.sqrt(var_i)
				out[i] = mu_i + std_i * torch.randn(batch, self.out_dim, device=self.device)
		return out

	def setup_optimizers(self, lr: float = 1e-3, weight_decay: float = 1e-5):
		self.optimizers = [optim.Adam(m.parameters(), lr=lr, weight_decay=weight_decay) for m in self.models]

	def train_batch(self, xb: torch.Tensor, yb: torch.Tensor, eps_adv: float = 0.0) -> float:
		if self.optimizers is None:
			raise RuntimeError("Optimizers not initialized. Call setup_optimizers() first.")

		xb = xb.to(self.device)
		yb = yb.to(self.device).float()

		total_loss = 0.0

		for i, m in enumerate(self.models):
			m.train()
			self.optimizers[i].zero_grad()

			mu, logvar = m(xb)
			loss = gaussian_nll(mu, logvar, yb).mean()

			if eps_adv > 0.0:
				xb_adv = fgsm_attack(m, xb, yb, eps_adv)
				mu2, logvar2 = m(xb_adv)
				loss = loss + gaussian_nll(mu2, logvar2, yb).mean()

			loss.backward()
			torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=1.0)
			self.optimizers[i].step()
			total_loss += loss.item()

		return total_loss


def train_ensemble(
	ensemble: EnsembleRegressor,
	x_train: torch.Tensor,
	y_train: torch.Tensor,
	epochs: int = 50,
	batch_size: int = 64,
	lr: float = 1e-3,
	eps_adv: float = 0.0,
	device: Optional[torch.device] = None,
):
	device = device or ensemble.device
	ensemble.to(device)

	dataset = TensorDataset(x_train, y_train)
	loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

	ensemble.setup_optimizers(lr=lr)

	try:
		y_var = float(torch.var(y_train).item())
	except Exception:
		y_var = 1.0
	init_logvar = math.log(max(1e-8, y_var))
	for m in ensemble.models:
		with torch.no_grad():
			if hasattr(m, 'logvar_head'):
				try:
					m.logvar_head.bias.data.fill_(init_logvar)
				except Exception:
					pass

	ensemble.train()

	for epoch in range(epochs):
		epoch_loss = 0.0
		n = 0
		for xb, yb in loader:
			batch_loss = ensemble.train_batch(xb, yb, eps_adv=eps_adv)
			epoch_loss += batch_loss
			n += ensemble.M


