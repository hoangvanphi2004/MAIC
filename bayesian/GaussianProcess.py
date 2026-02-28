import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
import numpy as np


class GPModel(ExactGP):
	def __init__(self, train_x, train_y, likelihood):
		super(GPModel, self).__init__(train_x, train_y, likelihood)
		self.mean_module = ConstantMean()
		self.covar_module = ScaleKernel(RBFKernel())

	def forward(self, x):
		mean_x = self.mean_module(x)
		covar_x = self.covar_module(x)
		return MultivariateNormal(mean_x, covar_x)
	
import gpytorch


class BatchedGPModel(gpytorch.models.ExactGP):
	def __init__(self, train_x, train_y, likelihood, output_dim):
		super().__init__(train_x, train_y, likelihood)

		batch_shape = torch.Size([output_dim])

		self.mean_module = gpytorch.means.ConstantMean(
			batch_shape=batch_shape
		)

		self.covar_module = gpytorch.kernels.ScaleKernel(
			gpytorch.kernels.RBFKernel(
				batch_shape=batch_shape
			),
			batch_shape=batch_shape
		)

	def forward(self, x):
		mean_x = self.mean_module(x)	 # (output_dim, N)
		covar_x = self.covar_module(x)   # (output_dim, N, N)
		return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class GaussianProcess:
	def __init__(self, output_dim: int, device=None):
		self.device = device or (
			torch.device("cuda") if torch.cuda.is_available()
			else torch.device("cpu")
		)

		# số chiều đầu ra (state_dim hoặc delta_state_dim)
		self.output_dim = output_dim

		# dùng chung cho toàn bộ batch GP
		self.likelihood = None
		self.model = None
		self.optimizer = None

	def fit(self, x_train, y_train, training_iter=50, lr=0.1):
		"""
		x_train: (N, input_dim)
		y_train: (N, output_dim)
		"""

		x_train = torch.tensor(x_train, dtype=torch.float32).to(self.device)
		y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)

		assert y_train.dim() == 2
		assert y_train.size(1) == self.output_dim

		# GPytorch batched GP yêu cầu shape (output_dim, N)
		y_train = y_train.transpose(0, 1)

		self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
			batch_shape=torch.Size([self.output_dim])
		).to(self.device)

		# noise floor để tránh numerical issue
		self.likelihood.noise_covar.register_constraint(
			"raw_noise", gpytorch.constraints.GreaterThan(1e-4)
		)

		self.model = BatchedGPModel(
			x_train,
			y_train,
			self.likelihood,
			output_dim=self.output_dim,
		).to(self.device)

		self.model.train()
		self.likelihood.train()

		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
		mll = gpytorch.mlls.ExactMarginalLogLikelihood(
			self.likelihood, self.model
		)

		for _ in range(training_iter):
			self.optimizer.zero_grad()
			output = self.model(x_train)
			loss = -mll(output, y_train)
			loss.backward()
			self.optimizer.step()

	def predict(self, x_test):
		"""
		x_test: (N, input_dim)

		return:
			mean: (N, output_dim)
			var:  (N, output_dim)
		"""
		is_torch = isinstance(x_test, torch.Tensor)
		if not is_torch:
			x_test = torch.tensor(x_test, dtype=torch.float32).to(self.device)
		else:
			x_test = x_test.to(self.device)

		self.model.eval()
		self.likelihood.eval()

		with torch.no_grad(), gpytorch.settings.fast_pred_var():
			pred = self.likelihood(self.model(x_test))
			mean = pred.mean		  # (output_dim, N)
			var = pred.variance	   # (output_dim, N)

		# numerical safety
		var = torch.clamp(var, min=1e-6)

		# đưa về (N, output_dim)
		mean = mean.transpose(0, 1)
		var = var.transpose(0, 1)

		if is_torch:
			return mean, var
		else:
			return mean.cpu().numpy(), var.cpu().numpy()
		
	def mixture_mean_var(self, x, return_decomposed: bool = False):
		"""
		x: (N, input_dim)

		return:
			mean: (N, output_dim)
			var:  (N, output_dim)

		nếu return_decomposed:
			std_total, std_ale, std_epi
		"""
		is_torch = isinstance(x, torch.Tensor)
		if not is_torch:
			x = torch.tensor(x, dtype=torch.float32).to(self.device)
		else:
			x = x.to(self.device)

		self.model.eval()
		self.likelihood.eval()

		with torch.no_grad(), gpytorch.settings.fast_pred_var():
			latent_pred = self.model(x)			   # f(x)
			obs_pred = self.likelihood(latent_pred)  # y = f(x) + noise

			mean = obs_pred.mean					 # (output_dim, N)

			var_total = obs_pred.variance			# total uncertainty
			var_epi = latent_pred.variance		   # epistemic
			var_ale = var_total - var_epi			# aleatoric

		# numerical safety
		var_total = torch.clamp(var_total, min=1e-6)
		var_epi = torch.clamp(var_epi, min=1e-6)
		var_ale = torch.clamp(var_ale, min=1e-6)

		# đưa về (N, output_dim)
		mean = mean.transpose(0, 1)
		var_total = var_total.transpose(0, 1)
		var_epi = var_epi.transpose(0, 1)
		var_ale = var_ale.transpose(0, 1)

		if return_decomposed:
			std_total = torch.sqrt(var_total)
			std_epi = torch.sqrt(var_epi)
			std_ale = torch.sqrt(var_ale)

			if is_torch:
				return mean, var_total, std_total, std_ale, std_epi
			else:
				return (
					mean.cpu().numpy(),
					var_total.cpu().numpy(),
					std_total.cpu().numpy(),
					std_ale.cpu().numpy(),
					std_epi.cpu().numpy(),
				)

		if is_torch:
			return mean, var_total
		else:
			return mean.cpu().numpy(), var_total.cpu().numpy()

	def sample(self, x_test, n_samples=1, mode="observation"):
		"""
		x_test: (N, input_dim)

		n_samples: số particle
		mode:
			- "observation": y = f(x) + noise  (dynamics rollout, imagination)
			- "latent":	   y = f(x)		 (chỉ epistemic, MPC / risk-sensitive)

		return:
			samples: (n_samples, N, output_dim)
		"""
		x_test = torch.tensor(x_test, dtype=torch.float32).to(self.device)

		self.model.eval()
		self.likelihood.eval()

		with torch.no_grad():
			if mode == "latent":
				# Posterior của f(x) → epistemic uncertainty
				latent_dist = self.model(x_test)
				samples = latent_dist.sample(
					sample_shape=torch.Size([n_samples])
				)  # (n_samples, output_dim, N)

			elif mode == "observation":
				# Predictive distribution y = f(x) + noise
				obs_dist = self.likelihood(self.model(x_test))
				samples = obs_dist.sample(
					sample_shape=torch.Size([n_samples])
				)  # (n_samples, output_dim, N)

			else:
				raise ValueError("mode must be 'latent' or 'observation'")

		# numerical safety
		samples = torch.nan_to_num(samples)

		# đưa về (n_samples, N, output_dim)
		samples = samples.permute(0, 2, 1)

		return samples.cpu().numpy()

