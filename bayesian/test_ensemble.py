import torch
import matplotlib.pyplot as plt
import numpy as np
from Bayesian_sampling import EnsembleRegressor, train_ensemble

torch.manual_seed(42)
np.random.seed(42)

in_dim = 5
out_dim = 3
M = 3
N_train = 200

x_train = torch.randn(N_train, in_dim)
y_train = torch.zeros(N_train, out_dim)
for i in range(N_train):
	y_train[i, 0] = torch.sin(x_train[i, 0]) + 0.1 * torch.randn(1)
	y_train[i, 1] = torch.cos(x_train[i, 1]) * x_train[i, 2] + 0.1 * torch.randn(1)
	y_train[i, 2] = x_train[i, 3] ** 2 - x_train[i, 4] + 0.1 * torch.randn(1)

ens = EnsembleRegressor(M=M, in_dim=in_dim, out_dim=out_dim, hidden=128, nlayers=3)
print(f"Ensemble: M={M}, in_dim={in_dim}, out_dim={out_dim}")

train_ensemble(ens, x_train, y_train, epochs=50, batch_size=64, lr=1e-3)

ens.eval()
x_test_1d = torch.linspace(-3, 3, 50)
x_test = torch.zeros(50, in_dim)
x_test[:, 0] = x_test_1d
x_test[:, 1:] = torch.randn(50, in_dim-1)
x_test = x_test.to(ens.device)

y_test_1d = torch.sin(x_test_1d)
print(f"\nTest input shape: {x_test.shape}")

mean, var = ens.mixture_mean_var(x_test)
print(f"Prediction mean shape: {mean.shape}")
print(f"Prediction var shape: {var.shape}")

samples = ens.sample_from_mixture(x_test, n_samples=10, use_gaussian_approx=False)
print(f"Samples shape: {samples.shape}")
print("✓ Multi-dimensional ensemble works!")

plt.style.use('seaborn-v0_8-darkgrid')
fig, axes = plt.subplots(1, out_dim, figsize=(16, 5))
if out_dim == 1:
	axes = [axes]

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

for d in range(out_dim):
	ax = axes[d]
	x_np = x_test[:, 0].detach().cpu().numpy()
	mean_np = mean[:, d].detach().cpu().numpy()
	std_np = torch.sqrt(var[:, d]).detach().cpu().numpy()
	
	ax.fill_between(x_np, mean_np - std_np, mean_np + std_np, 
					alpha=0.3, color=colors[d], label='±1 Std')
	ax.plot(x_np, mean_np, 'o-', linewidth=2, markersize=6, 
			color=colors[d], label='Ensemble Mean')
	
	for s in range(min(5, samples.shape[0])):
		sample_np = samples[s, :, d].detach().cpu().numpy()
		ax.plot(x_np, sample_np, '--', alpha=0.4, linewidth=1, color='gray')
	
	ax.set_title(f'Output Dimension {d+1}', fontsize=12, fontweight='bold')
	ax.set_xlabel('Input Feature', fontsize=11)
	ax.set_ylabel('Output Value', fontsize=11)
	ax.legend(loc='best', fontsize=10)
	ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ensemble_predictions.png')
plt.show()
print("Plot saved to ensemble_predictions.png")
