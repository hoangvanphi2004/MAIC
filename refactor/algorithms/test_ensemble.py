import torch
import numpy as np
import matplotlib.pyplot as plt
from ensemble import EnsembleRegressor, train_ensemble

def synthetic_data(n=200):
    # y = sin(x) + noise
    x = np.linspace(-4, 4, n).reshape(-1, 1)
    y = np.sin(x) + 0.2 * np.random.randn(n, 1)
    return x.astype(np.float32), y.astype(np.float32)

def main():
    # Generate synthetic data
    x_train, y_train = synthetic_data(200)
    x_test = np.linspace(-5, 5, 200).reshape(-1, 1).astype(np.float32)
    x_train_t = torch.from_numpy(x_train)
    y_train_t = torch.from_numpy(y_train)
    x_test_t = torch.from_numpy(x_test)
    print("Data generated.")
    # Create and train ensemble
    ens = EnsembleRegressor(M=5, in_dim=1, out_dim=1, hidden=128, nlayers=2)
    train_ensemble(ens, x_train_t, y_train_t, epochs=100, batch_size=32, lr=3e-4)
    print("Ensemble trained.")

    # Predict on test set
    ens.eval()
    with torch.no_grad():
        mean, var, std_total, std_ale, std_epi = ens.mixture_mean_var(x_test_t, return_decomposed=True)
        mean = mean.cpu().numpy().squeeze()
        std_total = std_total.cpu().numpy().squeeze()
        std_ale = std_ale.cpu().numpy().squeeze()
        std_epi = std_epi.cpu().numpy().squeeze()

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_train, y_train, 'kx', alpha=0.2, label='Train data')
    plt.plot(x_test, np.sin(x_test), 'g--', label='True function')
    plt.plot(x_test, mean, 'b', label='Ensemble mean')
    plt.fill_between(x_test.squeeze(), mean - std_total, mean + std_total, color='b', alpha=0.2, label='Total std')
    plt.fill_between(x_test.squeeze(), mean - std_ale, mean + std_ale, color='r', alpha=0.2, label='Aleatoric std')
    plt.fill_between(x_test.squeeze(), mean - std_epi, mean + std_epi, color='g', alpha=0.2, label='Epistemic std')
    plt.legend()
    plt.title('Ensemble Regression: Fit, Aleatoric, Epistemic Uncertainty')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
