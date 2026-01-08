"""Plotting helper for Bayesian_sampling demo output.

Loads `bayes_demo_out.json` produced by `Bayesian_sampling.py` and plots:
- training points (scatter)
- ensemble predictive mean (line)
- ±1 std shaded region
- several sample functions drawn from the predictive mixture

Saves plot to `bayes_demo_plot.png` and displays it.
"""
import json
import os
import matplotlib.pyplot as plt


def load_demo_json(path='bayes_demo_out.json'):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Demo output JSON not found at '{path}'. Run Bayesian_sampling.py first.")
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def plot_demo(data, out_png='bayes_demo_plot.png', n_samples_to_plot: int = 10):
    x = data['x']
    mean = data['mean']
    std = data['std']
    samples = data.get('samples', [])
    train_x = data.get('train_x', None)
    train_y = data.get('train_y', None)
    true_std = data.get('true_std', None)

    # ensure lists
    x = list(x)
    mean = list(mean)
    std = list(std)

    plt.figure(figsize=(10, 6))
    # plot samples
    if samples:
        nplot = min(n_samples_to_plot, len(samples))
        for i in range(nplot):
            plt.plot(x, samples[i], color='C2', alpha=0.5, linewidth=0.8)
    # mean
    plt.plot(x, mean, color='C0', lw=2, label='Ensemble mean')
    # std band
    upper = [m + s for m, s in zip(mean, std)]
    lower = [m - s for m, s in zip(mean, std)]
    plt.fill_between(x, lower, upper, color='C0', alpha=0.2, label='±1 std')

    # true noise std (if present)
    if true_std is not None:
        true_upper = [m + s for m, s in zip(mean, true_std)]
        true_lower = [m - s for m, s in zip(mean, true_std)]
        plt.fill_between(x, true_lower, true_upper, color='C1', alpha=0.12, label='True ±1 std')

    # training points
    if train_x is not None and train_y is not None:
        plt.scatter(train_x, train_y, s=10, color='k', alpha=0.6, label='Train data')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Ensemble predictive mean, std, and samples')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    print(f"Saved plot to {out_png}")
    plt.show()


if __name__ == '__main__':
    try:
        data = load_demo_json()
    except FileNotFoundError as e:
        print(e)
        raise
    plot_demo(data)
