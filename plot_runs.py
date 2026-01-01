import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
def load_json_files(runs_dir='runs'):
    """Load all episode_rewards.json files from runs directory"""
    data = {}
    runs_path = Path(runs_dir)
    for subdir in runs_path.iterdir():
        if subdir.is_dir():
            json_file = subdir / 'episode_rewards.json'
            if json_file.exists():
                try:
                    with open(json_file, 'r') as f:
                        data[subdir.name] = json.load(f)
                    print(f"Loaded: {subdir.name}/episode_rewards.json")
                except Exception as e:
                    print(f"Error loading {json_file}: {e}")
    return data
def running_mean_asym_std(values, window):
    """Compute running mean and asymmetric std (upper/lower) per window.
    Upper std uses only positive residuals; lower std uses only negative residuals.
    """
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return None, None, None
    w = max(1, min(window, len(arr)))
    cumsum = np.cumsum(np.insert(arr, 0, 0.0))
    mean = (cumsum[w:] - cumsum[:-w]) / w
    upper_std = np.empty_like(mean)
    lower_std = np.empty_like(mean)
    for i in range(len(mean)):
        win = arr[i:i+w]
        mu = mean[i]
        resid = win - mu
        pos = resid[resid > 0]
        neg = resid[resid < 0]
        if pos.size > 0:
            upper_std[i] = np.sqrt(np.mean(pos**2))
        else:
            upper_std[i] = 0.0
        if neg.size > 0:
            lower_std[i] = np.sqrt(np.mean(neg**2))
        else:
            lower_std[i] = 0.0
    return mean, upper_std, lower_std
def plot_rewards(data, window=50, save_path='runs_comparison.png'):
    """Plot running average with std band for each method"""
    plt.figure(figsize=(12, 6))
    for method_name, method_data in data.items():
        display_name = method_name.split('_', 1)[0]
        episodes = method_data.get('episodes', [])
        rewards = method_data.get('rewards', [])
        if len(episodes) > 0 and len(rewards) > 0:
            mean, upper_std, lower_std = running_mean_asym_std(rewards, window)
            if mean is None:
                continue
            stats_episodes = episodes[window-1:][:len(mean)]
            upper = mean + upper_std
            lower = mean - lower_std
            plt.plot(stats_episodes, mean, linewidth=2, label=f'{display_name.upper()} mean')
            plt.fill_between(stats_episodes, lower, upper, alpha=0.15, label=f'{display_name.upper()} band')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    plt.show()
def plot_individual_methods(data, window=50):
    """Create separate plots for each method with running average and std"""
    n_methods = len(data)
    if n_methods == 0:
        print("No data to plot")
        return
    fig, axes = plt.subplots(n_methods, 1, figsize=(12, 4*n_methods))
    if n_methods == 1:
        axes = [axes]
    for idx, (method_name, method_data) in enumerate(data.items()):
        display_name = method_name.split('_', 1)[0]
        episodes = method_data.get('episodes', [])
        rewards = method_data.get('rewards', [])
        if len(episodes) > 0 and len(rewards) > 0:
            mean, upper_std, lower_std = running_mean_asym_std(rewards, window)
            if mean is None:
                continue
            stats_episodes = episodes[window-1:][:len(mean)]
            ax = axes[idx]
            upper = mean + upper_std
            lower = mean - lower_std
            ax.plot(stats_episodes, mean, linewidth=2, color='red', label=f'{display_name.upper()} mean (window={window})')
            ax.fill_between(stats_episodes, lower, upper, alpha=0.15, color='blue', label=f'{display_name.upper()} band')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Reward')
            ax.legend()
            ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('runs_individual.png', dpi=300, bbox_inches='tight')
    print(f"Individual plots saved to: runs_individual.png")
    plt.show()
def print_statistics(data, running_window=100):
    """Print std and running average for each method"""
    print("\n" + "="*70)
    print("Statistics Summary")
    print("="*70)
    for method_name, method_data in data.items():
        display_name = method_name.split('_', 1)[0]
        rewards = method_data.get('rewards', [])
        if len(rewards) > 0:
            window = min(running_window, len(rewards))
            running_avg = np.mean(rewards[-window:])
            print(f"\n{display_name.upper()}:")
            print(f"  Std reward: {np.std(rewards):.4f}")
            print(f"  Running average (last {window} episodes): {running_avg:.4f}")
if __name__ == '__main__':
    data = load_json_files('runs')
    if not data:
        print("No JSON files found in runs directory!")
    else:
        print_statistics(data)
        print("\nCreating comparison plot...")
        plot_rewards(data, window=200)
        print("\nDone!")