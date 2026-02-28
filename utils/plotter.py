import warnings
import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

def plot_metrics(metrics, run_path, moving_average_func):
    if plt is None:
        warnings.warn('matplotlib not available; skipping plots')
        return
    try:
        fig, axes = plt.subplots(3, 3, figsize=(18, 12))
        axes_flat = axes.flatten()
        ax_reward, ax_policy, ax_entropy = axes_flat[0], axes_flat[1], axes_flat[2]
        ax_critic, ax_q, ax_alpha = axes_flat[3], axes_flat[4], axes_flat[5]
        ax_alpha1, ax_alpha2, ax_empty = axes_flat[6], axes_flat[7], axes_flat[8]
        
        rewards = metrics.get('episode_rewards', [])
        if rewards:
            x = np.arange(len(rewards))
            ax_reward.plot(x, rewards, color='tab:blue', alpha=0.35, label='Reward')
            ma = moving_average_func(np.array(rewards), window=20)
            if ma.size:
                ax_reward.plot(np.arange(len(ma)) + 19, ma, color='tab:orange', label='20-ep MA')
            ax_reward.set_title('Episode Reward')
            ax_reward.legend()
            ax_reward.grid(True, alpha=0.3)
        
        for k, ax, title, color in [
            ('actor_losses', ax_policy, 'Policy Loss', 'tab:blue'),
            ('critic_losses', ax_critic, 'Critic Loss', 'tab:blue'),
            ('q_values', ax_q, 'Q-value', 'tab:blue'),
            ('entropies', ax_entropy, 'Entropy', 'tab:blue'),
            ('alpha_values', ax_alpha, 'Alpha', 'purple'),
            ('alpha1_values', ax_alpha1, 'Alpha1 (Entropy)', 'red'),
            ('alpha2_values', ax_alpha2, 'Alpha2 (Info Bonus)', 'green'),
        ]:
            vals = metrics.get(k, [])
            if vals:
                ax.plot(vals, marker='o', ms=3, color=color)
                ax.set_title(title)
                ax.grid(True, alpha=0.3)
            else:
                ax.set_axis_off()
        
        ax_empty.set_axis_off()
        plt.tight_layout()
        out = run_path / 'metrics_grid.png'
        plt.savefig(str(out), dpi=150)
        plt.close(fig)
    except Exception as e:
        warnings.warn(f'Failed to save metric plots: {e}')
