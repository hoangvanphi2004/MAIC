import numpy as np
import json

def moving_average(vals, window=20):
    if len(vals) < 1:
        return np.array([])
    window = max(1, min(window, len(vals)))
    cumsum = np.cumsum(np.insert(vals, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / float(window)

def save_rewards_json(rewards, path):
    with open(path, 'w') as f:
        json.dump({
            'episodes': list(range(len(rewards))),
            'rewards': rewards
        }, f, indent=2)

def update_metrics(metrics, replay_buffer, agent, batch_size, steps_per_update, updates_num, ep):
    if len(replay_buffer) > batch_size and (ep + 1) % steps_per_update == 0:
        cycle_critic_loss = []
        cycle_q = []
        cycle_alpha = []
        cycle_alpha1 = []
        cycle_alpha2 = []
        
        update_fn = getattr(agent, 'update_sac', None)
        if update_fn is None:
            update_fn = getattr(agent, 'update', None)
        if update_fn is None:
            update_fn = getattr(agent, 'update_value_net', None)

        for _ in range(updates_num):
            if update_fn:
                stats = update_fn(replay_buffer, batch_size)
                if stats:
                    cycle_critic_loss.append(stats.get('critic_loss', stats.get('value_loss', 0.0)))
                    cycle_q.append(stats.get('q_value', stats.get('loss', 0.0)))
                    if 'alpha_value' in stats: cycle_alpha.append(stats['alpha_value'])
                    if 'alpha1_value' in stats: cycle_alpha1.append(stats['alpha1_value'])
                    if 'alpha2_value' in stats: cycle_alpha2.append(stats['alpha2_value'])
        
        if cycle_critic_loss: metrics['critic_losses'].append(np.mean(cycle_critic_loss))
        if cycle_q: metrics['q_values'].append(np.mean(cycle_q))
        if cycle_alpha: metrics['alpha_values'].append(np.mean(cycle_alpha))
        if cycle_alpha1: metrics['alpha1_values'].append(np.mean(cycle_alpha1))
        if cycle_alpha2: metrics['alpha2_values'].append(np.mean(cycle_alpha2))
