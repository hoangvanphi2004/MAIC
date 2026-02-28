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