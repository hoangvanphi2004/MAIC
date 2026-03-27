import json
import matplotlib.pyplot as plt
import numpy as np

def load_rewards(filename, key='episode_rewards', n=2500):
	with open(filename, 'r') as f:
		data = json.load(f)
	rewards = data[key][:n]
	return rewards

coma_rewards = load_rewards('COMA_passspare.json')
maic_rewards = load_rewards('MAIC_passspare.json')
masac_rewards = load_rewards('MASAC_passspare.json')
qmix_rewards = load_rewards('QMIX_passspare.json')


# Compute running averages
window = 200
def running_avg(data, window):
	return np.convolve(data, np.ones(window)/window, mode='valid')

coma_avg = running_avg(coma_rewards, window)
maic_avg = running_avg(maic_rewards, window)
masac_avg = running_avg(masac_rewards, window)
qmix_avg = running_avg(qmix_rewards, window)

plt.figure(figsize=(12,6))
plt.plot(coma_avg, label='COMA (avg200)')
plt.plot(maic_avg, label='MAIC (avg200)')
plt.plot(masac_avg, label='MASAC (avg200)')
plt.plot(qmix_avg, label='QMIX (avg200)')
plt.xlabel('Episode')
plt.ylabel('Reward')
# plt.title('Running Average Reward (window=200) for 3000 Episodes (16x16)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('reward_plot.png')
plt.show()
