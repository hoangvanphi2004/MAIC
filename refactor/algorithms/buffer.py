import random
import numpy as np
from collections import deque

class ReplayBuffer:
	def __init__(self, capacity):
		self.buffer = deque(maxlen=capacity)
	def push(self, obs, state, action, reward, next_obs, next_state, done):
		self.buffer.append((obs, state, action, reward, next_obs, next_state, done))
	def sample(self, batch_size):
		batch = random.sample(self.buffer, batch_size)
		obs, state, action, reward, next_obs, next_state, done = zip(*batch)
		return (np.array(obs), np.array(state), np.array(action), np.array(reward),
				np.array(next_obs), np.array(next_state), np.array(done))
	def __len__(self):
		return len(self.buffer)