import numpy as np
import json


class TrainingMetrics:
	def __init__(self):
		self.data = {
			'episode_rewards': [],
			'episode_lengths': [],
			'unique_states_seen': [],
			'critic_losses': [],
			'actor_losses': [],
			'entropies': [],
			'q_values': [],
			'alpha1_losses': [],
			'alpha2_losses': [],
			'alpha1_values': [],
			'alpha2_values': [],
			'information_gains': [],
			'kl_divergences': [],
		}

	def __getitem__(self, key):
		return self.data[key]

	def _mean_or_none(self, values):
		if not values:
			return None
		return float(np.mean(values))

	def _fmt_metric(self, value, precision=4):
		if value is None:
			return 'n/a'
		return f'{value:.{precision}f}'

	def _mean_from_stats(self, stats_list, key):
		values = [s[key] for s in stats_list if key in s and s[key] is not None]
		return self._mean_or_none(values)

	def append_episode(self, ep_reward, ep_len, ep_sac_stats, unique_states_seen=None):
		self.data['episode_rewards'].append(ep_reward)
		self.data['episode_lengths'].append(ep_len)
		self.data['unique_states_seen'].append(unique_states_seen)
		self.data['actor_losses'].append(self._mean_from_stats(ep_sac_stats, 'policy_loss'))
		self.data['critic_losses'].append(self._mean_from_stats(ep_sac_stats, 'critic_loss'))
		self.data['alpha1_losses'].append(self._mean_from_stats(ep_sac_stats, 'alpha1_loss'))
		self.data['alpha2_losses'].append(self._mean_from_stats(ep_sac_stats, 'alpha2_loss'))
		self.data['q_values'].append(self._mean_from_stats(ep_sac_stats, 'q_value'))
		self.data['entropies'].append(self._mean_from_stats(ep_sac_stats, 'entropy'))
		self.data['alpha1_values'].append(self._mean_from_stats(ep_sac_stats, 'alpha1_value'))
		self.data['alpha2_values'].append(self._mean_from_stats(ep_sac_stats, 'alpha2_value'))
		self.data['information_gains'].append(self._mean_from_stats(ep_sac_stats, 'information_gain'))
		self.data['kl_divergences'].append(self._mean_from_stats(ep_sac_stats, 'kl_divergence'))

	def as_dict(self):
		return self.data

	def to_serializable_dict(self, episode=None):
		payload = {}
		for key, values in self.data.items():
			if isinstance(values, np.ndarray):
				values = values.tolist()
			if isinstance(values, list):
				cleaned = []
				for v in values:
					if isinstance(v, (np.floating, np.integer)):
						cleaned.append(v.item())
					elif isinstance(v, np.ndarray):
						cleaned.append(v.tolist())
					else:
						cleaned.append(v)
				payload[key] = cleaned
			else:
				if isinstance(values, (np.floating, np.integer)):
					payload[key] = values.item()
				elif isinstance(values, np.ndarray):
					payload[key] = values.tolist()
				else:
					payload[key] = values
		if episode is not None:
			payload['episode'] = int(episode)
		return payload

	def save_json(self, run_path, episode, filename='metrics_latest.json'):
		out_path = run_path / filename
		payload = self.to_serializable_dict(episode=episode)
		with open(out_path, 'w', encoding='utf-8') as f:
			json.dump(payload, f, indent=2)
		print('Saved metrics JSON to', out_path)

	def save_rewards_json(self, run_path, filename='episode_rewards.json'):
		out_path = run_path / filename
		rewards = self.data['episode_rewards']
		payload = {
			'episodes': list(range(len(rewards))),
			'rewards': rewards,
		}
		with open(out_path, 'w', encoding='utf-8') as f:
			json.dump(payload, f, indent=2)
		print('Rewards saved to', out_path)

	def _recent_non_none(self, key, window=10):
		return [x for x in self.data[key][-window:] if x is not None]

	def print_episode_metrics(self, ep_idx, ep_reward, ep_len, elapsed, alpha_kl_value, buffer_size, total_steps):
		recent_rewards = self.data['episode_rewards'][-10:]
		recent_lengths = self.data['episode_lengths'][-10:]
		recent_actor = self._recent_non_none('actor_losses')
		recent_critic = self._recent_non_none('critic_losses')
		recent_alpha1_loss = self._recent_non_none('alpha1_losses')
		recent_alpha2_loss = self._recent_non_none('alpha2_losses')
		recent_q = self._recent_non_none('q_values')
		recent_entropy = self._recent_non_none('entropies')
		recent_alpha1 = self._recent_non_none('alpha1_values')
		recent_alpha2 = self._recent_non_none('alpha2_values')
		recent_kl = self._recent_non_none('kl_divergences')

		avg_recent_reward = float(np.mean(recent_rewards)) if recent_rewards else 0.0
		avg_recent_len = float(np.mean(recent_lengths)) if recent_lengths else 0.0
		ep_num = ep_idx + 1

		print(f'Reward: {ep_reward:.3f} (Avg 10: {avg_recent_reward:.3f}) | Length: {ep_len} (Avg 10: {avg_recent_len:.1f})')
		print(
			f'{elapsed:.1f}s\t{ep_num}\t'
			f'Actor Loss: {self._fmt_metric(self._mean_or_none(recent_actor), 6)} | '
			f'Critic Loss: {self._fmt_metric(self._mean_or_none(recent_critic), 6)} | '
			f'Alpha1 Loss: {self._fmt_metric(self._mean_or_none(recent_alpha1_loss), 6)} | '
			f'Alpha2 Loss: {self._fmt_metric(self._mean_or_none(recent_alpha2_loss), 6)}'
		)
		print(
			f'{elapsed:.1f}s\t{ep_num}\t'
			f'Q-Value: {self._fmt_metric(self._mean_or_none(recent_q), 4)} | '
			f'Entropy: {self._fmt_metric(self._mean_or_none(recent_entropy), 4)} | '
			f'KL: {self._fmt_metric(self._mean_or_none(recent_kl), 6)} | '
			f'Alpha1: {self._fmt_metric(self._mean_or_none(recent_alpha1), 4)} | '
			f'Alpha2: {self._fmt_metric(self._mean_or_none(recent_alpha2), 4)} | '
			f'AlphaKL: {alpha_kl_value:.4f}'
		)
		print(f'{elapsed:.1f}s\t{ep_num}\tBuffer Size: {buffer_size} | Total Steps: {total_steps}')
