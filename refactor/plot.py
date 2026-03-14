import numpy as np
import matplotlib.pyplot as plt


def moving_average(vals, window=20):
	"""Calculate moving average of values."""
	if len(vals) < 1:
		return np.array([])
	window = max(1, min(window, len(vals)))
	cumsum = np.cumsum(np.insert(vals, 0, 0))
	return (cumsum[window:] - cumsum[:-window]) / float(window)


def save_plots(metrics, run_path, ep):
	"""Generate and save metric plots.
	
	Args:
		metrics: Dictionary containing training metrics
		run_path: Path object to save directory
		ep: Episode number for filename
	"""
	fig, axes = plt.subplots(3, 3, figsize=(18, 10))
	(ax_reward, ax_policy, ax_info), (ax_critic, ax_q, ax_entropy), (ax_alpha1_alpha2, ax_unique_states, _) = axes

	# Plot episode rewards
	rewards = metrics.get('episode_rewards', [])
	if rewards:
		x = np.arange(len(rewards))
		ax_reward.plot(x, rewards, color='tab:blue', alpha=0.35, label='Episode Reward')
		ma = moving_average(np.array(rewards), window=20)
		if ma.size:
			ax_reward.plot(np.arange(len(ma)) + 19, ma, color='tab:orange', label='20-ep MA')
		ax_reward.set_title('Episode Reward')
		ax_reward.legend()
		ax_reward.grid(True, alpha=0.3)

	# Plot policy losses
	policy_losses = metrics.get('actor_losses', [])
	if policy_losses:
		ax_policy.plot(policy_losses, marker='o', ms=3)
		ax_policy.set_title('Policy Loss (avg per log interval)')
		ax_policy.grid(True, alpha=0.3)

	# Plot critic losses
	critic_losses = metrics.get('critic_losses', [])
	if critic_losses:
		ax_critic.plot(critic_losses, marker='o', ms=3)
		ax_critic.set_title('Critic Loss (avg per log interval)')
		ax_critic.grid(True, alpha=0.3)

	# Plot Q-values
	q_values = metrics.get('q_values', [])
	if q_values:
		ax_q.plot(q_values, marker='o', ms=3)
		ax_q.set_title('Q-value (avg per log interval)')
		ax_q.grid(True, alpha=0.3)

	# Plot entropy
	entropy_vals = metrics.get('entropies', [])
	if entropy_vals:
		ax_entropy.plot(entropy_vals, marker='o', ms=3)
		ax_entropy.set_title('Entropy (avg per episode)')
		ax_entropy.grid(True, alpha=0.3)
	else:
		ax_entropy.set_axis_off()

	# Plot information gains
	info_gains = metrics.get('information_gains', [])
	if info_gains:
		ax_info.plot(info_gains, marker='o', ms=3, color='tab:green')
		ax_info.set_title('Information Gain (avg per log interval)')
		ax_info.grid(True, alpha=0.3)
	else:
		ax_info.set_axis_off()

	# Plot alpha values
	alpha1_vals = metrics.get('alpha1_values', [])
	alpha2_vals = metrics.get('alpha2_values', [])
	if alpha1_vals or alpha2_vals:
		if alpha1_vals:
			ax_alpha1_alpha2.plot(alpha1_vals, marker='o', ms=3, label='alpha1', color='tab:blue')
		if alpha2_vals:
			ax_alpha1_alpha2.plot(alpha2_vals, marker='o', ms=3, label='alpha2', color='tab:orange')
		ax_alpha1_alpha2.set_title('Alpha1 & Alpha2')
		ax_alpha1_alpha2.legend()
		ax_alpha1_alpha2.grid(True, alpha=0.3)
	else:
		ax_alpha1_alpha2.set_axis_off()

	# Plot cumulative unique states encountered over episodes
	unique_states = metrics.get('unique_states_seen', [])
	if unique_states:
		ax_unique_states.plot(unique_states, marker='o', ms=3, color='tab:red')
		ax_unique_states.set_title('Unique States Seen (cumulative)')
		ax_unique_states.grid(True, alpha=0.3)
	else:
		ax_unique_states.set_axis_off()

	plt.tight_layout()
	out = run_path / f'metrics.png'
	plt.savefig(str(out), dpi=150)
	plt.close(fig)
	print('Saved metric plots to', out)
