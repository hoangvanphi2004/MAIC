import warnings
import numpy as np
import torch
from torch.distributions import Categorical

from mac.information_bonus import information_bonus

try:
	import matplotlib.pyplot as plt
except ImportError:
	plt = None

def plot_metrics(metrics, run_path, moving_average_func):
	if plt is None:
		warnings.warn('matplotlib not available; skipping plots')
		return
	try:
		fig, axes = plt.subplots(4, 3, figsize=(18, 16))
		axes_flat = axes.flatten()
		ax_reward, ax_policy, ax_entropy = axes_flat[0], axes_flat[1], axes_flat[2]
		ax_critic, ax_q, ax_info_gain = axes_flat[3], axes_flat[4], axes_flat[5]
		ax_alpha, ax_alpha1, ax_alpha2 = axes_flat[6], axes_flat[7], axes_flat[8]
		ax_alpha1_loss, ax_alpha2_loss, ax_empty = axes_flat[9], axes_flat[10], axes_flat[11]
		
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
			('information_gains', ax_info_gain, 'Information Gain', 'tab:blue'),
			('alpha_values', ax_alpha, 'Alpha', 'purple'),
			('alpha1_values', ax_alpha1, 'Alpha1 Value (Entropy)', 'red'),
			('alpha2_values', ax_alpha2, 'Alpha2 Value (Info)', 'green'),
			('alpha1_losses', ax_alpha1_loss, 'Alpha1 Loss', 'red'),
			('alpha2_losses', ax_alpha2_loss, 'Alpha2 Loss', 'green'),
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

def plot_q_heatmaps(agent, env, ep, run_path):
	if plt is None:
		return
	if agent.num_agents != 1:
		return
		
	env_unwrapped = getattr(env, 'unwrapped', env)
	width = env_unwrapped.grid.width
	height = env_unwrapped.grid.height
	n_actions = agent.n_actions
		
	# Directions: 0: Right, 1: Down, 2: Left, 3: Up
	dir_names = ['Right', 'Down', 'Left', 'Up']
		
	fig_q, axes_q = plt.subplots(2, 2, figsize=(10, 10))
	fig_entropy, axes_entropy = plt.subplots(2, 2, figsize=(10, 10))
	fig_info, axes_info = plt.subplots(2, 2, figsize=(10, 10))
	axes_q_flat = axes_q.flatten()
	axes_entropy_flat = axes_entropy.flatten()
	axes_info_flat = axes_info.flatten()
		
	# Save original state
	orig_pos = env_unwrapped.agents[0].state.pos
	orig_dir = env_unwrapped.agents[0].state.dir
	orig_terminated = env_unwrapped.agents[0].state.terminated
	env_unwrapped.agents[0].state.terminated = False
		
	for d in range(4):
		q_max_grid = np.full((height, width), np.nan)
		entropy_grid = np.full((height, width), np.nan)
		info_gain_grid = np.full((height, width), np.nan)
		for y in range(height):
			for x in range(width):
				cell = env_unwrapped.grid.get(x, y)
				if cell is not None and cell.type == 'wall':
					continue
					
				env_unwrapped.agents[0].state.pos = (x, y)
				env_unwrapped.agents[0].state.dir = d
				
				# Get state
				state = env_unwrapped.get_state()
				obs_dict = env_unwrapped.gen_obs()
				obs_img = np.array(obs_dict[0]['image'], dtype=np.float32)
				obs_batch = np.expand_dims(obs_img, 0)
				state_batch = np.expand_dims(state, 0)
				obs_tensor = torch.FloatTensor(obs_batch).permute(0, 3, 1, 2).to(agent.device)
				state_tensor = torch.FloatTensor(state_batch).permute(0, 3, 1, 2).to(agent.device)
				
				max_q = -1e9
				info_gain_per_action = np.zeros(n_actions, dtype=np.float32)
				with torch.no_grad():
					action_probs = agent.actors[0].forward(obs_tensor).squeeze(0)
					action_probs = action_probs + 1e-8
					action_probs = action_probs / action_probs.sum()
					entropy_grid[y, x] = Categorical(action_probs).entropy().item()

					for a in range(n_actions):
						if agent.critic_type == 'centralized':
							action_oh = torch.zeros((1, 1, n_actions)).to(agent.device)
							action_oh[0, 0, a] = 1.0
							q1, q2 = agent.critic(state_tensor, action_oh)
						else:
							action_oh = torch.zeros((1, n_actions)).to(agent.device)
							action_oh[0, a] = 1.0
							print(f"State tensor shape: {state_tensor.shape}, Action OH shape: {action_oh.shape}")
							q1, q2 = agent.critics[0](state_tensor, action_oh)
							
						q = torch.min(q1, q2).item()
						if q > max_q:
							max_q = q

						info_gain_a = information_bonus(agent, state_tensor, action_oh).item()
						info_gain_per_action[a] = info_gain_a
							
				q_max_grid[y, x] = max_q
				info_gain_grid[y, x] = float((action_probs.detach().cpu().numpy() * info_gain_per_action).sum())

		ax_q = axes_q_flat[d]
		im_q = ax_q.imshow(q_max_grid, cmap='hot', interpolation='nearest')
		ax_q.set_title(f'ep_{ep} - Q max ({dir_names[d]})')
		ax_q.set_xlabel('x')
		ax_q.set_ylabel('y')
		fig_q.colorbar(im_q, ax=ax_q)

		ax_entropy = axes_entropy_flat[d]
		im_entropy = ax_entropy.imshow(entropy_grid, cmap='viridis', interpolation='nearest')
		ax_entropy.set_title(f'ep_{ep} - Action Entropy ({dir_names[d]})')
		ax_entropy.set_xlabel('x')
		ax_entropy.set_ylabel('y')
		fig_entropy.colorbar(im_entropy, ax=ax_entropy)

		ax_info = axes_info_flat[d]
		im_info = ax_info.imshow(info_gain_grid, cmap='magma', interpolation='nearest')
		ax_info.set_title(f'ep_{ep} - Info Gain ({dir_names[d]})')
		ax_info.set_xlabel('x')
		ax_info.set_ylabel('y')
		fig_info.colorbar(im_info, ax=ax_info)
		
		# Restore original state
	env_unwrapped.agents[0].state.pos = orig_pos
	env_unwrapped.agents[0].state.dir = orig_dir
	env_unwrapped.agents[0].state.terminated = orig_terminated
		
	fig_q.tight_layout()
	fig_entropy.tight_layout()
	fig_info.tight_layout()

	q_path = run_path / f'q_heatmaps.png'
	entropy_path = run_path / f'entropy_heatmaps.png'
	info_path = run_path / f'info_gain_heatmaps.png'

	fig_q.savefig(str(q_path), dpi=150)
	fig_entropy.savefig(str(entropy_path), dpi=150)
	fig_info.savefig(str(info_path), dpi=150)
	plt.close(fig_q)
	plt.close(fig_entropy)
	plt.close(fig_info)
